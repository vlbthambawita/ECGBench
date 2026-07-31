"""
Unified PyTorch Dataset for loading any ECG dataset supported by ECGBench.

Uses the dataset's YAML config to determine how to load signals and metadata.
Adding a new dataset requires only a config file — no changes to this class.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError as _torch_err:
    raise ImportError(
        "PyTorch is required for ECGDataset. "
        "Install with: pip install ecgbench[torch]"
    ) from _torch_err

logger = logging.getLogger(__name__)


def _require_wfdb():
    """Lazily import wfdb."""
    try:
        import wfdb

        return wfdb
    except ImportError:
        raise ImportError(
            "wfdb is required to load ECG data. "
            "Install with: pip install ecgbench[torch]"
        )


class WindowOutOfRangeError(ValueError):
    """The requested sample window does not fit inside the record."""


def _window_error(
    record_path: str, start: int, length: int | None, available: int | None
) -> WindowOutOfRangeError:
    """Build one message for both formats, naming the record and its true length.

    wfdb's own error ("sampto must be shorter than the signal length") names
    neither the record nor how long it actually is, and the csv reader does not
    complain at all — it just returns fewer samples.
    """
    wanted = "to the end" if length is None else f"{length} samples"
    have = "unknown" if available is None else f"{available} samples"
    return WindowOutOfRangeError(
        f"window start={start} ({wanted}) does not fit record {record_path!r}, "
        f"which has {have}. Reduce the window, or drop records shorter than it — "
        "record length varies within several ECGBench datasets."
    )


def _record_length(record_path: str, signal_format: str) -> int | None:
    """Best-effort record length, used only to build a good error message."""
    if signal_format == "wfdb":
        try:
            import wfdb

            return int(wfdb.rdheader(record_path).sig_len)
        except Exception:
            return None
    return None


def _load_signal(
    record_path: str,
    signal_format: str,
    unit_scale: float = 1.0,
    window: tuple[int, int | None] | None = None,
) -> np.ndarray:
    """Load ECG signal from file. Returns shape (leads, samples) in millivolts.

    ``window`` is ``(start, length)`` in samples, with ``length=None`` meaning
    "to the end of the record". It is pushed down into the reader rather than
    applied as a slice afterwards, so only the requested samples are decoded —
    worth 13x on 30-minute records and nothing on 10-second ones.
    """
    start, length = window if window is not None else (0, None)
    sampto = None if length is None else start + length

    if signal_format == "wfdb":
        import wfdb

        try:
            record = wfdb.rdrecord(record_path, sampfrom=start, sampto=sampto)
        except ValueError as e:
            if window is None:
                raise
            raise _window_error(
                record_path, start, length, _record_length(record_path, signal_format)
            ) from e
        if record.p_signal is None:
            raise ValueError(f"Signal is None for record: {record_path}")
        signal = record.p_signal.T.astype(np.float32)
    elif signal_format == "csv":
        # One column per lead, one row per sample, with a header row naming the
        # leads — transposed relative to what ECGBench returns. Rows are samples,
        # so the window pushes down to skiprows/max_rows.
        signal = np.loadtxt(
            record_path,
            delimiter=",",
            skiprows=1 + start,
            max_rows=length,
            dtype=np.float32,
            ndmin=2,
        ).T
        # Unlike wfdb, loadtxt returns a short array instead of raising.
        if window is not None and (
            signal.size == 0 or (length is not None and signal.shape[1] != length)
        ):
            available = start + (0 if signal.size == 0 else signal.shape[1])
            raise _window_error(record_path, start, length, available)
    else:
        raise NotImplementedError(
            f"Signal format '{signal_format}' not yet supported. "
            "Currently supported: wfdb, csv"
        )

    if unit_scale != 1.0:
        signal = signal * np.float32(unit_scale)
    return signal


def _parse_dict_string(value: str) -> dict | str:
    """Try to parse a Python dict literal string."""
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


#: Output units the loader can produce. Signals reach here in millivolts, because
#: signal_unit_scale has already normalised whatever the source stored.
_UNIT_FACTORS = {"mv": 1.0, "uv": 1000.0, "\u00b5v": 1000.0}


def _resolve_units(units: str) -> float:
    factor = _UNIT_FACTORS.get(str(units).strip().lower())
    if factor is None:
        raise ValueError(
            f"units must be one of 'mV', 'uV' (or '\u00b5V'), got {units!r}"
        )
    return factor


def _resolve_window(window: Any) -> tuple[int, int | None] | None:
    """Validate a ``(start, length)`` sample window.

    ``length=None`` means "to the end of the record". Validated once here rather
    than per ``__getitem__``, so a typo fails at construction instead of on the
    first batch.
    """
    if window is None:
        return None
    if isinstance(window, int):
        raise TypeError(
            f"window must be a (start, length) tuple, got a bare int {window!r}. "
            f"Use (0, {window}) for the first {window} samples."
        )
    try:
        start, length = window
    except (TypeError, ValueError):
        raise ValueError(
            f"window must be a (start, length) tuple of 2 items, got {window!r}"
        ) from None
    start = int(start)
    if start < 0:
        raise ValueError(f"window start must be >= 0, got {start}")
    if length is not None:
        length = int(length)
        if length <= 0:
            raise ValueError(f"window length must be > 0 or None, got {length}")
    return start, length


def _resolve_leads(
    requested: list[str], available: list[str] | None, slug: str
) -> tuple[list[int], list[str]]:
    """Map requested lead names to row indices in the stored signal.

    Matching is case-insensitive because the catalogue is not consistent: PTB-XL
    spells the augmented leads AVR/AVL/AVF, everything else aVR/aVL/aVF. Names
    rather than indices are the whole point — MIMIC-IV-ECG stores aVF and aVL
    transposed, so row 4 is a different physical lead there than elsewhere.
    """
    if not available:
        raise ValueError(
            f"Config '{slug}' does not declare lead_names, so leads= cannot be "
            "resolved. Add lead_names to the dataset's YAML, in the order the "
            "files store them."
        )

    lookup: dict[str, int] = {}
    for position, name in enumerate(available):
        lookup.setdefault(str(name).strip().lower(), position)

    indices: list[int] = []
    resolved: list[str] = []
    for name in requested:
        key = str(name).strip().lower()
        if key not in lookup:
            raise ValueError(
                f"Lead {name!r} is not in '{slug}'. Available: {list(available)}"
            )
        if lookup[key] in indices:
            raise ValueError(f"Lead {name!r} requested more than once")
        indices.append(lookup[key])
        resolved.append(available[lookup[key]])
    return indices, resolved


class ECGDataset(_TorchDataset):
    """PyTorch Dataset for loading any ECG dataset supported by ECGBench.

    This class uses the dataset's YAML config to determine how to load
    signals and metadata. Adding a new dataset requires only a config file.

    Args:
        dataset: Dataset slug (e.g., "ptbxl") or a DatasetConfig object
        split: "train", "val", "test", or None. None selects records purely by
               ``fold_numbers``, ignoring the default split boundaries — use it
               for custom cross-validation. It requires ``fold_numbers``.
        version: "clean" (default) or "original"
        data_path: Path to the dataset's signal files on disk.
                   If None, attempts auto-download from config.download_url.
        sampling_rate: Which sampling rate to load (default: config.default_sampling_rate)
        fold_numbers: Specific fold(s) to load. None = all folds for the split.
                      With a named ``split`` the folds must belong to it (folds
                      1-8 are train, 9 val, 10 test); pass ``split=None`` to cross
                      that boundary.
        window: ``(start, length)`` in samples, e.g. ``(0, 2500)`` for the first
                2500 and ``(2500, 2500)`` for the next. ``length=None`` reads to
                the end. Pushed down into the reader, so only these samples are
                decoded — much faster than cropping afterwards on long records,
                and unlike a lambda ``transform`` it survives a DataLoader with
                ``num_workers>0`` under the "spawn" start method. Raises
                ``WindowOutOfRangeError`` if the window does not fit.
        transform: Optional callable applied to the signal tensor, after
                   ``window``, ``leads`` and ``units``
        metadata_source: "hf" (download fold CSVs from HuggingFace) or "local".
        leads: select and reorder leads by name, e.g. ``["I", "II", "V5"]``.
               Case-insensitive; needs ``lead_names`` in the dataset's config.
               The signal's first dimension becomes ``len(leads)``.
        units: "mV" (default) or "uV" — applied after lead selection and before
               ``transform``. Never affects validation or the exported folds.
        labels: attach per-record labels and metadata as ``sample["labels"]``.
                Needs a local copy of the source dataset — fold CSVs on the Hub
                carry identifiers only, never labels.

    Example:
        >>> train_ds = ECGDataset("ptbxl", split="train", data_path="/data/ptb-xl/1.0.3/")
        >>> loader = DataLoader(train_ds, batch_size=32, collate_fn=ecg_collate_fn)

        >>> ds = ECGDataset("ptbxl", split="train", data_path="...", labels=True)
        >>> ds[0]["labels"]["superclasses"]
        ['NORM']
    """

    def __init__(
        self,
        dataset: str | Any,  # str or DatasetConfig
        split: str | None = "train",
        version: str = "clean",
        data_path: Path | str | None = None,
        sampling_rate: int | None = None,
        fold_numbers: list[int] | None = None,
        transform: Callable | None = None,
        metadata_source: str = "hf",
        labels: bool = False,
        leads: list[str] | None = None,
        units: str = "mV",
        window: tuple[int, int | None] | None = None,
    ):
        super().__init__()

        from ecgbench.config import DatasetConfig, load_config

        if isinstance(dataset, str):
            self.config = load_config(dataset)
        elif isinstance(dataset, DatasetConfig):
            self.config = dataset
        else:
            raise TypeError(f"dataset must be str or DatasetConfig, got {type(dataset)}")

        self.split = split.lower() if isinstance(split, str) else None
        self.version = version
        self.sampling_rate = sampling_rate or self.config.default_sampling_rate
        self.transform = transform
        self.metadata_source = metadata_source
        self.window = _resolve_window(window)

        if self.split is not None and self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', 'test', or None, got '{split}'")
        # split=None selects purely by fold number, across the default split
        # boundaries — the only way to express custom cross-validation, since
        # fold_numbers alone is scoped to one split's directory.
        if self.split is None and not fold_numbers:
            raise ValueError(
                "split=None selects records by fold number across all splits, so "
                "fold_numbers is required with it. For example "
                "ECGDataset(..., split=None, fold_numbers=[1, 2, 3])."
            )
        if self.version not in ("clean", "original"):
            raise ValueError(f"version must be 'clean' or 'original', got '{version}'")

        # Resolve signal data path
        from ecgbench.download import resolve_data_path

        self.data_path = resolve_data_path(data_path, self.config)

        # Load fold metadata
        self.metadata_df = self._load_metadata(fold_numbers)

        # Determine signal path column
        self.signal_col = self.config.signal_path_columns.get(self.sampling_rate)
        if not self.signal_col:
            raise ValueError(
                f"No signal_path_column for rate {self.sampling_rate}. "
                f"Available: {list(self.config.signal_path_columns.keys())}"
            )

        # Lead selection and output units are read-time adapters: they shape the
        # tensor __getitem__ returns and never touch the files, the fold CSVs, or
        # validation — a record excluded for a bad V6 stays excluded even if you
        # never load V6.
        self._unit_factor = _resolve_units(units)
        self.units = "mV" if self._unit_factor == 1.0 else "uV"

        self.lead_names = tuple(self.config.lead_names) if self.config.lead_names else None
        self._lead_index: list[int] | None = None
        if leads is not None:
            self._lead_index, names = _resolve_leads(
                leads, self.config.lead_names, self.config.slug
            )
            self.lead_names = tuple(names)

        # Labels: loaded once and aligned to this split, never per __getitem__.
        self.labels_df = self._load_labels() if labels else None

    def _load_metadata(self, fold_numbers: list[int] | None) -> pd.DataFrame:
        """Load fold CSV metadata from HuggingFace Hub or local disk."""
        if self.metadata_source == "hf":
            return self._load_from_hf(fold_numbers)
        elif self.metadata_source == "local":
            return self._load_from_local(fold_numbers)
        else:
            raise ValueError(
                f"metadata_source must be 'hf' or 'local', got '{self.metadata_source}'"
            )

    def _load_from_hf(self, fold_numbers: list[int] | None) -> pd.DataFrame:
        """Download fold CSVs from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HF metadata. "
                "Install with: pip install ecgbench[hf]"
            )

        repo_id = "vlbthambawita/ECGBench"

        # split=None means "by fold, ignoring the default split", which only the
        # master folds.csv can answer.
        if fold_numbers is None or self.split is None:
            master_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{self.config.slug}/{self.version}/folds.csv",
                repo_type="dataset",
            )
            return self._filter_master(pd.read_csv(master_path), fold_numbers)

        files_to_load = [
            f"{self.config.slug}/{self.version}/{self.split}/fold_{n}.csv" for n in fold_numbers
        ]

        dfs = []
        for file_path in files_to_load:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
            )
            dfs.append(pd.read_csv(local_path))

        return pd.concat(dfs, ignore_index=True)

    def _filter_master(self, df: pd.DataFrame, fold_numbers: list[int] | None) -> pd.DataFrame:
        """Filter the master folds.csv by split and/or fold.

        Shared by the HF and local paths so the two cannot drift. With
        ``split=None`` the ``default_split`` filter is skipped, which is what
        makes cross-split fold selection possible.
        """
        if self.split is not None:
            df = df[df["default_split"] == self.split]
        if fold_numbers is not None:
            known = {int(n) for n in df["fold"].unique()}
            unknown = [n for n in fold_numbers if int(n) not in known]
            if unknown:
                raise ValueError(
                    f"Fold(s) {unknown} hold no records"
                    + (f" in split '{self.split}'" if self.split else "")
                    + f". Available: {sorted(known)}."
                )
            df = df[df["fold"].isin(fold_numbers)]
        return df.reset_index(drop=True)

    def _load_from_local(self, fold_numbers: list[int] | None) -> pd.DataFrame:
        """Load fold CSVs from local disk."""
        # Look for fold CSVs in the data_path following standard structure
        splits_base = self.data_path

        # Per-split fold files are the fast path, but they cannot answer
        # split=None — fold N lives in exactly one split's directory.
        if self.split is not None:
            for candidate in [
                splits_base / self.version / self.split,
                splits_base / self.split,
            ]:
                if candidate.exists():
                    return self._read_fold_csvs(candidate, fold_numbers)

        # Fallback: the master folds.csv, which also serves split=None
        for candidate in [
            splits_base / self.version / "folds.csv",
            splits_base / "folds.csv",
        ]:
            if candidate.exists():
                return self._filter_master(pd.read_csv(candidate), fold_numbers)

        raise FileNotFoundError(
            f"Could not find fold CSVs for split '{self.split}' "
            f"in {splits_base}. Run the split pipeline first or use metadata_source='hf'."
        )

    def _read_fold_csvs(
        self, split_dir: Path, fold_numbers: list[int] | None
    ) -> pd.DataFrame:
        """Read fold CSV files from a split directory."""
        if fold_numbers is not None:
            files = [split_dir / f"fold_{n}.csv" for n in fold_numbers]
            missing = [f for f in files if not f.exists()]
            if missing:
                present = sorted(int(p.stem.split("_")[1]) for p in split_dir.glob("fold_*.csv"))
                raise FileNotFoundError(
                    f"Fold(s) {[int(f.stem.split('_')[1]) for f in missing]} are not in "
                    f"split '{self.split}' (it holds folds {present}). Each fold belongs "
                    "to exactly one split, so to take folds across split boundaries — "
                    "for custom cross-validation — pass split=None with fold_numbers."
                )
        else:
            files = sorted(split_dir.glob("fold_*.csv"))
            if not files:
                raise FileNotFoundError(f"No fold_*.csv files in {split_dir}")

        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def _load_labels(self) -> pd.DataFrame:
        """Load per-record labels, reindexed to this split's records in order.

        Row i of the result corresponds to row i of ``metadata_df``, so
        ``__getitem__`` is a positional lookup with no per-item join.
        """
        from ecgbench.labels import load_labels

        label_df = load_labels(self.config, self.data_path)

        record_ids = self.metadata_df[self.config.record_id_column]
        if label_df.index.dtype != record_ids.dtype:
            # Fold CSVs and source CSVs can disagree on int vs str for the same
            # IDs, which would silently reindex to all-NaN.
            label_df = label_df.set_index(label_df.index.astype(str))
            record_ids = record_ids.astype(str)

        aligned = label_df.reindex(record_ids)
        missing = int(aligned.isna().all(axis=1).sum())
        if missing == len(aligned):
            raise ValueError(
                f"No record in split '{self.split}' matched a label row. Check that "
                f"'{self.config.labels.join_column}' in "
                f"{self.config.labels.source_csv} holds the same IDs as "
                f"'{self.config.record_id_column}' in the fold CSVs."
            )
        if missing:
            logger.warning(
                "%d of %d records in split '%s' have no label row",
                missing, len(aligned), self.split,
            )
        return aligned.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single ECG record with signal and metadata.

        Returns:
            dict with:
              - "signal": torch.Tensor, float32, shape (leads, samples). ``leads``
                is ``len(self.lead_names)`` after any ``leads=`` selection, and
                ``samples`` is the ``window=`` length when one is set.
              - "record_id": record identifier
              - "split": the dataset's split, or — when constructed with
                ``split=None`` — this record's own ``default_split``
              - "fold": int (if available)
              - "labels": dict of label fields (only with ``labels=True``)
              - All other metadata columns

        Raises:
            WindowOutOfRangeError: ``window=`` does not fit this record. Record
                length is not constant in every dataset, so a window can fit most
                records and not all.
        """
        if idx < 0 or idx >= len(self.metadata_df):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.metadata_df)}"
            )

        row = self.metadata_df.iloc[idx]

        # Load signal
        signal_path = str(row[self.signal_col])
        if self.config.signal_format == "wfdb":
            signal_path = str(Path(signal_path).with_suffix(""))
        full_path = str(self.data_path / signal_path)

        signal = _load_signal(
            full_path,
            self.config.signal_format,
            self.config.signal_unit_scale,
            self.window,
        )

        if self._lead_index is not None:
            if signal.shape[0] <= max(self._lead_index):
                raise ValueError(
                    f"Record {row.get(self.config.record_id_column)!r} has "
                    f"{signal.shape[0]} leads, too few for the requested "
                    f"{list(self.lead_names)}"
                )
            signal = signal[self._lead_index]

        signal_tensor = torch.from_numpy(signal).float()
        if self._unit_factor != 1.0:
            signal_tensor = signal_tensor * self._unit_factor

        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)

        # Build result dict
        result: dict[str, Any] = {
            "signal": signal_tensor,
            "record_id": row.get(self.config.record_id_column),
            # With split=None the rows come from several splits, so reporting a
            # single split name would be a lie — give the record's own instead.
            "split": self.split if self.split is not None else row.get("default_split"),
        }

        # Add fold if available
        if "fold" in row.index:
            result["fold"] = int(row["fold"])

        # Labels stay in their own dict: a nested key cannot collide with
        # "signal"/"fold"/"split", and ecg_collate_fn keeps dicts as a list.
        if self.labels_df is not None:
            result["labels"] = self.labels_df.iloc[idx].to_dict()

        # Add all other metadata
        for col in self.metadata_df.columns:
            if col in (self.signal_col, self.config.record_id_column, "default_split"):
                continue
            if col in ("fold",):
                continue  # Already added

            value = row[col]
            if isinstance(value, str):
                value = _parse_dict_string(value)

            if isinstance(value, (int, float, np.integer, np.floating)):
                if not np.isnan(value) if isinstance(value, (float, np.floating)) else True:
                    result[col] = torch.tensor(float(value), dtype=torch.float32)
                else:
                    result[col] = value
            elif isinstance(value, dict):
                result[col] = value
            else:
                result[col] = value

        return result


def ecg_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for ECG dataset batches.

    Stacks tensors, keeps dicts and strings as lists.

    Args:
        batch: List of samples from the dataset

    Returns:
        Batched dictionary
    """
    from torch.utils.data._utils.collate import default_collate

    if not batch:
        return {}

    all_keys = set(batch[0].keys())
    collatable = {}
    non_collatable = {}

    for key in all_keys:
        values = [sample[key] for sample in batch]

        if all(isinstance(v, dict) for v in values):
            non_collatable[key] = values
        elif all(isinstance(v, (str, type(None))) for v in values):
            non_collatable[key] = values
        else:
            collatable[key] = values

    # default_collate expects a list of dicts, not a dict of lists
    if collatable:
        collatable_batch = [
            {k: collatable[k][i] for k in collatable}
            for i in range(len(batch))
        ]
        result = default_collate(collatable_batch)
    else:
        result = {}
    result.update(non_collatable)

    return result
