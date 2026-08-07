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


class UnitConversionError(ValueError):
    """The dataset's samples are not in a physical unit, so units= cannot apply.

    Raised for sources whose publisher standardised the waveforms — see
    ``DatasetConfig.signal_units``. Scaling them by 1000 would produce a number
    that looks like microvolts and means nothing.
    """


class SplitsNotPublishedError(RuntimeError):
    """The dataset's splits are deliberately not on the Hub.

    Raised instead of a bare 404 for credentialed or restricted sources, whose
    identifiers ECGBench will not republish. The message carries the command
    that regenerates the identical split locally.
    """


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
    if signal_format == "npy":
        try:
            path, _ = _parse_npy_ref(record_path)
            return int(np.load(path, mmap_mode="r").shape[-2])
        except Exception:
            return None
    if signal_format == "csv_lead_rows":
        # Samples run along the row, so the length is one row's field count.
        try:
            with open(record_path) as handle:
                return handle.readline().count(",") + 1
        except Exception:
            return None
    if signal_format == "hdf5":
        try:
            import h5py

            path, key, _ = _parse_hdf5_ref(record_path)
            with h5py.File(path, "r") as handle:
                # Samples are axis 1 for both the 2-D and the 3-D layout.
                return int(_hdf5_dataset(handle, key, record_path).shape[1])
        except Exception:
            return None
    return None


#: Extensions an HDF5 record file may carry. Used to tell a ``:<dataset>``
#: suffix from a colon that happens to be part of the path.
_HDF5_SUFFIXES = (".h5", ".hdf5", ".he5", ".hdf")


def _parse_hdf5_ref(record_path: str) -> tuple[str, str | None, int | None]:
    """Split an HDF5 reference into file, optional dataset key and optional row.

    Three forms, in decreasing specificity:

    ``<file>.h5:<dataset>:<row>``
        Row ``<row>`` of a **3-D** ``(records, samples, leads)`` array — the
        layout CODE-15% and CODE-test use, where one file holds tens of
        thousands of records. Both the key and the row are required here, so
        that a dataset literally named ``417`` can never be read as a row.
    ``<file>.h5:<dataset>``
        A named **2-D** ``(leads, samples)`` array.
    ``<file>.h5``
        The file's sole 2-D array — SPH's ``A00001.h5`` holds only ``ecg``.

    A colon is only ever read as a separator when what precedes it ends in an
    HDF5 extension, so a directory named with a colon stays part of the path.
    """
    text = str(record_path)
    head, separator, tail = text.rpartition(":")

    if separator and tail.lstrip("-").isdigit():
        path, inner, key = head.rpartition(":")
        if inner and path.lower().endswith(_HDF5_SUFFIXES):
            return path, key, int(tail)

    if separator and head.lower().endswith(_HDF5_SUFFIXES):
        return head, tail, None

    return text, None, None


def _hdf5_dataset(handle, key: str | None, record_path: str):
    """Resolve the dataset to read out of an open HDF5 file.

    Returns the raw h5py dataset, 2-D or 3-D; :func:`_hdf5_signal_view` is what
    turns it into one record. Both shapes put samples on **axis 1**, which is
    what :func:`_record_length` relies on.
    """
    import h5py

    if key is not None:
        if key not in handle:
            raise ValueError(
                f"HDF5 record {record_path!r} names dataset {key!r}, which is not in "
                f"the file. Found: {sorted(handle.keys())}"
            )
        dataset = handle[key]
    else:
        names = [name for name, node in handle.items() if isinstance(node, h5py.Dataset)]
        if len(names) != 1:
            raise ValueError(
                f"HDF5 file {record_path!r} holds {len(names)} root datasets "
                f"({sorted(names)}), so which one carries the signal is ambiguous. "
                "Name it in the path as '<file>.h5:<dataset>'."
            )
        dataset = handle[names[0]]

    if dataset.ndim not in (2, 3):
        raise ValueError(
            f"HDF5 dataset in {record_path!r} has shape {dataset.shape}; expected "
            "2-D (leads, samples) or 3-D (records, samples, leads)."
        )
    return dataset


def _hdf5_signal_view(dataset, row: int | None, record_path: str, sl: slice) -> np.ndarray:
    """Read ``sl`` samples of one record out of a resolved HDF5 dataset.

    The two shapes have **opposite** orientations, which is not an inconsistency
    this code invented — it is what the releases ship:

    * 2-D is ``(leads, samples)``, already what ECGBench returns (SPH).
    * 3-D is ``(records, samples, leads)``, matching the ``npy`` convention and
      needing a transpose (CODE).

    Slicing happens inside h5py, so a window costs only the samples it asks for
    even though the files are contiguous and uncompressed.
    """
    if dataset.ndim == 2:
        if row is not None:
            raise ValueError(
                f"HDF5 record {record_path!r} names row {row}, but the dataset is "
                f"2-D {dataset.shape} — a row index only applies to a 3-D "
                "(records, samples, leads) array."
            )
        return np.asarray(dataset[:, sl], dtype=np.float32)

    if row is None:
        raise ValueError(
            f"HDF5 dataset in {record_path!r} is 3-D {dataset.shape}, so it holds "
            "many records and the reference must name one as "
            "'<file>.h5:<dataset>:<row>'."
        )
    if not -dataset.shape[0] <= row < dataset.shape[0]:
        raise ValueError(
            f"HDF5 record {record_path!r} names row {row}, but the dataset holds "
            f"{dataset.shape[0]} records."
        )
    return np.asarray(dataset[row, sl, :], dtype=np.float32).T


def _parse_npy_ref(record_path: str) -> tuple[str, int]:
    """Split a ``<file>.npy:<row>`` reference into its file and row index.

    Datasets in ``npy`` format do not give each record its own file — they ship
    one array per split holding every record, so a "path" has to carry the row
    too. ``EchoNext_test_waveforms.npy:417`` is row 417 of that array.
    """
    path, separator, row = str(record_path).rpartition(":")
    if not separator or not row.lstrip("-").isdigit():
        raise ValueError(
            f"npy record path {record_path!r} must end in ':<row>' — records are "
            "rows of a shared array, so the row index is part of the reference "
            "(e.g. 'EchoNext_test_waveforms.npy:417')."
        )
    return path, int(row)


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
    elif signal_format == "csv_lead_rows":
        # The transpose of "csv": one ROW per lead, one column per sample, and no
        # header at all — already the orientation ECGBench returns, so no .T.
        # Samples run along the row, which numpy's text reader cannot seek into,
        # so unlike every other branch here the window is a slice after the read
        # rather than a push-down. That costs nothing on the 10 s records this
        # format is used for; revisit if a long-record dataset ever adopts it.
        signal = np.loadtxt(record_path, delimiter=",", dtype=np.float32, ndmin=2)
        if window is not None:
            available = signal.shape[1]
            if start >= available or (length is not None and sampto > available):
                raise _window_error(record_path, start, length, available)
            signal = signal[:, start:sampto]
    elif signal_format == "npy":
        # One array per split holding every record, so the reference names a row
        # rather than a file of its own. Memory-mapped: the window slices before
        # anything is materialised, so a 17 GB array costs one record's worth of
        # reads. Stored (..., samples, leads) — transposed from what we return.
        path, row = _parse_npy_ref(record_path)
        array = np.load(path, mmap_mode="r")
        record = array[row]
        # A leading singleton channel axis is conventional in these releases
        # (N, 1, samples, leads); squeeze it rather than demanding one shape.
        while record.ndim > 2 and record.shape[0] == 1:
            record = record[0]
        if record.ndim != 2:
            raise ValueError(
                f"npy record {record_path!r} has shape {record.shape}; expected "
                "(samples, leads) after squeezing leading singleton axes."
            )
        if start >= record.shape[0] or (
            length is not None and start + length > record.shape[0]
        ):
            raise _window_error(record_path, start, length, int(record.shape[0]))
        signal = np.asarray(record[start:sampto], dtype=np.float32).T
    elif signal_format == "hdf5":
        # Either one file per record holding a (leads, samples) array, or one
        # file holding many records as (records, samples, leads) — see
        # _hdf5_signal_view, which normalises both to (leads, samples). h5py
        # slices on read, so a window decodes only the samples it asks for.
        import h5py

        path, key, row = _parse_hdf5_ref(record_path)
        with h5py.File(path, "r") as handle:
            dataset = _hdf5_dataset(handle, key, record_path)
            n_samples = int(dataset.shape[1])
            if start >= n_samples or (length is not None and sampto > n_samples):
                raise _window_error(record_path, start, length, n_samples)
            signal = _hdf5_signal_view(
                dataset, row, record_path, slice(start, sampto)
            )
    else:
        raise NotImplementedError(
            f"Signal format '{signal_format}' not yet supported. "
            "Currently supported: wfdb, csv, csv_lead_rows, npy, hdf5"
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


def _resolve_units(units: str, source_units: str = "mV") -> float:
    """Factor converting a record from ``source_units`` to the requested units.

    ``source_units`` is the config's ``signal_units``. Anything other than mV
    cannot be converted at all, so this refuses rather than scaling numbers whose
    units it does not know \u2014 the failure would otherwise be a silent 1000x.
    """
    factor = _UNIT_FACTORS.get(str(units).strip().lower())
    if factor is None:
        raise ValueError(
            f"units must be one of 'mV', 'uV' (or '\u00b5V'), got {units!r}"
        )
    if str(source_units).strip().lower() not in _UNIT_FACTORS:
        if factor != 1.0:
            raise UnitConversionError(
                f"This dataset's samples are stored as {source_units!r}, not a "
                f"physical unit, so they cannot be converted to {units!r}. Its "
                "publisher standardised the waveforms and did not release the "
                "mean and standard deviation, so millivolts are unrecoverable. "
                "Drop the units= argument to get the values as shipped."
            )
        # units="mV" is the default nobody passed on purpose; leave the samples
        # untouched rather than claiming they are millivolts.
        return 1.0
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


def _layout_union(layouts: list[list[str]]) -> list[str]:
    """Every lead name any layout holds, first-seen order, case-insensitively unique.

    Used only to validate a ``leads=`` request at construction time when the
    release uses several layouts: a name in none of them is a typo and should
    fail immediately, while a name in some of them is legitimate and is resolved
    per record later.
    """
    seen: dict[str, str] = {}
    for layout in layouts:
        for name in layout:
            seen.setdefault(str(name).strip().lower(), name)
    return list(seen.values())


def _stored_lead_names(record_path: str, signal_format: str, slug: str) -> list[str]:
    """The lead names *this* record stores, read from its own header.

    Only WFDB names its leads per record. A release that declares
    ``record_lead_layouts`` in any other format has no way to say which layout a
    given record uses, so this refuses rather than picking one.
    """
    if signal_format != "wfdb":
        raise ValueError(
            f"'{slug}' declares record_lead_layouts, but signal_format "
            f"'{signal_format}' stores no lead names per record, so there is "
            "nothing to resolve a request against. Only wfdb records name their "
            "own leads."
        )
    import wfdb

    names = list(wfdb.rdheader(record_path).sig_name or [])
    if not names:
        raise ValueError(f"Header for {record_path!r} names no leads.")
    return names


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
               The signal's first dimension becomes ``len(leads)``. Where a
               release uses more than one layout (``zzu_pecg``, ``mitdb``) the
               names are re-resolved per record, and a record whose layout lacks
               a requested lead raises rather than substituting another.
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
        self._unit_factor = _resolve_units(units, self.config.signal_units)
        if self.config.signal_units.strip().lower() in _UNIT_FACTORS:
            self.units = "mV" if self._unit_factor == 1.0 else "uV"
        else:
            # Report what the samples actually are, not the mV default nobody
            # chose. sample["units"] is what a user checks before plotting.
            self.units = self.config.signal_units

        self.lead_names = tuple(self.config.lead_names) if self.config.lead_names else None
        self._lead_index: list[int] | None = None
        # What the user asked for, kept so a record storing a different layout can
        # be re-resolved against it at read time. See _lead_index_for().
        self._requested_leads: list[str] | None = None
        self._declared_n_leads = len(self.config.lead_names) if self.config.lead_names else None
        self._alt_lead_index: dict[int, list[int]] = {}
        # Per-record-path indices, for a release whose records store the same
        # number of leads under different names. See _lead_index_for().
        self._path_lead_index: dict[str, list[int]] = {}
        if leads is not None:
            # With several layouts in play the declared order answers for none of
            # them, so the request is checked against the union — a name in no
            # layout is a typo and fails here — and the indices are re-resolved
            # per record at read time. _lead_index is then only a "selection is
            # active" marker; _lead_index_for never returns it.
            available = (
                _layout_union(self.config.record_lead_layouts)
                if self.config.record_lead_layouts
                else self.config.lead_names
            )
            self._lead_index, names = _resolve_leads(leads, available, self.config.slug)
            self._requested_leads = list(leads)
            # The resolved *names* are the same whatever layout a record uses —
            # that is the point of selecting by name — so this stays valid even
            # when the indices differ per record.
            self.lead_names = tuple(names)

        # Labels: loaded once and aligned to this split, never per __getitem__.
        self.labels_df = self._load_labels() if labels else None

    def _lead_index_for(
        self, n_stored: int, record_id: Any, record_path: str | None = None
    ) -> list[int]:
        """Row indices for the requested leads in a record storing ``n_stored`` of them.

        Almost always the indices resolved once in ``__init__``. Two releases in
        the catalogue do not use one layout throughout, and they break it in
        different ways:

        - ``zzu_pecg`` varies the lead **count** — 12 leads for 12,334 records and
          9 for the other 1,856, dropping V2, V4 and V6, so position 7 is V2 in
          the first layout and V3 in the second. ``alternate_lead_names`` maps the
          count to the layout.
        - ``mitdb`` varies the **names** at a constant count — all 48 records store
          2 leads, but 8 store something other than MLII/V1 and record 114 stores
          them reversed. A count-keyed map cannot see that at all, so
          ``record_lead_layouts`` instead says "layout varies, read it from the
          record", and the names come from the record's own header.

        Either way, taking the declared indices would return a different physical
        lead with no error, which is the failure this method exists to prevent. A
        dataset declaring neither field asserts one layout throughout and keeps
        the previous behaviour exactly, including the caller's "too few leads"
        check on an out-of-range index.
        """
        assert self._lead_index is not None  # only called when selection is active

        if self.config.record_lead_layouts:
            assert self._requested_leads is not None
            if record_path is None:
                raise ValueError(
                    f"'{self.config.slug}' resolves leads from each record's own "
                    "header, but no record path was supplied."
                )
            cached = self._path_lead_index.get(record_path)
            if cached is not None:
                return cached
            stored = _stored_lead_names(
                record_path, self.config.signal_format, self.config.slug
            )
            try:
                index, _ = _resolve_leads(
                    self._requested_leads, stored, self.config.slug
                )
            except ValueError as e:
                # The zzu_pecg "absent lead" case, one layer down: refuse rather
                # than hand back whichever lead happens to sit at that index.
                # .item() because a numeric record id arrives as a numpy scalar,
                # and "Record np.int64(102)" helps nobody.
                name = record_id.item() if hasattr(record_id, "item") else record_id
                raise ValueError(
                    f"Record {name!r} stores {stored}, and this dataset uses "
                    f"more than one lead layout. {e}"
                ) from e
            self._path_lead_index[record_path] = index
            return index

        alternates = self.config.alternate_lead_names or {}
        if not alternates or self._declared_n_leads is None:
            return self._lead_index
        if n_stored == self._declared_n_leads:
            return self._lead_index

        cached = self._alt_lead_index.get(n_stored)
        if cached is not None:
            return cached

        layout = alternates.get(n_stored)
        if layout is None:
            raise ValueError(
                f"Record {record_id!r} stores {n_stored} leads, but "
                f"'{self.config.slug}' declares {self._declared_n_leads} "
                f"({list(self.config.lead_names or [])}) and its "
                f"alternate_lead_names covers only {sorted(alternates)}. This "
                "dataset is known to use more than one lead layout, so selecting "
                "by name refuses to assume one — add the layout to the YAML."
            )
        assert self._requested_leads is not None
        index, _ = _resolve_leads(self._requested_leads, layout, self.config.slug)
        self._alt_lead_index[n_stored] = index
        return index

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

        if not self.config.publish_fold_csvs:
            raise SplitsNotPublishedError(
                f"ECGBench does not publish fold CSVs for '{self.config.slug}'.\n"
                f"{self.config.no_publish_reason.strip()}\n"
                f"Then load with metadata_source=\"local\", pointing data_path at the "
                "directory holding the generated original/ and clean/ trees."
            )

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
            lead_index = self._lead_index_for(
                signal.shape[0], row.get(self.config.record_id_column), full_path
            )
            if signal.shape[0] <= max(lead_index):
                raise ValueError(
                    f"Record {row.get(self.config.record_id_column)!r} has "
                    f"{signal.shape[0]} leads, too few for the requested "
                    f"{list(self.lead_names)}"
                )
            signal = signal[lead_index]

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
