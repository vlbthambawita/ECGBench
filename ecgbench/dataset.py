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


def _load_signal(record_path: str, signal_format: str) -> np.ndarray:
    """Load ECG signal. Returns shape (leads, samples)."""
    if signal_format == "wfdb":
        wfdb = _require_wfdb()
        record = wfdb.rdrecord(record_path)
        if record.p_signal is None:
            raise ValueError(f"Signal is None for record: {record_path}")
        return record.p_signal.T.astype(np.float32)
    else:
        raise NotImplementedError(
            f"Signal format '{signal_format}' not yet supported. "
            "Currently supported: wfdb"
        )


def _parse_dict_string(value: str) -> dict | str:
    """Try to parse a Python dict literal string."""
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


class ECGDataset(_TorchDataset):
    """PyTorch Dataset for loading any ECG dataset supported by ECGBench.

    This class uses the dataset's YAML config to determine how to load
    signals and metadata. Adding a new dataset requires only a config file.

    Args:
        dataset: Dataset slug (e.g., "ptbxl") or a DatasetConfig object
        split: "train", "val", or "test"
        version: "clean" (default) or "original"
        data_path: Path to the dataset's signal files on disk.
                   If None, attempts auto-download from config.download_url.
        sampling_rate: Which sampling rate to load (default: config.default_sampling_rate)
        fold_numbers: Specific fold(s) to load. None = all folds for the split.
        transform: Optional callable applied to the signal tensor
        metadata_source: "hf" (download fold CSVs from HuggingFace) or "local".

    Example:
        >>> train_ds = ECGDataset("ptbxl", split="train", data_path="/data/ptb-xl/1.0.3/")
        >>> loader = DataLoader(train_ds, batch_size=32, collate_fn=ecg_collate_fn)
    """

    def __init__(
        self,
        dataset: str | Any,  # str or DatasetConfig
        split: str = "train",
        version: str = "clean",
        data_path: Path | str | None = None,
        sampling_rate: int | None = None,
        fold_numbers: list[int] | None = None,
        transform: Callable | None = None,
        metadata_source: str = "hf",
    ):
        super().__init__()

        from ecgbench.config import DatasetConfig, load_config

        if isinstance(dataset, str):
            self.config = load_config(dataset)
        elif isinstance(dataset, DatasetConfig):
            self.config = dataset
        else:
            raise TypeError(f"dataset must be str or DatasetConfig, got {type(dataset)}")

        self.split = split.lower()
        self.version = version
        self.sampling_rate = sampling_rate or self.config.default_sampling_rate
        self.transform = transform
        self.metadata_source = metadata_source

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
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

        if fold_numbers is not None:
            files_to_load = [
                f"{self.config.slug}/{self.version}/{self.split}/fold_{n}.csv"
                for n in fold_numbers
            ]
        else:
            # Download the master folds.csv and filter by split
            master_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{self.config.slug}/{self.version}/folds.csv",
                repo_type="dataset",
            )
            master_df = pd.read_csv(master_path)
            if fold_numbers is None:
                return master_df[master_df["default_split"] == self.split].reset_index(
                    drop=True
                )

        dfs = []
        for file_path in files_to_load:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
            )
            dfs.append(pd.read_csv(local_path))

        return pd.concat(dfs, ignore_index=True)

    def _load_from_local(self, fold_numbers: list[int] | None) -> pd.DataFrame:
        """Load fold CSVs from local disk."""
        # Look for fold CSVs in the data_path following standard structure
        splits_base = self.data_path

        # Try a few common locations
        for candidate in [
            splits_base / self.version / self.split,
            splits_base / self.split,
        ]:
            if candidate.exists():
                return self._read_fold_csvs(candidate, fold_numbers)

        # Fallback: try master folds.csv
        for candidate in [
            splits_base / self.version / "folds.csv",
            splits_base / "folds.csv",
        ]:
            if candidate.exists():
                df = pd.read_csv(candidate)
                result = df[df["default_split"] == self.split]
                if fold_numbers is not None:
                    result = result[result["fold"].isin(fold_numbers)]
                return result.reset_index(drop=True)

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
                raise FileNotFoundError(f"Fold files not found: {missing}")
        else:
            files = sorted(split_dir.glob("fold_*.csv"))
            if not files:
                raise FileNotFoundError(f"No fold_*.csv files in {split_dir}")

        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single ECG record with signal and metadata.

        Returns:
            dict with:
              - "signal": torch.Tensor of shape (leads, samples), float32
              - "record_id": record identifier
              - "split": str
              - "fold": int (if available)
              - All other metadata columns
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

        signal = _load_signal(full_path, self.config.signal_format)
        signal_tensor = torch.from_numpy(signal).float()

        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)

        # Build result dict
        result: dict[str, Any] = {
            "signal": signal_tensor,
            "record_id": row.get(self.config.record_id_column),
            "split": self.split,
        }

        # Add fold if available
        if "fold" in row.index:
            result["fold"] = int(row["fold"])

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
