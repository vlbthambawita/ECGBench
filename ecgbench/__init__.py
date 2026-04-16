"""
ECGBench: Reproducible ECG benchmark datasets with standardised splits,
validation, and Croissant metadata.
"""

from __future__ import annotations

import importlib

try:
    from ecgbench._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__author__ = "Vajira Thambawita"

# --- Lightweight imports (always available) ---
from .catalogue import categories, get_dataset, list_datasets, search, to_dataframe
from .config import DatasetConfig, list_available_configs, load_config

# --- Lazy imports (heavy dependencies) ---
_LAZY_IMPORTS: dict[str, str] = {
    # PyTorch
    "ECGDataset": ".dataset",
    "ecg_collate_fn": ".dataset",
    # Validation
    "validate_dataset": ".validation",
    "ValidationResult": ".validation",
    # Splitting
    "split_dataset": ".splitting",
    "SplitResult": ".splitting",
    "get_splitter": ".splitting",
    "export_splits": ".splitting",
    # Croissant
    "generate_croissant": ".croissant",
    "save_croissant": ".croissant",
    "validate_croissant": ".croissant",
    # Download
    "download_dataset": ".download",
    "resolve_data_path": ".download",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package="ecgbench")
        return getattr(module, name)
    raise AttributeError(f"module 'ecgbench' has no attribute {name!r}")


__all__ = [
    # Config
    "load_config",
    "list_available_configs",
    "DatasetConfig",
    # Catalogue
    "list_datasets",
    "search",
    "get_dataset",
    "to_dataframe",
    "categories",
    # Dataset
    "ECGDataset",
    "ecg_collate_fn",
    # Validation
    "validate_dataset",
    "ValidationResult",
    # Splitting
    "split_dataset",
    "SplitResult",
    "get_splitter",
    "export_splits",
    # Croissant
    "generate_croissant",
    "save_croissant",
    "validate_croissant",
    # Download
    "download_dataset",
    "resolve_data_path",
]
