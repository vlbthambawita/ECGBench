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
from .metadata import DatasetMeta, MetadataQueryError, MetadataStore, SearchHit, open_store
from .metadata import get as get_metadata
from .metadata import related as related_metadata
from .metadata import search as search_metadata

# --- Lazy imports (heavy dependencies) ---
_LAZY_IMPORTS: dict[str, str] = {
    # PyTorch
    "ECGDataset": ".dataset",
    "ecg_collate_fn": ".dataset",
    "WindowOutOfRangeError": ".dataset",
    "SplitsNotPublishedError": ".dataset",
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
    # Manifests
    "verify_splits": ".manifest",
    # Labels
    "load_labels": ".labels",
    # Download
    "download_dataset": ".download",
    "resolve_data_path": ".download",
    # High-level pipelines (also exposed as the `ecgbench` CLI)
    "run_splits": ".cli",
    "run_croissant": ".cli",
    "run_upload": ".cli",
    "run_list": ".cli",
    "run_info": ".cli",
    "run_related": ".cli",
    "run_search": ".cli",
    "run_metadata_build": ".cli",
    "run_metadata_check": ".cli",
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
    # Metadata layer (catalogue + configs merged, any alias resolves)
    "DatasetMeta",
    "MetadataStore",
    "MetadataQueryError",
    "SearchHit",
    "open_store",
    "get_metadata",
    "search_metadata",
    "related_metadata",
    # Dataset
    "ECGDataset",
    "ecg_collate_fn",
    "WindowOutOfRangeError",
    "SplitsNotPublishedError",
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
    # Manifests
    "verify_splits",
    # Labels
    "load_labels",
    # Download
    "download_dataset",
    "resolve_data_path",
    # High-level pipelines
    "run_splits",
    "run_croissant",
    "run_upload",
    "run_list",
    "run_info",
    "run_related",
    "run_search",
    "run_metadata_build",
    "run_metadata_check",
]
