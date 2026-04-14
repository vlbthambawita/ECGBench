"""
ECGBench: Reproducible ECG Benchmark data from Open access datasets
"""

try:
    from ecgbench._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__author__ = "Vajira Thambawita"

# Catalogue functions (no heavy dependencies)
from .catalogue import list_datasets, search, get_dataset, to_dataframe, categories


def __getattr__(name):
    """Lazy-import ECGDataset and ecg_collate_fn to avoid requiring torch at import time."""
    if name == "ECGDataset":
        from .dataset import ECGDataset
        return ECGDataset
    if name == "ecg_collate_fn":
        from .dataset import ecg_collate_fn
        return ecg_collate_fn
    raise AttributeError(f"module 'ecgbench' has no attribute {name!r}")


__all__ = [
    "ECGDataset",
    "ecg_collate_fn",
    "list_datasets",
    "search",
    "get_dataset",
    "to_dataframe",
    "categories",
]

