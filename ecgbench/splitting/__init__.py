"""ECG dataset splitting framework."""

# Trigger strategy registration
from . import strategies  # noqa: F401
from .base import DatasetSplitter, SplitResult
from .engine import split_dataset
from .export import export_splits
from .registry import get_splitter, register

__all__ = [
    "DatasetSplitter",
    "SplitResult",
    "split_dataset",
    "export_splits",
    "get_splitter",
    "register",
]
