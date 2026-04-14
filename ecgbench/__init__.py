"""
ECGBench: Reproducible ECG Benchmark data from Open access datasets
"""

try:
    from ecgbench._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__author__ = "Vajira Thambawita"

from .dataset import ECGDataset, ecg_collate_fn

__all__ = ["ECGDataset", "ecg_collate_fn"]

