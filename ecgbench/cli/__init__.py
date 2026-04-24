"""ECGBench command-line interface.

Exposes:
- ``main`` — entry point for the ``ecgbench`` console script.
- ``run_splits`` — full validate + split + Croissant pipeline.
- ``run_croissant`` — standalone Croissant JSON-LD generation.
- ``run_upload`` — upload fold CSVs + metadata to HuggingFace Hub.
"""

from ecgbench.cli._main import main
from ecgbench.cli.croissant import run_croissant
from ecgbench.cli.splits import run_splits
from ecgbench.cli.upload import run_upload

__all__ = [
    "main",
    "run_splits",
    "run_croissant",
    "run_upload",
]
