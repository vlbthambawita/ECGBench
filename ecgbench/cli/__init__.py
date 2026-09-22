"""ECGBench command-line interface.

Exposes:
- ``main`` — entry point for the ``ecgbench`` console script.
- ``run_splits`` — full validate + split + Croissant pipeline.
- ``run_croissant`` — standalone Croissant JSON-LD generation.
- ``run_upload`` — upload fold CSVs + metadata to HuggingFace Hub.
- ``run_list`` / ``run_info`` / ``run_related`` / ``run_search`` / ``run_fields`` —
  read the merged dataset metadata (``ecgbench list``, ``info``, ``related``,
  ``search``, ``fields``).
- ``run_metadata_build`` / ``run_metadata_check`` — rebuild or verify the
  derived metadata files (``ecgbench metadata build [--check]``).
"""

from ecgbench.cli._main import main
from ecgbench.cli.catalog import run_fields, run_info, run_list, run_related, run_search
from ecgbench.cli.croissant import run_croissant
from ecgbench.cli.metadata import run_metadata_build, run_metadata_check
from ecgbench.cli.splits import run_splits
from ecgbench.cli.upload import run_upload

__all__ = [
    "main",
    "run_splits",
    "run_croissant",
    "run_upload",
    "run_list",
    "run_info",
    "run_related",
    "run_search",
    "run_fields",
    "run_metadata_build",
    "run_metadata_check",
]
