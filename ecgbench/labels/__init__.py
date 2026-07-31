"""
Per-record labels and metadata.

Exported fold CSVs are identification-only — record ID, patient ID, signal
paths, fold, split — so ground truth always lives in the source dataset.
``load_labels()`` is the one place that knows, per dataset, which file holds it
and how to reach it.

Two paths:

- **Declarative.** Most datasets need a column select and a join, described by
  the ``labels:`` block in the dataset's YAML. No Python.
- **Per-dataset module.** Datasets needing a real derivation get a module here
  (see ``ptbxl.py``, which reduces SCP codes to diagnostic superclasses).
  Dispatch is an explicit dict, not a decorator registry — these modules are
  meant to be imported directly by per-dataset scripts too.

The loader is authoritative: splitters derive their stratification labels from
it rather than keeping a private copy of the same mapping. That is deliberate —
PTB-XL previously had two derivations that had silently drifted apart.

Labels are **not** published to the HuggingFace Hub. The Hub tree carries only
fold CSVs, so ``load_labels`` needs a local copy of the source dataset. That is
also a licensing boundary: redistributing labels is fine for CC-BY datasets and
not for credentialed ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

__all__ = [
    "LabelsUnavailableError",
    "LabelSourceMissingError",
    "load_labels",
]


class LabelsUnavailableError(RuntimeError):
    """The dataset genuinely ships no labels (not a configuration mistake)."""


class LabelSourceMissingError(FileNotFoundError):
    """The dataset has labels, but the file holding them is not on disk."""


def _load_declarative(data_path: Path, config: DatasetConfig) -> pd.DataFrame:
    """Read labels straight out of a source CSV, per the config's labels block."""
    spec = config.labels
    if not spec.source_csv or not spec.join_column:
        raise ValueError(
            f"Config '{config.slug}' has a labels block without 'source_csv' and "
            "'join_column'. Either complete it or add a module in ecgbench/labels/."
        )

    csv_path = data_path / spec.source_csv
    if not csv_path.exists():
        raise LabelSourceMissingError(
            f"Labels for '{config.slug}' come from {spec.source_csv}, which is not in "
            f"{data_path}. ECGBench publishes fold CSVs only — labels stay with the "
            f"source dataset, so point data_path at a full local copy "
            f"(see {config.url})."
        )

    df = pd.read_csv(csv_path, sep=spec.separator)
    if spec.join_column not in df.columns:
        raise ValueError(
            f"Join column '{spec.join_column}' not in {csv_path}. "
            f"Found: {list(df.columns)}"
        )

    df = df.set_index(spec.join_column)
    df.index.name = config.record_id_column

    if spec.columns:
        missing = [c for c in spec.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Label columns {missing} not in {csv_path}")
        df = df[list(spec.columns)]

    return df


def _custom_loaders() -> dict[str, Callable[[Path, DatasetConfig], pd.DataFrame]]:
    """Slug -> loader, for datasets whose labels need more than a column select.

    Imported lazily so ``import ecgbench.labels`` stays cheap and a broken
    per-dataset module cannot take the whole package down.
    """
    from ecgbench.labels import challenge2021, incartdb, ludb, ptbdb, ptbxl

    return {
        "ptbxl": ptbxl.load_labels,
        "ptbdb": ptbdb.load_labels,
        "ludb": ludb.load_labels,
        "challenge2021": challenge2021.load_labels,
        "incartdb": incartdb.load_labels,
    }


def load_labels(
    dataset: str | DatasetConfig,
    data_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load per-record labels and metadata for a dataset.

    Args:
        dataset: Dataset slug or a DatasetConfig.
        data_path: Root of a local copy of the source dataset. Resolved through
            ``resolve_data_path`` when omitted.

    Returns:
        DataFrame indexed by ``config.record_id_column``, one row per record in
        the source dataset — not per record in a split. Reindex it against a
        split's record IDs to align the two.

    Raises:
        LabelsUnavailableError: the dataset ships no labels at all.
        LabelSourceMissingError: it has labels, but the source file is absent.
    """
    from ecgbench.config import DatasetConfig, load_config

    config = load_config(dataset) if isinstance(dataset, str) else dataset
    if not isinstance(config, DatasetConfig):
        raise TypeError(f"dataset must be str or DatasetConfig, got {type(dataset)}")

    spec = config.labels
    if spec is None:
        raise LabelsUnavailableError(
            f"Config '{config.slug}' has no labels block, so ECGBench does not know "
            "where its labels live. Add one (see ecgbench/data/configs/_template.yaml)."
        )
    if not spec.available:
        raise LabelsUnavailableError(
            f"'{config.slug}' ships no labels. {spec.unavailable_reason}".strip()
        )

    from ecgbench.download import resolve_data_path

    resolved = resolve_data_path(Path(data_path) if data_path else None, config)

    loader = _custom_loaders().get(config.slug)
    df = loader(resolved, config) if loader else _load_declarative(resolved, config)

    if df.index.has_duplicates:
        n = int(df.index.duplicated().sum())
        raise ValueError(
            f"Label source for '{config.slug}' has {n} duplicate record IDs in "
            f"'{df.index.name}'; a join on it would multiply rows."
        )
    return df
