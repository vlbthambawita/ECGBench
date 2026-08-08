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

    # The join column holds record ids, so for a zero-padded source it has to be
    # read as strings — the fold CSVs it joins against are read the same way, and
    # "00735" must not become 735 on one side only. Empty for every other dataset;
    # see DatasetConfig.zero_padded_identifiers.
    dtypes = dict(config.identifier_dtypes())
    if dtypes:
        dtypes[spec.join_column] = "str"
    df = pd.read_csv(csv_path, sep=spec.separator, dtype=dtypes)
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
    from ecgbench.labels import (
        afdb,
        challenge2017,
        challenge2020,
        challenge2021,
        code15,
        code_test,
        cpsc_2018,
        ecgcipa,
        ecgdmmld,
        ecgrdvq,
        echonext,
        ikem,
        incartdb,
        leipzig_heart_center_ecg,
        ltafdb,
        ludb,
        medalcare_xl,
        mhd_effect_ecg_mri,
        mimic_iv_ecg,
        mitdb,
        ningbo_iva,
        norwegian_athlete_ecg,
        ptbdb,
        ptbxl,
        sami_trop,
        sph,
        staffiii,
        wctecgdb,
        zzu_pecg,
    )

    return {
        "ptbxl": ptbxl.load_labels,
        "ptbdb": ptbdb.load_labels,
        "ludb": ludb.load_labels,
        "medalcare_xl": medalcare_xl.load_labels,
        "mhd_effect_ecg_mri": mhd_effect_ecg_mri.load_labels,
        "norwegian_athlete_ecg": norwegian_athlete_ecg.load_labels,
        "challenge2017": challenge2017.load_labels,
        "challenge2020": challenge2020.load_labels,
        "challenge2021": challenge2021.load_labels,
        "cpsc_2018": cpsc_2018.load_labels,
        "code15": code15.load_labels,
        "code_test": code_test.load_labels,
        "incartdb": incartdb.load_labels,
        "mitdb": mitdb.load_labels,
        "afdb": afdb.load_labels,
        "ltafdb": ltafdb.load_labels,
        "leipzig_heart_center_ecg": leipzig_heart_center_ecg.load_labels,
        "mimic_iv_ecg": mimic_iv_ecg.load_labels,
        "wctecgdb": wctecgdb.load_labels,
        "echonext": echonext.load_labels,
        "ecgcipa": ecgcipa.load_labels,
        "ecgdmmld": ecgdmmld.load_labels,
        "ecgrdvq": ecgrdvq.load_labels,
        "staffiii": staffiii.load_labels,
        "sph": sph.load_labels,
        "ningbo_iva": ningbo_iva.load_labels,
        "ikem": ikem.load_labels,
        "sami_trop": sami_trop.load_labels,
        "zzu_pecg": zzu_pecg.load_labels,
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
