"""
Leipzig Heart Center splitting strategy.

Two per-subject CSVs ship rather than one, with different columns, and neither
carries a signal-path column — so ``load_metadata`` builds a single normalised
metadata CSV via ``ecgbench.labels.leipzig_heart_center_ecg``, the same loader
users get from ``load_labels``, so the stratification label and the exposed labels
cannot drift.

Writing that frame to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata at
all.

**There is nothing to group by.** 39 records come from 39 subjects, one each, so
``patient_id_column`` is null and a record-level split is already patient-level.

Stratification is necessarily coarse. The shipped ``diagnosis`` has seven classes
over 39 records, three of them singletons (``AVRT-PJRT``, ``TOF without VT``,
``TOF with nsVT``), which cannot be spread across ten folds. The label loader
therefore derives a ``diagnosis_family`` — AVRT (16 records), AVNRT (13), TOF (10)
— and that is what folds are built on. It is a fold-construction label, not a
clinical grouping: train on ``diagnosis``, and on the beat-level ``tachy_*``
columns, which are richer than either.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column the label loader attaches, used for stratification.
STRATIFY_COLUMN = "stratify_class"

#: Columns that must stay strings when the cached CSV is re-read. ``subject_id`` is
#: zero-padded ('001', '0010') and would lose the padding as an int; ``signal_path``
#: and ``record_name`` are record stems like 'x001'; ``age_raw`` holds the one
#: malformed value verbatim.
_STRING_COLUMNS = ("subject_id", "record_name", "signal_path", "age_raw", "channel_names")


@register("leipzig_heart_center_ecg")
class LeipzigHeartCenterSplitter(DatasetSplitter):
    """Leipzig strategy: two subject CSVs joined into one, diagnosis-family folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype=dict.fromkeys(_STRING_COLUMNS, str),
            )

        from ecgbench.labels.leipzig_heart_center_ecg import load_labels

        df = load_labels(data_path, config).reset_index()
        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            # validate_dataset re-reads this file, so a read-only data directory
            # leaves validation with no metadata at all. Fail loudly instead.
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the diagnosis family attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("diagnosis_family")
        logger.info(
            "Diagnosis family distribution over %d records:\n%s",
            len(df),
            labels.value_counts().to_string(),
        )
        if "diagnosis" in df.columns:
            logger.info(
                "The full diagnosis has %d classes and is NOT what folds use:\n%s",
                df["diagnosis"].nunique(),
                df["diagnosis"].value_counts().to_string(),
            )
        return labels
