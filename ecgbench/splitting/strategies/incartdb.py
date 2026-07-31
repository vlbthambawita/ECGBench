"""
St Petersburg INCART splitting strategy.

Nothing machine-readable ships with this dataset, so ``load_metadata`` builds a
metadata CSV from the header comments and the ``.atr`` reference annotations via
``ecgbench.labels.incartdb`` — the same loader users get from ``load_labels``, so
the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Patient grouping is mandatory here.** 75 records come from only 32 patients —
one patient contributed 4 records, 11 contributed 3 and 18 contributed 2. A
record-level split puts the same patient in train and test, and with a cohort this
small that is most of the signal. ``engine.py`` groups on
``config.patient_id_column``, which the label loader fills in from the
``# patient N`` header comment.

Stratification is deliberately coarse. The patient-level ``<diagnoses>`` field is
empty for 14 of the 32 patients, and with 32 groups over 10 folds only two classes
can hold enough patients to be spread across folds at all. Grouping, not
stratification, is what matters for this dataset.
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


@register("incartdb")
class INCARTDBSplitter(DatasetSplitter):
    """INCART splitting strategy: header + annotation metadata, patient-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # patient_id is 'patientNN' and signal_path is 'I01' — both must
                # stay strings, and age is numeric-looking but kept verbatim.
                dtype={"patient_id": str, "signal_path": str, "age": str},
            )

        from ecgbench.labels.incartdb import load_labels

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
        """Return the pooled diagnosis attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("diagnosis")

        if config.patient_id_column and config.patient_id_column in df.columns:
            n_patients = df[config.patient_id_column].nunique()
            logger.info(
                "Grouping %d records into folds by %d patients (mandatory: %d patients "
                "contributed more than one record)",
                len(df),
                n_patients,
                int((df[config.patient_id_column].value_counts() > 1).sum()),
            )

        logger.info("Diagnosis distribution:\n%s", labels.value_counts().to_string())
        return labels
