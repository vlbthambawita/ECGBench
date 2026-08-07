"""
MIT-BIH Arrhythmia splitting strategy.

Nothing machine-readable ships with this dataset, so ``load_metadata`` builds a
metadata CSV from the header comments and the ``.atr`` reference annotations via
``ecgbench.labels.mitdb`` — the same loader users get from ``load_labels``, so the
stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split.**

Patient grouping matters exactly once, but it is not optional. 48 records come
from 47 subjects — records 201 and 202 were cut from the same analog tape (1960)
and the shipped directory says so. Ungrouped, those two land in different folds
about 90% of the time, and a model then sees the same subject in train and test.
``engine.py`` groups on ``config.patient_id_column``, which the label loader fills
from the tape number.

Stratification uses the database's own two halves rather than any clinical label.
Records 100-124 were chosen at random to be representative; records 200-234 were
selected for rare but clinically important phenomena. With 48 records over 10
folds, a fold of 4-5 records drawn without regard to that split can easily be all
one half, which is the difference between a test set with ventricular flutter in
it and one without. Anything finer — dominant rhythm has 6 values, one of them a
single record — cannot be spread over 10 folds at all.
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


@register("mitdb")
class MITDBSplitter(DatasetSplitter):
    """MIT-BIH splitting strategy: header + annotation metadata, tape-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "100" and signal_path likewise — both must stay
                # strings or wfdb gets handed an int. patient_id is "tape1960"
                # and recorder is "654" or the literal "N/A".
                dtype={
                    "record_name": str,
                    "signal_path": str,
                    "patient_id": str,
                    "recorder": str,
                },
            )

        from ecgbench.labels.mitdb import load_labels

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
        """Return the record group attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("record_group")

        if config.patient_id_column and config.patient_id_column in df.columns:
            shared = df[config.patient_id_column].value_counts()
            shared = shared[shared > 1]
            logger.info(
                "Grouping %d records into folds by %d subjects (%d subject(s) "
                "contributed more than one record: %s)",
                len(df),
                df[config.patient_id_column].nunique(),
                len(shared),
                shared.to_dict(),
            )

        logger.info("Record group distribution:\n%s", labels.value_counts().to_string())
        return labels
