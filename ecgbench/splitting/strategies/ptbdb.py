"""
PTB Diagnostic ECG Database splitting strategy.

The release ships no metadata file, so ``load_metadata`` builds one from the
record headers via ``ecgbench.labels.ptbdb`` — the same loader users get from
``load_labels``, so the stratification label and the exposed label cannot drift.

Grouping by patient is mandatory here: 113 of the 290 patients contributed more
than one recording (up to 7), so a record-level split leaks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

STRATIFY_COLUMN = "primary_diagnosis"


@register("ptbdb")
class PTBDBSplitter(DatasetSplitter):
    """PTBDB splitting strategy: header-derived metadata, patient-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        from ecgbench.labels.ptbdb import load_labels

        df = load_labels(data_path, config).reset_index()
        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            # validate_dataset re-reads this file, so a read-only data directory
            # leaves validation with no metadata at all.
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return the pooled diagnosis attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("diagnosis")
        logger.info("Diagnosis distribution:\n%s", labels.value_counts().to_string())
        return labels
