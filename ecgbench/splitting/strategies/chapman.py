"""
Chapman-Shaoxing specific splitting strategy.

No predefined splits. Uses Rhythm column directly for stratification.
Groups by PatientId to prevent patient leakage.
Signal paths need "ECGData/" prefix prepended.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)


@register("chapman_shaoxing")
class ChapmanSplitter(DatasetSplitter):
    """Chapman-Shaoxing splitting strategy.

    - Uses Rhythm column directly for stratification
    - Groups by PatientId to prevent leakage
    - Prepends "ECGData/" prefix to signal paths
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv
        df = pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        # Prepend ECGData/ prefix to the signal path column
        signal_col = config.signal_path_columns.get(config.default_sampling_rate)
        if signal_col and signal_col in df.columns:
            df[signal_col] = df[signal_col].apply(
                lambda x: f"ECGData/{x}" if not str(x).startswith("ECGData/") else x
            )

        logger.info("Loaded Chapman-Shaoxing metadata: %d records", len(df))
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Use the Rhythm column directly for stratification."""
        labels = df[config.label_column].copy()
        labels.name = "rhythm"

        dist = labels.value_counts()
        logger.info("Rhythm distribution:\n%s", dist.to_string())

        return labels
