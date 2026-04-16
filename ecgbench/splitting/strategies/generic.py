"""
Config-driven fallback splitter for datasets with simple structure.

Reads everything from DatasetConfig — no custom Python needed.

Handles:
- Loading CSV with configurable separator and column names
- Direct stratification (label_column used as-is)
- Predefined splits if specified in config
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)


@register("generic")
class GenericSplitter(DatasetSplitter):
    """Config-driven splitter for datasets with simple structure."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv
        df = pd.read_csv(csv_path, sep=config.metadata_csv_separator)
        logger.info("Loaded metadata from %s: %d records", csv_path.name, len(df))
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Use the label column directly for stratification."""
        labels = df[config.label_column].copy()
        labels.name = config.label_column

        dist = labels.value_counts()
        logger.info("Label distribution:\n%s", dist.to_string())

        return labels
