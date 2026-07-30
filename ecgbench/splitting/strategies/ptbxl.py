"""
PTB-XL specific splitting strategy.

Stratification labels come from ``ecgbench.labels.ptbxl``, which derives
diagnostic superclasses from the shipped ``scp_statements.csv``. This module used
to hardcode its own SCP code -> superclass dict; that copy had drifted from the
statement table (five diagnostic codes missing, seven non-diagnostic codes
treated as diagnostic), so the two label sources disagreed. There is now one.

Fold assignment itself is unaffected either way: PTB-XL sets
``has_predefined_splits: true``, so folds come from the official ``strat_fold``
column and the stratification label only names what the folds were balanced on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column added by ``load_metadata`` holding the single-label superclass.
STRATIFY_COLUMN = "primary_superclass"


@register("ptbxl")
class PTBXLSplitter(DatasetSplitter):
    """PTB-XL specific splitting logic.

    - Reads the official metadata CSV and normalises the signal path columns
    - Attaches the diagnostic superclass derived by ``ecgbench.labels.ptbxl``
    - Uses the strat_fold column for the official 10-fold assignment
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv
        df = pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        # Rename signal path columns to use sampling rate as key
        # PTB-XL uses filename_lr and filename_hr
        if "filename_lr" in df.columns and 100 in config.signal_path_columns:
            expected_col = config.signal_path_columns[100]
            if expected_col not in df.columns:
                df = df.rename(columns={"filename_lr": expected_col})
        if "filename_hr" in df.columns and 500 in config.signal_path_columns:
            expected_col = config.signal_path_columns[500]
            if expected_col not in df.columns:
                df = df.rename(columns={"filename_hr": expected_col})

        # Attach the superclass from the authoritative label loader rather than
        # re-deriving it here.
        from ecgbench.labels.ptbxl import load_labels

        label_df = load_labels(data_path, config)
        df[STRATIFY_COLUMN] = (
            df[config.record_id_column].map(label_df[STRATIFY_COLUMN]).to_numpy()
        )

        logger.info("Loaded PTB-XL metadata: %d records", len(df))
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return the diagnostic superclass attached by ``load_metadata``."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        superclasses = df[STRATIFY_COLUMN].rename("diagnostic_superclass")
        logger.info(
            "Superclass distribution:\n%s", superclasses.value_counts().to_string()
        )
        return superclasses
