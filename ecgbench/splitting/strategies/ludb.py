"""
LUDB splitting strategy.

``ludb.csv`` holds the labels but no signal path, and every string cell in it has
a trailing newline. ``load_metadata`` therefore writes a normalised metadata CSV
carrying a real ``signal_path`` column, built from the cleaned labels that
``ecgbench.labels.ludb`` produces — so the stratification label and the label a
user gets from ``load_labels`` come from one place.

One record per patient (200 of each), so no grouping column.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Records live in data/<ID>, with the .dat/.hea pair named by the bare integer.
SIGNAL_DIR = "data"

STRATIFY_COLUMN = "primary_rhythm"


@register("ludb")
class LUDBSplitter(DatasetSplitter):
    """LUDB splitting strategy: cleaned CSV labels, rhythm-stratified folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        signal_dir = data_path / SIGNAL_DIR
        if not signal_dir.is_dir():
            raise FileNotFoundError(
                f"Expected the signal directory {signal_dir}. Point --data-path at the "
                "dataset root, the directory holding ludb.csv and data/."
            )

        from ecgbench.labels.ludb import load_labels

        df = load_labels(data_path, config).reset_index()
        signal_col = config.signal_path_columns[config.default_sampling_rate]
        df[signal_col] = df[config.record_id_column].map(
            lambda rid: f"{SIGNAL_DIR}/{rid}"
        )
        # List columns would round-trip through CSV as their repr; keep the
        # newline-joined multi-label fields readable instead.
        for column in df.columns:
            if df[column].map(lambda v: isinstance(v, list)).any():
                df[column] = df[column].map(
                    lambda v: ";".join(v) if isinstance(v, list) else v
                )

        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return the pooled rhythm attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("rhythm")
        logger.info("Rhythm distribution:\n%s", labels.value_counts().to_string())
        return labels
