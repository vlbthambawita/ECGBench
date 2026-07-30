"""
MIMIC-IV-ECG Demo splitting strategy.

The demo subset ships ``record_list.csv`` with identifiers and timestamps only —
no diagnoses, no age or sex. (The full MIMIC-IV-ECG release adds
``machine_measurements.csv``; the demo does not include it.) There is therefore
nothing to stratify on, so this splitter emits a constant label and lets
``StratifiedGroupKFold`` degenerate to patient-grouped splitting, which is the
property that actually matters here: 659 records come from only 92 subjects.

Signal paths in ``record_list.csv`` are already relative to the dataset root and
carry no file extension, which is exactly what ``wfdb.rdrecord`` wants — no path
fix-up is needed, so the validation engine reads the shipped CSV directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Value of the synthetic label column. The demo carries no diagnostic labels;
#: a single class is what makes the split purely patient-grouped.
UNLABELLED = "UNLABELLED"


@register("mimic_iv_ecg_demo")
class MimicIVECGDemoSplitter(DatasetSplitter):
    """MIMIC-IV-ECG Demo splitting strategy.

    - Reads the shipped ``record_list.csv`` as-is (paths need no fix-up)
    - Adds the constant label column the config declares
    - Groups by ``subject_id``; stratification is a no-op by design
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv
        df = pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        signal_col = config.signal_path_columns[config.default_sampling_rate]
        missing = [
            c
            for c in (config.record_id_column, config.patient_id_column, signal_col)
            if c and c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{csv_path} is missing expected columns {missing}. "
                f"Found: {list(df.columns)}"
            )

        # The config declares label_column, so materialise it rather than leaving
        # a column name that resolves to nothing.
        df[config.label_column] = UNLABELLED

        logger.info(
            "Loaded MIMIC-IV-ECG Demo metadata: %d records from %d subjects",
            len(df),
            df[config.patient_id_column].nunique(),
        )
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return a single-class label series — the demo has no labels to use.

        Grouping by ``subject_id`` is what keeps a subject's studies inside one
        fold. Stratifying on records-per-subject was tried and rejected: it makes
        fold sizes markedly *less* even (std 19.4 vs 7.5 records over 10 folds),
        because balancing bucket counts fights the group-size balancing.
        """
        labels = pd.Series(UNLABELLED, index=df.index, name="unlabelled")
        logger.info(
            "No diagnostic labels in this dataset — splitting on subject groups only "
            "(%d subjects over %d records)",
            df[config.patient_id_column].nunique(),
            len(df),
        )
        return labels
