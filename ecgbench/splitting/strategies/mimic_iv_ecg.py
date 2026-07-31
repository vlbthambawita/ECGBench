"""
MIMIC-IV-ECG splitting strategy — the full 800,035-record credentialed release.

``record_list.csv`` ships usable as-is: ``path`` is already relative to the
dataset root and extension-free, exactly what ``wfdb.rdrecord`` wants. So unlike
Chapman or INCART there is no path fix-up, and the validation engine can read the
shipped CSV directly.

What this splitter adds is the stratification label. It joins
``machine_measurements.csv`` through ``ecgbench.labels.mimic_iv_ecg`` — the same
loader users get from ``load_labels`` — and attaches the class **in memory only**.
No normalised CSV is written back, and that is safe here only because paths need
no fix-up: ``validate_dataset`` re-reads the shipped ``record_list.csv`` and
rebuilds the same paths this frame holds. A dataset that rewrote paths would have
to persist them (see ``chapman.py``).

**Patient grouping is the point.** 800,035 studies come from 161,352 subjects:
64.5% of subjects contributed more than one, the busiest contributed 260, and the
mean is 5. A record-level split leaks a majority of patients across train and
test.

The measurements file is optional at split time. If it is absent — a fold-only
run on a partial copy — every record falls into ``UNKNOWN`` and the split
degenerates to purely patient-grouped, which is still correct. That is logged,
loudly, rather than failing: the split is reproducible either way, and the labels
are what require the credentialed file.
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


@register("mimic_iv_ecg")
class MimicIVECGSplitter(DatasetSplitter):
    """MIMIC-IV-ECG splitting strategy: report-stratified, patient-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected {config.metadata_csv} at {csv_path}. Point --data-path at the "
                "dataset root — the directory holding record_list.csv and files/."
            )

        df = pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        signal_col = config.signal_path_columns[config.default_sampling_rate]
        missing = [
            c
            for c in (config.record_id_column, config.patient_id_column, signal_col)
            if c and c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{csv_path} is missing expected columns {missing}. " f"Found: {list(df.columns)}"
            )

        df = self._attach_stratification(df, data_path, config)

        logger.info(
            "Loaded MIMIC-IV-ECG metadata: %d records from %d subjects "
            "(max %d studies for one subject)",
            len(df),
            df[config.patient_id_column].nunique(),
            int(df[config.patient_id_column].value_counts().max()),
        )
        return df

    def _attach_stratification(
        self, df: pd.DataFrame, data_path: Path, config: DatasetConfig
    ) -> pd.DataFrame:
        """Join the label loader's stratify_class onto the record list.

        Derived through ``ecgbench.labels.mimic_iv_ecg`` rather than reimplemented,
        so the fold label and the label users see cannot drift apart.
        """
        from ecgbench.labels import LabelSourceMissingError
        from ecgbench.labels.mimic_iv_ecg import UNKNOWN, load_labels

        try:
            labels = load_labels(data_path, config)
        except LabelSourceMissingError as e:
            logger.warning(
                "%s — every record will be stratified as '%s', so the split becomes "
                "purely patient-grouped. Fold membership stays reproducible; only the "
                "class balance across folds is lost.",
                e,
                UNKNOWN,
            )
            df[STRATIFY_COLUMN] = UNKNOWN
            return df

        mapped = df[config.record_id_column].map(labels[STRATIFY_COLUMN])
        n_missing = int(mapped.isna().sum())
        if n_missing:
            # A filtered local machine_measurements.csv is the usual cause; the
            # official file has one row per study, so this should be zero.
            logger.warning(
                "%d of %d records have no row in the measurements file and are "
                "stratified as '%s'. The official machine_measurements.csv covers "
                "every study — check your copy against the release's SHA256SUMS.txt.",
                n_missing,
                len(df),
                UNKNOWN,
            )
        df[STRATIFY_COLUMN] = mapped.fillna(UNKNOWN)
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the pooled first-report-line label attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("primary_report")
        counts = labels.value_counts()
        logger.info(
            "Stratifying on %d classes; top 5:\n%s",
            len(counts),
            counts.head(5).to_string(),
        )
        return labels
