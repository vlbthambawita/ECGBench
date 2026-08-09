"""
MIT-BIH Normal Sinus Rhythm splitting strategy.

Nothing machine-readable ships with this dataset beyond a one-line ``# <age>
<sex>`` header comment, so ``load_metadata`` builds a metadata CSV from the
headers and the ``.atr`` annotations via ``ecgbench.labels.nsrdb`` — the same
loader users get from ``load_labels``, so the stratification label and the
exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split.**

There is no clinical label to stratify on, and that is what the database is for.
Every subject was found to have no significant arrhythmia, and the release ships
no rhythm annotations at all — ``cohort_label`` is ``normal_sinus_rhythm`` for all
18 records. Folds are therefore stratified on the one axis the release documents
about its cohort: sex, 13 women and 5 men. See
``ecgbench.labels.nsrdb.attach_stratify_class`` for why nothing finer survives 18
records over 10 folds.

There is no patient grouping to do. PhysioNet describes 18 recordings from 18
subjects and the headers carry no subject identifier, so ``patient_id_column`` is
null and ``engine.py`` uses plain ``StratifiedKFold``. Folds hold one or two
records each.
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


@register("nsrdb")
class NSRDBSplitter(DatasetSplitter):
    """MIT-BIH NSR splitting strategy: header + annotation metadata, sex-balanced folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "16265" and signal_path likewise; both are handed
                # to wfdb as record stems, so neither may arrive as an int.
                dtype={"record_name": str, "signal_path": str},
            )

        from ecgbench.labels.nsrdb import load_labels

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
        """Return the subject sex attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("sex")

        counts = labels.value_counts()
        logger.info("Fold classes (sex):\n%s", counts.to_string())
        # StratifiedKFold raises only when EVERY class is smaller than n_folds, so
        # 13/5 is fine and 5/5 would not be. Its message names neither the config
        # nor the column, so say it here instead.
        if counts.max() < 10:
            logger.warning(
                "Largest fold class holds %d records, fewer than the 10 folds "
                "ECGBench generates; StratifiedKFold will fail. Widen the classes in "
                "ecgbench.labels.nsrdb.attach_stratify_class.",
                int(counts.max()),
            )
        logger.info(
            "%d records, %.1f h of signal; %.1f%% of it carries beat annotations",
            len(df),
            df["duration_secs"].sum() / 3600 if "duration_secs" in df else float("nan"),
            (
                100 * df["annotated_secs"].sum() / df["duration_secs"].sum()
                if {"annotated_secs", "duration_secs"} <= set(df.columns)
                else float("nan")
            ),
        )
        return labels
