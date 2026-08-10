"""
BIDMC Congestive Heart Failure splitting strategy.

Nothing machine-readable ships with this dataset beyond a one-line
``#Age: <n>  Sex: <X>  NYHA class: <c>`` header comment, so ``load_metadata``
builds a metadata CSV from the headers and the ``.ecg`` annotations via
``ecgbench.labels.chfdb`` — the same loader users get from ``load_labels``, so the
stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split.**

There is no clinical label to stratify on. All 15 subjects have severe congestive
heart failure, NYHA class III–IV, so ``cohort_label`` and ``nyha_class`` are one
value each across the release. Folds are stratified on the one axis PhysioNet
documents about the cohort: sex, 11 men and 4 women. The clinically interesting
axis — ventricular ectopy burden, which spans 0.017% to 20.52% here — is
arithmetically unusable over 15 records and 10 folds, and it rests on unaudited
detector output besides; see ``ecgbench.labels.chfdb.attach_stratify_class`` for
the three candidate cuts and which of them ``StratifiedKFold`` rejects.

There is no patient grouping to do. PhysioNet describes 15 recordings from 15
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


@register("chfdb")
class CHFDBSplitter(DatasetSplitter):
    """BIDMC CHF splitting strategy: header + annotation metadata, sex-balanced folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "chf01" and signal_path likewise; both are handed
                # to wfdb as record stems, so neither may arrive as anything else.
                dtype={"record_name": str, "signal_path": str},
            )

        from ecgbench.labels.chfdb import load_labels

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
        # 11/4 is fine and 8/7 would not be. Its message names neither the config
        # nor the column, so say it here instead.
        if counts.max() < 10:
            logger.warning(
                "Largest fold class holds %d records, fewer than the 10 folds "
                "ECGBench generates; StratifiedKFold will fail. Widen the classes in "
                "ecgbench.labels.chfdb.attach_stratify_class.",
                int(counts.max()),
            )
        logger.info(
            "%d records, %.1f h of signal; %d unaudited beats, %.2f%% ventricular; "
            "%d records carry rhythm annotation",
            len(df),
            df["duration_secs"].sum() / 3600 if "duration_secs" in df else float("nan"),
            int(df["n_beats"].sum()) if "n_beats" in df else -1,
            (
                100 * df["n_veb"].sum() / df["n_beats"].sum()
                if {"n_veb", "n_beats"} <= set(df.columns) and df["n_beats"].sum()
                else float("nan")
            ),
            int(df["has_rhythm_annotation"].sum()) if "has_rhythm_annotation" in df else -1,
        )
        return labels
