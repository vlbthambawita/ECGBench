"""
Apnea-ECG splitting strategy.

No machine-readable metadata ships with this database — the record list, the
per-minute apnea annotations, the polysomnography indices and the demographics
live in four different places — so ``load_metadata`` builds a metadata CSV via
``ecgbench.labels.apnea_ecg``, the same loader users get from ``load_labels``, so
the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split, and the second is the reason
this splitter exists at all.**

The partition is the 70 single-channel ECG records. ``RECORDS`` lists 86 names,
but 8 of them (``a01r``, …) hold respiration and SpO2 with no ECG, and 8 more
(``a01er``, …) point their headers at the very same ``.dat`` as the plain record.
``scan_records`` keeps only records that two independent filters agree on.

**The release's own learning/test split leaks subjects, and folds are grouped to
avoid it.** Apnea-ECG publishes no subject identifier, so nothing warns a user
that its 70 records come from 30 subjects, 27 of whom contributed more than one
night. Recovering the grouping from the published demographics (and from two
bit-identical duplicate recordings) shows that 18 of those 30 subjects — 49 of
the 70 records — have recordings in *both* the challenge learning set and the
challenge test set. ``has_predefined_splits`` is therefore false and
``patient_id_column`` is ``subject_id``, which routes ``engine.py`` through
``StratifiedGroupKFold``. ``challenge_set`` survives as a label column so the
original 2000 challenge result stays reproducible; it is not a split.
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

#: Record names ("a01") and signal paths ("a01") are handed to wfdb as record
#: stems and must not arrive as anything but strings. They carry no leading
#: zeros, so ``zero_padded_identifiers`` stays false — this is only about not
#: letting pandas guess a dtype for a column it has never seen.
_IDENTIFIER_DTYPES = {"record_name": str, "signal_path": str, "subject_id": str}


@register("apnea_ecg")
class ApneaECGSplitter(DatasetSplitter):
    """Apnea-ECG splitting strategy: derived metadata, subject-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype=_IDENTIFIER_DTYPES,
            )

        from ecgbench.labels.apnea_ecg import load_labels

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
        """Return the A/B/C apnea class attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("apnea_class")

        counts = labels.value_counts().sort_index()
        logger.info("Fold classes (apnea_class):\n%s", counts.to_string())
        # StratifiedGroupKFold raises only when EVERY class is smaller than
        # n_folds; 40/10/20 clears it comfortably. Its message names neither the
        # config nor the column, so say it here instead.
        if counts.max() < 10:
            logger.warning(
                "Largest fold class holds %d records, fewer than the 10 folds "
                "ECGBench generates; StratifiedGroupKFold will fail.",
                int(counts.max()),
            )

        if "subject_id" in df.columns:
            n_subjects = df["subject_id"].nunique()
            logger.info(
                "%d records from %d subjects (%.1f h of signal, %d annotated minutes "
                "of which %.1f%% apnea); folds are grouped on subject_id",
                len(df),
                n_subjects,
                df["duration_secs"].sum() / 3600 if "duration_secs" in df else float("nan"),
                int(df["n_annotated_minutes"].sum()) if "n_annotated_minutes" in df else 0,
                (
                    100 * df["n_apnea_minutes"].sum() / df["n_annotated_minutes"].sum()
                    if {"n_apnea_minutes", "n_annotated_minutes"} <= set(df.columns)
                    else float("nan")
                ),
            )
            if n_subjects < 10:
                logger.warning(
                    "Only %d subject groups for 10 folds; StratifiedGroupKFold cannot "
                    "fill every fold.",
                    n_subjects,
                )

        return labels
