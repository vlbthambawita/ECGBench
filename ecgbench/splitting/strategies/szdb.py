"""
Post-Ictal Heart Rate Oscillations in Partial Epilepsy splitting strategy.

Nothing tabular ships with this dataset, so ``load_metadata`` builds a metadata
CSV from the headers, the ``.ari`` annotations and ``times.seize`` via
``ecgbench.labels.szdb`` — the same loader users get from ``load_labels``, so the
stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split, and they are both unusual.**

**The subject grouping is reconstructed, and without it this database leaks.**
The release ships 7 records and no subject identifier of any kind, but the paper
describes five patients. sz02, sz03 and sz04 are three recordings of the same
woman — established from beat morphology, confirmed by both counts the paper
states — so ``patient_id_column`` is ``subject_id`` and ``engine.py`` uses
``StratifiedGroupKFold``. Reading the absent column as "one record per patient"
would put the same woman in train and test. See
``ecgbench.labels.szdb.SUBJECT_IDS`` for the derivation.

**Five folds, not ten, and that is a property of the dataset rather than a flag.**
Seven records over five subjects cannot make ten folds: ``StratifiedGroupKFold``
raises outright once ``n_splits`` exceeds the record count, and produces silently
*empty* folds once it exceeds the number of groups. ``szdb.yaml`` therefore sets
``n_folds: 5``, which ``run_splits`` picks up when ``--n-folds`` is not passed, so
the canonical partition is one subject per fold and reproducible from the shipped
config alone. With the default mapping that is train = folds 1-3, val = fold 4,
test = fold 5 — five records to train on, one to validate on, one to test on.

There is nothing to stratify on. Every record is one cohort and every fold is one
subject; see ``ecgbench.labels.szdb.attach_stratify_class`` for the four candidate
axes and the arithmetic that rejects each.
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


@register("szdb")
class SZDBSplitter(DatasetSplitter):
    """szdb splitting strategy: generated metadata, one reconstructed subject per fold."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "sz01" and signal_path likewise; both are handed
                # to wfdb as record stems, so neither may arrive as anything else.
                # seizure_*_secs are pipe-joined lists, not numbers.
                dtype={
                    "record_name": str,
                    "signal_path": str,
                    "subject_id": str,
                    "seizure_starts_secs": str,
                    "seizure_ends_secs": str,
                    "seizure_durations_secs": str,
                },
            )

        from ecgbench.labels.szdb import load_labels

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
        """Return the constant cohort label attached by the label loader.

        A constant is only a legitimate stratification label because the split is
        grouped: it reduces ``StratifiedGroupKFold`` to a plain partition of the
        five subjects, which is the leave-one-subject-out structure this database
        wants. ``attach_stratify_class`` documents the axes that were rejected and
        why each fails arithmetically.
        """
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("cohort")

        n_subjects = df["subject_id"].nunique() if "subject_id" in df else 0
        # StratifiedGroupKFold raises only when EVERY class is smaller than
        # n_folds, which one class of 7 records over 5 folds clears; but it emits
        # empty folds without complaint once n_folds exceeds the group count, and
        # that failure is silent. Say it here instead.
        if n_subjects and n_subjects < config.n_folds:
            logger.warning(
                "%d subjects over %d folds: StratifiedGroupKFold keeps groups intact, so "
                "%d fold(s) will come out EMPTY without raising. Lower n_folds in "
                "szdb.yaml.",
                n_subjects,
                config.n_folds,
                config.n_folds - n_subjects,
            )
        logger.info(
            "%d records from %d reconstructed subjects (%s); %.2f h of signal, %d seizures, "
            "%d unaudited beat detections",
            len(df),
            n_subjects,
            df["subject_id"].value_counts().to_dict() if "subject_id" in df else "-",
            df["duration_secs"].sum() / 3600 if "duration_secs" in df else float("nan"),
            int(df["n_seizures"].sum()) if "n_seizures" in df else -1,
            int(df["n_beats"].sum()) if "n_beats" in df else -1,
        )
        return labels
