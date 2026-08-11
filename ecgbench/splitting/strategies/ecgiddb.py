"""
ECG-ID splitting strategy.

Nothing machine-readable ships with this dataset except ``RECORDS``. The
demographics are three comment lines per ``.hea`` (``# Age:``, ``# Sex:``,
``# ECG date:``) and the beat positions are in the ``.atr`` files, so
``load_metadata`` builds a metadata CSV from all three via
``ecgbench.labels.ecgiddb`` — the same loader users get from ``load_labels``, so
the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split.**

**Record names are not unique, so the record id is synthesised.** Every
``Person_NN/`` directory numbers its records from ``rec_1``, so ``rec_1`` names 90
different recordings. ``record_id`` is ``"Person_01_rec_1"`` and ``signal_path``
keeps the release's own ``"Person_01/rec_1"``, which is what wfdb is handed.

**Grouping is by subject and that is what makes the folds unusable for this
database's own task.** ECG-ID exists to test identification of the 90 people in
it, so its label is ``subject_id`` — which is also ``patient_id_column``. Every
group is therefore exactly one class, and ``StratifiedGroupKFold`` puts each
subject's records wholly inside one fold, so no fold's model has ever seen the
person it would be asked to recognise. That is the right default for any other use
of these recordings — a model trained on Person_02's 22 records and tested on
Person_02 again is measuring nothing — and the wrong one for identification, which
needs a *within*-subject split. ``session_index`` and ``is_multi_session`` are
exposed by the label loader for exactly that: hold out a subject's later sessions.
See ``ecgbench.labels.ecgiddb.load_labels`` and the dataset page.

Folds are stratified on ``stratify_class``, sex crossed with a single age cut at 30
— ``female_le30`` (36 subjects, 124 records), ``male_le30`` (27/99), ``male_gt30``
(17/55), ``female_gt30`` (10/32). ``attach_stratify_class`` documents why that
cross and not a finer one.
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


@register("ecgiddb")
class ECGIDDBSplitter(DatasetSplitter):
    """ECG-ID strategy: header + annotation metadata, subject-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # All three carry the "Person_"/"rec_" prefixes, so none of them is
                # all-digits and pandas would not coerce them anyway. Pinned as str
                # so that stays true if a re-release ever drops a prefix.
                dtype={"record_id": str, "subject_id": str, "signal_path": str},
            )

        from ecgbench.labels.ecgiddb import load_labels

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
        """Return the sex x age-cut class attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("sex_x_age")

        counts = labels.value_counts()
        logger.info("Fold classes (sex x age cut at 30):\n%s", counts.to_string())
        if "subject_id" in df.columns:
            subjects = df.groupby(STRATIFY_COLUMN)["subject_id"].nunique()
            logger.info("Subjects per fold class:\n%s", subjects.to_string())
            # StratifiedGroupKFold keeps groups intact, so a class with fewer
            # subjects than folds simply cannot appear in every fold. Its own
            # message names neither the config nor the column, so say it here.
            if int(subjects.min()) < 10:
                logger.warning(
                    "Smallest fold class holds %d subjects, fewer than the 10 folds "
                    "ECGBench generates; some folds will contain none of it. Widen the "
                    "cells in ecgbench.labels.ecgiddb.attach_stratify_class.",
                    int(subjects.min()),
                )
            logger.info(
                "%d records from %d subjects, 1 to %d each; %d subjects were recorded on "
                "more than one day. Folds group by subject, so they cannot be used for "
                "the identification task this database was built for — see "
                "ecgbench.labels.ecgiddb.load_labels.",
                len(df),
                df["subject_id"].nunique(),
                int(df["n_records_for_subject"].max()) if "n_records_for_subject" in df else -1,
                (
                    int(df.groupby("subject_id")["is_multi_session"].first().sum())
                    if "is_multi_session" in df
                    else -1
                ),
            )
        return labels
