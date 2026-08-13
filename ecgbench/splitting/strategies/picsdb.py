"""
Preterm Infant Cardio-Respiratory Signals splitting strategy.

Nothing tabular ships with this database, so ``load_metadata`` builds a metadata
CSV from the headers, the ``.atr`` bradycardia onsets and the ``.qrsc`` R peaks
via ``ecgbench.labels.picsdb`` — the same loader users get from ``load_labels``,
so the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata. It
also saves re-scanning 1.58 billion samples for converter clipping and constant
runs on every later run.

**Three things about this dataset shape the split.**

**Only the ten ECG records are split.** ``RECORDS`` lists twenty names — an
``infantN_ecg`` and an ``infantN_resp`` for each infant. The respiration records
are not ECG, carry no beat annotation and get no row; each infant's row points at
its own through ``resp_path``.

**Ten records from ten infants make ten folds of one infant each**, so the
partition is leave-one-infant-out and the default mapping gives train = folds 1-8,
val = fold 9, test = fold 10 — one infant to validate on and one to test on. That
is the arithmetic of the release, not a defect: use ``split=None`` with
``fold_numbers=[...]`` for anything that needs a real evaluation set.

**There is nothing to stratify on, and that is measured.** Every fold is one
infant, and ``StratifiedGroupKFold`` requires every class to hold at least
``n_folds`` records — which over ten records admits exactly one class. See
``ecgbench.labels.picsdb.attach_stratify_class`` for the three axes that were
tried and the error each produces.
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


@register("picsdb")
class PICSDBSplitter(DatasetSplitter):
    """picsdb strategy: generated metadata, one infant per fold, constant label."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "infant1_ecg" and signal_path likewise; both are
                # handed to wfdb as record stems, so neither may arrive as anything
                # else. bradycardia_onsets_secs is a pipe-joined list, not a number,
                # and a record with no onsets would otherwise read back as NaN.
                # config.identifier_dtypes() is empty for this dataset; see the
                # zero_padded_identifiers comment in the YAML.
                dtype={
                    "record_name": str,
                    "subject_id": str,
                    "signal_path": str,
                    "resp_path": str,
                    "bradycardia_onsets_secs": str,
                },
            )

        from ecgbench.labels.picsdb import load_labels

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
        ten infants, which is the leave-one-infant-out structure this database
        wants. ``attach_stratify_class`` documents the axes that were rejected and
        the error each raises.
        """
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("cohort")

        n_subjects = df["subject_id"].nunique() if "subject_id" in df else 0
        # StratifiedGroupKFold raises only when EVERY class is smaller than
        # n_folds, which one class of 10 records over 10 folds clears exactly; but
        # it emits empty folds without complaint once n_folds exceeds the group
        # count, and that failure is silent. Say it here instead.
        if n_subjects and n_subjects < config.n_folds:
            logger.warning(
                "%d infants over %d folds: StratifiedGroupKFold keeps groups intact, so "
                "%d fold(s) will come out EMPTY without raising. Lower n_folds in "
                "picsdb.yaml.",
                n_subjects,
                config.n_folds,
                config.n_folds - n_subjects,
            )
        logger.info(
            "%d ECG records from %d infants; %.1f h of signal at %s Hz, %d manually "
            "validated bradycardia onsets, %d verified R peaks covering %.1f%% of the "
            "recorded time; %.1f h clipped at a converter rail",
            len(df),
            n_subjects,
            df["duration_secs"].sum() / 3600 if "duration_secs" in df else float("nan"),
            sorted(df["sampling_rate"].unique().tolist()) if "sampling_rate" in df else "-",
            int(df["n_bradycardias"].sum()) if "n_bradycardias" in df else -1,
            int(df["n_rpeaks"].sum()) if "n_rpeaks" in df else -1,
            100
            * (df["annotated_fraction"] * df["duration_secs"]).sum()
            / df["duration_secs"].sum()
            if "annotated_fraction" in df
            else float("nan"),
            df["rail_secs"].sum() / 3600 if "rail_secs" in df else float("nan"),
        )
        return labels
