"""
BUT QDB splitting strategy.

Nothing in the release can be split as shipped. There is no metadata file at all:
``subject-info.csv`` holds five demographic columns keyed by *recording* and carries
no path, no record geometry and nothing about the annotations, while the quality
labels live in 18 separate headerless 12-column CSVs of sample-index intervals. So
``load_metadata`` builds a metadata CSV via ``ecgbench.labels.butqdb``, the same
loader users get from ``load_labels``, so the stratification label and the exposed
labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata. It
also saves re-scanning 1.72 billion samples for converter saturation on every later
run.

**Three things about this dataset shape the split.**

**The subject grouping has to be recovered from the record name.** The release
documents the six digits as ``<3-digit subject><3-digit session>``, so 18 records
reduce to 15 subjects — 100 recorded twice, 103 three times. Nothing in the shipped
files states it as a column, and ``subject-info.csv`` hides it by repeating the same
demographics under each recording's ID. The label loader recovers it and checks it,
by requiring that records sharing a prefix carry identical demographics. Without the
grouping, two 24-hour recordings of the same chest a day apart could land on opposite
sides of a fold boundary.

**Folds are stratified on class-3 burden, not on sex.** Six of the 18 records have
more than 1% of their annotated time graded unusable, so at most six of ten folds can
hold one — and this axis achieves exactly six. Sex reaches five and buys nothing,
because one or two records per fold make any demographic fraction 0, 0.5 or 1
regardless. ``ecgbench.labels.butqdb.attach_stratify_class`` tabulates the five
alternatives that were measured.

**The ten folds are for cross-validation, not for the default 8/1/1 split.** 18
records over 10 folds leaves one record in ``val`` and one in ``test``. That is the
arithmetic of the release, not a defect in the partition: use ``split=None`` with
``fold_numbers=[...]`` for anything that needs an evaluation set.
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


@register("butqdb")
class BUTQDBSplitter(DatasetSplitter):
    """BUT QDB strategy: generated metadata, subject-grouped, class-3-balanced folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # LOAD-BEARING, not defensive. record_id and subject_id are
                # all-digits, so a plain read gives int64 here and the label loader
                # gives str — and StratifiedGroupKFold orders its groups by value, so
                # the first run (built from the loader) and every later one (built
                # from this cache) would partition differently and stamp different
                # fold_digests into manifest.json for identical data. signal_path and
                # acc_path go to wfdb as record stems and must not arrive as numbers
                # either. config.identifier_dtypes() is empty for this dataset; see
                # the zero_padded_identifiers comment in the YAML.
                dtype={
                    "record_id": str,
                    "subject_id": str,
                    "signal_path": str,
                    "acc_path": str,
                },
            )

        from ecgbench.labels.butqdb import load_labels

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
        """Return the class-3 burden band attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("class3_burden")

        counts = labels.value_counts()
        logger.info("Fold classes (class-3 burden):\n%s", counts.to_string())
        # StratifiedGroupKFold raises only when EVERY class holds fewer records than
        # there are folds, and 12/6 clears that on the larger class. Its message names
        # neither the config nor the column, so say it here.
        if int(counts.max()) < config.n_folds:
            logger.warning(
                "Largest fold class holds %d records, fewer than the %d folds "
                "ECGBench generates; StratifiedGroupKFold will raise. Widen the "
                "classes in ecgbench.labels.butqdb.attach_stratify_class.",
                int(counts.max()), config.n_folds,
            )
        if "subject_id" in df.columns:
            subjects = df.groupby(STRATIFY_COLUMN)["subject_id"].nunique()
            logger.info("Subjects per fold class:\n%s", subjects.to_string())
            logger.info(
                "%d records from %d subjects, %.1f h of signal of which %.1f h "
                "(%.1f%%) carries a sample-level quality annotation; %d records are "
                "annotated end to end and hold %.1f%% of that. `clean` is all %d "
                "records by design — see point 7 of butqdb.yaml.",
                len(df),
                df["subject_id"].nunique(),
                df["duration_secs"].sum() / 3600,
                df["annotated_secs"].sum() / 3600,
                100 * df["annotated_secs"].sum() / df["duration_secs"].sum(),
                int(df["fully_annotated"].sum()),
                100
                * df.loc[df["fully_annotated"], "annotated_secs"].sum()
                / df["annotated_secs"].sum(),
                len(df),
            )
        return labels
