"""
MIT-BIH Atrial Fibrillation splitting strategy.

Nothing machine-readable ships with this dataset — not even the header comments
``mitdb`` has — so ``load_metadata`` builds a metadata CSV from the ``.atr``
rhythm annotations and the ``.qrs`` beat annotations via ``ecgbench.labels.afdb``,
the same loader users get from ``load_labels``. The stratification label is a
column that loader attaches, so the exposed label and the fold label cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Three things about this dataset shape the split.**

All 25 records go into the partition, including the two whose ECG was never
released. 00735 and 03665 are in the shipped ``RECORDS`` file and carry real
rhythm labels; what they lack is a ``.dat``. Dropping them here would make the
``original`` version disagree with the release's own record count for a reason
that belongs in ``quality_issues``, so instead validation fails them on
``corrupt_header`` and ``clean`` holds the 23 that can be read. Fold membership is
identical between the two versions either way.

There is no patient grouping to do, and that is a fact about the release rather
than an omission. The headers carry no subject identifier of any kind — no age,
no sex, no tape number — so one record per subject is the most that can be
asserted, and ``config.patient_id_column`` is null. ``engine.py`` then uses plain
``StratifiedKFold``.

Stratification is binary because 25 records admit nothing else. See
``ecgbench.labels.afdb.attach_stratify_class``: ``StratifiedKFold`` needs at least
``n_folds`` members in every class, so with 10 folds no class may hold fewer than
10 records. The three-level ``af_class`` has 3 ``sustained`` records and
``dominant_rhythm`` has 1 ``J``; a single cut at 20% AF burden gives 14 / 11 and
is the only split of this data with margin over that floor.
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


@register("afdb")
class AFDBSplitter(DatasetSplitter):
    """MIT-BIH AFDB splitting strategy: annotation-derived metadata, AF-burden folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "00735" and signal_path likewise. Both MUST stay
                # strings: read as numbers they lose the leading zeros, and
                # wfdb is then handed a record called "735" that does not exist.
                dtype=config.identifier_dtypes(),
            )

        from ecgbench.labels.afdb import load_labels

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
        """Return the AF-burden class attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("af_burden_class")

        counts = labels.value_counts()
        logger.info("AF burden fold classes:\n%s", counts.to_string())
        # A class smaller than n_folds is what makes StratifiedKFold raise, and the
        # message it raises with does not mention the config. Say it here instead.
        if counts.min() < 10:
            logger.warning(
                "Smallest fold class holds %d records, fewer than the 10 folds "
                "ECGBench generates; StratifiedKFold will fail. Widen the cut in "
                "ecgbench.labels.afdb.STRATIFY_AF_BURDEN.",
                int(counts.min()),
            )

        if "has_signals" in df.columns:
            logger.info(
                "%d of %d records ship signals; the other %d are annotation-only "
                "and validation excludes them from the clean version",
                int(df["has_signals"].sum()),
                len(df),
                int((~df["has_signals"].astype(bool)).sum()),
            )
        return labels
