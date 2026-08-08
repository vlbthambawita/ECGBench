"""
Long Term AF Database splitting strategy.

Nothing machine-readable ships with this dataset — the headers carry no comment
lines at all — so ``load_metadata`` builds a metadata CSV from the ``.atr``
reference annotations and the ``.qrs`` detections via ``ecgbench.labels.ltafdb``,
the same loader users get from ``load_labels``. The stratification label is a
column that loader attaches, so the exposed label and the fold label cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Three things about this dataset shape the split.**

Stratification is on ``af_class`` directly — ``minimal`` (18 records), ``paroxysmal``
(33) and ``sustained`` (33). ``afdb`` had to coarsen the same quantity to a binary
label because ``StratifiedKFold`` needs at least ``n_folds`` members per class and
25 records could not spare 10 for a third class. 84 records can, so here the label
a reader wants and the label the folds use are the same label. ``dominant_rhythm``
would not work: 42 records are dominated by ``N``, 41 by ``AFIB`` and exactly one
by ``SBR``, and a class of one cannot be spread over ten folds.

There is no patient grouping to do, and that is a fact about the release rather
than an omission. The headers carry no subject identifier of any kind — no age,
no sex, no tape number — so one record per subject is the most that can be
asserted, and ``config.patient_id_column`` is null. ``engine.py`` then uses plain
``StratifiedKFold``.

Record ids are zero-padded and must stay strings. ``00``, ``01``, ``03``, ``05``,
``06``, ``07`` and ``08`` become ``0``, ``1``, ``3``, ``5``, ``6``, ``7`` and ``8``
under pandas' default inference, at which point they no longer name a record and
``data_path / "0"`` is not a file. Every read here passes
``config.identifier_dtypes()``; ``export_splits`` refuses to write the fold CSVs
if the config ever loses ``zero_padded_identifiers``.
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


@register("ltafdb")
class LTAFDBSplitter(DatasetSplitter):
    """Long Term AF Database strategy: annotation-derived metadata, AF-class folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "00" and signal_path likewise. Both MUST stay
                # strings: read as numbers they lose the leading zeros, and wfdb
                # is then handed a record called "0" that does not exist.
                dtype=config.identifier_dtypes(),
            )

        from ecgbench.labels.ltafdb import load_labels

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
        """Return the AF-class label attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("af_class")

        counts = labels.value_counts()
        logger.info("AF fold classes:\n%s", counts.to_string())
        # A class smaller than n_folds is what makes StratifiedKFold raise, and the
        # message it raises with does not mention the config. Say it here instead.
        if counts.min() < 10:
            logger.warning(
                "Smallest fold class holds %d records, fewer than the 10 folds "
                "ECGBench generates; StratifiedKFold will fail. Widen the cuts in "
                "ecgbench.labels.ltafdb.MINIMAL_AF_BURDEN / SUSTAINED_AF_BURDEN.",
                int(counts.min()),
            )
        return labels
