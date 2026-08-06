"""
PhysioNet/CinC Challenge 2020 splitting strategy.

The release ships no metadata file, so ``load_metadata`` builds one from the WFDB
headers via ``ecgbench.labels.challenge2020`` — the same loader users get from
``load_labels``, so the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

Stratification uses ``stratify_dx_abbreviation``: the globally rarest SNOMED code
each record carries. Records are multi-label with no clinically meaningful primary
code (``#Dx`` ordering differs by cohort), so some reduction is unavoidable, and
rarest-first is the one that keeps the tail representable — it spreads all 111
classes across folds, where taking the first listed code collapses to 102 and
leaves 39 classes below ten records instead of 35.

**No patient grouping is possible.** No patient identifiers are published and
records were renamed. Four of the six cohorts are one-record-per-patient, but the
516 ``ptb`` records and 74 ``st_petersburg_incart`` records are not, so those 590
records (1.37%) can place one patient in several folds.
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
STRATIFY_COLUMN = "stratify_dx_abbreviation"

#: Stratification classes smaller than this are pooled into "OTHER" so 10-fold
#: stratification stays well defined. 35 of the 111 classes fall below it,
#: covering 127 records.
MIN_CLASS_SIZE = 10

OTHER = "OTHER"

#: Cohorts whose source datasets have several recordings per patient, so folds
#: can split a patient. Logged on every run rather than buried in the docs.
_MULTI_RECORD_COHORTS = ("ptb", "st_petersburg_incart")


@register("challenge2020")
class Challenge2020Splitter(DatasetSplitter):
    """Challenge 2020 splitting strategy.

    - Builds (and caches) the metadata CSV from the per-record WFDB headers
    - Stratifies on the rarest SNOMED code per record, pooling rare classes
    - No patient grouping: no patient identifiers are published
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype={"dx": str, "stratify_dx": str, "age": str},
            )

        from ecgbench.labels.challenge2020 import load_labels

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
        """Return the rarest-code label from the loader, pooling rare classes."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        if "source" in df.columns:
            present = [c for c in _MULTI_RECORD_COHORTS if c in set(df["source"])]
            if present:
                n = int(df["source"].isin(present).sum())
                logger.warning(
                    "No patient identifiers are published for this dataset, so the %d "
                    "records from %s (whose source datasets have several recordings per "
                    "patient) may be split across folds.",
                    n,
                    " and ".join(present),
                )

        labels = df[STRATIFY_COLUMN].fillna(OTHER).replace("", OTHER)

        counts = labels.value_counts()
        rare = set(counts[counts < MIN_CLASS_SIZE].index)
        if rare:
            logger.info(
                "Pooling %d classes with <%d records into '%s': %s",
                len(rare),
                MIN_CLASS_SIZE,
                OTHER,
                sorted(rare),
            )
            labels = labels.where(~labels.isin(rare), OTHER)

        labels = labels.rename("stratify_class")
        logger.info("Stratification class distribution:\n%s", labels.value_counts().to_string())
        return labels
