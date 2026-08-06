"""
Shandong Provincial Hospital (SPH) splitting strategy.

The release ships ``metadata.csv`` with no signal-path column — records are found
by convention, ``records/<ECG_ID>.h5`` — and its ``AHA_Code`` strings need the
``code.csv`` codebook before they mean anything. So ``load_metadata`` builds the
frame from ``ecgbench.labels.sph``, the same loader users get from
``load_labels``, and the stratification label and the exposed labels cannot drift.

Writing that frame to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata and
no way to resolve a single signal path.

Stratification uses ``stratify_code``: the globally rarest of the 44 AHA primary
codes each record carries. Records are multi-label (3,724 of 25,770) and the
statement order is not a documented ranking, so there is no primary diagnosis to
take instead and some reduction is unavoidable. Rarest-first keeps the tail
representable — it spreads all 44 codes across folds, where taking the first
listed statement collapses to 40 and leaves 12 codes below ten records instead of
9. The 9 that remain below ten (49 records in total) are pooled into ``OTHER``.

**Patient grouping is real here.** 24,666 patients hold the 25,770 records, and
1,066 of them contributed between two and five, so ``patient_id`` grouping is
what keeps the same heart out of both train and test.
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
STRATIFY_COLUMN = "stratify_code"

#: Stratification classes smaller than this are pooled into "OTHER" so 10-fold
#: stratification stays well defined. 9 of the 44 codes fall below it, covering
#: 49 records.
MIN_CLASS_SIZE = 10

OTHER = "OTHER"


@register("sph")
class SPHSplitter(DatasetSplitter):
    """SPH splitting strategy.

    - Builds (and caches) the metadata CSV from ``metadata.csv`` + ``code.csv``,
      adding the ``records/<ECG_ID>.h5`` signal path the release leaves implicit
    - Stratifies on the rarest AHA primary code per record, pooling rare codes
    - Groups folds on ``patient_id``: 1,066 patients have more than one record
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # All four are codes, not numbers: leading zeros do not occur but
                # pandas would still read aha_code "1" as an int and break the
                # string operations downstream.
                dtype={
                    "ecg_id": str,
                    "patient_id": str,
                    "aha_code": str,
                    "aha_statements": str,
                    "aha_primary_codes": str,
                    "aha_modifier_codes": str,
                    STRATIFY_COLUMN: str,
                },
            )

        from ecgbench.labels.sph import load_labels

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
        """Return the rarest-code label from the loader, pooling rare codes."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).replace({"": OTHER, "nan": OTHER})

        counts = labels.value_counts()
        rare = set(counts[counts < MIN_CLASS_SIZE].index)
        if rare:
            logger.info(
                "Pooling %d AHA code(s) with <%d records into '%s': %s",
                len(rare),
                MIN_CLASS_SIZE,
                OTHER,
                sorted(rare),
            )
            labels = labels.where(~labels.isin(rare), OTHER)

        labels = labels.rename("stratify_class")
        logger.info("Stratification class distribution:\n%s", labels.value_counts().to_string())
        return labels
