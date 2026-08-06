"""
CPSC 2018 splitting strategy.

The mirror ships no metadata file, so ``load_metadata`` builds one from the WFDB
headers via ``ecgbench.labels.cpsc_2018`` — the same loader users get from
``load_labels``, so the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

Stratification uses ``stratify_dx_abbreviation``: the globally rarest of the nine
CPSC classes each record carries. Records are multi-label (476 of 6,877) and the
shipped ``#Dx`` order is a sort by class index rather than CPSC's original
First/Second/Third labelling, so there is no primary diagnosis to take instead
and some reduction is unavoidable. Rarest-first keeps the tail representable:
the smallest resulting class is STE with 220 records, comfortably above ten
folds, so unlike Challenge 2020 nothing needs pooling into an OTHER bucket.

**No patient grouping is possible.** No patient identifiers are published. The
challenge describes 6,877 recordings from 11 hospitals and never mentions repeat
patients, but nothing in the files proves one record per patient either.
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


@register("cpsc_2018")
class CPSC2018Splitter(DatasetSplitter):
    """CPSC 2018 splitting strategy.

    - Builds (and caches) the metadata CSV from the per-record WFDB headers
    - Stratifies on the rarest of the nine CPSC classes per record
    - No patient grouping: no patient identifiers are published
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype={"dx": str, "stratify_dx": str, "dx_class_indices": str, "age": str},
            )

        from ecgbench.labels.cpsc_2018 import load_labels

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
        """Return the rarest-class label from the loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].fillna("UNMAPPED").replace("", "UNMAPPED")
        labels = labels.rename("stratify_class")
        logger.info("Stratification class distribution:\n%s", labels.value_counts().to_string())
        return labels
