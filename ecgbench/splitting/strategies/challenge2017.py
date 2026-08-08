"""
PhysioNet/CinC Challenge 2017 splitting strategy.

No metadata file ships — the release has ``training/RECORDS`` (a path list) and
``training/REFERENCE.csv`` (headerless ``record,code`` pairs) and nothing else —
so ``load_metadata`` builds one from those plus the WFDB headers via
``ecgbench.labels.challenge2017``, the same loader users get from
``load_labels``, so the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Stratification is the four-class label unchanged.** No reduction is needed —
records are single-label, none is unlabelled, and the rarest class (``noisy``,
279 records) spreads over ten folds at about 28 per fold. This is one of the few
datasets here where the fold label and the training target are the same thing.

**No patient grouping is possible, and that is worse here than usual.** No
identifiers ship, and unlike a hospital release this one cannot even assert one
record per person: the recordings came from members of the public who had bought
an AliveCor handheld device, and nothing in the release says whether anyone
contributed more than one. A repeat contributor would straddle folds
undetectably. Say so wherever these folds are used.

**The shipped ``validation/`` directory is deliberately not a split.** Its 300
``.mat`` files are byte-identical to their ``training/`` counterparts, so it is a
duplicate subset, not held-out data. Those records take part in the folds like
any other and carry ``in_challenge_validation_subset`` so they can be excluded
from a comparison against published challenge results.
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


@register("challenge2017")
class Challenge2017Splitter(DatasetSplitter):
    """Challenge 2017 splitting strategy: header + REFERENCE metadata, ungrouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # class_code is "~" for the noisy class, which pandas is happy
                # with, but the version columns must not be inferred either.
                dtype={
                    "record_name": str,
                    "signal_path": str,
                    "class_code": str,
                    "class_code_v0": str,
                    "class_code_v1": str,
                    "class_code_v2": str,
                    "class_code_v3": str,
                },
            )

        from ecgbench.labels.challenge2017 import load_labels

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
        """Return the four-class rhythm label attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("rhythm_class")
        logger.info("Rhythm class distribution:\n%s", labels.value_counts().to_string())
        logger.info(
            "No patient grouping: this release ships no identifiers and does not "
            "claim one recording per person."
        )
        return labels
