"""
St. Vincent's/UCD Sleep Apnea Database splitting strategy.

The only tabular file in the release is ``SubjectDetails.xls`` — a 2003-era BIFF
spreadsheet, which ``pandas.read_csv`` cannot open and which carries no signal
path. So ``load_metadata`` builds a metadata CSV from it, the sleep-stage files,
the respiratory-event files and the EDF headers, via ``ecgbench.labels.ucddb`` —
the same loader users get from ``load_labels``, so the stratification label and
the exposed labels cannot drift apart.

Writing that cache to disk is load-bearing rather than a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself instead
of reusing this DataFrame, so an in-memory-only frame would leave validation with
no metadata and every record failing ``corrupt_header``.

**Two things shape the split, and neither is visible in the metadata.**

**ucddb014 and ucddb028 share one Holter recording, so they share one fold.**
Their ``_lifecard.edf`` payloads are bit-identical over all 20,782,080 bytes;
only the four-byte start-time field differs. Their polysomnograms, annotations
and demographics are all different, so the release presents them as two subjects,
and a subject-level grouping would put the same waveform in train and in test.
``patient_id_column`` is therefore ``recording_group``, which merges the pair into
``"ucddb014+ucddb028"`` and leaves the other 23 records as themselves — 24 groups
over 25 records. The column is deliberately not called ``subject_id``: it is a
recording identity, and these are two different people. See
``ecgbench.labels.ucddb.HOLTER_DUPLICATES`` for how it was established.

**The stratification label is coarser than the clinical grade, because 25
subjects cannot carry four classes over ten folds.** ``ahi_severity`` splits
normal 1 / mild 10 / moderate 6 / severe 8, and a class of one cannot be spread
over ten folds. ``stratify_class`` pools at the moderate-or-severe cut point
(AHI >= 15): 13 groups against 11, both comfortably above the fold count. See
``ecgbench.labels.ucddb.attach_stratify_class``.
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

#: Columns that must survive the CSV round trip as strings. ``record_name`` and
#: ``recording_group`` are handed straight to path building and fold grouping.
_STRING_COLUMNS = {
    "record_name": str,
    "recording_group": str,
    "signal_path": str,
    "psg_path": str,
    "stage_path": str,
    "respevt_path": str,
    "subject_number": str,
    "sex": str,
    "ahi_severity": str,
    "holter_duplicate_of": str,
    "stratify_class": str,
}


@register("ucddb")
class UCDDBSplitter(DatasetSplitter):
    """ucddb splitting strategy: generated metadata, duplicate-aware grouping."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path, sep=config.metadata_csv_separator, dtype=_STRING_COLUMNS
            )

        from ecgbench.labels.ucddb import load_labels

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
        """Return the moderate-or-severe OSA class attached by the label loader.

        Not ``config.label_column`` (``ahi_severity``), and the difference is the
        point: the four clinical grades put one subject in ``normal``, which ten
        folds cannot accommodate. See the module docstring.
        """
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("osa_class")

        group_column = config.patient_id_column
        if group_column and group_column in df.columns:
            n_groups = df[group_column].nunique()
            # StratifiedGroupKFold emits silently EMPTY folds once n_folds exceeds
            # the group count, so say it here rather than let a user find two
            # empty fold CSVs.
            if n_groups < config.n_folds:
                logger.warning(
                    "%d recording groups over %d folds: StratifiedGroupKFold keeps groups "
                    "intact, so some folds will be empty.",
                    n_groups,
                    config.n_folds,
                )
            per_group = df.groupby(group_column)[STRATIFY_COLUMN].first().value_counts()
            logger.info("OSA class by recording group:\n%s", per_group.to_string())

        logger.info("OSA class distribution:\n%s", labels.value_counts().to_string())
        return labels
