"""
STAFF III splitting strategy.

Nothing machine-readable ships with this dataset — the annotations are an
``.xlsx`` with one wide row per patient, holding merged header cells and up to
five balloon inflations per row. ``load_metadata`` unpivots that into one row per
record via ``ecgbench.labels.staffiii``, the same loader users get from
``load_labels``, so the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, and it reads it with ``pandas.read_csv``, which cannot open an .xlsx
at all. An in-memory-only frame would leave validation with no metadata.

**Patient grouping is mandatory here.** 520 records come from only 104 patients,
five each on average, and a patient's records are the same heart under the same
electrodes within one session — a baseline, an occlusion and a recovery minutes
apart. A record-level split would put all three in different folds and report a
score that is mostly patient identity.

**Stratification is on the occluded territory, not the protocol phase.** That
looks backwards, since ``recording_type`` is the dataset's actual label, but it
follows from the grouping: every patient contributes roughly the same mix of
phases, so the phase distribution comes out balanced whatever the split does,
while the occluded vessel varies between patients and will not balance itself.
The derivation lives in ``labels/staffiii.py:attach_stratify_class``.
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


@register("staffiii")
class STAFFIIISplitter(DatasetSplitter):
    """STAFF III splitting strategy: spreadsheet metadata, patient-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name ("001a"), patient_id ("patient001") and signal_path
                # ("data/001a") are all zero-padded or prefixed and must stay
                # strings. age is numeric-looking but blank for two patients, and
                # the event-time columns are semicolon-joined lists that pandas
                # would otherwise coerce to float when a record has exactly one.
                dtype={
                    "record_name": str,
                    "patient_id": str,
                    "signal_path": str,
                    "age": str,
                    "recording_index": str,
                    "inflation_start_s": str,
                    "deflation_s": str,
                    "inflation_duration_s": str,
                    "injection_s": str,
                },
            )

        from ecgbench.labels.staffiii import load_labels

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
        """Return the pooled occluded-artery territory attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("artery_territory")

        if config.patient_id_column and config.patient_id_column in df.columns:
            counts = df[config.patient_id_column].value_counts()
            logger.info(
                "Grouping %d records into folds by %d patients (mandatory: %d "
                "patients contributed more than one record, up to %d)",
                len(df),
                len(counts),
                int((counts > 1).sum()),
                int(counts.max()),
            )

        if "recording_type" in df.columns:
            # Not the stratification label, but the one users train on — worth
            # seeing in the run log next to the label that folds were built from.
            logger.info(
                "Protocol phase distribution (the dataset's own label, NOT the "
                "stratification target):\n%s",
                df["recording_type"].value_counts().to_string(),
            )

        logger.info(
            "Stratifying on primary occluded territory:\n%s",
            labels.value_counts().to_string(),
        )
        return labels
