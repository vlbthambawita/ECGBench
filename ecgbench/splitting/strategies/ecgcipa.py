"""
CiPA ECG Validation Study splitting strategy.

The release ships four CDISC-style clinical analysis datasets and **no table
mapping records to signal files** — ``adeg.csv`` is a long-format measurement
table, and the record-to-subject link lives in each WFDB header's comment block.
``load_metadata`` therefore builds one from ``ecgbench.labels.ecgcipa``, which
joins all four, and caches it in the dataset root as ``config.metadata_csv``.

Writing that cache to disk is load-bearing rather than a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself instead of
reusing this frame, so an in-memory-only table would leave validation with no
metadata at all.

**Grouping on ``patient_id`` is not optional here, and patient count understates
why.** 5,749 records come from 60 subjects, 24 to 168 each — but they also come in
near-duplicate *triplicates*: three 10-second segments were extracted per subject
per nominal timepoint for 1,916 of the 1,917 timepoint groups, seconds apart, from
the same person in the same posture at the same plasma concentration. Split
per-record and roughly two thirds of any test set has a near-copy of itself in
train. The effective sample size is closer to 1,917 than 5,749, and closer still
to 60 for anything subject-specific.

**Stratification is on ``treatment``, and that choice needs stating** because this
dataset has no diagnosis to stratify on. All 60 participants were healthy
volunteers; what varies is the drug. ``treatment`` is ``adeg``'s ``TRTA``, the
treatment actually in effect when the record was taken, and it is what a
drug-effect model is asked to separate.

Three consequences worth knowing before trusting the fold balance:

- **The label is record-level while the group is patient-level.** The ten crossover
  subjects (2001-2010) each contribute a dofetilide block *and* a
  diltiazem+dofetilide block, so grouping by patient necessarily puts two
  treatments in the same fold. There is no split of this dataset in which those two
  arms are disjoint by subject.
- **The arithmetic happens to work out, but only just.** Every treatment has
  exactly 10 patients and there are 10 folds, so 60 patients divide into 6 per
  fold and the shipped v1.0.0 split does put all 7 treatments in all 10 folds —
  84 records per treatment per fold in 61 of the 70 cells. The thin cells are
  Verapamil in fold 5 (24 records) and Diltiazem+Dofetilide in folds 8 and 10
  (51 each), which is why fold sizes run 528-588 rather than a uniform 575. A
  per-treatment evaluation on one fold rests on as few as 24 records from one
  subject, so read the crosstab before quoting a per-arm result.
- **``StratifiedGroupKFold`` does not warn.** It emits none of the "least populated
  class" messages ``StratifiedKFold`` does, so silence is not evidence of balance —
  read the per-fold distribution this module logs. That the balance came out well
  here is a property of this cohort's arithmetic, not a guarantee.

``treatment`` is a stratification target, not the study's endpoint. The published
analysis regresses *change* in QTcF and J-Tpeakc against plasma concentration, and
those change-from-baseline values exist only on ``adeg.csv``'s triplicate-average
rows, which carry no record identifier at all. See
``ecgbench/labels/ecgcipa.py`` — this is the dataset's central gotcha and no
splitter can work around it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column carrying the stratification label, attached by the label loader.
STRATIFY_COLUMN = "treatment"


@register("ecgcipa")
class ECGCIPASplitter(DatasetSplitter):
    """CiPA ECG Validation Study splitting strategy.

    - Builds (and caches) the metadata CSV by joining adeg, adpc, adsl and addm
    - Points ``signal_path`` at ``raw/``, never at the derived ``medians/``
    - Stratifies on the treatment in effect for the record
    - Groups on ``patient_id``: 5,749 records from 60 subjects, in triplicate
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        from ecgbench.labels.ecgcipa import load_labels

        df = load_labels(data_path, config).reset_index()

        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return the treatment in effect for each record, from the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename(STRATIFY_COLUMN)
        logger.info(
            "Treatment distribution (records):\n%s", labels.value_counts().to_string()
        )
        if config.patient_id_column in df.columns:
            # The grouped splitter balances patients, not records; the record counts
            # above are weighted by how many timepoints each subject completed.
            per_treatment = (
                df.groupby(STRATIFY_COLUMN)[config.patient_id_column]
                .nunique()
                .to_string()
            )
            logger.info("Treatment distribution (patients):\n%s", per_treatment)
            timepoints = df.drop_duplicates(
                subset=[config.patient_id_column, "period", "timepoint_n"]
            )
            logger.info(
                "%d records over %d (subject, period, timepoint) groups — "
                "records within a group are near-duplicate triplicates",
                len(df), len(timepoints),
            )
        return labels
