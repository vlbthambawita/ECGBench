"""
ECGDMMLD splitting strategy.

The release ships a well-formed metadata table — ``SCR-003.Clinical.Data.csv``,
one row per record, ``EGREFID`` equal to the signal filename — but **no signal
path column**. The path is ``raw/<RANDID>/<EGREFID>``, so ``load_metadata`` builds
the frame through ``ecgbench.labels.ecgdmmld``, which adds it (and the derived
median beat's path, and heart rate, and QTcF), then caches the result in the
dataset root as ``config.metadata_csv``.

Writing that cache to disk is load-bearing rather than a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself and
rebuilds record paths from the raw column instead of reusing this frame, so an
in-memory-only path fix-up would leave every record failing ``corrupt_header``.

**Grouping on ``patient_id`` is not optional here, and 22 patients understates
why.** 4,211 records come from 22 subjects, 42 to 210 each — but they also come in
near-duplicate *triplicates*: three 10-second segments were extracted per subject
per nominal timepoint for 1,403 of the 1,404 timepoint groups, seconds apart, from
the same person in the same posture at the same plasma concentration. Split
per-record and roughly two thirds of any test set has a near-copy of itself in
train. The effective sample size is closer to 1,404 than 4,211, and closer still to
22 for anything subject-specific.

**Stratification is on ``treatment``, and here that is close to a formality.**
This dataset has no diagnosis to stratify on — all 22 participants were healthy
volunteers, and what varies is the drug. But it is also a *complete 5-period
crossover*: 19 of the 22 subjects passed through all five arms, so once records are
grouped by patient every fold receives every treatment automatically. Three
consequences worth knowing before reading anything into the fold balance:

- **No split of this dataset separates the arms.** Treatment is a record-level
  label while the group is patient-level, and each patient carries all five
  treatments, so per-arm balance across folds is a property of the design rather
  than of the stratifier. What stratifying still buys is keeping the three early
  withdrawals — subjects 2011, 2015 and 2021 completed 2, 1 and 3 periods, so
  their treatment mixes are lopsided — from clustering into one fold.
- **Fold sizes are uneven, and patient grouping is the reason.** 22 patients over
  10 folds is 2-3 per fold while records per subject run 42-210, so a fold holding a
  withdrawn subject is a fraction of the size of one holding two completers. That is
  the correct trade: the alternative leaks triplicates. Read the per-fold counts
  this module logs before quoting a per-fold result.
- **``StratifiedGroupKFold`` does not warn.** It emits none of the "least populated
  class" messages ``StratifiedKFold`` does, so silence is not evidence of balance.

``treatment`` is a stratification target, not a training target, and this dataset
is the one where that distinction bites hardest: the drugs were staged hours apart
within each period, so a record labelled ``Mexiletine + Dofetilide`` at the 2-hour
timepoint has no dofetilide in it at all. See ``ecgbench/labels/ecgdmmld.py`` —
train on the plasma concentration columns.
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

#: Columns identifying one triplicate — the group patient-level splitting must
#: keep intact. Logged rather than asserted, because the assertion belongs to the
#: split engine, not to a strategy.
TRIPLICATE_KEY = ["patient_id", "period", "timepoint_hours"]


@register("ecgdmmld")
class ECGDMMLDSplitter(DatasetSplitter):
    """ECGDMMLD splitting strategy.

    - Builds (and caches) the metadata CSV, adding the signal paths the release omits
    - Points ``signal_path`` at ``raw/``, never at the derived ``medians/``
    - Stratifies on the period's randomised treatment arm
    - Groups on ``patient_id``: 4,211 records from 22 subjects, in triplicate
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        from ecgbench.labels.ecgdmmld import load_labels

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
        """Return the treatment arm for each record, from the label loader."""
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
            # above are weighted by how many periods each subject completed. In a
            # complete crossover these per-treatment patient counts are all but
            # identical, which is the point made in the module docstring.
            per_treatment = (
                df.groupby(STRATIFY_COLUMN)[config.patient_id_column]
                .nunique()
                .to_string()
            )
            logger.info("Treatment distribution (patients):\n%s", per_treatment)
            if all(c in df.columns for c in TRIPLICATE_KEY):
                groups = df.drop_duplicates(subset=TRIPLICATE_KEY)
                logger.info(
                    "%d records over %d (subject, period, timepoint) groups — "
                    "records within a group are near-duplicate triplicates",
                    len(df), len(groups),
                )
        return labels
