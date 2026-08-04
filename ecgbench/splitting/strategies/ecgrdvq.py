"""
ECGRDVQ splitting strategy.

The release ships a well-formed metadata table — ``SCR-002.Clinical.Data.csv``,
one row per record, ``EGREFID`` equal to the signal filename — but **no signal
path column**. The path is ``raw/<RANDID>/<EGREFID>``, so ``load_metadata`` builds
the frame through ``ecgbench.labels.ecgrdvq``, which adds it (and the median
beat's path, and heart rate, and QTcF), then caches the result in the dataset root
as ``config.metadata_csv``.

Writing that cache to disk is load-bearing rather than a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself and
rebuilds record paths from the raw column instead of reusing this frame, so an
in-memory-only path fix-up would leave every record failing ``corrupt_header``.

**Grouping on ``patient_id`` is not optional here, and 22 patients understates
why.** 5,232 records come from 22 subjects, 192 or 240 each — but they also come in
near-duplicate *triplicates*: three 10-second segments were extracted per subject
per nominal timepoint, and here that structure is **exact**, with all 1,744
timepoint groups holding precisely 3 records (5,232 = 1,744 x 3; ECGDMMLD manages
1,403 of 1,404). The three are the same person in the same posture at the same
plasma concentration, seconds apart. Split per-record and two thirds of any test
set has a near-copy of itself in train. The effective sample size is closer to
1,744 than 5,232, and closer still to 22 for anything subject-specific.

**Stratification is on ``treatment``, and here that is close to a formality.**
This dataset has no diagnosis to stratify on — all 22 participants were healthy
volunteers, and what varies is the drug. But it is also an almost-complete 5-period
crossover: 21 of the 22 subjects passed through all five arms, so once records are
grouped by patient every fold receives every treatment automatically. Three
consequences worth knowing before reading anything into the fold balance:

- **No split of this dataset separates the arms.** Treatment is a record-level
  label while the group is patient-level, and each patient carries four or five
  treatments, so per-arm balance across folds is a property of the design rather
  than of the stratifier. What stratifying still buys is placing subject **1002**,
  the one early withdrawal (4 periods, no quinidine), rather than letting it fall
  wherever.
- **Fold sizes are uneven, and patient grouping is the reason.** 22 patients over
  10 folds is 2-3 per fold while records per subject are 192 or 240, so a fold
  holding three subjects is half again the size of one holding two. That is the
  correct trade: the alternative leaks triplicates. Read the per-fold counts this
  module logs before quoting a per-fold result.
- **``StratifiedGroupKFold`` does not warn.** It emits none of the "least populated
  class" messages ``StratifiedKFold`` does, so silence is not evidence of balance.

Unlike the sibling ``ecgdmmld``, ``treatment`` here really is the drug that was
administered — each period dosed a single agent, so there is no staged-combination
trap. It remains a *stratification* target rather than a training target for a
smaller reason: the 327 pre-dose records carry their period's drug name while
containing no drug. See ``ecgbench/labels/ecgrdvq.py``.
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


@register("ecgrdvq")
class ECGRDVQSplitter(DatasetSplitter):
    """ECGRDVQ splitting strategy.

    - Builds (and caches) the metadata CSV, adding the signal paths the release omits
    - Points ``signal_path`` at ``raw/``, never at the derived ``medians/``
    - Stratifies on the period's randomised single-agent treatment
    - Groups on ``patient_id``: 5,232 records from 22 subjects, in exact triplicates
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        from ecgbench.labels.ecgrdvq import load_labels

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
        """Return the treatment for each record, from the label loader."""
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
            # The grouped splitter balances patients, not records; in an almost
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
