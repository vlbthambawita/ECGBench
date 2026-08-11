"""
Long-Term ST Database splitting strategy.

Nothing machine-readable ships with this dataset, so ``load_metadata`` builds a
metadata CSV from the header comment tree and the ``.atr``/``.sta``/``.stb``/``.stc``
annotations via ``ecgbench.labels.ltstdb`` — the same loader users get from
``load_labels``, so the stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split.**

Patient grouping is necessary and, unlike ``edb``, the release supplies it. The
record name is ``sXYYYZ`` where ``YYY`` is the subject: 80 subjects over 86
records, with s20271-s20274 all one person and s30731/s30732, s30741/s30742,
s30751/s30752 three more pairs. Ungrouped, subject 027 alone would put four
records holding 416 of the release's 1,795 ischaemic episodes across several
folds, and any of them landing in both train and test is the same day of the same
heart on both sides of the split. ``engine.py`` groups on
``config.patient_id_column``.

Stratification uses the criterion-A ischaemic episode count, banded at 1, 6 and 21
episodes, giving 18 / 14 / 25 / 29 records. Ischaemic ST change is the quantity
this database was built to evaluate, and the burden is enormously uneven — 18
records hold none and s20274 holds 143 — so a fold drawn without regard to it can
easily be all quiet records or all busy ones. Nothing clinical works better:
``st_class`` puts 55 of 86 records in one class and 2 in another, the header
findings describe the subject rather than the recording, and ``diagnoses`` is free
text with more distinct values than there are records to spread over folds.

Grouping and stratification pull against each other here, and grouping wins:
``StratifiedGroupKFold`` keeps subjects whole and balances the bands as well as it
can, so fold sizes and band proportions are approximate rather than exact.
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


@register("ltstdb")
class LTSTDBSplitter(DatasetSplitter):
    """Long-Term ST strategy: annotation-derived metadata, subject-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "s20011" and signal_path likewise, both handed to
                # wfdb as record stems; patient_id is the zero-padded subject number
                # "027", which pandas would otherwise read as 27. Same three columns
                # config.identifier_dtypes() protects everywhere else.
                dtype=config.identifier_dtypes() or {"record_name": str, "signal_path": str},
            )

        from ecgbench.labels.ltstdb import load_labels

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
        """Return the ischaemic-burden band attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("ischemic_burden")

        if config.patient_id_column and config.patient_id_column in df.columns:
            shared = df[config.patient_id_column].value_counts()
            shared = shared[shared > 1]
            logger.info(
                "Grouping %d records into folds by %d subjects "
                "(%d subject(s) contributed more than one record: %s)",
                len(df),
                df[config.patient_id_column].nunique(),
                len(shared),
                shared.to_dict(),
            )

        counts = labels.value_counts()
        logger.info("Fold classes (ischaemic episode burden):\n%s", counts.to_string())
        # StratifiedGroupKFold raises only when EVERY class is smaller than n_folds,
        # but a class below it still cannot be spread across all ten folds. Its
        # message names neither the config nor the column, so say it here instead.
        if counts.min() < 10:
            logger.warning(
                "Ischaemic burden band %r holds %d records, fewer than the 10 folds "
                "ECGBench generates, so it cannot appear in every fold. Widen the "
                "edges in ecgbench.labels.ltstdb.ISCHEMIC_BURDEN_EDGES to change it.",
                str(counts.idxmin()),
                int(counts.min()),
            )
        logger.info(
            "%d records, %.0f h of signal; %d beats, %d ischaemic and %d "
            "rate-related ST episodes under criterion A (75 uV / 30 s)",
            len(df),
            df["duration_hours"].sum() if "duration_hours" in df else float("nan"),
            int(df["n_beats"].sum()) if "n_beats" in df else -1,
            int(df["n_ischemic_episodes"].sum()) if "n_ischemic_episodes" in df else -1,
            int(df["n_rate_related_episodes"].sum())
            if "n_rate_related_episodes" in df
            else -1,
        )
        return labels
