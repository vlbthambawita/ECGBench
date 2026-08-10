"""
QT Database splitting strategy.

Nothing machine-readable ships with this dataset, so ``load_metadata`` builds a
metadata CSV from the headers and all nine annotation layers via
``ecgbench.labels.qtdb`` — the same loader users get from ``load_labels``, so the
stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Three things about this dataset shape the split.**

Stratification is on ``source_database``, the release's own Table 1 breakdown — 33
European ST-T, 23 sudden-death, 15 MIT-BIH Arrhythmia, 13 Supraventricular, 10 NSR,
6 ST Change, 4 Long-Term and 1 matched control. It is the axis every other property
follows from (original sampling rate, electrode placement, whether reference
annotations exist, whether the amplitude is calibrated), and there is no clinical
alternative: the release has no record-level diagnosis at all. ``ltdb`` (4) and
``bih_control`` (1) are smaller than the 10 folds ECGBench generates, so neither can
appear in every fold; see ``attach_stratify_class`` for why they are kept separate
rather than pooled.

Patient grouping is necessary for exactly two pairs, and the release does not supply
it. Two European ST-T subjects each contributed two recordings that both ended up
here — e0121 with e0122 and e0124 with e0126 — so ungrouped folds put the same
person in train and test. ``patient_id`` collapses them; everything else is its own
subject, giving 103 subjects over 105 records. ``engine.py`` groups on
``config.patient_id_column``.

**The split does not protect against this dataset's real leakage risk, and cannot.**
Every record is a fifteen-minute excerpt of a recording published in another
database, six of which ECGBench also partitions. Folds drawn here are disjoint
*within* qtdb and say nothing about ``edb``, ``sddb``, ``mitdb``, ``svdb``, ``nsrdb``
or ``stdb``: 30 of the 33 European ST-T excerpts are bit-identical to their edb
source and 22 of the 23 sudden-death excerpts reproduce sddb exactly. Anyone
combining qtdb with any of those must filter on ``source_database`` themselves. That
is what the ``related:`` block in the catalogue entry is for.
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


@register("qtdb")
class QTDBSplitter(DatasetSplitter):
    """QT Database strategy: annotation-derived metadata, subject-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "sel100" and signal_path likewise — both are
                # handed to wfdb as record stems. patient_id holds the naming
                # record of each subject group, so it is a record id too.
                dtype={
                    "record_name": str,
                    "signal_path": str,
                    "patient_id": str,
                    # "0104", "30", "e0104" — the source record ids are a mix of
                    # bare integers and prefixed strings, and 30 must not become
                    # an int in a column that also holds "e0104".
                    "source_record": str,
                },
            )

        from ecgbench.labels.qtdb import load_labels

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

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return the source database attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("source_database")

        if config.patient_id_column and config.patient_id_column in df.columns:
            shared = df[config.patient_id_column].value_counts()
            shared = shared[shared > 1]
            logger.info(
                "Grouping %d records into folds by %d subjects (%d subject(s) "
                "contributed more than one record: %s)",
                len(df),
                df[config.patient_id_column].nunique(),
                len(shared),
                shared.to_dict(),
            )

        counts = labels.value_counts()
        logger.info("Fold classes (source database):\n%s", counts.to_string())
        # StratifiedKFold raises only when EVERY class is smaller than n_folds, but a
        # class below it still cannot be spread across all ten folds. Its message
        # names neither the config nor the column, so say it here instead.
        small = counts[counts < 10]
        if len(small):
            logger.warning(
                "Source database(s) %s hold fewer than the 10 folds ECGBench "
                "generates (%s), so they cannot appear in every fold. This is "
                "expected — the release contributes only 6 MIT-BIH ST Change and 4 "
                "MIT-BIH Long-Term excerpts, plus 1 matched control. See "
                "ecgbench.labels.qtdb.attach_stratify_class.",
                sorted(small.index),
                small.to_dict(),
            )

        if "n_annotated_beats" in df:
            logger.info(
                "%d records, %.1f h of signal; %d manually annotated beats "
                "(%d with a measurable QT), %d record(s) with a second annotator",
                len(df),
                df["duration_secs"].sum() / 3600.0,
                int(df["n_annotated_beats"].sum()),
                int(df["n_t_ends"].sum()),
                int(df["has_second_annotator"].sum()),
            )
        # This is the one warning worth repeating at split time: the folds are
        # disjoint inside qtdb and mean nothing across the six source databases
        # ECGBench also partitions.
        if "source_catalogue_slug" in df:
            partners = df.loc[df["source_catalogue_slug"].notna(), "source_catalogue_slug"]
            partners = partners[partners.astype(str) != ""]
            if len(partners):
                logger.warning(
                    "%d of %d records are excerpts of recordings in %d other "
                    "ECGBench dataset(s): %s. These folds are disjoint within qtdb "
                    "only — filter on source_database before combining.",
                    len(partners),
                    len(df),
                    partners.nunique(),
                    ", ".join(sorted(partners.unique())),
                )
        return labels
