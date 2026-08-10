"""
European ST-T Database splitting strategy.

Nothing machine-readable ships with this dataset, so ``load_metadata`` builds a metadata
CSV from the header comments and the ``.atr`` reference annotations via
``ecgbench.labels.edb`` — the same loader users get from ``load_labels``, so the
stratification label and the exposed labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this DataFrame,
so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split.**

Patient grouping is necessary and the release does not supply it. ``edb.txt`` says the
90 records come from 79 subjects; nothing in the files says which. Seven subjects
contributed more than one record — e0118, e0119, e0121 and e0122 are one 51-year-old man
— so ungrouped folds put the same person in train and test. ``patient_id`` is therefore
reconstructed from the header by ``ecgbench.labels.edb.reconstruct_patient_ids``, whose
docstring is where the justification and the limits live. ``engine.py`` groups on
``config.patient_id_column``.

Stratification uses the ST-episode count, banded at fixed edges of 1, 3 and 6 episodes,
giving 4 / 32 / 29 / 25 records. ST change is the quantity this database was built to
evaluate and the burden is very uneven, so a fold drawn without regard to it can easily
be all quiet records or all busy ones. Nothing clinical works better here: the header
findings are subject background rather than per-record annotation, ``dominant_rhythm`` is
sinus for all 90 records, and ``st_t_class`` has two classes of two records.

The ``none`` band holds 4 records, fewer than the 10 folds ECGBench generates, so it
cannot appear in every fold; see ``attach_stratify_class`` for why it is kept separate
anyway.
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


@register("edb")
class EDBSplitter(DatasetSplitter):
    """European ST-T splitting strategy: annotation-derived metadata, subject-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "e0103" and signal_path likewise — both are handed to
                # wfdb as record stems. patient_id holds the naming record of each
                # reconstructed subject, so it is a record id too.
                dtype={
                    "record_name": str,
                    "signal_path": str,
                    "patient_id": str,
                },
            )

        from ecgbench.labels.edb import load_labels

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
        """Return the ST-episode burden band attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("st_burden")

        if config.patient_id_column and config.patient_id_column in df.columns:
            shared = df[config.patient_id_column].value_counts()
            shared = shared[shared > 1]
            logger.info(
                "Grouping %d records into folds by %d reconstructed subjects "
                "(%d subject(s) contributed more than one record: %s)",
                len(df),
                df[config.patient_id_column].nunique(),
                len(shared),
                shared.to_dict(),
            )

        counts = labels.value_counts()
        logger.info("Fold classes (ST episode burden):\n%s", counts.to_string())
        # StratifiedKFold raises only when EVERY class is smaller than n_folds, but a
        # class below it still cannot be spread across all ten folds. Its message names
        # neither the config nor the column, so say it here instead.
        if counts.min() < 10:
            logger.warning(
                "ST burden band %r holds %d records, fewer than the 10 folds ECGBench "
                "generates, so it cannot appear in every fold. This is expected for "
                "'none' (4 records with no ST episode at all); widen the edges in "
                "ecgbench.labels.edb.ST_BURDEN_EDGES to change it.",
                str(counts.idxmin()),
                int(counts.min()),
            )
        logger.info(
            "%d records, %.0f h of signal; %d beats, %d ST and %d T episodes",
            len(df),
            len(df) * 2.0,
            int(df["n_beats"].sum()) if "n_beats" in df else -1,
            int(df["n_st_episodes"].sum()) if "n_st_episodes" in df else -1,
            int(df["n_t_episodes"].sum()) if "n_t_episodes" in df else -1,
        )
        return labels
