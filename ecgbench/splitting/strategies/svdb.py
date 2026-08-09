"""
MIT-BIH Supraventricular Arrhythmia splitting strategy.

Nothing machine-readable ships with this dataset — not a metadata file, and not
even the header comments ``mitdb`` and ``nsrdb`` carry, since every SVDB header is
four lines of signal specification. ``load_metadata`` therefore builds a metadata
CSV from the ``.atr`` annotations via ``ecgbench.labels.svdb`` — the same loader
users get from ``load_labels``, so the stratification label and the exposed labels
cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Two things about this dataset shape the split.**

The only axis to stratify on is how much supraventricular ectopy each record
holds. There are no demographics, no diagnoses and — with a single ``+``
annotation in the whole release — no rhythm labels either. Folds are balanced on
the SVEB burden band, at fixed edges of 1%, 3% and 10% of beats, which gives
21/20/23/14 records. See ``ecgbench.labels.svdb.attach_stratify_class`` for why
fixed edges rather than quantiles.

There is no patient grouping to do. The release ships no subject identifier of any
kind and does not say how many subjects its 78 records represent, so
``patient_id_column`` is null and ``engine.py`` uses plain ``StratifiedKFold``.
Folds hold seven or eight records each.
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


@register("svdb")
class SVDBSplitter(DatasetSplitter):
    """MIT-BIH SVDB splitting strategy: annotation-derived metadata, SVEB-balanced folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "800" and signal_path likewise; both are handed to
                # wfdb as record stems, so neither may arrive as an int.
                dtype={"record_name": str, "signal_path": str},
            )

        from ecgbench.labels.svdb import load_labels

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
        """Return the SVEB burden band attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("sveb_burden")

        counts = labels.value_counts()
        logger.info("Fold classes (SVEB burden):\n%s", counts.to_string())
        # StratifiedKFold raises only when EVERY class is smaller than n_folds, but
        # a class below it still cannot be spread across all ten folds. Its message
        # names neither the config nor the column, so say it here instead.
        if counts.min() < 10:
            logger.warning(
                "SVEB burden band %r holds %d records, fewer than the 10 folds "
                "ECGBench generates, so it cannot appear in every fold. Widen the "
                "edges in ecgbench.labels.svdb.SVEB_BURDEN_EDGES.",
                str(counts.idxmin()),
                int(counts.min()),
            )
        logger.info(
            "%d records, %.1f h of signal; %d beats of which %d supraventricular",
            len(df),
            df["duration_secs"].sum() / 3600 if "duration_secs" in df else float("nan"),
            int(df["n_beats"].sum()) if "n_beats" in df else -1,
            int(df["n_sveb"].sum()) if "n_sveb" in df else -1,
        )
        return labels
