"""
Sudden Cardiac Death Holter splitting strategy.

Nothing machine-readable ships with this dataset — no metadata file, and the only
header comments are a provenance line and, in 20 of 23 records, ``#vfon:
HH:MM:SS``. Even the clinical table is published only on the landing page. So
``load_metadata`` builds a metadata CSV from the headers, both annotation layers
and that transcribed table via ``ecgbench.labels.sddb`` — the same loader users
get from ``load_labels``, so the stratification label and the exposed labels
cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata.

**Three things about this dataset shape the split.**

The only clinical axis is the rhythm underneath the terminal event. Every subject
sustained a ventricular tachyarrhythmia, so ``cohort_label`` is one value across
the release and there is no diagnostic contrast. Folds are stratified on
``rhythm_class`` — 18 sinus, 4 atrial fibrillation, 1 continuously paced — which
is PhysioNet's own description of the cohort. Ventricular ectopy burden, the axis
``svdb`` and ``chfdb`` use, is unusable here: 11 of the 23 records have no audited
annotation, so the bands would be measuring the detector in half the release and
a cardiologist in the other half. See
``ecgbench.labels.sddb.attach_stratify_class`` for the four candidates.

There is no patient grouping to do, and for an unusual reason: the release's
subject identifier *is* the record name. The landing page's clinical table is
keyed "Subject Number" with values 30-52, one record per subject, so
``patient_id_column`` is null because a patient column would copy the index, not
because the subjects are unidentified. ``engine.py`` uses plain
``StratifiedKFold``; folds hold two or three records each.

The label scan reads both annotators for all 23 records but does **not** touch the
signals, so this is seconds rather than the minutes a 1.6 GB read would take. The
consequence is that the metadata CSV carries no NaN counts — validation produces
those, and ``ecgbench.labels.sddb.scan_invalid_samples`` computes them on request.
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


@register("sddb")
class SDDBSplitter(DatasetSplitter):
    """Sudden Cardiac Death Holter: header + annotation metadata, rhythm-balanced folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # record_name is "30" and signal_path likewise; both are handed to
                # wfdb as record stems, so neither may arrive as an int64.
                dtype={"record_name": str, "signal_path": str},
            )

        from ecgbench.labels.sddb import load_labels

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
        """Return the underlying cardiac rhythm attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("rhythm_class")

        counts = labels.value_counts()
        logger.info("Fold classes (underlying rhythm):\n%s", counts.to_string())
        # StratifiedKFold raises only when EVERY class is smaller than n_folds, so
        # 18/4/1 is fine. Its message names neither the config nor the column, so
        # say it here instead.
        if counts.max() < 10:
            logger.warning(
                "Largest fold class holds %d records, fewer than the 10 folds "
                "ECGBench generates; StratifiedKFold will fail. Widen the classes in "
                "ecgbench.labels.sddb.attach_stratify_class.",
                int(counts.max()),
            )
        logger.info(
            "%d records, %.1f h of signal; %d unaudited beats, %d audited beats over "
            "%d records; %d records carry a VF-onset comment",
            len(df),
            df["duration_secs"].sum() / 3600 if "duration_secs" in df else float("nan"),
            int(df["ari_n_beats"].sum()) if "ari_n_beats" in df else -1,
            int(df["atr_n_beats"].sum()) if "atr_n_beats" in df else -1,
            int(df["has_audited_annotation"].sum()) if "has_audited_annotation" in df else -1,
            int(df["has_vf_onset"].sum()) if "has_vf_onset" in df else -1,
        )
        return labels
