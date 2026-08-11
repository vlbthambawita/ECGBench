"""
tOLIet splitting strategy.

Nothing in the release can be split as shipped. ``DataSet.csv`` is
semicolon-separated with a UTF-8 BOM and sixteen unnamed empty columns, it lists
four IDs that have no signal file, it carries no path column, it has one row per
*sitting* where ECGBench wants one per electrode channel, and — the part that
matters most — it says nothing about which of the four seat electrodes actually
made contact, which is the fact that decides whether a record is valid. So
``load_metadata`` builds a metadata CSV via ``ecgbench.labels.tollet``, the same
loader users get from ``load_labels``, so the stratification label and the exposed
labels cannot drift.

Writing that cache to disk is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
DataFrame, so an in-memory-only frame would leave validation with no metadata. It
also saves a 20-second rescan of 17.9 million samples on every later run.

**Three things about this dataset shape the split.**

**One record is one electrode channel.** The 145 OpenSignals files each hold four
differential channels from four electrode textures, and three quarters of those
channels made no contact. Kept as 4-lead records, a single dead electrode sinks
the whole sitting through ``flat_line`` and ``clean`` comes out at 5 records; per
channel it is 342 of 580. See ``ecgbench.labels.tollet``.

**The signal path names one column of one file.** ``signal_path`` is
``ECG_EXP/15_1.txt:A2`` — the OpenSignals reference ``_parse_opensignals_ref``
resolves, picking the sinusoidal-electrode column out of a file that also stores a
sequence number, four digital I/O columns and two 6-bit analog channels that are
zero throughout. That suffix has to be in the *file on disk*, not just in this
frame, for the same reason as above.

**Stratification balances the clean subset, not just the demographics.** Folds are
stratified on ``stratify_class``, sex crossed with ``signal_active`` —
``F_active`` (50 subjects, 192 records), ``M_active`` (36/150), ``F_flat``
(47/148), ``M_flat`` (34/90). Liveness is in the cross because it is what decides
membership of the ``clean`` version; stratifying on sex alone spreads the 342 live
channels across the ten folds at 28 to 39 each.
``ecgbench.labels.tollet.attach_stratify_class`` tabulates the alternatives that
were measured and rejected.
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


@register("tollet")
class TolletSplitter(DatasetSplitter):
    """tOLIet strategy: one record per electrode channel, subject-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # LOAD-BEARING, not defensive. subject_id is all-digits, so a
                # plain read gives int64 here and the label loader gives str —
                # and StratifiedGroupKFold orders its groups by value, so the
                # first run (built from the loader) and every later one (built
                # from this cache) would partition differently and stamp
                # different fold_digests into manifest.json for identical data.
                # config.identifier_dtypes() is empty for this dataset; see the
                # zero_padded_identifiers comment in the YAML.
                dtype={"record_id": str, "subject_id": str, "signal_path": str},
            )

        from ecgbench.labels.tollet import load_labels

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
        """Return the sex x signal_active class attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("sex_x_signal_active")

        logger.info("Fold classes (sex x signal_active):\n%s",
                    labels.value_counts().to_string())
        if "subject_id" in df.columns:
            subjects = df.groupby(STRATIFY_COLUMN)["subject_id"].nunique()
            logger.info("Subjects per fold class:\n%s", subjects.to_string())
            # StratifiedGroupKFold keeps groups intact, so a class with fewer
            # subjects than folds cannot appear in every fold. Its own message
            # names neither the config nor the column, so say it here.
            if int(subjects.min()) < config.n_folds:
                logger.warning(
                    "Smallest fold class holds %d subjects, fewer than the %d folds "
                    "ECGBench generates; some folds will contain none of it.",
                    int(subjects.min()), config.n_folds,
                )
            logger.info(
                "%d channel records from %d sittings by %d subjects. %d channels "
                "carry a signal and are the whole of the `clean` version; the "
                "other %d are electrode pairs that made no contact and stay in "
                "`original` only.",
                len(df),
                df["source_record"].nunique() if "source_record" in df else -1,
                df["subject_id"].nunique(),
                int(df["signal_active"].sum()) if "signal_active" in df else -1,
                int((~df["signal_active"]).sum()) if "signal_active" in df else -1,
            )
        return labels
