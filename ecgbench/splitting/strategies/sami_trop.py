"""
SaMi-Trop splitting strategy — one HDF5 array, joined by row position.

The release gives no record a file: ``exams.hdf5`` holds a single
``(1631, 4096, 12)`` ``tracings`` array, so a record's "path" has to name a row
as well as a file — ``exams.hdf5:tracings:417``. That much is CODE-15%'s problem
too, but SaMi-Trop is harder in one respect and easier in two others.

**Harder: there is no key.** CODE-15%'s parts each carry an ``exam_id`` dataset,
so its splitter recovers the row mapping from the file itself — which it must,
because that release's ``exams.csv`` is not in file order. SaMi-Trop's HDF5 has
only ``tracings``. Row position is the only join available, so it was validated
statistically instead (see :mod:`ecgbench.labels.sami_trop`) and the row count
is asserted on every run. :func:`build_signal_paths` opens the file to check the
array's real shape rather than trusting the CSV's length alone, so a truncated
or re-exported copy fails here instead of silently shifting every label.

**Easier: no parts and no padding row.** One file, 1,631 rows, 1,631 records —
no all-zero trailing row like CODE-15%'s, and nothing to reconcile across parts.

**Easier: no patient grouping.** The release is each patient's *first* exam, so
``exam_id`` is patient-unique, the config sets ``patient_id_column: null`` and
the folds are a plain stratified 10-way split. This is the rare dataset in the
catalogue where that is genuinely safe rather than merely unchecked.

As everywhere else, the resolved reference has to reach disk: ``validate_dataset``
re-reads ``config.metadata_csv`` and rebuilds paths from the raw column, so a
fix-up living only in the returned DataFrame is invisible to it and every record
fails ``corrupt_header``. The normalised frame is written to
``ecgbench_metadata.csv`` in the dataset root, as ``chapman.py`` and
``code15.py`` do.

Stratification uses ``stratify_class`` from :mod:`ecgbench.labels.sami_trop`:
``DEATH`` (104), ``NORMAL`` (283) or ``ABNORMAL_ALIVE`` (1,244). Death comes
first so the rare mortality endpoint — the reason this dataset exists — is what
the folds balance.
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


def build_signal_paths(data_path: Path, rows: pd.Series, config: DatasetConfig) -> pd.Series:
    """Map each record to ``exams.hdf5:tracings:<row>``, checking the array shape.

    Args:
        data_path: Dataset root holding ``exams.hdf5``.
        rows: exam_id -> row index in the tracings array, i.e. the label
            loader's ``row`` column.
        config: Used only for its URL in error messages.

    Returns:
        exam_id -> signal path, in the order ``rows`` was given.

    Raises:
        FileNotFoundError: ``exams.hdf5`` is not on disk.
        ValueError: the array is missing, not 3-D, or does not have one row per
            metadata record.
    """
    import h5py

    from ecgbench.labels.sami_trop import N_RECORDS, N_SAMPLES, TRACINGS_DATASET, TRACINGS_HDF5

    path = data_path / TRACINGS_HDF5
    if not path.exists():
        raise FileNotFoundError(
            f"SaMi-Trop waveforms come from {TRACINGS_HDF5}, which is not in {data_path}. "
            f"It ships inside exams.zip on Zenodo (see {config.url}); extract it into the "
            "dataset root."
        )

    with h5py.File(path, "r") as handle:
        if TRACINGS_DATASET not in handle:
            raise ValueError(
                f"{path} has no '{TRACINGS_DATASET}' dataset. Found: {sorted(handle.keys())}."
            )
        shape = handle[TRACINGS_DATASET].shape

    if len(shape) != 3:
        raise ValueError(
            f"{path}:{TRACINGS_DATASET} has shape {shape}; expected a 3-D "
            "(records, samples, leads) array."
        )
    if shape[0] != len(rows):
        raise ValueError(
            f"{path}:{TRACINGS_DATASET} holds {shape[0]} records but the metadata has "
            f"{len(rows)}. This release has no identifier inside the HDF5, so records are "
            "matched to exams.csv by row position and the two lengths must agree exactly "
            f"({N_RECORDS} in the published release). A copy that disagrees would be "
            "mis-joined, not partially joined."
        )
    if shape[1] != N_SAMPLES:
        raise ValueError(
            f"{path}:{TRACINGS_DATASET} has {shape[1]} samples per record, expected "
            f"{N_SAMPLES}. The config's expected_samples and duration_seconds assume "
            f"{N_SAMPLES} at 400 Hz."
        )

    logger.info("Resolved %d records to rows of %s %s", shape[0], TRACINGS_HDF5, shape)
    return rows.map(lambda row: f"{TRACINGS_HDF5}:{TRACINGS_DATASET}:{int(row)}").rename(
        "signal_path"
    )


@register("sami_trop")
class SamiTropSplitter(DatasetSplitter):
    """SaMi-Trop splitting strategy.

    - Builds (and caches) the metadata CSV from ``exams.csv`` plus the row
      reference the release leaves implicit
    - Stratifies on mortality first, then on the ``normal_ecg`` flag
    - Does **not** group: one recording per patient, so there is nothing to group
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype={STRATIFY_COLUMN: str},
            )

        from ecgbench.labels.sami_trop import load_labels

        labels = load_labels(data_path, config)
        labels["signal_path"] = build_signal_paths(data_path, labels["row"], config)
        df = labels.reset_index()

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

        logger.info(
            "SaMi-Trop: %d records, one per patient, %d deaths",
            len(df),
            int(df["death"].sum()),
        )
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the mortality-first stratification label from the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("stratify_class")
        logger.info("Stratification class distribution:\n%s", labels.value_counts().to_string())
        return labels
