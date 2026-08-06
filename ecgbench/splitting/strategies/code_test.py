"""
CODE-test splitting strategy — a dataset with no identifiers at all.

``ecg_tracings.hdf5`` holds one 3-D ``tracings`` array of shape
``(827, 4096, 12)`` and nothing else: no exam_id, no patient id, no filenames.
``attributes.csv`` and the seven annotation CSVs each hold 827 unkeyed rows and
are aligned to the array purely by position — "the i-th line corresponds to the
i-th tracing", per the bundled README. So the record id ECGBench uses **is** the
row index, 0-826, and the signal path is a row reference into the shared array:
``ecg_tracings.hdf5:tracings:417``.

That makes the row count the only integrity check available, and it is a strict
one: ``ecgbench.labels.code_test`` refuses any source file that is not exactly
827 rows, and this splitter additionally checks the HDF5 array itself. A
positional join against a table of the wrong length does not fail loudly on its
own — it produces 827 confidently wrong labels.

As with the other generated-metadata datasets, the frame is written to
``ecgbench_metadata.csv`` in the dataset root: ``validate_dataset`` re-reads
``config.metadata_csv`` from disk and never sees an in-memory fix-up.

**No patient grouping.** The README describes the 827 tracings as coming from
827 different patients and publishes no patient identifier, so there is nothing
to group on and nothing that needs it.

**This is an evaluation set that nevertheless gets ten folds.** That is the
ECGBench convention applied uniformly, not advice — the release's intended use
is all 827 records as one hold-out set. See the note at the top of
``code_test.yaml``.

Stratification uses ``stratify_class`` from ``ecgbench.labels.code_test``: the
rarest gold-standard abnormality a record carries, or ``NONE``. Rarest-wins
matters more here than anywhere else in ECGBench — AF appears in 13 of 827
records, so any reduction that let a commoner class win would leave folds
without a single AF example.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.labels.code_test import N_RECORDS, TRACINGS_DATASET, TRACINGS_HDF5
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column the label loader attaches, used for stratification.
STRATIFY_COLUMN = "stratify_class"


def build_signal_paths(data_path: Path, n_records: int = N_RECORDS) -> list[str]:
    """Return one ``ecg_tracings.hdf5:tracings:<row>`` reference per record.

    Checks the array's own length first: the record ids are row numbers, so an
    array of a different size means the positional alignment the whole release
    depends on no longer holds.
    """
    import h5py

    path = data_path / TRACINGS_HDF5
    if not path.exists():
        raise FileNotFoundError(
            f"CODE-test waveforms {TRACINGS_HDF5} are not in {data_path}. The Zenodo "
            "archive (https://doi.org/10.5281/zenodo.3765780) extracts to a 'data/' "
            "subdirectory holding ecg_tracings.hdf5, attributes.csv and annotations/ "
            "— point --data-path at that directory, not at its parent."
        )

    with h5py.File(path, "r") as handle:
        if TRACINGS_DATASET not in handle:
            raise ValueError(
                f"{path} has no '{TRACINGS_DATASET}' dataset. " f"Found: {sorted(handle.keys())}."
            )
        shape = handle[TRACINGS_DATASET].shape

    if len(shape) != 3 or shape[0] != n_records:
        raise ValueError(
            f"{path} holds a {TRACINGS_DATASET} array of shape {shape}, expected "
            f"({n_records}, 4096, 12). Every table in this release is aligned to that "
            "array by row position and has no key of its own, so a different row "
            "count means nothing can be joined."
        )

    return [f"{TRACINGS_HDF5}:{TRACINGS_DATASET}:{row}" for row in range(n_records)]


@register("code_test")
class CODETestSplitter(DatasetSplitter):
    """CODE-test splitting strategy.

    - Builds (and caches) the metadata CSV, since the release ships no table
      carrying a record identifier
    - Assigns record ids 0-826, the row indices the release's own documentation
      defines its alignment by
    - Stratifies on the rarest gold-standard abnormality; no patient grouping,
      because one record per patient and no patient id ships
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # The code lists are empty for 681 of 827 records; without this
                # pandas reads those columns as float NaN.
                keep_default_na=False,
                na_values=[""],
            ).pipe(
                lambda d: d.assign(
                    **{c: d[c].fillna("") for c in d.columns if c.endswith("abnormality_codes")}
                )
            )

        from ecgbench.labels.code_test import load_labels

        labels = load_labels(data_path, config)
        labels["signal_path"] = build_signal_paths(data_path, len(labels))
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
            "CODE-test: %d records, %d with a gold-standard abnormality",
            len(df),
            int((df["n_abnormalities"] > 0).sum()),
        )
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the rarest gold-standard abnormality per record."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("stratify_class")
        logger.info("Stratification class distribution:\n%s", labels.value_counts().to_string())
        return labels
