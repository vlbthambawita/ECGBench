"""
CODE-15% splitting strategy — resolving exam_id to a row of an HDF5 part.

The release does not give each record a file. 18 HDF5 parts each hold one 3-D
``tracings`` array of shape ``(N, 4096, 12)``, so a record's "path" has to name
a row as well as a file: ``exams_part0.hdf5:tracings:417``. Building that
reference is what this splitter is for, and there are three traps in it.

**1. exams.csv is not in file order.** Its ``trace_file`` column says which part
holds a record but not where in it, and the rows within a part do not follow the
part's own ``exam_id`` dataset — part 0's CSV rows and its HDF5 order disagree.
The row index therefore has to be read out of each part's ``exam_id`` dataset;
using the CSV's row number would mislabel almost every record.

**2. Every part has one more row than it has records.** Parts 0-16 hold 20,001
rows for 20,000 records and part 17 holds 5,780 for 5,779. The extra row is all
zeros with ``exam_id`` 0 and appears in no CSV. Keying on ``exam_id`` skips it
for free — which is the point of doing it this way — but anything positional
would shift by one from that row onwards.

**3. The reference has to reach disk.** ``validate_dataset`` re-reads
``config.metadata_csv`` itself and rebuilds paths from the raw column, so a
fix-up living only in this DataFrame is invisible to it and every record fails
``corrupt_header``. The normalised frame is written to ``ecgbench_metadata.csv``
in the dataset root and the config points at that, as ``chapman.py``,
``echonext.py`` and ``sph.py`` do.

Reading the 18 ``exam_id`` datasets costs about 2.8 MB of I/O in total — they
are 1-D int64 arrays — so the resolution is cheap despite the parts being 66 GB
between them. It doubles as an integrity check: the exam_id set of each part
must equal the set exams.csv assigns to it, and a mismatch raises rather than
dropping records quietly. That check matters because Zenodo publishes checksums
only for the ``exams_part*.zip`` archives, so an extracted copy cannot otherwise
be verified against the provider.

**Patient grouping is real and large here.** 233,770 patients hold the 345,779
records and 66,929 of them contributed more than one, up to 38 — without
grouping, roughly a third of the records could put the same patient in both
train and test. Note the grouping does not catch everything: the source contains
a small number of byte-identical duplicate recordings filed under different
patient_ids (47 records among part 0's 20,000).

Stratification uses ``stratify_class`` from ``ecgbench.labels.code15``: the
globally rarest of the six abnormalities a record carries, or ``NORMAL`` /
``OTHER`` for the 308,004 records carrying none. Those two are kept apart
deliberately — 173,347 records have no abnormality flag *and* no normal flag,
and pooling them with the genuine normals would stratify on a label that means
two different things.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: The dataset inside each part that holds the waveforms.
TRACINGS_DATASET = "tracings"

#: The 1-D dataset giving each row's exam_id. The whole resolution rests on it.
EXAM_ID_DATASET = "exam_id"

#: Sentinel exam_id of the all-zero padding row that ends every part.
PADDING_EXAM_ID = 0

#: Column the label loader attaches, used for stratification.
STRATIFY_COLUMN = "stratify_class"


def build_signal_paths(data_path: Path, trace_files: pd.Series) -> pd.Series:
    """Map each exam_id to ``<part>:tracings:<row>``, keyed on the part's own ids.

    Args:
        data_path: Dataset root holding the ``exams_part*.hdf5`` files.
        trace_files: exam_id -> part filename, i.e. ``exams.csv``'s
            ``trace_file`` column indexed by ``exam_id``.

    Returns:
        exam_id -> signal path, in the order ``trace_files`` was given.

    Raises:
        FileNotFoundError: a part named in the metadata is not on disk.
        ValueError: a part's exam_id set disagrees with the metadata's.
    """
    import h5py

    paths: dict[int, str] = {}
    for part in sorted(trace_files.unique()):
        path = data_path / part
        if not path.exists():
            raise FileNotFoundError(
                f"CODE-15% part {part} is named in exams.csv but is not in "
                f"{data_path}. All 18 exams_part*.hdf5 files are needed; they ship "
                "as separate zips on Zenodo (https://doi.org/10.5281/zenodo.4916206)."
            )

        with h5py.File(path, "r") as handle:
            if EXAM_ID_DATASET not in handle:
                raise ValueError(
                    f"{path} has no '{EXAM_ID_DATASET}' dataset, so its rows cannot "
                    f"be matched to exam_ids. Found: {sorted(handle.keys())}."
                )
            ids = handle[EXAM_ID_DATASET][:]
            n_rows = int(handle[TRACINGS_DATASET].shape[0])

        if len(ids) != n_rows:
            raise ValueError(
                f"{path} has {n_rows} tracings but {len(ids)} exam_ids; the row "
                "mapping would be ambiguous."
            )

        # The trailing all-zero padding row. Dropped by id, so it cannot be
        # confused with a real record even if a future release moves it.
        in_part = {
            int(exam_id): row for row, exam_id in enumerate(ids) if int(exam_id) != PADDING_EXAM_ID
        }

        expected = set(trace_files.index[trace_files == part].astype(int))
        if set(in_part) != expected:
            only_file = sorted(set(in_part) - expected)[:5]
            only_csv = sorted(expected - set(in_part))[:5]
            raise ValueError(
                f"{path} and exams.csv disagree about which records it holds: "
                f"{len(set(in_part) - expected)} only in the file (e.g. {only_file}), "
                f"{len(expected - set(in_part))} only in the CSV (e.g. {only_csv}). "
                "The local copy is incomplete or has been modified — Zenodo's md5 "
                "for exams.csv is 0107516d3f63864498fb77d15799cc95."
            )

        paths.update(
            {exam_id: f"{part}:{TRACINGS_DATASET}:{row}" for exam_id, row in in_part.items()}
        )
        logger.debug("Resolved %d records in %s (%d rows)", len(in_part), part, n_rows)

    logger.info(
        "Resolved %d exam_ids to rows across %d HDF5 part(s)",
        len(paths),
        trace_files.nunique(),
    )
    return trace_files.index.to_series().map(paths).rename("signal_path")


@register("code15")
class CODE15Splitter(DatasetSplitter):
    """CODE-15% splitting strategy.

    - Builds (and caches) the metadata CSV from ``exams.csv`` plus the exam_id ->
      row resolution the release leaves implicit
    - Stratifies on the rarest of the six abnormalities, keeping NORMAL and
      OTHER apart
    - Groups folds on ``patient_id``: 66,929 patients have more than one record
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # abnormality_codes is empty for 89% of records; without this
                # pandas reads the column as float NaN and the label list stops
                # being a string.
                dtype={"abnormality_codes": str, STRATIFY_COLUMN: str},
                keep_default_na=False,
                na_values=[""],
            ).assign(abnormality_codes=lambda d: d["abnormality_codes"].fillna(""))

        from ecgbench.labels.code15 import load_labels

        labels = load_labels(data_path, config)
        labels["signal_path"] = build_signal_paths(data_path, labels["trace_file"])
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
            "CODE-15%%: %d records, %d patients, %d parts",
            len(df),
            df["patient_id"].nunique(),
            df["trace_file"].nunique(),
        )
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the rarest-abnormality label from the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("stratify_class")
        logger.info("Stratification class distribution:\n%s", labels.value_counts().to_string())
        return labels
