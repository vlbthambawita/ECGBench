"""
IKEM splitting strategy — resolving exam_id to a row of one of three HDF5 parts.

Same shape of problem as CODE-15%: no record has a file of its own. Three parts
hold one 3-D ``tracings`` array each — ``(48264, 4096, 8)``,
``(48683, 4096, 8)`` and ``(1183, 4096, 8)`` — so a record's "path" must name a
row as well as a file: ``exams_part_1.hdf5:tracings:417``.

Two differences from CODE-15% make this the easier of the two, and both were
checked rather than assumed:

**exams.csv has no `trace_file` column**, so there is no hint which part holds a
record and all parts are scanned. That costs three reads of a 1-D int32 array
(about 400 kB) against 6.6 GB of waveforms, so it is free.

**The parts are exactly the records.** 48,264 + 48,683 + 1,183 = 98,130, which
is the CSV's row count, and the exam_id sets agree exactly. CODE-15% needs a
sentinel rule because every one of its parts carries a trailing all-zero row
with ``exam_id`` 0; IKEM has no such padding row. Keying on ``exam_id`` rather
than on position means a future release that adds one is handled anyway.

**Integrity is unusually cheap to check here.** Each part carries a ``hashes``
dataset holding the SHA-1 of every record's raw int16 bytes. Verifying them
would mean reading all 6.6 GB, which this splitter does not do, but their
*uniqueness* is a 1-D read: all 98,130 are distinct in the published release, so
a copy with duplicated or truncated rows is caught without touching a waveform.
That is checked on every run.

**Patient grouping is mandatory and large.** 30,290 patients hold the 98,130
records, 19,078 of them contributed more than one and the largest contributed
96 — 86,918 records (88.6%) belong to a multi-record patient. Ungrouped folds
would leak almost the whole dataset. Unlike CODE-15% there are no byte-identical
duplicate recordings hiding behind different patient_ids, so grouping on
``patient_id`` is sufficient here rather than merely necessary.

As everywhere else the resolved reference has to reach disk: ``validate_dataset``
re-reads ``config.metadata_csv`` and rebuilds paths from the raw column, so a
fix-up living only in the returned DataFrame is invisible to it and every record
fails ``corrupt_header``.

Stratification uses ``stratify_class`` from :mod:`ecgbench.labels.ikem` — the
cart's ventricular rate banded into BRADY/NORMAL/TACHY. This release ships no
diagnoses, so there is no diagnostic class to stratify on; see that module.
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

#: Column the label loader attaches, used for stratification.
STRATIFY_COLUMN = "stratify_class"


def build_signal_paths(data_path: Path, exam_ids: pd.Index, config: DatasetConfig) -> pd.Series:
    """Map each exam_id to ``<part>:tracings:<row>``, keyed on the parts' own ids.

    Args:
        data_path: Dataset root holding the ``exams_part_*.hdf5`` files.
        exam_ids: The exam_ids to resolve, i.e. the label frame's index.
        config: Used for its URL in error messages.

    Returns:
        exam_id -> signal path, in the order ``exam_ids`` was given.

    Raises:
        FileNotFoundError: no part is on disk.
        ValueError: a part is malformed, the record hashes are not unique, or the
            parts and the metadata disagree about which records exist.
    """
    import h5py

    from ecgbench.labels.ikem import (
        EXAM_ID_DATASET,
        HASHES_DATASET,
        N_RECORDS,
        PART_GLOB,
    )

    parts = sorted(data_path.glob(PART_GLOB))
    if not parts:
        raise FileNotFoundError(
            f"No {PART_GLOB} found in {data_path}. IKEM ships three parts totalling "
            f"6.6 GB on Zenodo (see {config.url}); all of them are needed."
        )

    paths: dict[int, str] = {}
    hashes: dict[bytes, int] = {}
    for part in parts:
        with h5py.File(part, "r") as handle:
            if EXAM_ID_DATASET not in handle:
                raise ValueError(
                    f"{part} has no '{EXAM_ID_DATASET}' dataset, so its rows cannot be "
                    f"matched to exam_ids. Found: {sorted(handle.keys())}."
                )
            ids = handle[EXAM_ID_DATASET][:]
            n_rows = int(handle[TRACINGS_DATASET].shape[0])
            part_hashes = handle[HASHES_DATASET][:] if HASHES_DATASET in handle else None

        if len(ids) != n_rows:
            raise ValueError(
                f"{part} has {n_rows} tracings but {len(ids)} exam_ids; the row mapping "
                "would be ambiguous."
            )

        for row, exam_id in enumerate(ids):
            exam_id = int(exam_id)
            if exam_id in paths:
                raise ValueError(
                    f"exam_id {exam_id} appears in more than one part ({paths[exam_id]} "
                    f"and {part.name}:{TRACINGS_DATASET}:{row}); the release's parts are "
                    "meant to be disjoint."
                )
            paths[exam_id] = f"{part.name}:{TRACINGS_DATASET}:{row}"

        # Waveform SHA-1s, which the release ships precisely so this is cheap.
        # Uniqueness alone catches a copy with duplicated or repeated rows
        # without reading any of the 6.6 GB of tracings.
        if part_hashes is not None:
            for row, digest in enumerate(part_hashes):
                digest = bytes(digest)
                if digest in hashes:
                    raise ValueError(
                        f"{part.name} row {row} has the same waveform SHA-1 as another "
                        "record in this release. The published IKEM v1.0.0 has 98,130 "
                        "distinct hashes, so this copy has duplicated rows."
                    )
                hashes[digest] = row
        else:
            logger.warning(
                "%s has no '%s' dataset; skipping the hash check", part.name, HASHES_DATASET
            )

    expected = set(int(i) for i in exam_ids)
    if set(paths) != expected:
        only_file = sorted(set(paths) - expected)[:5]
        only_csv = sorted(expected - set(paths))[:5]
        raise ValueError(
            f"The HDF5 parts in {data_path} and exams.csv disagree about which records "
            f"exist: {len(set(paths) - expected)} only in the files (e.g. {only_file}), "
            f"{len(expected - set(paths))} only in the CSV (e.g. {only_csv}). The local "
            "copy is incomplete or modified — all five published files have md5s on "
            "Zenodo and ship as zenodo_md5sums.txt, so run `md5sum -c` against it."
        )
    if len(paths) != N_RECORDS:
        logger.warning(
            "Resolved %d records; the published v1.0.0 has %d. Proceeding, but this is "
            "not the release ECGBench's reference manifest was built from.",
            len(paths),
            N_RECORDS,
        )

    logger.info(
        "Resolved %d exam_ids to rows across %d HDF5 part(s); %d distinct waveform hashes",
        len(paths),
        len(parts),
        len(hashes),
    )
    return exam_ids.to_series().map(paths).rename("signal_path")


@register("ikem")
class IKEMSplitter(DatasetSplitter):
    """IKEM splitting strategy.

    - Builds (and caches) the metadata CSV from ``exams.csv`` plus the exam_id ->
      (part, row) resolution the release leaves implicit
    - Checks the shipped waveform SHA-1s are unique, cheaply
    - Stratifies on the cart's ventricular rate, banded; no diagnoses ship
    - Groups folds on ``patient_id``: 88.6% of records share a patient with
      another record
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype={STRATIFY_COLUMN: str, "patient_id": str},
            )

        from ecgbench.labels.ikem import load_labels

        labels = load_labels(data_path, config)
        labels["signal_path"] = build_signal_paths(data_path, labels.index, config)
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
            "IKEM: %d records, %d patients, %d parts",
            len(df),
            df["patient_id"].nunique(),
            df["signal_path"].str.split(":").str[0].nunique(),
        )
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the banded ventricular-rate label from the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("stratify_class")
        logger.info("Stratification class distribution:\n%s", labels.value_counts().to_string())
        return labels
