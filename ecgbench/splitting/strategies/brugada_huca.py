"""
Brugada-HUCA splitting strategy.

Only one thing stops this dataset using ``GenericSplitter``: the shipped
``metadata.csv`` carries the labels but **no signal-path column**, and the
waveforms live in a per-subject directory. ``load_metadata`` therefore derives
``files/<patient_id>/<patient_id>`` and writes a normalised CSV to disk as the
config's ``metadata_csv``.

Writing it out is load-bearing rather than a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` and rebuilds record paths from it,
so a fix-up that lived only in this frame would leave every record failing
``corrupt_header``.

Stratification uses the ``brugada`` column verbatim, so unlike the derived
labels elsewhere in ECGBench there is nothing here that can drift from what
``load_labels`` exposes.

Two quirks of the shipped release are worked around here, both confirmed against
``SHA256SUMS.txt`` so they are in the release itself and not in one download:

- ``RECORDS`` has 364 lines but only 363 distinct records — ``files/596382/596382``
  is listed twice. We enumerate subjects from ``metadata.csv`` instead, which is
  authoritative, carries the labels, and has one row per subject.
- a macOS ``files/.DS_Store`` ships inside the release (it is even checksummed),
  so anything globbing ``files/*`` has to filter to directories.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Directory (relative to the dataset root) holding the per-subject record dirs.
RECORDS_DIR = "files"

#: The label file shipped with the release.
SOURCE_CSV = "metadata.csv"

#: Human-readable meanings of the ``brugada`` code, for logs and example scripts.
#: The release documents these in README.md but ships them only as 0/1/2.
BRUGADA_CLASSES = {
    0: "healthy",
    1: "confirmed Brugada syndrome",
    2: "other/atypical",
}


@register("brugada_huca")
class BrugadaHUCASplitter(DatasetSplitter):
    """Brugada-HUCA splitting strategy: derive signal paths, stratify on ``brugada``."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        source = data_path / SOURCE_CSV
        if not source.exists():
            raise FileNotFoundError(
                f"Expected {SOURCE_CSV} at {source}. Point --data-path at the dataset "
                "root — the directory holding metadata.csv, RECORDS and files/."
            )

        df = pd.read_csv(source)
        record_id = config.record_id_column
        if record_id not in df.columns:
            raise ValueError(f"{source} has no '{record_id}' column. Found: {list(df.columns)}")

        signal_col = config.signal_path_columns[config.default_sampling_rate]
        df[signal_col] = df[record_id].map(lambda pid: f"{RECORDS_DIR}/{pid}/{pid}")

        self._check_records_present(df, data_path, config, signal_col)

        df = df.sort_values(record_id).reset_index(drop=True)
        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            # validate_dataset re-reads this file, so a read-only data directory
            # leaves validation rebuilding paths that do not exist.
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def _check_records_present(
        self,
        df: pd.DataFrame,
        data_path: Path,
        config: DatasetConfig,
        signal_col: str,
    ) -> None:
        """Warn if metadata rows have no header on disk, or vice versa.

        Cheap at 363 records, and it catches a partial download before the
        validation engine reports every record as a corrupt header.
        """
        missing = [
            str(row[config.record_id_column])
            for _, row in df.iterrows()
            if not (data_path / f"{row[signal_col]}.hea").exists()
        ]
        if missing:
            logger.warning(
                "%d of %d subjects in %s have no .hea on disk (e.g. %s)",
                len(missing),
                len(df),
                SOURCE_CSV,
                missing[:5],
            )

        # Directories only: the release ships a files/.DS_Store alongside them.
        records_root = data_path / RECORDS_DIR
        if records_root.is_dir():
            on_disk = {p.name for p in records_root.iterdir() if p.is_dir()}
            unlisted = on_disk - {str(v) for v in df[config.record_id_column]}
            if unlisted:
                logger.warning(
                    "%d record director(ies) under %s/ have no row in %s: %s",
                    len(unlisted),
                    RECORDS_DIR,
                    SOURCE_CSV,
                    sorted(unlisted)[:5],
                )

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the ``brugada`` diagnosis, used verbatim as the fold label."""
        column = config.label_column
        if column not in df.columns:
            raise ValueError(
                f"'{column}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[column].rename("brugada")
        counts = labels.value_counts().sort_index()
        logger.info(
            "Brugada diagnosis distribution:\n%s",
            "\n".join(
                f"  {code} ({BRUGADA_CLASSES.get(code, '?')}): {n}" for code, n in counts.items()
            ),
        )
        rare = counts[counts < 10]
        if not rare.empty:
            # Not pooled on purpose: 'other/atypical' is neither healthy nor
            # confirmed, so folding it into either class would be wrong.
            logger.warning(
                "Class(es) %s have fewer records than the fold count, so they "
                "cannot appear in every fold. They are kept unpooled because the "
                "Brugada codes are not interchangeable.",
                dict(rare),
            )
        return labels
