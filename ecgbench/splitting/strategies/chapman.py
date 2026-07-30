"""
Chapman-Shaoxing (figshare release) splitting strategy.

The figshare collection ships its metadata as ``Diagnostics.xlsx`` and its signals
as one CSV per record under ``ECGData/``, named by ``FileName`` without the
extension. Neither shape suits the pipeline directly:

- ``pandas.read_csv`` cannot open .xlsx, and ``validate_dataset`` re-reads
  ``config.metadata_csv`` from disk itself, so an in-memory conversion would leave
  validation with nothing.
- ``FileName`` is a bare stem, so the signal path needs both the ``ECGData/``
  prefix and a ``.csv`` suffix. Putting that fix-up only in ``load_metadata``
  (as this module used to) meant validation built paths from the raw column and
  every record failed ``corrupt_header``.

So ``load_metadata`` writes a normalised ``ecgbench_metadata.csv`` into the
dataset root, with a real ``signal_path`` column, and the config points at that.
Both the splitter and the validation engine then read the same file.

Stratification uses ``Rhythm``, the single-label rhythm diagnosis. ``Beat`` is
space-separated multi-label and is exposed through the label loader instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Directory holding one CSV per record, relative to the dataset root.
SIGNAL_DIR = "ECGData"

#: Denoised variant shipped alongside; not used for splitting or validation.
DENOISED_DIR = "ECGDataDenoised"

#: Source metadata, in preference order. The download script converts the .xlsx
#: to .csv; reading the .xlsx directly needs openpyxl, so it is the fallback.
SOURCE_METADATA = ("Diagnostics.csv", "Diagnostics.xlsx")


def _read_source_metadata(data_path: Path) -> pd.DataFrame:
    """Read Diagnostics.csv if present, else Diagnostics.xlsx."""
    for name in SOURCE_METADATA:
        path = data_path / name
        if not path.exists():
            continue
        if path.suffix == ".csv":
            return pd.read_csv(path)
        try:
            return pd.read_excel(path)
        except ImportError as e:
            raise ImportError(
                f"Reading {name} needs openpyxl (pip install openpyxl), or convert it "
                "to Diagnostics.csv first — examples/download_chapman_figshare.py does "
                "that as part of fetching the dataset."
            ) from e

    raise FileNotFoundError(
        f"Expected one of {SOURCE_METADATA} in {data_path}. Fetch the dataset with "
        "examples/download_chapman_figshare.py, or point --data-path at the "
        "directory holding Diagnostics.* and ECGData/."
    )


def build_metadata(data_path: Path, config: DatasetConfig) -> pd.DataFrame:
    """Normalise the source metadata into the frame the pipeline expects."""
    df = _read_source_metadata(data_path)

    record_col = config.record_id_column
    if record_col not in df.columns:
        raise ValueError(
            f"Expected column '{record_col}' in the Chapman metadata. "
            f"Found: {list(df.columns)}"
        )

    signal_dir = data_path / SIGNAL_DIR
    if not signal_dir.is_dir():
        raise FileNotFoundError(
            f"Expected the signal directory {signal_dir}. Point --data-path at the "
            "dataset root, not at ECGData/ itself."
        )

    signal_col = config.signal_path_columns[config.default_sampling_rate]
    df[signal_col] = df[record_col].astype(str).map(lambda name: f"{SIGNAL_DIR}/{name}.csv")

    df = df.sort_values(record_col).reset_index(drop=True)
    logger.info("Loaded Chapman-Shaoxing metadata: %d records", len(df))
    return df


@register("chapman_shaoxing")
class ChapmanSplitter(DatasetSplitter):
    """Chapman-Shaoxing (figshare) splitting strategy.

    - Normalises Diagnostics.xlsx/.csv into a generated metadata CSV on first run
    - Builds ``signal_path`` as ``ECGData/<FileName>.csv``
    - Stratifies on ``Rhythm``; no patient grouping (one record per patient)
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        df = build_metadata(data_path, config)
        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            # validate_dataset re-reads this file, so a read-only data directory
            # means validation cannot resolve a single signal path.
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Use the Rhythm column directly — it is already single-label."""
        labels = df[config.label_column].astype(str).rename("rhythm")
        logger.info("Rhythm distribution:\n%s", labels.value_counts().to_string())
        return labels
