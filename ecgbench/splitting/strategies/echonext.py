"""
EchoNext splitting strategy — predefined splits over rows of shared .npy arrays.

EchoNext does not give each record a file. It ships **one array per split**,
``EchoNext_<split>_waveforms.npy`` of shape ``(N, 1, 2500, 12)``, and the metadata
CSV's rows line up with those arrays row for row *within each split*. So a record's
"path" has to name a row as well as a file, which is what the ``npy`` signal format
reads: ``EchoNext_test_waveforms.npy:417``.

Three things this splitter has to get right.

**1. The row reference must be written to disk, not just built in memory.**
``validate_dataset`` re-reads ``config.metadata_csv`` itself and rebuilds paths from
the raw column, so a fix-up that lives only in ``load_metadata`` is invisible to it
and every record fails ``corrupt_header``. This module therefore writes a
normalised ``ecgbench_metadata.csv`` into the dataset root and the config points at
that, the same arrangement ``chapman.py`` uses.

**2. The 17,457 ``no_split`` records are excluded from the partition.** They are the
non-latest ECGs of patients whose latest ECG is in val or test, and the publisher
excludes them from training. ECGBench drops them because the alternative is silent
leakage: ``export.py`` maps any fold outside ``fold_mapping`` to ``"train"``, which
would put 2,499 test patients' and 2,119 val patients' earlier recordings into the
training set. With them excluded the three splits are patient-disjoint — verified,
0 patients shared between any pair. The records stay reachable through
``ecgbench.labels.echonext``; they are simply not part of the ECGBench partition.

**3. Stratification is the composite SHD label.** ``shd_moderate_or_greater_flag``
is the dataset's headline target: moderate-or-greater structural heart disease.
The ten per-condition flags are exposed through the label loader, not re-derived
here.

Sizes after exclusion: 82,543 records over 36,286 patients —
train 72,475 / val 4,626 / test 5,442.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: The release's own metadata table. The bundled README calls this
#: ``EchoNext_metadata_100k.csv``, but the shipped and checksummed file is
#: lowercase; the capitalised name does not exist.
SOURCE_METADATA = "echonext_metadata_100k.csv"

#: Normalised table this splitter writes, and the one the config names.
GENERATED_METADATA = "ecgbench_metadata.csv"

#: Waveform array per split, holding every record of that split.
WAVEFORM_TEMPLATE = "EchoNext_{split}_waveforms.npy"

#: Split name -> ECGBench fold number. ``no_split`` deliberately has none: those
#: rows are dropped before folds are assigned (see the module docstring).
SPLIT_TO_FOLD = {"train": 1, "val": 2, "test": 3}

#: Rows the publisher assigned to no split at all.
EXCLUDED_SPLIT = "no_split"

#: Composite target: moderate or greater structural heart disease.
STRATIFY_COLUMN = "shd_moderate_or_greater_flag"


@register("echonext")
class EchoNextSplitter(DatasetSplitter):
    """Honours EchoNext's own train/val/test assignment, minus ``no_split``."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        source = Path(data_path) / SOURCE_METADATA
        if not source.exists():
            raise FileNotFoundError(
                f"EchoNext metadata not found at {source}. Point --data-path at the "
                "1.1.0 directory holding echonext_metadata_100k.csv and the "
                "EchoNext_*.npy arrays. EchoNext is credentialed: download it from "
                "https://physionet.org/content/echonext/1.1.0/ ."
            )

        df = pd.read_csv(source)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])

        # The row index within a split is the row index into that split's array,
        # so it must be taken BEFORE any filtering or reordering.
        df["array_row"] = df.groupby("split").cumcount()

        excluded = int((df["split"] == EXCLUDED_SPLIT).sum())
        df = df[df["split"] != EXCLUDED_SPLIT].reset_index(drop=True)
        logger.info(
            "Excluded %d '%s' records; they share patients with val/test and the "
            "publisher excludes them from training",
            excluded,
            EXCLUDED_SPLIT,
        )

        df["signal_path"] = [
            f"{WAVEFORM_TEMPLATE.format(split=split)}:{row}"
            for split, row in zip(df["split"], df["array_row"])
        ]
        df["fold"] = df["split"].map(SPLIT_TO_FOLD).astype(int)

        # Written to disk so validate_dataset reads the same paths this returns.
        generated = Path(data_path) / GENERATED_METADATA
        df.to_csv(generated, index=False)
        logger.info(
            "Wrote %s (%d records, %d patients)",
            generated,
            len(df),
            df["patient_key"].nunique(),
        )

        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' not in EchoNext metadata; found "
                f"{list(df.columns)[:12]}..."
            )
        return df[STRATIFY_COLUMN].astype(int)
