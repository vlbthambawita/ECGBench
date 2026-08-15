"""
EDGAR splitting strategy.

Two jobs the generic splitter cannot do.

**Build the metadata, and the files it points at.** EDGAR ships no table of any
kind — 26 experiments, one free-text README each — and every recording lives
inside a zip. ``load_metadata`` therefore unpacks the 24 authoritative archives
into ``ecgbench_extracted/`` and scans every ``.mat`` in them, via
``ecgbench.labels.edgar``, which is the same loader ``load_labels`` uses so the
stratification label and the exposed labels cannot drift apart.

Writing that cache to disk is load-bearing rather than a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself instead
of reusing this DataFrame, and it opens the signal files by path, so an
in-memory-only frame would leave validation with no metadata and no files.

**Stratify on a coarser axis than the label, because the release cannot carry
the label.** ``recording_surface`` has five values and is what users want, but
two of them come from a single subject each — all 190 intramural recordings are
one dog's plunge needles, all 16 transmembrane runs are one simulated anatomy —
and a class living in one patient group cannot be spread over ten folds.
Measured over the real table with ``StratifiedGroupKFold(10)``, stratifying on
``recording_surface`` leaves **all ten** folds missing at least one class;
stratifying on the body-surface/cardiac-surface split leaves three. So
``stratify_class`` is that binary, and ``recording_surface`` stays the label.

**What no stratification can fix, and what nobody should be surprised by.** Four
subjects hold 92% of the recordings (charles_pstov_pat1 alone has 944 of 2,943)
and five hold two each, so patient-safe folds are necessarily unequal — the
default fold-10 test split is a couple of dozen records, not a tenth of the
dataset. That is a property of a repository of 20 experiments, not of the
splitter, and the alternative — splitting one subject's 944 recordings across
train and test — would make pacing-site localisation look solved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column the splitter adds and stratifies on. Deliberately coarser than
#: ``config.label_column`` — see the module docstring.
STRATIFY_COLUMN = "stratify_class"


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add the body-surface / cardiac-surface class used to build folds.

    Everything that is not ``torso`` is pooled: epicardial socks and cages,
    endocardial catheters, plunge needles and simulated cardiac sources. They
    are physically different measurements, and the pooling is a concession to
    twenty patient groups rather than a claim that they are alike. Train on
    ``recording_surface``; this column exists to build folds.
    """
    out = df.copy()
    out[STRATIFY_COLUMN] = out["recording_surface"].map(
        lambda surface: "body_surface" if surface == "torso" else "cardiac_surface"
    )
    return out


@register("edgar")
class EDGARSplitter(DatasetSplitter):
    """edgar splitting strategy: generated metadata, subject-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        from ecgbench.labels.edgar import _STRING_COLUMNS, extract_archives, scan_records

        csv_path = data_path / config.metadata_csv
        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return attach_stratify_class(
                pd.read_csv(
                    csv_path, sep=config.metadata_csv_separator, dtype=_STRING_COLUMNS
                )
            )

        logger.info("Unpacking EDGAR archives under %s", data_path)
        extract_archives(data_path)
        df = scan_records(data_path)
        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            # validate_dataset re-reads this file, so a read-only data directory
            # leaves validation with no metadata at all. Fail loudly instead.
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because EDGAR's recordings have "
                "to be unpacked from their archives and the validation engine reads "
                "the metadata CSV from disk."
            ) from e
        return attach_stratify_class(df)

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return the body-surface/cardiac-surface class, not ``recording_surface``.

        The difference is the point; see the module docstring for the measurement
        that chose it.
        """
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("surface_class")

        group_column = config.patient_id_column
        if group_column and group_column in df.columns:
            n_groups = df[group_column].nunique()
            # StratifiedGroupKFold emits silently EMPTY folds once n_folds
            # exceeds the group count, so say it here rather than let a user
            # find two empty fold CSVs.
            if n_groups < config.n_folds:
                raise ValueError(
                    f"edgar has {n_groups} subjects but n_folds={config.n_folds}. "
                    "StratifiedGroupKFold would emit empty folds; reduce n_folds in "
                    "the config."
                )
        return labels
