"""
MHD-effect (ECG in MRI scanners) splitting strategy.

The release ships no metadata table — 53 ``.dat``/``.hea``/``.qrs`` triples plus
README, RECORDS, ANNOTATORS, LICENSE.txt and SHA256SUMS.txt. ``load_metadata``
builds one from ``ecgbench.labels.mhd_effect_ecg_mri``, which parses each header's
``#--Key:Value`` block and counts its QRS annotations, and caches it next to the
data as ``config.metadata_csv``.

Writing that cache is load-bearing, not a convenience: ``validate_dataset``
re-reads ``data_path / config.metadata_csv`` itself rather than reusing this
frame, so an in-memory-only table would leave validation with no metadata.

**Folds are grouped by ``subject_key``, not by the filename's subject number.**
The filename number is scoped per scanner — ``ECGMRI1T01`` and ``ECGMRI3T01`` are
different people — and three subjects were recorded in more than one scanner
(``3T01 == 7T04``; ``1T01 == 3T02 == 7T05``). Grouping on the number would put one
person's 3T record in train and their 7T record in test, which is precisely the
leakage that matters for a dataset built to compare field strengths. The label
loader derives ``subject_key`` from sex/age/weight/height instead, giving 26
groups over 53 records; read its docstring for the ways that key can still be
wrong.

Stratification is on ``condition`` — reference / 1T / 3T / 7T — the dataset's
independent variable.

Two consequences of 53 records and 26 subjects, stated rather than left to be
discovered:

- **Stratification is necessarily approximate, and rare conditions concentrate.**
  ``StratifiedGroupKFold`` assigns whole subjects to folds, and a subject can span
  conditions: the ``1T01 == 3T02 == 7T05`` group alone carries 1T, 3T, 7T *and*
  reference records. Since both 1T records belong to that one subject, all of 1T
  necessarily lands in a single fold. Coverage per condition is 23 subjects at 3T,
  7 with a reference, 5 at 7T and 1 at 1T, so nothing can spread the tail evenly.
  Rare classes are deliberately **not** pooled: collapsing 1T, 7T and reference
  into one bucket would balance the folds by destroying the only distinction the
  dataset is about.
- **Records vary in length by a factor of 30 and in channel count.** Folds say
  nothing about either; batch with ``window=`` and filter on ``lead_config`` or
  ``n_signals`` first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column carrying the stratification label, attached by the label loader.
STRATIFY_COLUMN = "condition"

#: Column carrying the derived patient ID. Must match config.patient_id_column.
SUBJECT_COLUMN = "subject_key"


@register("mhd_effect_ecg_mri")
class MHDEffectECGMRISplitter(DatasetSplitter):
    """MHD-effect ECG-in-MRI splitting strategy.

    - Builds (and caches) the metadata CSV from the WFDB headers and .qrs files
    - Groups folds by the derived ``subject_key``, so no subject spans folds
    - Stratifies on the acquisition condition (reference / 1T / 3T / 7T)
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            # subject_number is a zero-padded string ('01'); left to pandas it
            # would come back as int 1 and stop matching the filename.
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype={"subject_number": str},
            )

        from ecgbench.labels.mhd_effect_ecg_mri import load_labels

        df = load_labels(data_path, config).reset_index()

        # Signals sit flat in the dataset root, named by the bare record stem.
        signal_col = config.signal_path_columns[config.default_sampling_rate]
        df[signal_col] = df[config.record_id_column]

        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return the acquisition condition attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        if SUBJECT_COLUMN not in df.columns:
            raise ValueError(
                f"'{SUBJECT_COLUMN}' missing — it is this dataset's patient ID, and "
                "without it folds would split single subjects across scanners."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("condition")
        summary = (
            df.groupby(labels, observed=True)[SUBJECT_COLUMN]
            .agg(records="size", subjects="nunique")
            .sort_values("records", ascending=False)
        )
        logger.info(
            "Condition distribution (subjects is what bounds fold balance):\n%s",
            summary.to_string(),
        )
        return labels
