"""
Wilson Central Terminal ECG Database splitting strategy.

The release ships no metadata table — only 540 ``.dat``/``.hea`` pairs under 92
``patientNNN/`` directories, plus ``RECORDS``, ``SHA256SUMS.txt`` and
``LICENSE.txt``. ``load_metadata`` therefore builds one from
``ecgbench.labels.wctecgdb``, which parses the ``#Age:``, ``#Sex:``,
``#Diagnosis report:`` and ``#Reconstruct Precordials:`` comment lines out of each
header, and caches it in the dataset root as ``config.metadata_csv``.

Writing that cache to disk is load-bearing rather than a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself instead of
reusing this frame, so an in-memory-only table would leave validation with no
metadata at all.

Two structural facts drive the rest of this module:

- **``seg01`` is not a unique record id.** Every one of the 92 patient directories
  contains a ``seg01``, so the record id is the RECORDS path with its slash
  replaced (``patient001_seg01``) while the signal path keeps the slash
  (``patient001/seg01``). Two columns, one derived from the other.
- **Segments are heavily clustered by patient**, 1 to 31 per patient with a median
  of 4, and age, sex and the diagnosis are constant within a patient. Grouping on
  ``patient_id`` is therefore not optional: without it a patient with 31 segments
  would put near-duplicate 10-second windows of the same admission on both sides of
  the split, and the diagnosis label would leak outright.

Stratification uses ``diagnosis_group``, the label loader's 8-way reduction of the
43 free-text admission diagnoses. The strings themselves cannot be stratified on:
28 of the 40 corrected values occur for exactly one patient, so most classes could
not be spread over 10
folds at all.

Even reduced, three of the eight groups hold 3-6 patients, fewer than the 10 folds,
so stratification is approximate in a way worth stating: on the shipped v1.0.1 the
3-patient ``Bradyarrhythmia or conduction block`` group reaches only 3 of the 10
folds, and the default test fold (10) holds no bradyarrhythmia and no
cardiomyopathy record at all. ``StratifiedGroupKFold`` does **not** emit the "least
populated class" warning that ``StratifiedKFold`` does, so silence here is not
evidence of balance — check the per-fold distribution. Fold sizes vary for the
same reason: 92 patients over 10 folds gives 7-11 patients and 49-58 records per
fold rather than a uniform 54.

``diagnosis_group`` is a stratification target only. It is a single-label reduction
of free text where six patients' strings name two conditions — train on the
``diagnosis`` column and group your own way.
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
STRATIFY_COLUMN = "diagnosis_group"

#: Separator for the list-valued ``reconstructed_precordials`` column when it
#: round-trips through CSV. A comma would collide with the channel list's own
#: commas ("V1, V1-raw, V2, V2-raw"), so it must not be one.
LIST_SEPARATOR = ";"


@register("wctecgdb")
class WCTECGSplitter(DatasetSplitter):
    """Wilson Central Terminal ECG splitting strategy.

    - Builds (and caches) the metadata CSV from the per-record WFDB header comments
    - Derives the nested signal path from the record id
    - Stratifies on the coarse admission-diagnosis group
    - Groups on ``patient_id``: 540 segments come from 92 patients, 1-31 each
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        from ecgbench.labels.wctecgdb import load_labels

        df = load_labels(data_path, config).reset_index()

        # Signals live in per-patient subdirectories, so the path is not the record
        # id: "patient001_seg01" identifies the record, "patient001/seg01" locates
        # it. Building the path here rather than in the label loader keeps the
        # loader free of layout knowledge.
        signal_col = config.signal_path_columns[config.default_sampling_rate]
        df[signal_col] = df["patient_id"] + "/" + df["segment"]

        # reconstructed_precordials is a list; a plain to_csv would write its Python
        # repr and read back as a string that looks parseable but is not.
        for column in df.columns:
            if df[column].map(lambda v: isinstance(v, list)).any():
                df[column] = df[column].map(
                    lambda v: LIST_SEPARATOR.join(v) if isinstance(v, list) else v
                )

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
        """Return the coarse admission-diagnosis group, from the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("diagnosis_group")
        logger.info(
            "Diagnosis-group distribution (records):\n%s", labels.value_counts().to_string()
        )
        if config.patient_id_column in df.columns:
            per_patient = (
                df.drop_duplicates(subset=[config.patient_id_column])[STRATIFY_COLUMN]
                .value_counts()
                .to_string()
            )
            # The per-patient counts are what the grouped splitter actually balances;
            # the record counts above are dominated by the 31-segment patients.
            logger.info("Diagnosis-group distribution (patients):\n%s", per_patient)
        return labels
