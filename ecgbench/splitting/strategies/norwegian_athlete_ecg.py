"""
Norwegian Endurance Athlete ECG Database splitting strategy.

The release ships no metadata table — only 28 ``.dat``/``.hea`` pairs, ``RECORDS``,
``SHA256SUMS.txt`` and ``LICENSE.txt``. ``load_metadata`` therefore builds one
from ``ecgbench.labels.norwegian_athlete_ecg``, which parses the two
interpretation comment lines out of each header, and caches it next to the data
as ``config.metadata_csv``.

Writing that cache to disk is load-bearing rather than a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself instead
of reusing this frame, so an in-memory-only table would leave validation with no
metadata at all.

Stratification uses ``cardiologist_primary_rhythm`` from the label loader — the
rhythm the cardiologist's reading opens with. The obvious alternative, the
cardiologist's overall verdict, is degenerate: 26 of the 28 records are "Normal
ECG" and the other 2 "Borderline ECG", which would put every borderline record in
a different fold and tell you nothing. The rhythm splits 16 / 7 / 5.

Two consequences of a 28-record dataset that are worth stating rather than
discovering:

- **Folds hold 2-3 records each** (3 in folds 1-8, 2 in folds 9 and 10), so the
  default mapping gives 24 train / 2 val / 2 test. Both val and test end up
  holding only ``Normal sinus rhythm`` records — with 2-record folds and a 16/7/5
  rhythm split there is no assignment that avoids it. Treat this as a
  cross-validation harness rather than a train/val/test split: pass
  ``ECGDataset(split=None, fold_numbers=[...])`` and rotate which folds you hold
  out. The rhythm is spread as evenly as 28 records permit across folds 1-8.
- **sklearn warns about the least populated class**, because 5 < 10. It does not
  raise (that needs *every* class below ``n_splits``), and the fold assignment is
  still deterministic and patient-disjoint. The warning is expected here.

One record per athlete, so there is no grouping column.
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
STRATIFY_COLUMN = "cardiologist_primary_rhythm"

#: Separator for the list-valued finding columns when they round-trip through CSV.
#: A comma would collide with the statements' own commas, which is exactly the
#: trap the label parser exists to avoid — do not change it to one.
LIST_SEPARATOR = ";"


@register("norwegian_athlete_ecg")
class NorwegianAthleteECGSplitter(DatasetSplitter):
    """Norwegian athlete ECG splitting strategy.

    - Builds (and caches) the metadata CSV from the per-record WFDB header comments
    - Stratifies on the cardiologist's opening rhythm
    - No patient grouping: 28 records, 28 athletes
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        from ecgbench.labels.norwegian_athlete_ecg import load_labels

        df = load_labels(data_path, config).reset_index()

        # Signals sit flat in the dataset root, named by the bare record stem, so
        # the path column is the record ID. Keeping it as its own column anyway
        # means the exported fold CSVs and ECGDataset resolve paths the same way
        # every other dataset does.
        signal_col = config.signal_path_columns[config.default_sampling_rate]
        df[signal_col] = df[config.record_id_column]

        # Findings are lists; a plain to_csv would write their Python repr and
        # read back as a string that looks parseable but is not.
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
        """Return the cardiologist's opening rhythm, attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("rhythm")
        logger.info("Rhythm distribution:\n%s", labels.value_counts().to_string())
        return labels
