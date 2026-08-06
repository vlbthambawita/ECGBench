"""
Ningbo First Hospital IVA splitting strategy.

The release ships ``Diagnosis.xlsx`` and one CSV per patient under
``PVCVTRawECGData/``, named by ``HospitalID`` with no path in the table. Neither
shape suits the pipeline directly:

- ``pandas.read_csv`` cannot open .xlsx, and ``validate_dataset`` re-reads
  ``config.metadata_csv`` from disk itself, so an in-memory conversion would leave
  validation with nothing.
- there is no signal-path column at all, so the ``PVCVTRawECGData/`` prefix and
  the ``.csv`` suffix have to be built. Putting that fix-up only in
  ``load_metadata`` would leave validation building paths from a column that does
  not exist — the failure mode Chapman-Shaoxing shipped with for months.

So ``load_metadata`` writes a normalised ``ecgbench_metadata.csv`` into the
dataset root, built by ``ecgbench.labels.ningbo_iva`` — the same loader users get
from ``load_labels``, so the stratification label and the exposed labels cannot
drift — and the config points at that file.

Stratification uses ``left_right``: RVOT (257) or LVOT (77), the
ablation-confirmed outflow tract and the canonical task on this dataset. It is
already one label per patient, so nothing is reduced. ``sublocation`` is the
finer label and is deliberately **not** used: 12 values over 334 patients, five of
them under ten cases and 40 cells blank, cannot support stratified ten-fold
splits.

**No patient grouping is configured, because none is possible or needed.**
``HospitalID`` is simultaneously the record id and the patient id — 334 patients,
334 records, one each — so ``patient_id_column`` stays null rather than pointing
at the record id and asserting a grouping that was never exercised.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column the label loader attaches, used for stratification.
STRATIFY_COLUMN = "left_right"


@register("ningbo_iva")
class NingboIVASplitter(DatasetSplitter):
    """Ningbo IVA splitting strategy.

    - Normalises Diagnosis.xlsx into a generated metadata CSV on first run,
      adding the ``PVCVTRawECGData/<HospitalID>.csv`` path the table omits
    - Stratifies on ``left_right`` (RVOT/LVOT), already single-label
    - No patient grouping: one record per patient, and the two ids are the same
      column
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # HospitalID is a 6-7 digit opaque identifier, not a number; read
                # as int it would still round-trip, but the record ids in the fold
                # CSVs would then differ in type from the ones ECGDataset compares
                # against.
                dtype={config.record_id_column: str},
            )

        from ecgbench.labels.ningbo_iva import load_labels

        df = load_labels(data_path, config).reset_index()
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
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Use ``left_right`` directly — one ablation-confirmed tract per patient."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("stratify_class")
        logger.info("Outflow tract distribution:\n%s", labels.value_counts().to_string())
        return labels
