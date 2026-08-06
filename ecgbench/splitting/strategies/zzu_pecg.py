"""
ZZU-pECG splitting strategy — normalising a metadata table that ships unusable.

``AttributesDictionary.csv`` cannot be pointed at by a config as it stands, for
four independent reasons, and every one of them has to be fixed **on disk**
rather than in memory: ``validate_dataset`` re-reads ``config.metadata_csv`` and
rebuilds signal paths from the raw column, so a fix-up living only in the frame
this returns is invisible to it and every record fails ``corrupt_header``. That
is the bug Chapman shipped with for months, and it is why ``chapman.py``,
``code15.py`` and this module all write a normalised CSV.

1. **The Filename column omits the directory the waveforms are in.** It reads
   ``P00/P00001/P00001_E01``; the files are under ``Child_ecg/``. Paths must
   resolve relative to ``data_path``.
2. **Age is ``"572d"``**, a string with a unit suffix.
3. **Gender is ``"'Female'"``**, quoted inside the field.
4. **Every label column is ``;``-packed**, and the two code columns mix codes
   with prose (see :mod:`ecgbench.labels.zzu_pecg`).

Two things about this dataset that the splitter has to get right:

**Patient grouping is required.** 11,643 patients hold the 14,190 records; 1,691
contributed more than one, up to 19, covering 4,238 records (29.9%). The patient
key is also embedded in the path (``P00/P00001/P00001_E01``), so an ungrouped
split is visibly wrong as well as wrong.

**Stratification cannot use the ECG findings.** They are multi-label with 99
distinct codes over 14,190 records, which does not partition ten ways. The folds
are stratified on ``primary_disease_group`` from the label loader instead — a
rarest-wins reduction of the four ICD-10 disease groups, so Cardiomyopathy (147
records) and Kawasaki disease (194) survive the split. The reduction lives in the
label loader, not here, so the exposed labels and the fold labels cannot drift.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column the label loader derives, used for stratification.
STRATIFY_COLUMN = "primary_disease_group"

#: Columns whose values are ``;``-packed lists or otherwise must stay strings
#: through the CSV round trip. Without this, a cached read turns an all-empty
#: label column into float NaN and the list stops being a string.
_STRING_COLUMNS = (
    "patient_id",
    "signal_path",
    "sex",
    "aha_codes",
    "aha_base_codes",
    "chn_codes",
    "chn_base_codes",
    "ecg_findings",
    "icd10_codes",
    "disease_groups",
    STRATIFY_COLUMN,
    "psqi_by_lead",
    "bassqi_by_lead",
    "bsqi_by_lead",
)


@register("zzu_pecg")
class ZZUPediatricSplitter(DatasetSplitter):
    """ZZU-pECG splitting strategy.

    - Builds (and caches) a normalised metadata CSV: prefixed signal paths,
      parsed age and sex, unpacked code lists
    - Stratifies on the rarest ICD-10 disease group a record carries
    - Groups folds on ``Patient_ID``: 4,238 records share a patient
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            df = pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype={c: str for c in _STRING_COLUMNS},
                keep_default_na=False,
                na_values=[""],
            )
            for column in _STRING_COLUMNS:
                if column in df.columns:
                    df[column] = df[column].fillna("")
            return df

        from ecgbench.labels.zzu_pecg import load_labels

        labels = load_labels(data_path, config)
        # The label loader already builds signal_path with the Child_ecg/ prefix,
        # since the prefix is a fact about the release rather than about folds.
        df = labels.reset_index().rename(
            columns={
                "patient_id": config.patient_id_column,
            }
        )

        missing = self._missing_signals(data_path, df, config)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(df)} ZZU-pECG records named in "
                f"{config.metadata_csv} have no .hea on disk (e.g. {missing[:3]}). The "
                "waveforms ship as a two-part split zip (Child_ecg.zip + Child_ecg.z01) "
                "which must be joined before extracting — unzipping only the .zip gives "
                "a partial tree."
            )

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
            "ZZU-pECG: %d records, %d patients, %d nine-lead, %d with a target diagnosis",
            len(df),
            df[config.patient_id_column].nunique(),
            int((df["n_leads"] == 9).sum()),
            int((df["n_disease_groups"] > 0).sum()),
        )
        return df

    @staticmethod
    def _missing_signals(
        data_path: Path, df: pd.DataFrame, config: DatasetConfig, limit: int = 10
    ) -> list[str]:
        """Names of records whose WFDB header is absent, up to ``limit``.

        Checked once here rather than record by record during validation, so an
        incompletely extracted split zip is reported as one clear error instead
        of 14,190 ``corrupt_header`` failures.
        """
        missing: list[str] = []
        for path in df["signal_path"]:
            if not (data_path / f"{path}.hea").exists():
                missing.append(str(path))
                if len(missing) >= limit:
                    break
        return missing

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the rarest-disease-group label from the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )
        labels = df[STRATIFY_COLUMN].astype(str).rename("stratify_class")
        logger.info("Stratification class distribution:\n%s", labels.value_counts().to_string())
        return labels
