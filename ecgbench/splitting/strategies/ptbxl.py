"""
PTB-XL specific splitting strategy.

Parses SCP codes from dict-string format and maps them to diagnostic
superclasses (NORM, MI, STTC, HYP, CD) for stratification.
Uses the official strat_fold column for 10-fold assignment.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

# Complete SCP code -> diagnostic superclass mapping
SCP_TO_SUPERCLASS: dict[str, str] = {
    "NORM": "NORM",
    # Myocardial Infarction
    "IMI": "MI", "AMI": "MI", "ALMI": "MI", "ASMI": "MI", "ILMI": "MI",
    "INJAS": "MI", "INJAL": "MI", "INJIN": "MI", "INJLA": "MI", "INJIL": "MI",
    "PMI": "MI", "LMI": "MI",
    # ST/T Changes
    "STD_": "STTC", "STE_": "STTC", "NST_": "STTC", "ISC_": "STTC",
    "ISCA": "STTC", "ISCI": "STTC", "ISCAL": "STTC", "ISCAS": "STTC",
    "ISCIN": "STTC", "ISCIL": "STTC", "ISCLA": "STTC", "DIG": "STTC",
    "LNGQT": "STTC", "APTS": "STTC", "NDT": "STTC", "NT_": "STTC", "TAB_": "STTC",
    # Hypertrophy
    "LVH": "HYP", "RVH": "HYP", "LAO/LAE": "HYP", "RAO/RAE": "HYP", "SEHYP": "HYP",
    # Conduction Disturbance
    "LAFB": "CD", "LPFB": "CD", "IRBBB": "CD", "CRBBB": "CD",
    "CLBBB": "CD", "1AVB": "CD", "2AVB": "CD", "3AVB": "CD",
    "WPW": "CD", "IVCD": "CD", "ILBBB": "CD",
}


def _parse_scp_codes(scp_string: str) -> dict[str, float]:
    """Parse SCP codes from a Python dict-string representation."""
    try:
        return ast.literal_eval(scp_string)
    except (ValueError, SyntaxError):
        return {}


def _get_superclass(scp_codes: dict[str, float]) -> str:
    """Determine the dominant diagnostic superclass from SCP codes.

    Returns the superclass with the highest confidence score.
    If no diagnostic SCP codes are found, returns "OTHER".
    """
    superclass_scores: dict[str, float] = {}
    for code, score in scp_codes.items():
        superclass = SCP_TO_SUPERCLASS.get(code)
        if superclass:
            superclass_scores[superclass] = (
                superclass_scores.get(superclass, 0.0) + score
            )
    if not superclass_scores:
        return "OTHER"
    return max(superclass_scores, key=superclass_scores.get)


@register("ptbxl")
class PTBXLSplitter(DatasetSplitter):
    """PTB-XL specific splitting logic.

    - Parses scp_codes from dict-string format
    - Maps SCP codes to diagnostic superclass (NORM, MI, STTC, HYP, CD)
    - Uses strat_fold column for official 10-fold assignment
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv
        df = pd.read_csv(csv_path, sep=config.metadata_csv_separator)

        # Rename signal path columns to use sampling rate as key
        # PTB-XL uses filename_lr and filename_hr
        if "filename_lr" in df.columns and 100 in config.signal_path_columns:
            expected_col = config.signal_path_columns[100]
            if expected_col not in df.columns:
                df = df.rename(columns={"filename_lr": expected_col})
        if "filename_hr" in df.columns and 500 in config.signal_path_columns:
            expected_col = config.signal_path_columns[500]
            if expected_col not in df.columns:
                df = df.rename(columns={"filename_hr": expected_col})

        logger.info("Loaded PTB-XL metadata: %d records", len(df))
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Map SCP codes to diagnostic superclasses for stratification."""
        superclasses = df[config.label_column].apply(
            lambda x: _get_superclass(_parse_scp_codes(str(x)))
        )
        superclasses.name = "diagnostic_superclass"

        # Log distribution
        dist = superclasses.value_counts()
        logger.info("Superclass distribution:\n%s", dist.to_string())

        return superclasses
