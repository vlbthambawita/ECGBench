"""
Ningbo IVA labels: the ablation-confirmed origin of each patient's arrhythmia.

This is not a diagnosis dataset. Every one of the 334 patients had a *successful*
catheter ablation, and the label is **where the arrhythmia actually came from** as
proven by that ablation — not a reading of the ECG. That is the whole point of the
release: the ground truth is invasive, so a model can be trained to predict the
origin from the 12-lead surface ECG before the procedure.

Three nested labels, coarsest first:

    left_right    RVOT (257) or LVOT (77) — the outflow tract, the canonical task
    sublocation   12 anatomic sites within the tract (LC, LCC, AMC, ...)
    arrhythmia_type  PVC (329) or VT (5) — what the patient presented with

``left_right`` is the label to use. ``Sublocation`` has 12 values over 334
patients, five of them with under ten cases, so it cannot support stratified
ten-fold splits and is exposed for analysis rather than as a split target.

Quirks worth knowing, all verified against the files:

- **One record per patient, and no age.** ``HospitalID`` is both the record and
  the patient identifier — 334 distinct values, 334 CSV files, an exact
  one-to-one match in both directions. The paper reports a mean age of
  46.1 +/- 13.1 years; ``Diagnosis.xlsx`` ships **no age column at all**, so age
  is not recoverable from the release. Sex is (``female`` 230, ``male`` 104,
  matching the paper's 230/104 exactly).
- **40 Sublocation cells are blank, and the paper's Table 2 explains all 40.**
  The table assigns 45 RVOT patients to "RVOTOther" and 1 LVOT patient to "NA";
  the shipped file has 6 explicit ``RVOTOther`` plus 39 blanks among the RVOT
  patients, and the single LVOT blank. 39 + 6 = 45 and the arithmetic closes on
  both tracts. The blanks are left blank here rather than filled with
  ``RVOTOther``, because the inference — however well the totals agree — is ours
  and not the providers'.
- **The Type column disagrees with the paper and the paper is not reproducible
  from the files.** Table 1 reports 325 frequent PVC and 9 sustained VT (RVOT
  251/6, LVOT 74/3); ``Diagnosis.xlsx`` reads 329 PVC and 5 VT (RVOT 254/3, LVOT
  75/2). Four patients are VT in the paper and PVC in the shipped file. There is
  no changelog and figshare shows one revision (v2), so which is right cannot be
  established here — the file is what this loader reports. Do not quote the
  paper's 9.
- **Sex is lower-case and spelled out** (``female``/``male``), unlike every other
  ECGBench dataset. ``sex`` keeps the shipped spelling; ``sex_code`` gives the
  ``F``/``M`` form the rest of the catalogue uses, so a cross-dataset join does
  not need a per-dataset special case.

The signal-path columns this loader attaches, and why there are two:

    signal_path            PVCVTRawECGData/<HospitalID>.csv  — the canonical one
    signal_path_denoised   PVCVTECGData/<HospitalID>.csv     — reference only

Only ``signal_path`` is wired into the config, validated and exported. The
denoised copy is a derived artefact and is **not** interchangeable with the raw
one: the release's wavelet denoiser ran on each lead independently, so
Einthoven's and Goldberger's relations no longer hold in it (III - (II - I) is
under 1% of III's RMS in the raw files and up to 14% in the denoised ones), and
106 of the 334 denoised files are *shorter* than their raw counterparts, by up to
7x. The two directories are therefore not sample-aligned and a window computed on
one does not transfer to the other.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DENOISED_DIR",
    "DIAGNOSIS_FILES",
    "SAMPLING_RATE",
    "SIGNAL_DIR",
    "load_labels",
    "read_diagnosis",
]

#: Directory holding the raw 12-lead CSVs — the canonical signals.
SIGNAL_DIR = "PVCVTRawECGData"

#: Wavelet-denoised copies of the same recordings, same filenames. Exposed as a
#: path only; see the module docstring for why it is not interchangeable.
DENOISED_DIR = "PVCVTECGData"

#: Source diagnosis table, in preference order. The release ships .xlsx only;
#: the .csv is accepted so a converted copy works without openpyxl.
DIAGNOSIS_FILES = ("Diagnosis.csv", "Diagnosis.xlsx")

#: Constant across the release (EP-WorkMate system, per the paper). Confirmed
#: from the data: R-peak intervals give a median 81.5 bpm at 2000 Hz, which would
#: be an impossible 20 bpm at 500 Hz.
SAMPLING_RATE = 2000

#: RVOT/LVOT, the names the paper uses, from the shipped Right/Left.
_TRACT = {"Right": "RVOT", "Left": "LVOT"}

#: female/male as shipped -> the F/M spelling the rest of the catalogue uses.
_SEX_CODE = {"female": "F", "male": "M"}


def read_diagnosis(data_path: Path | str) -> pd.DataFrame:
    """Read Diagnosis.csv if present, else Diagnosis.xlsx."""
    from ecgbench.labels import LabelSourceMissingError

    root = Path(data_path)
    for name in DIAGNOSIS_FILES:
        path = root / name
        if not path.exists():
            continue
        if path.suffix == ".csv":
            return pd.read_csv(path)
        try:
            return pd.read_excel(path)
        except ImportError as e:
            raise ImportError(
                f"Reading {name} needs openpyxl (pip install openpyxl), or convert it "
                "to Diagnosis.csv first."
            ) from e

    raise LabelSourceMissingError(
        f"Ningbo IVA labels come from one of {DIAGNOSIS_FILES}, and neither is in "
        f"{root}. ECGBench publishes fold CSVs only — labels stay with the source "
        "dataset, so point data_path at a full local copy "
        "(https://doi.org/10.6084/m9.figshare.c.4668086.v2)."
    )


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Ningbo IVA labels indexed by ``hospital_id``.

    Columns:
        ``left_right`` (RVOT/LVOT), ``sublocation`` (12 sites, 40 blank),
        ``arrhythmia_type`` (PVC/VT), ``sex``, ``sex_code``, ``sampling_rate``,
        ``signal_path``, ``signal_path_denoised``.

    Single-label throughout — one ablation-confirmed origin per patient, and one
    record per patient. ``left_right`` is the split and training target; see the
    module docstring for why ``sublocation`` is not.
    """
    root = Path(data_path)
    raw = read_diagnosis(root)

    expected = {"HospitalID", "Type", "LeftRight", "Sublocation", "Gender"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(
            f"Ningbo IVA diagnosis table is missing column(s) {sorted(missing)}. "
            f"Found: {list(raw.columns)}"
        )

    record_ids = raw["HospitalID"].astype(str).to_numpy()
    # .to_numpy() throughout: passing index= alongside Series values *reindexes*
    # them against the new labels rather than relabelling, which silently yields
    # a frame of NaN.
    df = pd.DataFrame(
        {
            "left_right": raw["LeftRight"].map(_TRACT).fillna(raw["LeftRight"]).to_numpy(),
            # Kept exactly as shipped, blanks included — see the module docstring.
            "sublocation": raw["Sublocation"].to_numpy(),
            "arrhythmia_type": raw["Type"].to_numpy(),
            "sex": raw["Gender"].to_numpy(),
            "sex_code": raw["Gender"].map(_SEX_CODE).fillna(raw["Gender"]).to_numpy(),
        },
        index=pd.Index(record_ids, name="hospital_id"),
    )
    df["sampling_rate"] = SAMPLING_RATE
    # Relative to the dataset root, so both resolve identically for the splitter,
    # the validation engine and ECGDataset.
    df["signal_path"] = [f"{SIGNAL_DIR}/{rid}.csv" for rid in df.index]
    df["signal_path_denoised"] = [f"{DENOISED_DIR}/{rid}.csv" for rid in df.index]

    df = df.sort_index()
    logger.info(
        "Loaded Ningbo IVA labels: %d patients, %s",
        len(df),
        df["left_right"].value_counts().to_dict(),
    )
    return df
