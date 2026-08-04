"""
Wilson Central Terminal ECG Database labels: demographics + one admission diagnosis.

The release ships no metadata table. Everything ECGBench can call a label lives in
comment lines appended to each record's ``.hea``:

    #Age: 46
    #Sex: M
    #Diagnosis report: Non ST segment elevation myocardial infarction (NSTEMI)
    #Reconstruct Precordials: V2, V2-raw          <- 8 records only

This module reads those headers directly, so ``load_labels`` works on a fresh copy
of the dataset with no prior ``ecgbench splits`` run; the splitter then builds its
metadata CSV from this loader rather than re-parsing the headers.

**The diagnosis is per patient, not per segment.** All 540 records were checked:
every one of the 92 patients carries a single age, sex and diagnosis string across
all of its segments, and segment counts run 1-31. So the diagnosis describes the
admission, not the beat content of a particular 10-second window — a segment
labelled "Ventricular tachycardia (VT)" need not contain VT, and the label repeats
up to 31 times in a record-level table. Weight or group by ``patient_id`` before
reporting any per-record rate.

Four things about the diagnosis strings, all verified against all 540 headers:

- **The headers are not UTF-8.** The dash in "ST segment elevation" is byte
  ``0xA0`` (a Windows-1252 non-breaking space), so a strict UTF-8 read raises and
  ``errors="replace"`` silently produces "ST�segment�elevation". They are
  decoded as cp1252 here and the NBSPs are folded to ordinary spaces.
- **43 distinct strings over 92 patients, 30 of them held by one patient** — 40
  distinct and 28 singletons once the misspellings below are corrected. The
  vocabulary is free text written per admission, not a coded taxonomy.
- **Some of it is misspelt or inconsistently cased**: "Atypica chest pain",
  "Type 2 Myocaridal infarctoin", "Congestive Cardic failure (CCF)", and
  "sinus bradycardia" alongside "Sinus bradycardia". ``diagnosis`` is the
  typo-corrected form; ``diagnosis_raw`` keeps the header string verbatim.
- **"not reported" is a real value, for 10 patients / 38 records** — matching the
  landing page's "10 patients with unreported diagnoses". It is exposed as
  ``diagnosis_reported = False`` rather than as NaN, because the distinction
  between "no diagnosis recorded" and "column missing" matters here.

:data:`DIAGNOSIS_GROUP` reduces the 40 corrected strings to 8 groups for
stratification.
That is a **single-label reduction of free text and a judgement call** — six
patients carry two conditions in one string (e.g. "Non ST segment elevation
myocardial infarction (NSTEMI)- rapid Atrial fibrillation"), and the group is
whichever the map assigns. Do not train on ``diagnosis_group``; use ``diagnosis``
and make your own grouping explicit.

Also exposed: ``reconstructed_precordials``, the list of channels that were
*synthesised* rather than recorded, from ``V = UV - WCT``. Eight records of five
patients (007, 008, 010, 014, 031) are affected. Those channels are not
measurements and must be excluded from any evaluation of precordial-lead
reconstruction, which is the headline use of this dataset — otherwise the method
under test is scored against its own output.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Newline-delimited list of ``patientNNN/segMM`` record paths in the dataset root.
RECORDS_FILE = "RECORDS"

#: Header encoding. The diagnosis strings contain byte 0xA0; utf-8 raises on them.
HEADER_ENCODING = "cp1252"

#: Comment key -> output column. Present in every one of the 540 headers.
REQUIRED_COMMENTS = {
    "Age": "age",
    "Sex": "sex",
    "Diagnosis report": "diagnosis_raw",
}

#: Comment key for the synthesised-channel note, present in 8 headers only.
RECONSTRUCT_COMMENT = "Reconstruct Precordials"

#: The string the release uses for a patient with no recorded diagnosis.
NOT_REPORTED = "not reported"

#: Header spelling -> corrected spelling. Applied after NBSP folding and
#: whitespace collapsing, so keys are written with ordinary single spaces.
#: Complete for v1.0.1; anything not listed passes through unchanged.
SPELLING_FIXES = {
    "Atypica chest pain": "Atypical chest pain",
    "Type 2 Myocaridal infarctoin": "Type 2 myocardial infarction",
    "Congestive Cardic failure (CCF)": "Congestive cardiac failure (CCF)",
    "sinus bradycardia": "Sinus bradycardia",
    "Pulmonary embolism.": "Pulmonary embolism",
    "Stable angina underwent PCI (Percutaneous Coronary Intervention )": (
        "Stable angina underwent PCI (percutaneous coronary intervention)"
    ),
    "Syncope unknown cause": "Syncope - undetermined cause",
    "Syncope - Undetermined cause": "Syncope - undetermined cause",
}

#: Corrected diagnosis string -> coarse group, for stratification only.
#:
#: Written out one string at a time rather than matched by regex so that every
#: assignment is auditable and a string the release did not contain raises instead
#: of being quietly bucketed. Patient counts as of v1.0.1 are in the comments.
#:
#: Six strings name two conditions; the group below is the one this map picked,
#: and the choice is arbitrary in the sense that a reader might pick the other.
DIAGNOSIS_GROUP = {
    # --- Myocardial infarction (28 patients) ---
    "Non ST segment elevation myocardial infarction (NSTEMI)": "Myocardial infarction",  # 21
    "ST segment elevation myocardial infarction (STEMI)": "Myocardial infarction",  # 4
    "Inferior STEMI (ST segment elevation myocardial infarction)": "Myocardial infarction",
    "Type 2 myocardial infarction": "Myocardial infarction",
    # MI wins over the AF also named in this string.
    "Non ST segment elevation myocardial infarction (NSTEMI)- rapid Atrial fibrillation": (
        "Myocardial infarction"
    ),
    # --- Angina or coronary artery disease (12 patients) ---
    "Stable angina": "Angina or coronary artery disease",  # 6
    "Coronary artery disease": "Angina or coronary artery disease",  # 3
    "Angina": "Angina or coronary artery disease",
    "Unstable angina": "Angina or coronary artery disease",
    "Stable angina underwent PCI (percutaneous coronary intervention)": (
        "Angina or coronary artery disease"
    ),
    # --- Atrial fibrillation or flutter (14 patients) ---
    "Atrial fibrillation": "Atrial fibrillation or flutter",  # 4
    "Atrial fibrillation with rapid ventricular response": "Atrial fibrillation or flutter",  # 2
    "Atrial flutter": "Atrial fibrillation or flutter",
    "Slow AF": "Atrial fibrillation or flutter",
    # AF wins over the second condition in each of these five.
    "Atrial fibrillation with tachy-brady syndrome-pericarditis": (
        "Atrial fibrillation or flutter"
    ),
    "Atrial fibrillation-cardiomyopathy": "Atrial fibrillation or flutter",
    "Atrial fibrillation-heart failure": "Atrial fibrillation or flutter",
    "Pulmonary embolism-Atrial fibrillation": "Atrial fibrillation or flutter",
    "Rapid Atrial fibrillation - pericarditis": "Atrial fibrillation or flutter",
    "Rapid atrial fibrillation with new cardiomyopathy": "Atrial fibrillation or flutter",
    # --- Other tachyarrhythmia (6 patients) ---
    "Ventricular tachycardia (VT)": "Other tachyarrhythmia",  # 3
    "Supraventricular tachycardia (SVT)": "Other tachyarrhythmia",  # 2
    "Supraventricular tachycardia (SVT): Atrioventricular nodal reentry tachycardia (AVNRT)": (
        "Other tachyarrhythmia"
    ),
    # --- Cardiomyopathy or heart failure (5 patients) ---
    "Hypertrophic obstructive cardiomyopathy": "Cardiomyopathy or heart failure",
    "Cardiomyopathy": "Cardiomyopathy or heart failure",
    "Congestive cardiac failure (CCF)": "Cardiomyopathy or heart failure",
    "Congestive cardiac failure (CHF) exacerbation": "Cardiomyopathy or heart failure",
    "Syncope-cardiomyopathy": "Cardiomyopathy or heart failure",
    # --- Bradyarrhythmia or conduction block (3 patients) ---
    "Sinus bradycardia": "Bradyarrhythmia or conduction block",  # 2
    "Complete Heart block": "Bradyarrhythmia or conduction block",
    # --- Other or non-cardiac (14 patients) ---
    "Atypical chest pain": "Other or non-cardiac",  # 5
    "Chest pain": "Other or non-cardiac",
    "Epigastric pain": "Other or non-cardiac",
    "Gastritis (non cardiac chest pain)": "Other or non-cardiac",
    "Fall secondary to alcohol intoxication": "Other or non-cardiac",
    "Urosepsis": "Other or non-cardiac",
    "Pulmonary embolism": "Other or non-cardiac",
    "Severe Mitral Stenosis": "Other or non-cardiac",
    "Syncope - undetermined cause": "Other or non-cardiac",  # 2
    # --- Not reported (10 patients) ---
    NOT_REPORTED: "Not reported",
}

#: Sexes in the release. Per patient: 65 M, 27 F — the landing page's 27 female.
SEXES = frozenset({"M", "F"})

_NBSP = "\xa0"


def normalise_diagnosis(raw: str) -> str:
    """Fold the NBSPs, collapse whitespace and correct the known misspellings."""
    text = re.sub(r"\s+", " ", raw.replace(_NBSP, " ")).strip()
    return SPELLING_FIXES.get(text, text)


def diagnosis_group(diagnosis: str) -> str:
    """Return the coarse stratification group for a corrected diagnosis string."""
    group = DIAGNOSIS_GROUP.get(diagnosis)
    if group is None:
        raise ValueError(
            f"Unmapped diagnosis {diagnosis!r}. This is the stratification label, so "
            "guessing a group would silently skew the folds. Add it to "
            "ecgbench.labels.wctecgdb.DIAGNOSIS_GROUP (and SPELLING_FIXES if it is a "
            f"variant spelling of one of: {sorted(set(DIAGNOSIS_GROUP))})."
        )
    return group


def _read_comments(hea_path: Path) -> dict[str, str]:
    """Return {comment key: value} for one header file."""
    found: dict[str, str] = {}
    with open(hea_path, encoding=HEADER_ENCODING) as f:
        for line in f:
            if not line.startswith("#"):
                continue
            key, separator, value = line[1:].partition(":")
            if separator:
                found[key.strip()] = value.strip()

    missing = [k for k in REQUIRED_COMMENTS if k not in found]
    if missing:
        raise ValueError(
            f"{hea_path.name} is missing the {missing} comment line(s). All 540 records "
            "in v1.0.1 carry #Age:, #Sex: and #Diagnosis report:."
        )
    return found


def _record_names(data_path: Path, config: DatasetConfig) -> list[str]:
    """``patientNNN/segMM`` paths, from the shipped RECORDS file if present."""
    records_file = data_path / RECORDS_FILE
    if records_file.exists():
        return [line.strip() for line in records_file.read_text().splitlines() if line.strip()]

    logger.warning("%s not found — falling back to globbing patient*/*.hea", records_file)
    names = sorted(
        f"{p.parent.name}/{p.stem}" for p in data_path.glob("patient*/*.hea")
    )
    if not names:
        raise FileNotFoundError(
            f"No patient*/*.hea header files under {data_path}. Point data_path at the "
            f"dataset root, the directory holding patient001/ and RECORDS (see {config.url})."
        )
    return names


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return demographics and the admission diagnosis of every record.

    Indexed by ``record_name`` (``patient001_seg01`` — the RECORDS path with its
    slash replaced, because ``seg01`` alone repeats in all 92 patient directories).

    Columns:

        patient_id                  ``patient001``, the directory name. Group on
                                    this: segment counts run 1-31, so a per-record
                                    average is dominated by a handful of patients.
        segment                     ``seg01``, the record stem within the patient.
        age                         integer years, 41-94 (patient-level: mean 65.23,
                                    SD 12.13)
        sex                         ``M`` / ``F`` (patient-level: 65 M, 27 F)
        diagnosis                   free-text admission diagnosis, NBSPs folded and
                                    the known misspellings corrected; 40 distinct
                                    values over 92 patients, 28 held by one patient
                                    (43 and 30 before correction)
        diagnosis_raw               the header string verbatim, byte 0xA0 and all
        diagnosis_reported          False for the 10 patients / 38 records whose
                                    diagnosis is the literal "not reported"
        diagnosis_group             8-way reduction of ``diagnosis``, for
                                    stratification only — see the module docstring
        reconstructed_precordials   list of channel names that were synthesised via
                                    ``V = UV - WCT`` rather than recorded; empty for
                                    532 of the 540 records
        has_reconstructed_precordials
                                    ``bool(reconstructed_precordials)`` — True for 8
                                    records of patients 007, 008, 010, 014 and 031

    The diagnosis is a **patient-level** admission label, not a description of the
    10 seconds in the segment, and it is single-label free text — six patients'
    strings name two conditions. There is no beat, rhythm or interval annotation in
    this release.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    if not data_path.is_dir():
        raise LabelSourceMissingError(
            f"Wilson Central Terminal labels come from the per-record .hea headers, but "
            f"{data_path} is not a directory. ECGBench publishes fold CSVs only — point "
            f"data_path at a full local copy (see {config.url})."
        )

    names = _record_names(data_path, config)
    rows: list[dict[str, object]] = []
    for name in names:
        hea_path = data_path / f"{name}.hea"
        if not hea_path.exists():
            raise LabelSourceMissingError(
                f"{RECORDS_FILE} lists {name}, but {hea_path} is missing. The labels for "
                f"this dataset are in the headers, so point data_path at a complete local "
                f"copy (see {config.url})."
            )

        comments = _read_comments(hea_path)
        patient_id, _, segment = name.partition("/")
        diagnosis = normalise_diagnosis(comments["Diagnosis report"])
        sex = comments["Sex"]
        if sex not in SEXES:
            raise ValueError(
                f"{name}: unexpected #Sex: {sex!r}; v1.0.1 uses only {sorted(SEXES)}."
            )

        reconstructed = [
            channel.strip()
            for channel in comments.get(RECONSTRUCT_COMMENT, "").split(",")
            if channel.strip()
        ]

        rows.append(
            {
                config.record_id_column: name.replace("/", "_"),
                "patient_id": patient_id,
                "segment": segment,
                "age": int(comments["Age"]),
                "sex": sex,
                "diagnosis": diagnosis,
                "diagnosis_raw": comments["Diagnosis report"],
                "diagnosis_reported": diagnosis != NOT_REPORTED,
                "diagnosis_group": diagnosis_group(diagnosis),
                "reconstructed_precordials": reconstructed,
                "has_reconstructed_precordials": bool(reconstructed),
            }
        )

    out = pd.DataFrame(rows).set_index(config.record_id_column)
    patients = out.groupby("patient_id")["diagnosis"].nunique()
    inconsistent = patients[patients > 1]
    if len(inconsistent):
        # v1.0.1 has none; a future release that introduced per-segment diagnoses
        # would invalidate the patient-level grouping this module documents.
        logger.warning(
            "%d patient(s) carry more than one diagnosis string: %s. The docstring's "
            "patient-level reading no longer holds for them.",
            len(inconsistent), list(inconsistent.index),
        )

    logger.info(
        "Loaded Wilson Central Terminal labels: %d records over %d patients; "
        "%d record(s) have synthesised precordial channels; %d record(s) have no "
        "reported diagnosis",
        len(out), out["patient_id"].nunique(),
        int(out["has_reconstructed_precordials"].sum()),
        int((~out["diagnosis_reported"]).sum()),
    )
    return out
