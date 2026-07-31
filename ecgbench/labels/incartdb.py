"""
INCART labels: header comments plus a summary of the reference beat annotations.

Nothing machine-readable ships with this dataset. Each record's ``.hea`` carries
exactly three comment lines, and the reference annotations live in a companion
``.atr``:

    #<age>: 65 <sex>: F <diagnoses> Coronary artery disease, arterial hypertension
    # patient 1
    # PVCs, noise

Line 1 is **patient-level** (age, sex, confirmed diagnoses), line 2 gives the
patient number that makes patient-grouped folds possible, and line 3 is a
free-text list of the ECG findings in *this* record. All three are exposed
verbatim, plus per-record beat-type counts derived from the ``.atr``.

Quirks worth knowing, all verified against the files:

- **The ``<diagnoses>`` token is absent entirely in 34 of the 75 records** (14 of
  the 32 patients), rather than present-and-empty. A parser that requires it
  fails on nearly half the dataset.
- **The README's demographics disagree with the files.** It states "17 men and 15
  women, aged 18-80; mean age: 58"; the headers give **18 men and 14 women**,
  ages 18-80, mean **56.2**. The recomputed values are what this loader returns.
- **The README's diagnosis table mixes two sources.** WPW, atrial fibrillation,
  AV nodal block and bundle branch block appear only in the per-record findings
  text, never in ``<diagnoses>``, which is why its categories sum past 32
  patients. Use ``record_features`` as well as ``diagnosis``.
- ``files-patients-diagnoses.txt`` and ``record-descriptions.txt`` restate the
  same content. Both were checked against all 75 headers and agree on every
  record — no patient-id, diagnosis or findings disagreement — so this loader
  reads the headers only and the two text files are redundant.
- Beat annotations were placed by an algorithm and corrected manually, but their
  **positions** were not corrected, so onsets may be slightly misaligned. The
  counts here are unaffected; anything time-sensitive is not.
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

#: Annotator extension, per the shipped ANNOTATORS file.
ANNOTATOR = "atr"

#: PhysioBank beat symbols that occur in this dataset, in descending frequency.
#: '+' is a rhythm-change marker, not a beat, and is counted separately.
BEAT_SYMBOLS = ("N", "V", "R", "A", "F", "j", "n", "S", "Q", "B")

#: Human-readable names for the beat symbols above, for the example script and docs.
BEAT_NAMES = {
    "N": "normal",
    "V": "premature ventricular contraction",
    "R": "right bundle branch block beat",
    "A": "atrial premature contraction",
    "F": "fusion of ventricular and normal",
    "j": "nodal (junctional) escape",
    "n": "supraventricular escape",
    "S": "premature or ectopic supraventricular",
    "Q": "unclassifiable",
    "B": "left or right bundle branch block",
}

#: Stratification classes with fewer than this many PATIENTS are pooled into
#: OTHER. Patients, not records, because folds are grouped by patient — a class
#: with 11 records spread over 4 patients cannot be spread across 10 folds.
MIN_CLASS_PATIENTS = 10

OTHER = "OTHER"
UNKNOWN = "UNKNOWN"

#: Header comment 1. The <diagnoses> group is optional: it is absent, token and
#: all, on 34 of the 75 records.
_DEMOGRAPHICS_RE = re.compile(
    r"^#\s*<age>:\s*(?P<age>\S+)\s+<sex>:\s*(?P<sex>\S+)\s*"
    r"(?:<diagnoses>\s*(?P<diagnoses>.*))?$"
)

#: Header comment 2.
_PATIENT_RE = re.compile(r"^#\s*patient\s+(?P<patient>\d+)\s*$")


def parse_header_comments(hea_path: Path) -> dict[str, str]:
    """Parse the three comment lines of one INCART header.

    Returns ``age``, ``sex``, ``diagnosis``, ``patient_id`` and
    ``record_features``. Missing pieces come back as empty strings rather than
    raising, so one malformed header cannot fail the whole scan — the validation
    engine flags genuinely broken records via ``corrupt_header``.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    comments = [line for line in lines if line.startswith("#")]

    out = {
        "age": "",
        "sex": "",
        "diagnosis": "",
        "patient_id": "",
        "record_features": "",
    }

    if comments:
        match = _DEMOGRAPHICS_RE.match(comments[0])
        if match:
            out["age"] = match.group("age").strip()
            out["sex"] = match.group("sex").strip()
            # Absent group -> None; normalise to "" so the column is always a str.
            out["diagnosis"] = (match.group("diagnoses") or "").strip()
        else:
            logger.warning("Unparsed demographics comment in %s: %r", hea_path.name, comments[0])

    if len(comments) > 1:
        match = _PATIENT_RE.match(comments[1])
        if match:
            out["patient_id"] = f"patient{int(match.group('patient')):02d}"
        else:
            logger.warning("Unparsed patient comment in %s: %r", hea_path.name, comments[1])

    if len(comments) > 2:
        # Free text, and several records have a trailing comma or double space.
        out["record_features"] = comments[2].lstrip("#").strip().rstrip(",").strip()

    return out


def count_beats(record_path: Path) -> dict[str, int]:
    """Summarise one record's reference annotations into per-symbol counts.

    Returns one ``beat_<symbol>`` column per symbol in :data:`BEAT_SYMBOLS`, plus
    ``n_beats`` (all beat annotations) and ``n_rhythm_changes`` (the ``+``
    markers, which are not beats and must not inflate the beat total).
    """
    import wfdb

    counts = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
    counts["n_beats"] = 0
    counts["n_rhythm_changes"] = 0

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .atr must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    beat_set = set(BEAT_SYMBOLS)
    unexpected: set[str] = set()
    for symbol in annotation.symbol:
        if symbol in beat_set:
            counts[f"beat_{symbol}"] += 1
            counts["n_beats"] += 1
        elif symbol == "+":
            counts["n_rhythm_changes"] += 1
        else:
            unexpected.add(symbol)
    if unexpected:
        # Worth seeing rather than silently dropping: it would mean this release
        # uses symbols the module does not know about.
        logger.warning(
            "%s: annotation symbols outside BEAT_SYMBOLS, not counted: %s",
            record_path.name,
            sorted(unexpected),
        )
    return counts


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    Adds ``signal_path`` (the bare record stem — the tree is flat) and
    ``pvc_fraction``, the share of beats annotated as premature ventricular
    contractions.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    headers = sorted(data_path.glob("I[0-9][0-9].hea"))
    if not headers:
        raise LabelSourceMissingError(
            f"No I??.hea headers under {data_path}. INCART labels live in the record "
            "headers and .atr annotation files, so point data_path at the dataset "
            "root — the flat directory holding I01.hea and RECORDS."
        )

    rows = []
    for hea in headers:
        row = {"record_name": hea.stem}
        row.update(parse_header_comments(hea))
        row.update(count_beats(hea.with_suffix("")))
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = hea.stem
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    df["pvc_fraction"] = (df["beat_V"] / df["n_beats"]).where(df["n_beats"] > 0)
    logger.info(
        "Parsed %d INCART records from %d patients; %d beat annotations",
        len(df),
        df["patient_id"].nunique(),
        int(df["n_beats"].sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: the patient-level diagnosis, pooled for folds.

    This is the **only** derivation of the stratification label —
    ``INCARTDBSplitter`` reads the column rather than recomputing it, so the
    exposed label and the fold label cannot drift.

    Pooling counts **patients**, not records, because folds are grouped by
    patient. With 32 patients over 10 folds this label is necessarily coarse; it
    exists so the folds are defined, and is not a clinical grouping. Train on
    ``diagnosis`` and ``record_features``.
    """
    out = df.copy()
    labels = out["diagnosis"].fillna("").replace("", UNKNOWN)

    patients_per_class = (
        pd.DataFrame({"label": labels, "patient": out["patient_id"]})
        .groupby("label")["patient"]
        .nunique()
    )
    rare = set(patients_per_class[patients_per_class < MIN_CLASS_PATIENTS].index)
    if rare:
        logger.info(
            "Pooling %d diagnosis class(es) with <%d patients into '%s': %s",
            len(rare),
            MIN_CLASS_PATIENTS,
            OTHER,
            sorted(rare),
        )
        labels = labels.where(~labels.isin(rare), OTHER)

    out["stratify_class"] = labels
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return INCART labels indexed by record name.

    Columns:

    - ``diagnosis`` — the patient-level confirmed diagnoses, verbatim from
      ``<diagnoses>``. **Empty for 34 of the 75 records** (14 of 32 patients);
      that is the source data, not a parse failure.
    - ``record_features`` — free-text ECG findings for this specific record. The
      only place WPW, atrial fibrillation, AV nodal block and bundle branch block
      are recorded, so do not treat ``diagnosis`` as the full label.
    - ``patient_id`` — ``patientNN`` from the header. 32 patients over 75 records;
      this is what folds are grouped by.
    - ``age``, ``sex`` — patient-level. Constant across a patient's records.
    - ``beat_N`` … ``beat_B`` — reference beat counts per symbol (see
      :data:`BEAT_NAMES`), with ``n_beats`` and ``pvc_fraction``.
      ``n_rhythm_changes`` counts the ``+`` markers, which are not beats.
    - ``stratify_class`` — pooled diagnosis, **for fold construction only**.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
