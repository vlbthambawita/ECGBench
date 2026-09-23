"""
PTB Diagnostic ECG Database labels: 47 clinical fields, all inside the headers.

The release ships no metadata file of any kind. Every record's `.hea` carries a
48-line comment block of ``# key: value`` pairs (CRLF-terminated) covering
demographics, the diagnosis, infarction details, a full haemodynamics panel and
the therapy history. Absence is encoded as the literal string ``n/a``, not as a
missing key — all 47 keys are present on all 549 records.

Every key is exposed under its original name, because dropping fields users
cannot recover is worse than a wide frame. Four normalised columns are added on
top: ``age``, ``sex``, ``diagnosis`` and ``primary_diagnosis``.

Quirks worth knowing, all verified against the files:

- ``Catheterization date`` appears **twice** per record, once under Hemodynamics
  and once under Therapy. A naive dict parse silently keeps one; this keeps the
  first and exposes the second as ``Catheterization date (2)``.
- ``laod`` is a real typo in the source data, not a transcription error here.
- Numeric values use European decimal commas (``4,34 l/min``), so the clinical
  panels stay as strings.
- 27 records carry ``n/a`` as their diagnosis, and one record (patient285/
  s0544_re) has an empty ``sex``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ecgbench.labels._fields import Field

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: The comment key holding the diagnosis.
DIAGNOSIS_KEY = "Reason for admission"

#: Diagnoses with fewer than this many records are pooled into OTHER, so that
#: 10-fold stratification is defined. Several classes have a single record.
MIN_CLASS_SIZE = 10
OTHER = "OTHER"
UNKNOWN = "UNKNOWN"

#: Absence marker used throughout the headers.
_NA = "n/a"


def parse_header_comments(hea_path: Path) -> dict[str, str]:
    """Parse the ``# key: value`` comment block of one PTBDB header.

    Headers are CRLF-terminated and use ``n/a`` for absent values, which is
    normalised to an empty string here. Repeated keys get a ``(2)`` suffix rather
    than overwriting.
    """
    text = hea_path.read_text(encoding="utf-8", errors="replace")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        key, sep, value = line[1:].partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.lower() == _NA:
            value = ""
        if key in fields:
            # 'Catheterization date' is listed under both Hemodynamics and Therapy.
            suffix = 2
            while f"{key} ({suffix})" in fields:
                suffix += 1
            key = f"{key} ({suffix})"
        fields[key] = value
    return fields


def scan_headers(data_path: Path) -> pd.DataFrame:
    """Parse every record header under ``patientNNN/`` into one frame.

    Adds ``record_name``, ``patient_id`` (the directory, since 113 of 290
    patients have more than one recording) and ``signal_path``.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    headers = sorted(data_path.glob("patient*/*.hea"))
    if not headers:
        raise LabelSourceMissingError(
            f"No patient*/*.hea headers under {data_path}. PTBDB labels live in the "
            "record headers — point data_path at the dataset root, the directory "
            "holding patient001/ and RECORDS."
        )

    rows = []
    for hea in headers:
        fields = parse_header_comments(hea)
        fields["record_name"] = hea.stem
        fields["patient_id"] = hea.parent.name
        fields["signal_path"] = f"{hea.parent.name}/{hea.stem}"
        rows.append(fields)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d PTBDB headers from %d patients",
        len(df), df["patient_id"].nunique(),
    )
    return df


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return PTBDB labels indexed by record name.

    Columns: all 47 header keys under their original names, plus

        age                 numeric where parseable, NaN otherwise
        sex                 'male' / 'female' / '' (one record has none)
        diagnosis           'Reason for admission' verbatim; '' for the 27 n/a
        primary_diagnosis   the same, with '' -> UNKNOWN and classes below
                            MIN_CLASS_SIZE pooled into OTHER — stratification
                            only, do not train on it
        patient_id          the patientNNN directory

    Single-label: each record has exactly one 'Reason for admission', and no
    patient has conflicting diagnoses across their recordings.
    """
    df = scan_headers(Path(data_path))
    record_col = config.record_id_column

    out = df.set_index(record_col)
    out.index.name = record_col

    out["age"] = pd.to_numeric(out.get("age"), errors="coerce")
    out["sex"] = out.get("sex", "").fillna("")
    out["diagnosis"] = out.get(DIAGNOSIS_KEY, "").fillna("")
    out["primary_diagnosis"] = _pool_rare(out["diagnosis"].tolist())

    logger.info(
        "Loaded PTBDB labels: %d records, %d without a diagnosis",
        len(out), int((out["diagnosis"] == "").sum()),
    )
    return out


def _pool_rare(diagnoses: list[str]) -> list[str]:
    """Label the undiagnosed UNKNOWN and pool classes too small to stratify."""
    named = [d if d else UNKNOWN for d in diagnoses]
    counts = pd.Series(named).value_counts()
    rare = set(counts[counts < MIN_CLASS_SIZE].index)
    if rare:
        logger.info(
            "Pooling %d diagnosis class(es) with <%d records into '%s': %s",
            len(rare), MIN_CLASS_SIZE, OTHER, sorted(rare),
        )
    return [OTHER if d in rare else d for d in named]


# --------------------------------------------------------------------------- fields

#: The columns ``load_labels`` returns, as data — see ``ecgbench.labels._fields``.
#: Literal on purpose: the metadata build reads it without importing this module.
FIELDS = (
    Field(
        "age",
        "integer",
        "Header 'age' as a number; missing where it does not parse",
        unit="year",
    ),
    Field(
        "sex",
        "string",
        "Header 'sex' as shipped, lower-case; empty for patient285/s0544_re",
        vocabulary=("male", "female", ""),
        nullable=False,
    ),
    Field(
        "ECG date",
        "string",
        "Header 'ECG date' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Diagnose",
        "string",
        "Header section marker 'Diagnose:', which carries no value",
        nullable=False,
    ),
    Field(
        "Reason for admission",
        "string",
        "Header 'Reason for admission', the diagnosis; empty ('n/a') for 27 records. Also "
        "returned as diagnosis.",
        nullable=False,
    ),
    Field(
        "Acute infarction (localization)",
        "string",
        "Header 'Acute infarction (localization)' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Former infarction (localization)",
        "string",
        "Header 'Former infarction (localization)' verbatim; empty where the source says "
        "n/a",
        nullable=False,
    ),
    Field(
        "Additional diagnoses",
        "string",
        "Header 'Additional diagnoses' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Smoker",
        "string",
        "Header 'Smoker' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Number of coronary vessels involved",
        "string",
        "Header 'Number of coronary vessels involved' verbatim; empty where the source says "
        "n/a",
        nullable=False,
    ),
    Field(
        "Infarction date (acute)",
        "string",
        "Header 'Infarction date (acute)' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Previous infarction (1) date",
        "string",
        "Header 'Previous infarction (1) date' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Previous infarction (2) date",
        "string",
        "Header 'Previous infarction (2) date' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Hemodynamics",
        "string",
        "Header section marker 'Hemodynamics:', which carries no value",
        nullable=False,
    ),
    Field(
        "Catheterization date",
        "string",
        "Header 'Catheterization date' under Hemodynamics (the first of its two "
        "occurrences)",
        nullable=False,
    ),
    Field(
        "Ventriculography",
        "string",
        "Header 'Ventriculography' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Chest X-ray",
        "string",
        "Header 'Chest X-ray' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Peripheral blood Pressure (syst/diast)",
        "string",
        "Header 'Peripheral blood Pressure (syst/diast)' verbatim; European decimal commas "
        "(e.g. '4,34 l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Pulmonary artery pressure (at rest) (syst/diast)",
        "string",
        "Header 'Pulmonary artery pressure (at rest) (syst/diast)' verbatim; European "
        "decimal commas (e.g. '4,34 l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Pulmonary artery pressure (at rest) (mean)",
        "string",
        "Header 'Pulmonary artery pressure (at rest) (mean)' verbatim; European decimal "
        "commas (e.g. '4,34 l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Pulmonary capillary wedge pressure (at rest)",
        "string",
        "Header 'Pulmonary capillary wedge pressure (at rest)' verbatim; European decimal "
        "commas (e.g. '4,34 l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Cardiac output (at rest)",
        "string",
        "Header 'Cardiac output (at rest)' verbatim; European decimal commas (e.g. '4,34 "
        "l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Cardiac index (at rest)",
        "string",
        "Header 'Cardiac index (at rest)' verbatim; European decimal commas (e.g. '4,34 "
        "l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Stroke volume index (at rest)",
        "string",
        "Header 'Stroke volume index (at rest)' verbatim; European decimal commas (e.g. "
        "'4,34 l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Pulmonary artery pressure (laod) (syst/diast)",
        "string",
        "Header 'Pulmonary artery pressure (laod) (syst/diast)' verbatim ('laod' is the "
        "source's own typo); European decimal commas (e.g. '4,34 l/min'), kept as text; "
        "empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Pulmonary artery pressure (laod) (mean)",
        "string",
        "Header 'Pulmonary artery pressure (laod) (mean)' verbatim ('laod' is the source's "
        "own typo); European decimal commas (e.g. '4,34 l/min'), kept as text; empty where "
        "the source says n/a",
        nullable=False,
    ),
    Field(
        "Pulmonary capillary wedge pressure (load)",
        "string",
        "Header 'Pulmonary capillary wedge pressure (load)' verbatim; European decimal "
        "commas (e.g. '4,34 l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Cardiac output (load)",
        "string",
        "Header 'Cardiac output (load)' verbatim; European decimal commas (e.g. '4,34 "
        "l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Cardiac index (load)",
        "string",
        "Header 'Cardiac index (load)' verbatim; European decimal commas (e.g. '4,34 "
        "l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Stroke volume index (load)",
        "string",
        "Header 'Stroke volume index (load)' verbatim; European decimal commas (e.g. '4,34 "
        "l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Aorta (at rest) (syst/diast)",
        "string",
        "Header 'Aorta (at rest) (syst/diast)' verbatim; European decimal commas (e.g. "
        "'4,34 l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Aorta (at rest) mean",
        "string",
        "Header 'Aorta (at rest) mean' verbatim; European decimal commas (e.g. '4,34 "
        "l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Left ventricular enddiastolic pressure",
        "string",
        "Header 'Left ventricular enddiastolic pressure' verbatim; European decimal commas "
        "(e.g. '4,34 l/min'), kept as text; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Left coronary artery stenoses (RIVA)",
        "string",
        "Header 'Left coronary artery stenoses (RIVA)' verbatim; empty where the source "
        "says n/a",
        nullable=False,
    ),
    Field(
        "Left coronary artery stenoses (RCX)",
        "string",
        "Header 'Left coronary artery stenoses (RCX)' verbatim; empty where the source says "
        "n/a",
        nullable=False,
    ),
    Field(
        "Right coronary artery stenoses (RCA)",
        "string",
        "Header 'Right coronary artery stenoses (RCA)' verbatim; empty where the source "
        "says n/a",
        nullable=False,
    ),
    Field(
        "Echocardiography",
        "string",
        "Header 'Echocardiography' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Therapy",
        "string",
        "Header section marker 'Therapy:', which carries no value",
        nullable=False,
    ),
    Field(
        "Infarction date",
        "string",
        "Header 'Infarction date' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Catheterization date (2)",
        "string",
        "Header 'Catheterization date' under Therapy, the repeated key that a dict parse "
        "would silently overwrite",
        nullable=False,
    ),
    Field(
        "Admission date",
        "string",
        "Header 'Admission date' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Medication pre admission",
        "string",
        "Header 'Medication pre admission' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Start lysis therapy (hh.mm)",
        "string",
        "Header 'Start lysis therapy (hh.mm)' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Lytic agent",
        "string",
        "Header 'Lytic agent' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Dosage (lytic agent)",
        "string",
        "Header 'Dosage (lytic agent)' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Additional medication",
        "string",
        "Header 'Additional medication' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "In hospital medication",
        "string",
        "Header 'In hospital medication' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "Medication after discharge",
        "string",
        "Header 'Medication after discharge' verbatim; empty where the source says n/a",
        nullable=False,
    ),
    Field(
        "patient_id",
        "string",
        "The patientNNN directory; 113 of 290 patients have more than one recording",
        nullable=False,
        example="patient001",
    ),
    Field(
        "signal_path",
        "string",
        "patientNNN/<record> WFDB stem",
        nullable=False,
    ),
    Field(
        "diagnosis",
        "string",
        "'Reason for admission' verbatim; empty for the 27 n/a",
        nullable=False,
    ),
    Field(
        "primary_diagnosis",
        "string",
        "diagnosis with empty as UNKNOWN and classes under 10 records pooled into OTHER; "
        "stratification only, do not train on it",
        nullable=False,
    ),
)
