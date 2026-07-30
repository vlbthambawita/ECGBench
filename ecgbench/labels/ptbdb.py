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
