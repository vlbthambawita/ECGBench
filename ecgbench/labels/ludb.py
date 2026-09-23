"""
LUDB labels: nine categorised diagnosis columns, most of them multi-label.

``ludb.csv`` is the structured source. The per-record ``.hea`` comments carry the
same content flattened into free text, losing the category grouping, so this
module reads the CSV.

Two things make a plain declarative block insufficient:

- **Every string cell has a trailing newline**, and multi-value cells join their
  values with newlines rather than a delimiter. Read raw, ``Ischemia`` looks like
  40 classes with at most 4 records each; split properly it is a handful of
  findings that co-occur.
- **``Age`` is not numeric for every record** — ID 34 is recorded as ``>89``.

So the loader strips, splits the multi-label columns into lists, and exposes age
both as the original string and as a nullable number.
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

SOURCE_CSV = "ludb.csv"

#: CSV column -> exposed name. The nine diagnosis columns are all multi-label
#: except the electric axis, which records a single orientation.
_MULTI_LABEL = {
    "Rhythms": "rhythms",
    "Conduction abnormalities": "conduction_abnormalities",
    "Extrasystolies": "extrasystolies",
    "Hypertrophies": "hypertrophies",
    "Cardiac pacing": "cardiac_pacing",
    "Ischemia": "ischemia",
    "Non-specific repolarization abnormalities": "repolarization_abnormalities",
    "Other states": "other_states",
}
_SINGLE = {"Electric axis of the heart": "electric_axis"}

#: Rhythm classes with fewer than this many records are pooled into OTHER for
#: stratification. 200 records over 11 rhythms leaves several singletons.
MIN_CLASS_SIZE = 10
OTHER = "OTHER"


def _clean(value: object) -> str:
    """Strip the trailing newlines and padding every cell in this CSV carries."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _split(value: object) -> list[str]:
    """Split a newline-joined multi-value cell into atomic labels."""
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split("\n") if part.strip()]


def _strip_prefix(value: str, prefix: str) -> str:
    """Drop the redundant 'Electric axis of the heart: ' prefix from its own column."""
    cleaned = _clean(value)
    lowered = cleaned.lower()
    if lowered.startswith(prefix.lower()):
        cleaned = cleaned[len(prefix):].strip()
    return cleaned.rstrip(".").strip()


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return LUDB labels indexed by record ID.

    Columns:
        sex                 'M' or 'F'
        age_raw             as recorded; ID 34 is ">89", not a number
        age                 numeric where parseable, NaN otherwise
        rhythms             list of rhythm findings (multi-label)
        primary_rhythm      first rhythm, rare classes pooled into "OTHER" —
                            stratification only, do not train on it
        electric_axis       single value, prefix stripped
        conduction_abnormalities, extrasystolies, hypertrophies, cardiac_pacing,
        ischemia, repolarization_abnormalities, other_states
                            lists, empty where the record has no such finding

    Multi-label by design: only ``Rhythms`` is populated for all 200 records.
    ``Other states`` covers 9, ``Cardiac pacing`` 10, ``Extrasystolies`` 14 — a
    model trained on the sparse columns has almost no signal to learn from.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    csv_path = data_path / SOURCE_CSV
    if not csv_path.exists():
        raise LabelSourceMissingError(
            f"LUDB labels come from {SOURCE_CSV}, which is not in {data_path}. "
            "ECGBench publishes fold CSVs only — point data_path at a full local "
            f"copy (see {config.url})."
        )

    raw = pd.read_csv(csv_path)
    record_col = config.record_id_column
    out = pd.DataFrame(index=pd.Index(raw[record_col], name=record_col))

    out["sex"] = raw["Sex"].map(_clean).to_numpy()
    out["age_raw"] = raw["Age"].map(_clean).to_numpy()
    out["age"] = pd.to_numeric(out["age_raw"], errors="coerce")

    for source, name in _MULTI_LABEL.items():
        out[name] = [_split(v) for v in raw[source]]
    for source, name in _SINGLE.items():
        out[name] = [
            _strip_prefix(v, "Electric axis of the heart:") for v in raw[source]
        ]

    out["primary_rhythm"] = _pool_rare(
        [row[0] if row else OTHER for row in out["rhythms"]]
    )

    unparseable = int(out["age"].isna().sum())
    if unparseable:
        logger.info(
            "LUDB: %d record(s) have a non-numeric age (kept in age_raw)", unparseable
        )
    logger.info("Loaded LUDB labels: %d records", len(out))
    return out


def _pool_rare(rhythms: list[str]) -> list[str]:
    """Pool rhythm classes below MIN_CLASS_SIZE so 10-fold stratification works."""
    counts = pd.Series(rhythms).value_counts()
    rare = set(counts[counts < MIN_CLASS_SIZE].index)
    if rare:
        logger.info(
            "Pooling %d rhythm(s) with <%d records into '%s': %s",
            len(rare), MIN_CLASS_SIZE, OTHER, sorted(rare),
        )
    return [OTHER if r in rare else r for r in rhythms]


# --------------------------------------------------------------------------- fields

#: The columns ``load_labels`` returns, as data — see ``ecgbench.labels._fields``.
#: Literal on purpose: the metadata build reads it without importing this module.
FIELDS = (
    Field(
        "sex",
        "string",
        "From the 'Sex' column, trailing newline stripped",
        vocabulary=("M", "F"),
        nullable=False,
    ),
    Field(
        "age_raw",
        "string",
        "The 'Age' cell verbatim after stripping; record 34 reads '>89'",
        nullable=False,
    ),
    Field(
        "age",
        "integer",
        "age_raw as a number; missing for '>89'",
        unit="year",
    ),
    Field(
        "rhythms",
        "array[string]",
        "Rhythm findings from the 'Rhythms' column ('Rhythms' column): the cell's "
        "newline-joined values split into a list, empty when blank",
        nullable=False,
    ),
    Field(
        "conduction_abnormalities",
        "array[string]",
        "Conduction abnormalities ('Conduction abnormalities' column): the cell's "
        "newline-joined values split into a list, empty when blank",
        nullable=False,
    ),
    Field(
        "extrasystolies",
        "array[string]",
        "Extrasystoles ('Extrasystolies' column): the cell's newline-joined values split "
        "into a list, empty when blank",
        nullable=False,
    ),
    Field(
        "hypertrophies",
        "array[string]",
        "Hypertrophies ('Hypertrophies' column): the cell's newline-joined values split "
        "into a list, empty when blank",
        nullable=False,
    ),
    Field(
        "cardiac_pacing",
        "array[string]",
        "Cardiac pacing findings ('Cardiac pacing' column): the cell's newline-joined "
        "values split into a list, empty when blank",
        nullable=False,
    ),
    Field(
        "ischemia",
        "array[string]",
        "Ischaemia findings (40 distinct strings raw, a handful once split) ('Ischemia' "
        "column): the cell's newline-joined values split into a list, empty when blank",
        nullable=False,
    ),
    Field(
        "repolarization_abnormalities",
        "array[string]",
        "Non-specific repolarisation abnormalities ('Non-specific repolarization "
        "abnormalities' column): the cell's newline-joined values split into a list, empty "
        "when blank",
        nullable=False,
    ),
    Field(
        "other_states",
        "array[string]",
        "Other states ('Other states' column): the cell's newline-joined values split into "
        "a list, empty when blank",
        nullable=False,
    ),
    Field(
        "electric_axis",
        "string",
        "'Electric axis of the heart' with the repeated prefix and trailing period removed; "
        "empty when blank",
        nullable=False,
        example="normal",
    ),
    Field(
        "primary_rhythm",
        "string",
        "First entry of rhythms, with rhythms held by fewer than 10 records pooled into "
        "OTHER; the config's label column",
        nullable=False,
    ),
)
