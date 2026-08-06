"""
PhysioNet/CinC Challenge 2020 labels: SNOMED-CT diagnoses from the headers.

The release ships **no metadata file and no code table**. Every record's ``.hea``
carries a short comment block — ``#Age``, ``#Sex``, ``#Dx`` and three fields that
are the literal string ``Unknown`` on every record (``#Rx``, ``#Hx``, ``#Sx``).
``#Dx`` is a comma-separated list of SNOMED-CT codes, so labels are genuinely
**multi-label**: 111 distinct codes over 43,101 records, 2.18 codes per record on
average and up to 10. No record is unlabelled.

Because no code table ships with the data, ECGBench packages the challenge's own
one as ``ecgbench/data/challenge2020_dx_mapping.csv`` — the concatenation of
``dx_mapping_scored.csv`` (27 classes) and ``dx_mapping_unscored.csv`` (84) from
the official scoring code at github.com/physionetchallenges/evaluation-2020
(BSD-2-Clause). It covers exactly the 111 codes present in the data, with no
duplicate code and no duplicate abbreviation.

The 27 ``scored`` classes are the ones the challenge metric evaluated (they
collapse to 24 evaluated classes, because three pairs were scored as equivalent —
CRBBB/RBBB, PAC/SVPB and PVC/VPB). ``scored_dx`` exposes that subset. Note this
is **not** the 2021 scored set: 2021's 30 classes are these 27 plus PRWP and
CLBBB, which do not occur in this release at all, and BBB, which does (137
records) but was unscored in 2020. Nothing scored in 2020 was dropped in 2021.

Quirks worth knowing, all verified against the files:

- **631 records repeat a code inside their own ``#Dx`` list**, and this loader
  deduplicates them. 596 records list ``284470004`` (PAC) twice, 30 list
  ``17338001`` (VPB) twice, and five others repeat one code; one Georgia record
  lists a code three times. This is not cosmetic — counting raw list entries
  inflates Georgia's PAC total from 639 records to 1,236, and it is the whole
  reason the shipped v1.0.2 data appears to disagree with the official code
  table. **After deduplication all 111 codes and all six per-cohort columns of
  ``dx_mapping_*.csv`` reproduce exactly** (93,843 code-record pairs). The 2021
  re-release had already deduplicated and numerically sorted these lists, which
  is why only the 2020 loader needs this.
- **``#Dx`` order carries no clinical meaning.** Unlike the PhysioNet
  ``ecg-arrhythmia`` release, where the first code is the rhythm diagnosis, here
  the order varies by cohort and, in Georgia, is not even internally consistent.
  There is therefore **no primary diagnosis** to read off, which is why the
  single-label reduction below is called ``stratify_dx`` rather than
  ``primary_dx``: it exists to make stratified folds well defined and is not
  ground truth. Train on ``dx``.
- **Age carries two sentinel values and they are left as they are.** 181 records
  have no age; beyond that, 204 ``ptb-xl`` records record ``300`` (PTB-XL's own
  convention for a patient older than 89) and 6 CPSC records record ``-1``.
  Genuine ages run 1-92. These are *not* converted to blank, because the release
  also has real blanks and collapsing the three states would lose the
  distinction between "unknown" and "over 89"; use ``AGE_SENTINELS`` to filter.
- **Sex is normalised.** The 74 ``st_petersburg_incart`` records spell it ``M``
  and ``F`` where the other five cohorts spell it ``Male`` and ``Female``; this
  loader maps them onto the long form, as the 2021 re-release did. One record
  has no sex at all.
- Each record exists at exactly **one** sampling rate — 500 Hz for 42,511
  records, 1000 Hz for the 516 ``ptb`` records, 257 Hz for the 74
  ``st_petersburg_incart`` ones. ``sampling_rate`` is therefore a per-record
  label here, not a dataset-wide constant.
- The six source cohorts are exposed as ``source``. They are not interchangeable:
  durations run from 5 s to 1800 s and four of them overlap datasets catalogued
  separately in ECGBench.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ecgbench.labels._challenge_headers import RECORDS_DIR, parse_header
from ecgbench.labels._challenge_headers import scan_headers as _scan_headers

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AGE_SENTINELS",
    "RECORDS_DIR",
    "UNMAPPED",
    "attach_dx_columns",
    "load_dx_mapping",
    "load_labels",
    "parse_header",
    "scan_headers",
]

#: Challenge code table packaged with ECGBench, since none ships with the data.
MAPPING_CSV = Path(__file__).parent.parent / "data" / "challenge2020_dx_mapping.csv"

#: Marker for a code absent from the table. Never expected: the packaged table
#: covers all 111 codes in this release. It would fire if PhysioNet reissued the
#: dataset with new classes, which is worth seeing rather than silently mapping.
UNMAPPED = "UNMAPPED"

#: Ages that are codes rather than measurements. ``300`` is PTB-XL's marker for
#: a patient over 89 (204 records); ``-1`` appears in 6 CPSC records. Kept in the
#: ``age`` column as shipped — see the module docstring — so filter on these
#: before computing any age statistic.
AGE_SENTINELS = ("-1", "300")

#: The 74 St Petersburg records use single letters; every other cohort spells the
#: word out. Normalised so ``sex`` has one vocabulary across the release.
_SEX_NORMALISATION = {"M": "Male", "F": "Female"}


@functools.cache
def load_dx_mapping() -> pd.DataFrame:
    """Return the challenge code table indexed by SNOMED-CT code.

    Columns: ``dx_name``, ``abbreviation``, ``scored``. 111 rows, 27 of them
    scored by the 2020 challenge metric.
    """
    df = pd.read_csv(MAPPING_CSV, dtype={"snomed_code": str})
    df["snomed_code"] = df["snomed_code"].str.strip()
    df = df.set_index("snomed_code")
    df["scored"] = df["scored"].astype(bool)
    return df


def scan_headers(data_path: Path | str) -> pd.DataFrame:
    """Parse every Challenge 2020 record header under ``training/`` into one frame.

    Thin wrapper over the shared scanner, which the 2021 release uses too.
    """
    return _scan_headers(data_path, release="Challenge 2020")


def _split_codes(dx: str) -> list[str]:
    """Split a ``#Dx`` field into codes, dropping repeats but keeping order.

    ``dict.fromkeys`` is the deduplication: 631 records in this release list one
    of their codes twice (one lists it three times). See the module docstring —
    counting the raw entries makes the data disagree with the official table.
    """
    return list(dict.fromkeys(c for c in str(dx).split(",") if c))


def _rarest_code(codes: list[str], frequency: dict[str, int]) -> str:
    """Pick the globally rarest code a record carries.

    Ties break on the lowest numeric SNOMED code, so the result is a pure
    function of the dataset and does not depend on scan order or dict ordering.
    """
    return min(codes, key=lambda c: (frequency.get(c, 0), int(c) if c.isdigit() else 0))


def attach_dx_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived diagnosis columns to a frame holding a raw ``dx`` column.

    Also rewrites ``dx`` itself to the deduplicated code list, so every consumer
    sees the same codes the official table counts.

    This is the **only** derivation of these columns in ECGBench —
    ``Challenge2020Splitter`` calls it through ``load_labels`` rather than
    repeating the mapping, so the stratification label and the exposed labels
    cannot drift apart.
    """
    mapping = load_dx_mapping()
    abbreviation = mapping["abbreviation"].to_dict()
    dx_name = mapping["dx_name"].to_dict()
    scored = set(mapping.index[mapping["scored"]])

    raw_lists = df["dx"].fillna("").astype(str).apply(lambda dx: [c for c in dx.split(",") if c])
    code_lists = df["dx"].fillna("").astype(str).apply(_split_codes)

    n_duplicated = int((raw_lists.apply(len) != code_lists.apply(len)).sum())
    if n_duplicated:
        logger.info(
            "Deduplicated repeated SNOMED codes inside #Dx for %d record(s); the "
            "official dx_mapping_*.csv totals count records, not list entries.",
            n_duplicated,
        )

    unknown = {c for codes in code_lists for c in codes} - set(mapping.index)
    if unknown:
        logger.warning(
            "%d SNOMED code(s) absent from the packaged challenge table, kept raw: %s",
            len(unknown),
            sorted(unknown),
        )

    # Frequencies over this dataset, used only for the stratification reduction.
    frequency: dict[str, int] = {}
    for codes in code_lists:
        for code in codes:
            frequency[code] = frequency.get(code, 0) + 1

    out = df.copy()
    out["dx"] = code_lists.apply(",".join)
    out["n_dx"] = code_lists.apply(len)
    out["dx_abbreviations"] = code_lists.apply(
        lambda codes: ",".join(abbreviation.get(c, UNMAPPED) for c in codes)
    )
    out["dx_names"] = code_lists.apply(
        lambda codes: "|".join(dx_name.get(c, UNMAPPED) for c in codes)
    )
    out["scored_dx"] = code_lists.apply(
        lambda codes: ",".join(abbreviation[c] for c in codes if c in scored)
    )
    out["n_scored_dx"] = out["scored_dx"].apply(lambda s: len(s.split(",")) if s else 0)

    # Single-label reduction for stratification ONLY — see the module docstring.
    strat = code_lists.apply(lambda codes: _rarest_code(codes, frequency) if codes else "")
    out["stratify_dx"] = strat
    out["stratify_dx_abbreviation"] = strat.apply(
        lambda c: abbreviation.get(c, UNMAPPED) if c else UNMAPPED
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Challenge 2020 labels indexed by record name.

    Columns:

    - ``dx`` — comma-separated SNOMED-CT codes, **deduplicated** (631 records
      repeat a code in the shipped headers). **This is the ground truth**; it is
      multi-label, and every record carries at least one code.
    - ``dx_abbreviations`` / ``dx_names`` — the same codes through the packaged
      challenge table (``dx_names`` is pipe-separated, because the names contain
      commas). ``n_dx`` is the count.
    - ``scored_dx`` / ``n_scored_dx`` — restricted to the 27 classes the 2020
      challenge metric scored. Not the same subset as Challenge 2021's 30.
    - ``stratify_dx`` / ``stratify_dx_abbreviation`` — a single-label reduction
      taking the globally rarest code each record carries, ties broken on the
      lowest numeric code. **For stratification only, not a clinical primary
      diagnosis** — ``#Dx`` order carries no clinical meaning in this release.
    - ``source`` — one of the six cohort directories under ``training/``.
    - ``age``, ``sex`` — from the headers. ``sex`` is normalised to
      ``Male``/``Female``; ``age`` is left exactly as shipped and still contains
      the ``AGE_SENTINELS`` values.
    - ``sampling_rate``, ``n_samples``, ``n_leads`` — per record, because rate
      and length both vary across this dataset.
    """
    df = scan_headers(data_path)
    df["sex"] = df["sex"].replace(_SEX_NORMALISATION)
    df = attach_dx_columns(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column

    n_sentinel = int(df["age"].isin(AGE_SENTINELS).sum())
    if n_sentinel:
        logger.info(
            "%d record(s) carry a sentinel age (%s) and %d have none; exclude both "
            "before computing age statistics.",
            n_sentinel,
            "/".join(AGE_SENTINELS),
            int((df["age"].fillna("") == "").sum()),
        )
    logger.info(
        "Loaded labels for %d records; %d distinct SNOMED codes",
        len(df),
        len({c for dx in df["dx"] for c in str(dx).split(",") if c}),
    )
    return df
