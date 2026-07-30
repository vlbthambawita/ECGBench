"""
PhysioNet/CinC Challenge 2021 labels: SNOMED-CT diagnoses from the headers.

The release ships **no metadata file and no code table**. Every record's ``.hea``
carries a short comment block — ``#Age``, ``#Sex``, ``#Dx`` and three fields that
are the literal string ``Unknown`` on every record (``#Rx``, ``#Hx``, ``#Sx``).
``#Dx`` is a comma-separated list of SNOMED-CT codes, so labels are genuinely
**multi-label**: 133 distinct codes over 88,253 records, 2.06 codes per record on
average and up to 12. No record is unlabelled.

Because no code table ships with the data, ECGBench packages the challenge's own
one as ``ecgbench/data/challenge2021_dx_mapping.csv`` — the concatenation of
``dx_mapping_scored.csv`` (30 classes) and ``dx_mapping_unscored.csv`` (103) from
the official scoring code at github.com/physionetchallenges/evaluation-2021
(BSD-2-Clause). It covers exactly the 133 codes present in the data, with no
duplicate code and no duplicate abbreviation. Its per-code totals were checked
against a full scan of this release and agree on all 133 codes.

The 30 ``scored`` classes are the ones the challenge metric evaluated (the
official table lists 30 rows collapsing to 26 evaluated classes, because four
pairs were scored as equivalent — CRBBB/RBBB, PAC/SVPB, PVC/VPB, and
STE-related pairs). ``scored_dx`` exposes that subset.

Quirks worth knowing, all verified against the files:

- **``#Dx`` order carries no clinical meaning.** Unlike the PhysioNet
  ``ecg-arrhythmia`` release, where the first code is the rhythm diagnosis, here
  the order varies by cohort — PTB-XL's lists are numerically sorted, Chapman's
  are not. There is therefore **no primary diagnosis** to read off, which is why
  the single-label reduction below is called ``stratify_dx`` rather than
  ``primary_dx``: it exists to make stratified folds well defined and is not
  ground truth. Train on ``dx``.
- Age is missing (``NaN``) in 236 records; sex is ``Unknown`` in 22 and empty in
  one.
- Each record exists at exactly **one** sampling rate — 500 Hz for 87,663
  records, 1000 Hz for the 516 ``ptb`` records, 257 Hz for the 74
  ``st_petersburg_incart`` ones. ``sampling_rate`` is therefore a per-record
  label here, not a dataset-wide constant.
- The eight source cohorts are exposed as ``source``. They are not
  interchangeable: durations run from 5 s to 1800 s and three of them overlap
  datasets catalogued separately in ECGBench.
"""

from __future__ import annotations

import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Directory (relative to the dataset root) holding the eight source cohorts.
RECORDS_DIR = "training"

#: Challenge code table packaged with ECGBench, since none ships with the data.
MAPPING_CSV = Path(__file__).parent.parent / "data" / "challenge2021_dx_mapping.csv"

#: Marker for a code absent from the table. Never expected: the packaged table
#: covers all 133 codes in this release. It would fire if PhysioNet reissued the
#: dataset with new classes, which is worth seeing rather than silently mapping.
UNMAPPED = "UNMAPPED"


@functools.cache
def load_dx_mapping() -> pd.DataFrame:
    """Return the challenge code table indexed by SNOMED-CT code.

    Columns: ``dx_name``, ``abbreviation``, ``scored``.
    """
    df = pd.read_csv(MAPPING_CSV, dtype={"snomed_code": str})
    df["snomed_code"] = df["snomed_code"].str.strip()
    df = df.set_index("snomed_code")
    df["scored"] = df["scored"].astype(bool)
    return df


def parse_header(hea_path: Path) -> dict[str, object]:
    """Parse the fields ECGBench needs out of one WFDB header.

    Headers here are under a kilobyte, so a text parse is far cheaper than a
    ``wfdb.rdheader`` call per record across 88k records.
    """
    with open(hea_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    record: dict[str, object] = {
        "record_name": hea_path.stem,
        "n_leads": None,
        "sampling_rate": None,
        "n_samples": None,
        "age": "",
        "sex": "",
        "dx": "",
    }

    if lines:
        # Record line: <name> <n_sig> <fs> <n_samples>
        parts = lines[0].split()
        for key, index in (("n_leads", 1), ("sampling_rate", 2), ("n_samples", 3)):
            if len(parts) > index:
                try:
                    record[key] = int(float(parts[index]))
                except ValueError:
                    pass  # corrupt header — validation flags it via corrupt_header

    for line in lines:
        if not line.startswith("#"):
            continue
        key, _, value = line[1:].partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value.lower() in ("unknown", "nan", "n/a"):
            value = ""
        if key in ("age", "sex", "dx"):
            record[key] = value.replace(" ", "") if key == "dx" else value

    return record


def scan_headers(data_path: Path | str) -> pd.DataFrame:
    """Parse every record header under ``training/`` into one frame.

    Adds ``source`` (the cohort directory) and ``signal_path`` (relative to
    ``data_path``, so it resolves identically for the splitter, the validation
    engine and ``ECGDataset``).
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_root = data_path / RECORDS_DIR
    if not records_root.is_dir():
        raise LabelSourceMissingError(
            f"Expected the record tree at {records_root}. Challenge 2021 labels live "
            "in the WFDB headers, so point data_path at the dataset root — the "
            "directory holding training/ and RECORDS."
        )

    hea_files = sorted(records_root.rglob("*.hea"))
    if not hea_files:
        raise LabelSourceMissingError(f"No .hea headers found under {records_root}")
    logger.info("Parsing %d WFDB headers under %s", len(hea_files), records_root)

    # Header reads are I/O bound and independent — threads are enough.
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(parse_header, hea_files))

    for row, hea in zip(rows, hea_files, strict=True):
        row["source"] = hea.relative_to(records_root).parts[0]
        row["signal_path"] = str(hea.with_suffix(".mat").relative_to(data_path))

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info("Parsed %d headers from %d source cohorts", len(df), df["source"].nunique())
    return df


def _rarest_code(codes: list[str], frequency: dict[str, int]) -> str:
    """Pick the globally rarest code a record carries.

    Ties break on the lowest numeric SNOMED code, so the result is a pure
    function of the dataset and does not depend on scan order or dict ordering.
    """
    return min(codes, key=lambda c: (frequency.get(c, 0), int(c) if c.isdigit() else 0))


def attach_dx_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived diagnosis columns to a frame holding a raw ``dx`` column.

    This is the **only** derivation of these columns in ECGBench —
    ``Challenge2021Splitter`` calls it through ``load_labels`` rather than
    repeating the mapping, so the stratification label and the exposed labels
    cannot drift apart.
    """
    mapping = load_dx_mapping()
    abbreviation = mapping["abbreviation"].to_dict()
    dx_name = mapping["dx_name"].to_dict()
    scored = set(mapping.index[mapping["scored"]])

    code_lists = df["dx"].fillna("").astype(str).apply(lambda dx: [c for c in dx.split(",") if c])

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
    """Return Challenge 2021 labels indexed by record name.

    Columns:

    - ``dx`` — raw comma-separated SNOMED-CT codes. **This is the ground truth**;
      it is multi-label, and every record carries at least one code.
    - ``dx_abbreviations`` / ``dx_names`` — the same codes through the packaged
      challenge table (``dx_names`` is pipe-separated, because the names contain
      commas). ``n_dx`` is the count.
    - ``scored_dx`` / ``n_scored_dx`` — restricted to the 30 classes the
      challenge metric scored.
    - ``stratify_dx`` / ``stratify_dx_abbreviation`` — a single-label reduction
      taking the globally rarest code each record carries, ties broken on the
      lowest numeric code. **For stratification only, not a clinical primary
      diagnosis** — ``#Dx`` order carries no clinical meaning in this release.
    - ``source`` — one of the eight cohort directories under ``training/``.
    - ``age``, ``sex`` — from the headers; blank where unknown.
    - ``sampling_rate``, ``n_samples``, ``n_leads`` — per record, because rate
      and length both vary across this dataset.
    """
    df = scan_headers(data_path)
    df = attach_dx_columns(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    logger.info(
        "Loaded labels for %d records; %d distinct SNOMED codes",
        len(df),
        len({c for dx in df["dx"] for c in str(dx).split(",") if c}),
    )
    return df
