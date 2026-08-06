"""
CPSC 2018 labels: the challenge's nine classes, read out of the WFDB headers.

The mirror ECGBench reads ships **no metadata file** — no ``REFERENCE.csv``, no
CSV, no ``RECORDS``. Every record's ``.hea`` carries a short comment block:
``#Age``, ``#Sex``, ``#Dx`` and three fields that are the literal string
``Unknown`` on every record (``#Rx``, ``#Hx``, ``#Sx``). ``#Dx`` is a
comma-separated list of SNOMED-CT codes, so labels are genuinely **multi-label**:
6,401 records carry one class, 470 carry two and 6 carry three, for 7,359
class-record pairs over 6,877 records. No record is unlabelled.

Exactly nine codes occur, and they are precisely CPSC's nine classes — the
mapping is closed and is hardcoded in ``CPSC_CLASSES`` below rather than read
from the packaged Challenge 2020 table. Two of the nine would otherwise be
renamed: the challenge table calls ``164884008`` "ventricular ectopics" (VEB) and
``426783006`` "sinus rhythm" (NSR), where CPSC's own published table calls them
"premature ventricular contraction" (PVC) and "Normal". This is the CPSC
dataset, so CPSC's vocabulary wins; ``dx`` still carries the SNOMED codes, which
join straight onto ``ecgbench.labels.challenge2020``.

Quirks worth knowing, all verified against the files:

- **There is no primary diagnosis, and the published class table cannot be
  reproduced from these files.** CPSC's ``REFERENCE.csv`` distinguished
  ``First_label``/``Second_label``/``Third_label``, and the per-class counts on
  the challenge page ("according to the 'First label' annotations", summing to
  6,877) are First_label counts. The WFDB conversion sorted each ``#Dx`` list by
  CPSC class index and dropped the distinction. The challenge page documents
  A0043 as First=5 (RBBB), Second=2 (AF); its header reads
  ``164889003,59118001`` — AF first. So the first code is an artefact of the
  sort, not a diagnosis, which is why the single-label reduction below is called
  ``stratify_dx`` and not ``primary_dx``. **Train on ``dx``.** What you *can*
  recompute is the any-label count per class, which exceeds the published
  First_label count by exactly the 482 second and third labels (470 x 1 + 6 x 2).
- **Age has a sentinel and a gap.** 5 records carry no age at all (A0608, A1549,
  A1876, A2299, A5990 — the header says ``NaN``, which the shared parser turns
  into an empty string) and 4 more carry ``-1``. Genuine ages run 1 to 104. The
  ``-1`` is left exactly as shipped, because blanking it would merge two
  distinct states; filter on ``AGE_SENTINELS`` before computing any age
  statistic.
- **Ages above 89 are unredacted here.** 125 records give an exact age over 89,
  up to 104. PhysioNet's Challenge 2020 re-release of these same waveforms rails
  all of them to 92, changing 104 values. If you need the two to agree, clip.
- **Sex is complete and needs no normalisation** — 3,699 ``Male`` and 3,178
  ``Female``, matching the challenge's published figures exactly.
- **Length varies by a factor of 24** and is exposed per record as ``n_samples``
  and ``duration_seconds``: 6 s to 144 s, 1,650 distinct lengths, median 12 s.
  The challenge page claims a 60 s maximum; 27 records exceed it. Anything
  batching these records needs ``window=`` or padding.
- **Sampling rate is a constant** (500 Hz in all 6,877 headers) and is exposed
  anyway, so a frame from this loader has the same shape as the challenge ones.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ecgbench.labels._challenge_headers import parse_header

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AGE_SENTINELS",
    "CPSC_CLASSES",
    "RECORDS_DIR",
    "attach_dx_columns",
    "load_labels",
    "parse_header",
    "scan_headers",
]

#: Directory (relative to the dataset root) holding the records. The mirror is
#: flat — 6,877 .hea/.mat pairs directly under here, no cohort subdirectories.
RECORDS_DIR = "Training_WFDB"

#: The nine CPSC 2018 classes, in the challenge's own numbering, with the
#: SNOMED-CT code the WFDB conversion encoded each one as. Closed set: these are
#: exactly the nine codes that occur in the 6,877 headers, and nothing else does.
#: Order matters — it is the order the conversion sorted each #Dx list into.
CPSC_CLASSES: tuple[tuple[int, str, str, str], ...] = (
    (1, "426783006", "NSR", "Normal"),
    (2, "164889003", "AF", "Atrial fibrillation"),
    (3, "270492004", "IAVB", "First-degree atrioventricular block"),
    (4, "164909002", "LBBB", "Left bundle branch block"),
    (5, "59118001", "RBBB", "Right bundle branch block"),
    (6, "284470004", "PAC", "Premature atrial contraction"),
    (7, "164884008", "PVC", "Premature ventricular contraction"),
    (8, "429622005", "STD", "ST-segment depression"),
    (9, "164931005", "STE", "ST-segment elevation"),
)

_ABBREVIATION = {code: abbr for _, code, abbr, _ in CPSC_CLASSES}
_CLASS_NAME = {code: name for _, code, _, name in CPSC_CLASSES}
_CLASS_INDEX = {code: index for index, code, _, _ in CPSC_CLASSES}

#: Marker for a code outside CPSC's nine. Never expected — the set is closed —
#: but a silent mismap would be worse than a visible marker if the mirror ever
#: changes.
UNMAPPED = "UNMAPPED"

#: Ages that are codes rather than measurements. ``-1`` appears in 4 records.
#: Kept in the ``age`` column as shipped — see the module docstring.
AGE_SENTINELS = ("-1",)


def scan_headers(data_path: Path | str) -> pd.DataFrame:
    """Parse every CPSC 2018 record header into one frame.

    Adds ``signal_path`` relative to ``data_path``, so it resolves identically
    for the splitter, the validation engine and ``ECGDataset``.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_root = data_path / RECORDS_DIR
    if not records_root.is_dir():
        raise LabelSourceMissingError(
            f"Expected the record tree at {records_root}. CPSC 2018 labels live in "
            "the WFDB headers, so point data_path at the dataset root — the "
            f"directory holding {RECORDS_DIR}/."
        )

    hea_files = sorted(records_root.glob("*.hea"))
    if not hea_files:
        raise LabelSourceMissingError(f"No .hea headers found in {records_root}")
    logger.info("Parsing %d WFDB headers in %s", len(hea_files), records_root)

    # Header reads are I/O bound and independent — threads are enough.
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(parse_header, hea_files))

    for row, hea in zip(rows, hea_files, strict=True):
        row["signal_path"] = str(hea.with_suffix(".mat").relative_to(data_path))

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info("Parsed %d headers", len(df))
    return df


def _split_codes(dx: str) -> list[str]:
    """Split a ``#Dx`` field into codes, keeping the shipped order."""
    return [c for c in str(dx).split(",") if c]


def _rarest_code(codes: list[str], frequency: dict[str, int]) -> str:
    """Pick the globally rarest class a record carries.

    Ties break on the lowest CPSC class index, so the result is a pure function
    of the dataset and does not depend on scan order.
    """
    return min(codes, key=lambda c: (frequency.get(c, 0), _CLASS_INDEX.get(c, 99)))


def attach_dx_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived diagnosis columns to a frame holding a raw ``dx`` column.

    This is the **only** derivation of these columns in ECGBench —
    ``CPSC2018Splitter`` calls it through ``load_labels`` rather than repeating
    the mapping, so the stratification label and the exposed labels cannot drift
    apart.
    """
    code_lists = df["dx"].fillna("").astype(str).apply(_split_codes)

    unknown = {c for codes in code_lists for c in codes} - set(_ABBREVIATION)
    if unknown:
        logger.warning(
            "%d SNOMED code(s) outside CPSC's nine classes, kept raw: %s",
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
        lambda codes: ",".join(_ABBREVIATION.get(c, UNMAPPED) for c in codes)
    )
    out["dx_names"] = code_lists.apply(
        lambda codes: "|".join(_CLASS_NAME.get(c, UNMAPPED) for c in codes)
    )
    out["dx_class_indices"] = code_lists.apply(
        lambda codes: ",".join(str(_CLASS_INDEX[c]) for c in codes if c in _CLASS_INDEX)
    )

    # Single-label reduction for stratification ONLY — see the module docstring.
    strat = code_lists.apply(lambda codes: _rarest_code(codes, frequency) if codes else "")
    out["stratify_dx"] = strat
    out["stratify_dx_abbreviation"] = strat.apply(
        lambda c: _ABBREVIATION.get(c, UNMAPPED) if c else UNMAPPED
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return CPSC 2018 labels indexed by record name.

    Columns:

    - ``dx`` — comma-separated SNOMED-CT codes. **This is the ground truth**; it
      is multi-label (476 of 6,877 records carry more than one class) and every
      record carries at least one.
    - ``dx_abbreviations`` / ``dx_names`` / ``dx_class_indices`` — the same codes
      as CPSC's abbreviations, full names (pipe-separated, matching the challenge
      loaders) and 1-9 class numbers. ``n_dx`` is the count.
    - ``stratify_dx`` / ``stratify_dx_abbreviation`` — a single-label reduction
      taking the globally rarest class each record carries, ties broken on the
      lowest CPSC class index. **For stratification only, not a clinical primary
      diagnosis** — the shipped ``#Dx`` order is a sort by class index and the
      real First/Second/Third labelling did not survive the WFDB conversion.
    - ``age``, ``sex`` — from the headers. ``age`` is left exactly as shipped and
      still contains the ``AGE_SENTINELS`` value; ``sex`` is complete.
    - ``sampling_rate``, ``n_samples``, ``n_leads``, ``duration_seconds`` — per
      record. Rate and lead count are constant here, but length is not: 6 s to
      144 s.
    """
    df = scan_headers(data_path)
    df = attach_dx_columns(df)
    df["duration_seconds"] = (df["n_samples"] / df["sampling_rate"]).round(3)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column

    n_sentinel = int(df["age"].isin(AGE_SENTINELS).sum())
    n_blank = int((df["age"].fillna("") == "").sum())
    if n_sentinel or n_blank:
        logger.info(
            "%d record(s) carry a sentinel age (%s) and %d have none; exclude both "
            "before computing age statistics.",
            n_sentinel,
            "/".join(AGE_SENTINELS),
            n_blank,
        )
    logger.info(
        "Loaded labels for %d records; %d distinct classes, %d multi-label records",
        len(df),
        len({c for dx in df["dx"] for c in str(dx).split(",") if c}),
        int((df["n_dx"] > 1).sum()),
    )
    return df
