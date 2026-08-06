"""
SPH labels: AHA/ACC/HRS diagnostic statements, decoded against the shipped table.

The release ships two CSVs and this module joins them. ``metadata.csv`` holds one
row per record with an ``AHA_Code`` string; ``code.csv`` is the codebook that
turns those numbers into categories and English descriptions. Neither is useful
alone — ``AHA_Code`` of ``"30+310"`` means nothing until ``code.csv`` says 30 is
"Atrial premature complex(es)" and 310 is the modifier "Frequent".

**The ``AHA_Code`` grammar, which nothing else in ECGBench uses.** A record's
diagnosis is a ``;``-separated list of *statements*, and each statement is one
**primary** code optionally followed by ``+``-joined **modifiers**::

    "1"          one statement: Normal ECG
    "22;23"      two statements: sinus bradycardia, sinus arrhythmia
    "30+310"     one statement: atrial premature complexes, Frequent
    "60+310;147" ventricular premature complexes (Frequent); T-wave abnormality

44 primary codes and 15 modifiers occur, and they are exactly the 44 + 15 rows of
``code.csv`` — the vocabulary is closed and every code in all 25,770 records
resolves, so ``UNMAPPED`` below should never appear.

Quirks worth knowing, all verified against the files:

- **Records are multi-label and there is no primary diagnosis.** 22,046 records
  carry one statement, 2,936 two, 665 three, 109 four, 12 five and 2 six —
  3,724 records (14.45%) with more than one, matching the paper exactly. The
  statement order is *nearly* a numeric sort of the primary code (24,961 of
  25,770 records are in ascending order, and 2,915 of the 3,724 multi-statement
  ones), so the first statement is neither reliably a sort artefact nor
  documented as a priority. Nothing in the release ranks the statements. That is
  why the single-label reduction here is called ``stratify_code`` and not
  ``primary_code``: it exists to make stratified folds well defined and is **not
  ground truth**. Train on ``aha_primary_codes``.
- **"Normal" is a code, not the absence of codes**, and two records list it
  twice. 13,905 records carry primary code 1 (Normal ECG) and nothing else; no
  record is unlabelled. But only 13,903 have ``AHA_Code`` equal to the string
  ``"1"`` — A02322 and A05000 read ``"1;1"``, the same statement repeated. That
  is why the primary-code lists are deduplicated with ``dict.fromkeys``, and why
  ``is_normal`` tests the deduplicated list rather than the raw string: it gives
  13,905 records, 53.96% normal and 46.04% abnormal, matching the paper's 46.04%
  exactly where a string comparison would miss by two.
- **Modifiers are not diagnoses and are exposed separately.** ``310`` ("Frequent")
  qualifies whichever primary it is attached to, so pooling modifiers into the
  label set would invent classes. ``aha_modifier_codes`` keeps them, and
  ``aha_statements`` keeps the primary-to-modifier attachment that both flat
  columns discard.
- **Age and sex are complete and carry no sentinels.** Ages run 18-95 in all
  25,770 rows with no blanks, and sex is ``M`` (14,265) or ``F`` (11,505). The
  paper says 18-100; the shipped maximum is 95.
- **Record length varies by a factor of 5.6** and is exposed as ``n_samples`` and
  ``duration_seconds``: 5,000 to 28,000 samples, i.e. 10 s to 56 s at 500 Hz, in
  39 distinct lengths. 18,842 records (73%) are exactly 10 s. The metadata's
  ``N`` agrees with the HDF5 array for all 25,770 records, so it is authoritative
  and nothing needs to open a signal file to learn a length. Anything batching
  these records needs ``window=`` or padding.
- **Patients repeat.** 24,666 patients over 25,770 records: 23,600 contributed
  one, 1,033 two, 29 three, 3 four and 1 five. Folds are grouped on
  ``patient_id`` — see the config.

List-valued columns are joined with ``;``, not ``,``, because four of the
descriptions contain a comma ("Atrial premature complexes, nonconducted", "AV
block, complete (third-degree)", …) and a comma-joined string could not be split
back apart.
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
    "CODE_CSV",
    "METADATA_CSV",
    "RECORDS_DIR",
    "SAMPLING_RATE",
    "UNMAPPED",
    "attach_aha_columns",
    "load_code_table",
    "load_labels",
]

#: One row per record: ECG_ID, AHA_Code, Patient_ID, Age, Sex, N, Date.
METADATA_CSV = "metadata.csv"

#: The AHA codebook: Category, Code, Description. 44 primary rows (categories
#: A and C-M; B, G and N do not occur) plus 15 rows in category "Modifier".
CODE_CSV = "code.csv"

#: Directory holding one HDF5 file per record, relative to the dataset root.
#: Ships as records.tar.gz — which is an *uncompressed* tar despite the name, so
#: `tar -xf` works and `tar -xzf` does not.
RECORDS_DIR = "records"

#: Constant across the release; exposed so a frame from this loader has the same
#: shape as the ones that genuinely vary.
SAMPLING_RATE = 500

#: The category value ``code.csv`` uses for the 15 qualifiers.
MODIFIER_CATEGORY = "Modifier"

#: Marker for a code outside the codebook. Never expected — the vocabulary is
#: closed — but a silent mismap would be worse than a visible marker if a future
#: release adds statements without adding rows to code.csv.
UNMAPPED = "UNMAPPED"

#: Separator for the list-valued derived columns. Not a comma: four descriptions
#: contain one.
LIST_SEPARATOR = ";"


def load_code_table(data_path: Path | str) -> pd.DataFrame:
    """Read ``code.csv``, indexed by code (as a string).

    Columns: ``category``, ``description``, ``is_modifier``.
    """
    from ecgbench.labels import LabelSourceMissingError

    path = Path(data_path) / CODE_CSV
    if not path.exists():
        raise LabelSourceMissingError(
            f"SPH's AHA codebook {CODE_CSV} is not in {data_path}. ECGBench publishes "
            "fold CSVs only — labels stay with the source dataset, so point data_path "
            "at a full local copy (https://doi.org/10.6084/m9.figshare.c.5779802.v1)."
        )

    df = pd.read_csv(path, dtype={"Code": str})
    missing = {"Category", "Code", "Description"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")

    # .to_numpy(), not the Series: passing index= alongside Series values
    # *reindexes* them against the new labels rather than relabelling, which
    # silently yields a frame of NaN.
    out = pd.DataFrame(
        {
            "category": df["Category"].astype(str).to_numpy(),
            "description": df["Description"].astype(str).to_numpy(),
        },
        index=pd.Index(df["Code"].astype(str).to_numpy(), name="code"),
    )
    out["is_modifier"] = out["category"] == MODIFIER_CATEGORY
    if out.index.has_duplicates:
        raise ValueError(f"{path} has duplicate codes: {sorted(out.index[out.index.duplicated()])}")
    return out


def _parse_statements(aha_code: str) -> list[tuple[str, list[str]]]:
    """Split one ``AHA_Code`` cell into ``[(primary, [modifier, ...]), ...]``.

    Empty fragments are dropped rather than becoming empty codes, so a trailing
    ``;`` — none occur today — could not create a phantom statement.
    """
    statements = []
    for fragment in str(aha_code).split(";"):
        parts = [p.strip() for p in fragment.split("+") if p.strip()]
        if parts:
            statements.append((parts[0], parts[1:]))
    return statements


def _rarest_code(codes: list[str], frequency: dict[str, int]) -> str:
    """Pick the globally rarest primary code a record carries.

    Ties break on the lowest numeric code, so the result is a pure function of
    the dataset and does not depend on row order or dict ordering.
    """
    return min(codes, key=lambda c: (frequency.get(c, 0), int(c) if c.isdigit() else 0))


def attach_aha_columns(df: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:
    """Add the decoded diagnosis columns to a frame holding a raw ``aha_code``.

    This is the **only** derivation of these columns in ECGBench — ``SPHSplitter``
    reaches them through ``load_labels`` rather than repeating the parse, so the
    stratification label and the exposed labels cannot drift apart.
    """
    category = codes["category"].to_dict()
    description = codes["description"].to_dict()
    modifiers = set(codes.index[codes["is_modifier"]])

    parsed = df["aha_code"].map(_parse_statements)

    def primaries(statements: list[tuple[str, list[str]]]) -> list[str]:
        # dict.fromkeys, not set(): a record listing the same primary twice keeps
        # its first-seen order rather than becoming order-dependent.
        return list(dict.fromkeys(p for p, _ in statements))

    def modifier_codes(statements: list[tuple[str, list[str]]]) -> list[str]:
        return list(dict.fromkeys(m for _, mods in statements for m in mods))

    primary_lists = parsed.map(primaries)
    modifier_lists = parsed.map(modifier_codes)

    unexpected = {
        code for lst in primary_lists for code in lst if code in modifiers or code not in category
    }
    if unexpected:
        logger.warning(
            "Primary position holds %d code(s) that are not primary codes in %s: %s",
            len(unexpected),
            CODE_CSV,
            sorted(unexpected),
        )

    out = df.copy()
    out["aha_statements"] = parsed.map(
        lambda stmts: LIST_SEPARATOR.join("+".join([p, *mods]) for p, mods in stmts)
    )
    out["n_statements"] = parsed.map(len)
    out["aha_primary_codes"] = primary_lists.map(LIST_SEPARATOR.join)
    out["aha_primary_descriptions"] = primary_lists.map(
        lambda lst: LIST_SEPARATOR.join(description.get(c, UNMAPPED) for c in lst)
    )
    out["aha_primary_categories"] = primary_lists.map(
        lambda lst: LIST_SEPARATOR.join(dict.fromkeys(category.get(c, UNMAPPED) for c in lst))
    )
    out["aha_modifier_codes"] = modifier_lists.map(LIST_SEPARATOR.join)
    out["aha_modifier_descriptions"] = modifier_lists.map(
        lambda lst: LIST_SEPARATOR.join(description.get(c, UNMAPPED) for c in lst)
    )
    out["n_primary_codes"] = primary_lists.map(len)
    # Deduplicated list == ["1"], not aha_code == "1": A02322 and A05000 read
    # "1;1". 13,905 records, which is the paper's 53.96%.
    out["is_normal"] = primary_lists.map(lambda lst: lst == ["1"])

    # Single-label reduction for stratification ONLY — see the module docstring.
    frequency: dict[str, int] = {}
    for lst in primary_lists:
        for code in lst:
            frequency[code] = frequency.get(code, 0) + 1
    strat = primary_lists.map(lambda lst: _rarest_code(lst, frequency) if lst else "")
    out["stratify_code"] = strat
    out["stratify_description"] = strat.map(
        lambda c: description.get(c, UNMAPPED) if c else UNMAPPED
    )
    out["stratify_category"] = strat.map(lambda c: category.get(c, UNMAPPED) if c else UNMAPPED)
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return SPH labels and metadata indexed by ``ecg_id``.

    Columns:
        ``patient_id``, ``age``, ``sex``, ``date``, ``n_samples``,
        ``duration_seconds``, ``sampling_rate``, ``aha_code`` (as shipped),
        ``aha_statements``, ``n_statements``, ``aha_primary_codes``,
        ``aha_primary_descriptions``, ``aha_primary_categories``,
        ``n_primary_codes``, ``aha_modifier_codes``,
        ``aha_modifier_descriptions``, ``is_normal``, ``stratify_code``,
        ``stratify_description``, ``stratify_category``, ``signal_path``.

    Multi-label: the ``aha_primary_*`` columns are ``;``-separated lists. Train
    on those, not on ``stratify_code`` — see the module docstring.
    """
    from ecgbench.labels import LabelSourceMissingError

    root = Path(data_path)
    metadata_path = root / METADATA_CSV
    if not metadata_path.exists():
        raise LabelSourceMissingError(
            f"SPH labels come from {METADATA_CSV}, which is not in {root}. ECGBench "
            "publishes fold CSVs only — labels stay with the source dataset, so point "
            f"data_path at a full local copy (see {config.url})."
        )

    raw = pd.read_csv(metadata_path, dtype={"ECG_ID": str, "Patient_ID": str, "AHA_Code": str})
    expected = {"ECG_ID", "AHA_Code", "Patient_ID", "Age", "Sex", "N", "Date"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"{metadata_path} is missing column(s) {sorted(missing)}")

    # .to_numpy() throughout: passing index= alongside Series values *reindexes*
    # them against the new labels rather than relabelling, which silently yields
    # a frame of NaN.
    df = pd.DataFrame(
        {
            "patient_id": raw["Patient_ID"].to_numpy(),
            "age": raw["Age"].to_numpy(),
            "sex": raw["Sex"].to_numpy(),
            "date": raw["Date"].to_numpy(),
            "n_samples": raw["N"].astype(int).to_numpy(),
            "aha_code": raw["AHA_Code"].fillna("").to_numpy(),
        },
        index=pd.Index(raw["ECG_ID"].to_numpy(), name="ecg_id"),
    )
    df["duration_seconds"] = df["n_samples"] / SAMPLING_RATE
    df["sampling_rate"] = SAMPLING_RATE
    # Relative to the dataset root, so it resolves identically for the splitter,
    # the validation engine and ECGDataset.
    df["signal_path"] = [f"{RECORDS_DIR}/{rid}.h5" for rid in df.index]

    df = attach_aha_columns(df, load_code_table(root))
    logger.info(
        "Loaded SPH labels: %d records, %d patients, %d abnormal",
        len(df),
        df["patient_id"].nunique(),
        int((~df["is_normal"]).sum()),
    )
    return df
