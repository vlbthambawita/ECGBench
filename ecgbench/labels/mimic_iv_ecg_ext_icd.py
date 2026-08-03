"""
MIMIC-IV-ECG-Ext-ICD — ICD-10-CM diagnostic labels **for MIMIC-IV-ECG's records**.

This release ships no waveforms. It is one 323 MB table,
``records_w_diag_icd10.csv``, whose 800,035 rows are exactly the 800,035 studies
of MIMIC-IV-ECG, keyed by that dataset's own ``study_id``. Verified against the
files: every ``study_id`` and every ``subject_id`` is present on both sides with
none missing and none extra, ``file_name`` is ``record_list.csv``'s ``path``
under a ``mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/`` prefix
in 100% of rows, and ``ecg_time`` agrees in 100% of rows.

**There is deliberately no ``mimic_iv_ecg_ext_icd`` dataset config.** Every row is
a MIMIC-IV-ECG record, so generating a ten-fold ECGBench partition for it would
create a *second* partition over recordings the ``mimic_iv_ecg`` config already
partitions, and a user who trained on one and evaluated on the other would be
testing on training data. So this module is a *label provider*: you load
MIMIC-IV-ECG on its own ECGBench folds and join these columns onto it.
:func:`load_ext_icd` returns a frame indexed by ``study_id`` for exactly that.

What the release adds, per record: up to five sets of ICD-10-CM discharge
diagnoses drawn from the MIMIC-IV emergency-department and hospital modules,
patient demographics, date of death, and the linkage ids that produced the join.
``all_diag_all`` is the one to train on — it is ``all_diag_hosp`` where a hospital
admission exists and ``ed_diag_ed`` otherwise.

Four properties of the shipped table that change what you can do with it, all
recomputed from the file (SHA-256
``834586ff4f34e96c07e05e4828053b21347a18bb4fd86c2d961ebce47dafd260``, matching the
release's own ``SHA256SUMS.txt``):

1. **Only 468,005 of the 800,035 records carry any diagnosis at all** — 58.5%.
   The other 332,030 were not part of an ED or hospital stay that MIMIC-IV holds
   a discharge diagnosis for, and their five diagnosis columns are all empty
   lists, *not* nulls. An empty list is a real value here ("this ECG has no
   linked discharge diagnosis"), never a parse failure, so filter on it
   explicitly rather than on ``notna()``.
2. **The release ships its own 20-fold split, and it is not ECGBench's.** ``fold``
   and ``strat_fold`` both run 0-19 and are patient-grouped (0 of 161,352 subjects
   span a fold in either). The upstream benchmark uses folds 0-17 for training,
   18 for validation and 19 for test. That partition is statistically
   *independent* of the 10 folds the ``mimic_iv_ecg`` config produces: the
   upstream test fold's 39,569 records land 79.3% in ECGBench's train split,
   10.4% in val and 10.4% in test, and 6,449 of its 8,067 patients appear in
   ECGBench's train split. Pick one partition and stay inside it; see
   :func:`upstream_fold_split`.
3. **``gender`` encodes missing as the string ``"missing"``, not as a null**
   (4,489 records). Those same records have null ``age``, ``anchor_age`` and
   ``anchor_year``. This loader converts ``"missing"`` to ``NaN``, which is
   lossless because the column has no genuine nulls.
4. **Ages above 89 are a de-identification artefact.** MIMIC-IV sets
   ``anchor_age`` to 91 for every patient older than 89, per its own
   documentation, and it shows: 26,267 records sit at exactly 91, which is also
   the maximum ``anchor_age`` in the file. ``age`` is ``anchor_age`` plus years
   elapsed and so reaches 101. Do not read an age over 89 as the patient's real
   age. Timestamps are date-shifted into the future for the same reason
   (``ecg_time`` spans 2097 to 2211).

Labels are **never** published to the HuggingFace Hub, and these especially: the
source is credentialed under the PhysioNet Credentialed Health Data Use
Agreement, so its ICD codes stay with the people who signed it. You need local
copies of **both** MIMIC-IV-ECG (waveforms and ECGBench folds) and this release.
"""

from __future__ import annotations

import ast
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Join key throughout: MIMIC-IV-ECG's own record identifier.
JOIN_COLUMN = "study_id"

#: The single table this release ships, as PhysioNet names it.
SOURCE_CSV = "records_w_diag_icd10.csv"

#: The five diagnosis columns, in the order the release derives them. Values are
#: Python-literal lists of ICD-10-CM codes — empty lists where no diagnosis is
#: linked, which is 41.5% of records.
DIAGNOSIS_COLUMNS: dict[str, str] = {
    "ed_diag_ed": "ED discharge diagnoses, from the MIMIC-IV-ED module",
    "ed_diag_hosp": "hospital discharge diagnoses reached via ed_hadm_id",
    "hosp_diag_hosp": "hospital discharge diagnoses reached via hosp_hadm_id",
    "all_diag_hosp": "ed_diag_hosp and hosp_diag_hosp, de-duplicated",
    "all_diag_all": "all_diag_hosp where it exists, otherwise ed_diag_ed",
}

#: The label column the upstream benchmark trains on.
DEFAULT_DIAGNOSIS_COLUMN = "all_diag_all"

#: The three ECG subsets of the upstream benchmark's T(A2B)-E(C2D) notation,
#: mapped to the boolean column that selects each.
ECG_SUBSETS: dict[str, str] = {
    "ALL": "ecg_taken_in_ed_or_hosp",
    "ED": "ecg_taken_in_ed",
    "HOSP": "ecg_taken_in_hosp",
}

#: The release's own fold assignment, as its Usage Notes prescribe. Folds are
#: 0-indexed here, unlike ECGBench's 1-indexed folds — another reason not to mix
#: the two. See :func:`upstream_fold_split`.
UPSTREAM_FOLDS: dict[str, tuple[int, ...]] = {
    "train": tuple(range(0, 18)),
    "val": (18,),
    "test": (19,),
}

#: ``gender``'s missing marker. A string, not a null — see the module docstring.
MISSING_GENDER = "missing"

#: ``ecg_no_within_stay`` for a record with no ED or hospital stay to enumerate
#: within. Left as-is rather than nulled: it is a meaningful category, and it is
#: exactly the 331,907 records where ``ecg_taken_in_ed_or_hosp`` is False.
NO_STAY = -1

#: Codes are truncated to this many characters before superclass propagation, per
#: the upstream benchmark.
TRUNCATE_TO = 5

#: Shortest ICD-10-CM code, and so the shortest superclass worth propagating.
MIN_CODE_LENGTH = 3

#: Upstream drops codes appearing in fewer than this many records. With the other
#: defaults here that reproduces the published label set exactly: 1,076 codes
#: (361 three-character, 466 four-character, 249 five-character).
MIN_CODE_COUNT = 2000


def _require(path: Path) -> None:
    from ecgbench.labels import LabelSourceMissingError

    if not path.exists():
        raise LabelSourceMissingError(
            f"MIMIC-IV-ECG-Ext-ICD labels come from {SOURCE_CSV}, which is not in "
            f"{path.parent}. Point data_path at the release root — the directory "
            f"holding {SOURCE_CSV} — from "
            "https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/ . "
            "Access is credentialed under the PhysioNet Credentialed Health Data Use "
            "Agreement, so ECGBench never redistributes these labels."
        )


def parse_codes(value: object) -> list[str]:
    """Parse one ``"['I10', 'E785']"`` cell into a list of code strings.

    An empty list is the shipped value for "no linked discharge diagnosis" in
    41.5% of records, so it is returned as such rather than treated as missing.
    """
    if isinstance(value, list):
        return [str(c) for c in value]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text or text[0] != "[":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        logger.warning("Unparseable diagnosis cell %r; treating as empty", text[:60])
        return []
    return [str(c) for c in parsed] if isinstance(parsed, (list, tuple)) else []


def propagate_superclasses(
    codes: list[str] | tuple[str, ...],
    truncate: int = TRUNCATE_TO,
) -> list[str]:
    """Truncate each code, drop ICD-10 placeholder Xs, and add every superclass.

    Three steps, in the order the upstream benchmark applies them
    (``prepare_mimic_ecg`` in the authors' ``ECG-MIMIC`` repository):

    1. cut each code to ``truncate`` characters — ``'S72092A'`` -> ``'S7209'``;
    2. strip **trailing** ``X`` placeholders, which ICD-10-CM uses to pad a code
       to its full width — ``'W19XXXA'`` truncates to ``'W19XX'`` and then to
       ``'W19'``. Skipping this step is why a naive reconstruction of the
       published label set comes out 13 codes too large;
    3. add every prefix of length 3 or more, so a record coded ``'I2510'`` is also
       positive for ``'I251'`` and ``'I25'``.

    Codes shorter than three characters are dropped, having no ICD-10 category.

    Returns a sorted list, so the result is deterministic.
    """
    out: set[str] = set()
    for code in codes:
        text = str(code).strip()[:truncate].rstrip("X")
        for n in range(MIN_CODE_LENGTH, len(text) + 1):
            out.add(text[:n])
    return sorted(out)


def load_ext_icd(
    data_path: Path | str,
    columns: list[str] | tuple[str, ...] | None = None,
    parse: bool = True,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Return the Ext-ICD table indexed by ``study_id``, ready to join onto MIMIC-IV-ECG.

    Args:
        data_path: the MIMIC-IV-ECG-Ext-ICD release root (holds ``SOURCE_CSV``).
        columns: which columns to keep, or None for all of them. ``study_id``
            always becomes the index and is never returned as a column.
        parse: turn the five diagnosis columns from Python-literal strings into
            real lists. Costs about a minute over 800,035 rows; pass False when
            you only want the demographics or the fold columns.
        prefix: prepend this to every column name. Worth setting when joining
            onto MIMIC-IV-ECG's own label frame, which also carries ``ecg_time``
            — an unprefixed join silently keeps one of the two.

    ``gender``'s ``"missing"`` marker becomes NaN (4,489 records). Everything else
    is returned as shipped, including the empty diagnosis lists of the 332,030
    records with no linked discharge diagnosis and the ``ecg_no_within_stay``
    value of -1 for the 331,907 records with no ED or hospital stay.
    """
    path = Path(data_path) / SOURCE_CSV
    _require(path)

    usecols = None
    if columns is not None:
        usecols = [JOIN_COLUMN] + [c for c in columns if c != JOIN_COLUMN]

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if JOIN_COLUMN not in df.columns:
        raise ValueError(f"{path} has no '{JOIN_COLUMN}' column. Found: {list(df.columns)}")

    if "gender" in df.columns:
        n_missing = int((df["gender"] == MISSING_GENDER).sum())
        df["gender"] = df["gender"].replace(MISSING_GENDER, np.nan)
        if n_missing:
            logger.debug("gender: %d '%s' marker(s) set to NaN", n_missing, MISSING_GENDER)

    if parse:
        for column in DIAGNOSIS_COLUMNS:
            if column in df.columns:
                df[column] = df[column].map(parse_codes)

    df = df.set_index(JOIN_COLUMN)
    if columns is not None:
        df = df[[c for c in columns if c != JOIN_COLUMN]]
    if prefix:
        df = df.add_prefix(prefix)

    diagnosis_column = (prefix or "") + DEFAULT_DIAGNOSIS_COLUMN
    if parse and diagnosis_column in df.columns:
        n_labelled = int(df[diagnosis_column].map(bool).sum())
        logger.info(
            "Loaded MIMIC-IV-ECG-Ext-ICD for %d studies; %d (%.1f%%) carry at least one "
            "%s code, %d carry none",
            len(df),
            n_labelled,
            100 * n_labelled / max(len(df), 1),
            DEFAULT_DIAGNOSIS_COLUMN,
            len(df) - n_labelled,
        )
    else:
        logger.info("Loaded MIMIC-IV-ECG-Ext-ICD for %d studies", len(df))
    return df


def _resolve(df: pd.DataFrame, column: str, prefix: str = "") -> str:
    """Find ``column`` in ``df``, allowing for the ``prefix`` load_ext_icd added.

    Every function below takes the *logical* column name plus the ``prefix`` you
    passed to :func:`load_ext_icd`. Guessing the prefix from the frame is not an
    option — ``fold`` and ``strat_fold`` share a suffix, so a suffix match on a
    prefixed frame is ambiguous.
    """
    name = f"{prefix}{column}" if prefix and not column.startswith(prefix) else column
    if name in df.columns:
        return name
    raise ValueError(
        f"Column '{name}' not in frame. Found: {list(df.columns)}. Pass "
        "prefix= matching the one given to load_ext_icd(), and load with "
        "columns=None (or include this column) so it is present."
    )


def label_set(
    df: pd.DataFrame,
    column: str = DEFAULT_DIAGNOSIS_COLUMN,
    min_count: int = MIN_CODE_COUNT,
    truncate: int = TRUNCATE_TO,
    prefix: str = "",
) -> list[str]:
    """The upstream benchmark's label set: frequent codes after superclass expansion.

    Reproduces the published construction — truncate every code to ``truncate``
    characters, strip trailing placeholder Xs, propagate superclasses, skip records
    with no diagnosis, then keep codes appearing in at least ``min_count`` records.
    On the full shipped table with the defaults this yields the published label set
    exactly: **1,076** codes, 361 of three characters, 466 of four and 249 of five.

    Args:
        df: a frame from :func:`load_ext_icd` with ``parse=True``, or any row
            subset of one. Pass the whole table to reproduce the published set;
            pass a fold to see what that fold contains.
        column: which of :data:`DIAGNOSIS_COLUMNS` to count.
        min_count: minimum number of records a code must appear in.
        truncate: passed to :func:`propagate_superclasses`.
        prefix: the prefix given to :func:`load_ext_icd`, if any.

    Returns:
        Codes ordered by descending record count, then alphabetically — so the
        order is stable and index 0 is the most common code.
    """
    name = _resolve(df, column, prefix)
    counts = _code_counts(df, name, truncate)
    kept = {code: n for code, n in counts.items() if n >= min_count}
    logger.info(
        "%s: %d distinct codes after truncate=%d + superclasses; %d occur in >=%d records",
        name,
        len(counts),
        truncate,
        len(kept),
        min_count,
    )
    return sorted(kept, key=lambda code: (-kept[code], code))


def _code_counts(df: pd.DataFrame, column: str, truncate: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for codes in df[column]:
        parsed = codes if isinstance(codes, list) else parse_codes(codes)
        if parsed:
            counts.update(propagate_superclasses(parsed, truncate))
    return counts


def multi_hot(
    df: pd.DataFrame,
    codes: list[str] | tuple[str, ...],
    column: str = DEFAULT_DIAGNOSIS_COLUMN,
    truncate: int = TRUNCATE_TO,
    prefix: str = "",
) -> pd.DataFrame:
    """Multi-hot targets for ``codes``, one row per row of ``df``.

    Superclasses are propagated first, so a record coded ``I2510`` is positive
    for ``I25``, ``I251`` and ``I2510`` alike.

    Pass a **row subset** — a fold, not the whole table. The full 800,035 records
    against the 1,076-code label set is a 861-million-cell frame, and that is
    rarely what you want in memory at once.

    Records with no diagnosis come back all-zero. They are 41.5% of the shipped
    table, so filter them out before training unless "no linked discharge
    diagnosis" is genuinely a negative for your task. A record *with* diagnoses can
    also come back all-zero if every one of its codes fell below ``min_count`` when
    the label set was built, so a positive count is the number to check rather than
    the number of records carrying codes.
    """
    name = _resolve(df, column, prefix)
    order = list(codes)
    position = {code: i for i, code in enumerate(order)}
    matrix = np.zeros((len(df), len(order)), dtype=np.int8)
    for row, cell in enumerate(df[name]):
        parsed = cell if isinstance(cell, list) else parse_codes(cell)
        for code in propagate_superclasses(parsed, truncate):
            index = position.get(code)
            if index is not None:
                matrix[row, index] = 1
    return pd.DataFrame(matrix, index=df.index, columns=order)


def ecg_subset(df: pd.DataFrame, subset: str, prefix: str = "") -> pd.DataFrame:
    """Rows of one upstream ECG subset: ``"ALL"``, ``"ED"`` or ``"HOSP"``.

    These are the A and C of the benchmark's T(A2B)-E(C2D) scenario notation:
    ALL is every ECG taken during an ED *or* hospital stay (468,128 records), ED
    only (184,720) and hospital only (298,258). The three overlap — an ECG taken
    in the ED of a patient who was then admitted counts in both ED and HOSP.

    Note that ALL is *not* "every record": the 331,907 records taken outside any
    ED or hospital stay are in no subset, which is close to but not identical
    with the 332,030 that carry no diagnosis code.
    """
    key = subset.upper()
    if key not in ECG_SUBSETS:
        raise ValueError(f"subset must be one of {sorted(ECG_SUBSETS)}, got {subset!r}")
    column = _resolve(df, ECG_SUBSETS[key], prefix)
    return df[df[column].astype(bool)]


def upstream_fold_split(
    df: pd.DataFrame,
    split: str,
    stratified: bool = False,
    prefix: str = "",
) -> pd.DataFrame:
    """Rows of the release's **own** train/val/test split — not ECGBench's.

    The upstream benchmark assigns patients to 20 folds and uses 0-17 for
    training, 18 for validation and 19 for test. Folds are patient-grouped
    (0 of 161,352 subjects span one), so the partition is leakage-free *on its own
    terms*.

    It is not, however, ECGBench's partition, and the two are independent rather
    than merely different: 79.3% of the upstream test fold's 39,569 records fall
    inside the ECGBench ``mimic_iv_ecg`` train split, and 6,449 of its 8,067
    patients appear there. Use this function to reproduce published numbers, and
    ``ECGDataset(..., split=...)`` to work on ECGBench's folds. Never mix them —
    selecting training data by one partition and evaluating by the other puts the
    same patient on both sides.

    Args:
        df: a frame from :func:`load_ext_icd` carrying the fold column.
        split: ``"train"``, ``"val"`` or ``"test"``.
        stratified: use ``strat_fold`` (multi-label stratified, and fixed in
            v1.0.1) instead of ``fold`` (random, and what the original benchmark
            actually used).
        prefix: the prefix given to :func:`load_ext_icd`, if any.
    """
    if split not in UPSTREAM_FOLDS:
        raise ValueError(f"split must be one of {sorted(UPSTREAM_FOLDS)}, got {split!r}")
    column = _resolve(df, "strat_fold" if stratified else "fold", prefix)
    return df[df[column].isin(UPSTREAM_FOLDS[split])]
