"""
Symile-MIMIC — a multimodal cohort **over MIMIC-IV-ECG's records**.

Symile-MIMIC pairs each of 11,622 hospital admissions with three modalities drawn
from MIMIC-IV: a chest X-ray taken 24-72 h post-admission (MIMIC-CXR-JPG), an ECG
taken within 24 h of admission (MIMIC-IV-ECG), and up to 50 blood-lab values
(MIMIC-IV hosp). It was built for the NeurIPS 2024 Symile paper's CXR-retrieval
task, where a query's CXR must be picked out of 10 candidates given the ECG and
the labs.

**The ECGs are MIMIC-IV-ECG's own recordings**, verified from the files: all
11,610 distinct ``ecg_study_id`` values are MIMIC-IV-ECG ``study_id`` values with
none extra, all 9,573 ``subject_id`` values are MIMIC-IV-ECG subjects, and
``ecg_path``, ``ecg_time`` and ``subject_id`` agree with ``record_list.csv`` in
100% of rows. The shipped ``data_npy/*/ecg_*.npy`` tensors are those same
waveforms min-max normalised (see :func:`load_split_tensors`), not new recordings.

**There is deliberately no ``symile_mimic`` dataset config.** Every ECG here is a
MIMIC-IV-ECG record, so generating a ten-fold ECGBench partition would create a
*second* partition over recordings the ``mimic_iv_ecg`` config already partitions,
and a user who trained on one and evaluated on the other would be testing on
training data. So this module is a **cohort and label provider**: you load
MIMIC-IV-ECG on its own ECGBench folds and join these columns onto it.
:func:`by_study_id` returns a frame indexed by ``study_id`` for exactly that.

Six properties of the release that change what you can do with it, all recomputed
from the files (every one of the 40 shipped files matches the release's own
``SHA256SUMS.txt``):

1. **The row unit is the admission, not the ECG.** ``hadm_id`` is unique across
   all 11,622 rows; ``ecg_study_id`` is not — 12 ECG studies each serve two
   admissions, always two admissions of the same patient hours or a day apart.
   That is why :func:`load_cohort` keys by ``hadm_id`` and :func:`by_study_id`
   makes the de-duplication policy explicit instead of silently dropping rows.
2. **The column literally named ``study_id`` is the CXR's, not the ECG's.**
   It is byte-identical to ``cxr_study_id`` in 100% of rows, while the ECG key —
   the one MIMIC-IV-ECG calls ``study_id`` — is ``ecg_study_id``. Joining
   MIMIC-IV-ECG on the shipped ``study_id`` matches nothing. :func:`load_cohort`
   drops the duplicate rather than leave the trap in place.
3. **The release's own train/val/test split is not ECGBench's, and the two are
   independent.** Symile's splits are patient-disjoint on their own terms (0
   subjects shared between any pair), but 75.6% of the 464 test studies land in
   the ECGBench ``mimic_iv_ecg`` *train* split, and 349 of the 461 test subjects
   appear there. Pick one partition and stay inside it; see :data:`SPLIT_CSVS`.
4. **The split CSVs are not a partition of the full table, and they drop
   columns.** 10,000 + 750 + 464 = 11,214 of 11,622 admissions appear in a split;
   the other 408 were discarded by the patient-disjointness filter. The split
   CSVs also carry only 6 of the 14 CheXpert labels and no demographics,
   ``ecg_study_id``, ``ecg_time`` or CXR metadata — those live only in
   ``symile_mimic_data.csv``. Use :func:`load_cohort` for metadata and
   :func:`load_split` for split membership.
5. **CheXpert labels use CheXpert's four-state encoding**, not booleans: 1.0
   positive, 0.0 negative, **-1.0 uncertain**, and NaN "not mentioned in the
   report". Reading -1.0 as negative or NaN as negative are different modelling
   choices with different results, so :func:`chexpert_targets` makes you choose.
6. **Timestamps and ages are de-identification artefacts.** MIMIC-IV shifts dates
   into the future (``ecg_time`` spans 2110-2208) and caps ``anchor_age`` at 91
   for everyone over 89 — 580 rows sit at exactly 91, the file maximum, and
   ``age`` reaches 100. Do not read an age above 89 as a real age.

The labs carry **no sentinel rails** — unlike MIMIC-IV-ECG's
``machine_measurements.csv``, missing labs here are genuine NaNs, so
``notna()`` is the right test. A handful of values are nonetheless
physiologically impossible rather than missing (10 of 5,254 Base Excess values
are below -30, the minimum being -413); they are MIMIC-IV data-entry errors
passed through unchanged, so filter on a physiologic range if you train on them.

Neither the labs, the CheXpert labels nor the waveforms are ever published to the
HuggingFace Hub: the source is credentialed under the PhysioNet Credentialed
Health Data Use Agreement, so it stays with the people who signed it. You need
local copies of **both** Symile-MIMIC and MIMIC-IV-ECG — this release has no
ECGBench folds, and MIMIC-IV-ECG has none of these columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The dataset whose recordings these are, and whose ECGBench folds to use.
HOST_CONFIG_SLUG = "mimic_iv_ecg"

#: MIMIC-IV-ECG's record identifier — the key to join on, and what
#: :func:`by_study_id` indexes by.
JOIN_COLUMN = "study_id"

#: The release's own row key: one row per hospital admission. Unique across all
#: 11,622 rows, which ``ecg_study_id`` is not.
ROW_KEY = "hadm_id"

#: Where the ECG key actually lives in the shipped CSVs. See point 2 above.
ECG_STUDY_COLUMN = "ecg_study_id"

#: The shipped column named ``study_id``, which is the **CXR** study id and a
#: byte-identical duplicate of ``cxr_study_id``. :func:`load_cohort` drops it.
AMBIGUOUS_STUDY_COLUMN = "study_id"
CXR_STUDY_COLUMN = "cxr_study_id"

#: The full cohort table: 11,622 admissions x 94 columns, all the metadata.
SOURCE_CSV = "symile_mimic_data.csv"

#: The release's own splits, and their shipped row counts. ``val_retrieval`` and
#: ``test`` are query x candidate expansions — 750 and 464 queries respectively,
#: each with :data:`RETRIEVAL_CANDIDATES` candidates — so their row counts are not
#: admission counts. See :func:`retrieval_queries`.
SPLIT_CSVS: dict[str, str] = {
    "train": "train.csv",
    "val": "val.csv",
    "val_retrieval": "val_retrieval.csv",
    "test": "test.csv",
}

#: Rows per split CSV as shipped, for a sanity check on a local copy.
SPLIT_SIZES: dict[str, int] = {
    "train": 10000,
    "val": 750,
    "val_retrieval": 7500,
    "test": 4640,
}

#: Distinct admissions per split — what ``val_retrieval`` and ``test`` actually
#: cover, as opposed to their expanded row counts.
SPLIT_ADMISSIONS: dict[str, int] = {
    "train": 10000,
    "val": 750,
    "val_retrieval": 750,
    "test": 464,
}

#: Candidates per retrieval query: 1 positive (the query itself) and 9 negatives
#: sampled from the same split.
RETRIEVAL_CANDIDATES = 10

#: The 50 most common MIMIC-IV blood labs, itemid -> name, exactly as the
#: release's own ``code/constants.py`` lists them. Three itemids carry the
#: single-letter names MIMIC-IV gives them (``H``, ``L``, ``I``) and are the
#: sparsest columns here at 10.5% coverage.
LABS: dict[str, str] = {
    "51221": "Hematocrit",
    "51265": "Platelet Count",
    "50912": "Creatinine",
    "50971": "Potassium",
    "51222": "Hemoglobin",
    "51301": "White Blood Cells",
    "51249": "MCHC",
    "51279": "Red Blood Cells",
    "51250": "MCV",
    "51248": "MCH",
    "51277": "RDW",
    "51006": "Urea Nitrogen",
    "50983": "Sodium",
    "50902": "Chloride",
    "50882": "Bicarbonate",
    "50868": "Anion Gap",
    "50931": "Glucose",
    "50960": "Magnesium",
    "50893": "Calcium, Total",
    "50970": "Phosphate",
    "51237": "INR(PT)",
    "51274": "PT",
    "51275": "PTT",
    "51146": "Basophils",
    "51256": "Neutrophils",
    "51254": "Monocytes",
    "51200": "Eosinophils",
    "51244": "Lymphocytes",
    "52172": "RDW-SD",
    "50934": "H",
    "51678": "L",
    "50947": "I",
    "50861": "Alanine Aminotransferase (ALT)",
    "50878": "Asparate Aminotransferase (AST)",
    "50813": "Lactate",
    "50863": "Alkaline Phosphatase",
    "50885": "Bilirubin, Total",
    "50820": "pH",
    "50862": "Albumin",
    "50802": "Base Excess",
    "50821": "pO2",
    "50804": "Calculated Total CO2",
    "50818": "pCO2",
    "52075": "Absolute Neutrophil Count",
    "52073": "Absolute Eosinophil Count",
    "52074": "Absolute Monocyte Count",
    "52069": "Absolute Basophil Count",
    "51133": "Absolute Lymphocyte Count",
    "50910": "Creatine Kinase (CK)",
    "52135": "Immature Granulocytes",
}

#: Suffix the release appends for a lab's train-set ECDF percentile. Present in
#: the split CSVs only — ``symile_mimic_data.csv`` carries raw values alone.
PERCENTILE_SUFFIX = "_percentile"

#: The three lab representations, mapped to how :func:`labs_frame` builds each.
LAB_KINDS = ("value", "percentile", "missingness")

#: The 14 CheXpert findings, as MIMIC-CXR-JPG's ``chexpert.csv`` names them.
#: All 14 are in :data:`SOURCE_CSV`; only :data:`SPLIT_CHEXPERT_LABELS` survive
#: into the split CSVs.
CHEXPERT_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
)

#: The 6 CheXpert labels the release's ``create_dataset_splits.py`` happens not to
#: drop. The other 8 exist only in :data:`SOURCE_CSV`.
SPLIT_CHEXPERT_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
)

#: CheXpert's encoding. NaN — "the radiology report does not mention this
#: finding" — is the majority state for most findings and is *not* in this dict,
#: because it is a real category rather than a value.
CHEXPERT_CODES: dict[float, str] = {
    1.0: "positive",
    0.0: "negative",
    -1.0: "uncertain",
}

#: MIMIC-IV-ECG's lead order, which these waveforms inherit. **aVF comes before
#: aVL** — ``signal[4]`` is aVF here, not aVL as in PTB-XL or Chapman. Pass
#: ``leads=`` to ``ECGDataset`` to select by name instead of position.
ECG_LEAD_NAMES: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVF",
    "aVL",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

#: Modality -> the ``data_npy/<split>/<modality>_<split>.npy`` file it names.
#: ``label`` and ``label_hadm_id`` exist for ``test`` and ``val_retrieval`` only.
TENSOR_MODALITIES: tuple[str, ...] = (
    "ecg",
    "cxr",
    "labs_percentiles",
    "labs_missingness",
    "hadm_id",
    "label",
    "label_hadm_id",
)

#: Splits whose ``data_npy`` directory carries the retrieval label tensors.
RETRIEVAL_SPLITS: tuple[str, ...] = ("val_retrieval", "test")

_PHYSIONET_URL = "https://physionet.org/content/symile-mimic/1.0.0/"


def _require(path: Path, what: str) -> None:
    from ecgbench.labels import LabelSourceMissingError

    if not path.exists():
        raise LabelSourceMissingError(
            f"Symile-MIMIC {what} comes from {path.name}, which is not in "
            f"{path.parent}. Point data_path at the release root — the directory "
            f"holding {SOURCE_CSV} — from {_PHYSIONET_URL} . Access is credentialed "
            "under the PhysioNet Credentialed Health Data Use Agreement, so ECGBench "
            "never redistributes this cohort."
        )


def _drop_ambiguous_study_id(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the shipped ``study_id`` column, which is the CXR's, not the ECG's.

    It duplicates ``cxr_study_id`` exactly in the shipped file, so dropping it
    loses nothing and removes a column whose name invites a join that matches no
    MIMIC-IV-ECG record. If a future release ever makes the two disagree, both are
    kept and a warning says so rather than discarding data on an assumption.
    """
    if AMBIGUOUS_STUDY_COLUMN not in df.columns:
        return df
    if CXR_STUDY_COLUMN not in df.columns:
        logger.warning(
            "'%s' is present without '%s'; keeping it, but note it is the CXR study "
            "id, not MIMIC-IV-ECG's — join on '%s' instead",
            AMBIGUOUS_STUDY_COLUMN,
            CXR_STUDY_COLUMN,
            ECG_STUDY_COLUMN,
        )
        return df
    if not df[AMBIGUOUS_STUDY_COLUMN].equals(df[CXR_STUDY_COLUMN]):
        logger.warning(
            "'%s' and '%s' disagree in this copy; keeping both. '%s' is the CXR "
            "study id — MIMIC-IV-ECG's key is '%s'",
            AMBIGUOUS_STUDY_COLUMN,
            CXR_STUDY_COLUMN,
            AMBIGUOUS_STUDY_COLUMN,
            ECG_STUDY_COLUMN,
        )
        return df
    return df.drop(columns=[AMBIGUOUS_STUDY_COLUMN])


def _index_by_row_key(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    if ROW_KEY not in df.columns:
        raise ValueError(f"{source} has no '{ROW_KEY}' column. Found: {list(df.columns)}")
    df = df.set_index(ROW_KEY)
    if df.index.has_duplicates:
        n = int(df.index.duplicated().sum())
        raise ValueError(
            f"{source} has {n} duplicate '{ROW_KEY}' values, which the shipped file "
            "does not — this copy has been altered or concatenated."
        )
    return df


def load_cohort(
    data_path: Path | str,
    columns: list[str] | tuple[str, ...] | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Return the full cohort table indexed by ``hadm_id``.

    This is ``symile_mimic_data.csv``: 11,622 admissions and every column the
    release ships, including the metadata its split CSVs drop — all 14 CheXpert
    labels, demographics, race, date of death, CXR view position, ``ecg_time`` and
    ``ecg_study_id``.

    Args:
        data_path: the Symile-MIMIC release root (holds :data:`SOURCE_CSV`).
        columns: which columns to keep, or None for all of them. ``hadm_id``
            always becomes the index and is never returned as a column.
        prefix: prepend this to every column name. Worth setting when joining
            onto MIMIC-IV-ECG's own label frame, which also carries ``subject_id``
            and ``ecg_time`` — an unprefixed join silently keeps one of the two.

    The shipped ``study_id`` column is dropped: it is the **CXR** study id, a
    duplicate of ``cxr_study_id``, and joining MIMIC-IV-ECG on it matches nothing.
    The ECG key stays under its own name, ``ecg_study_id``; pass the result to
    :func:`by_study_id` to index by it.

    Everything else is returned as shipped, NaNs and all: missing labs are genuine
    NaNs, and a NaN CheXpert label means "not mentioned in the report".
    """
    path = Path(data_path) / SOURCE_CSV
    _require(path, "cohort metadata")

    usecols = None
    if columns is not None:
        wanted = [c for c in columns if c != ROW_KEY]
        usecols = [ROW_KEY] + wanted
        # study_id is dropped below, but only safely if its twin came along too.
        if AMBIGUOUS_STUDY_COLUMN in wanted and CXR_STUDY_COLUMN not in usecols:
            usecols.append(CXR_STUDY_COLUMN)

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = _drop_ambiguous_study_id(df)
    df = _index_by_row_key(df, path)

    if columns is not None:
        df = df[[c for c in columns if c != ROW_KEY and c in df.columns]]
    if prefix:
        df = df.add_prefix(prefix)

    logger.info("Loaded Symile-MIMIC cohort: %d admissions, %d columns", len(df), df.shape[1])
    return df


def _derive_study_id(paths: pd.Series) -> pd.Series:
    """Recover MIMIC-IV-ECG's ``study_id`` from an ``ecg_path``.

    The split CSVs drop ``ecg_study_id`` but keep ``ecg_path``, whose last segment
    is the record name — and MIMIC-IV-ECG's ``record_list.csv`` sets ``file_name``
    equal to ``study_id`` and ``path`` to a directory ending in it. Verified
    against both files: the stem equals ``ecg_study_id`` in 100% of the 11,622
    shipped rows, and ``ecg_path`` equals ``record_list.csv``'s ``path`` in 100%.

    A non-numeric stem means the path layout changed, so this raises rather than
    producing NaN keys that would silently fail to join.
    """
    stems = paths.astype(str).str.rsplit("/", n=1).str[-1]
    bad = stems[~stems.str.fullmatch(r"\d+")]
    if len(bad):
        raise ValueError(
            f"{len(bad)} 'ecg_path' value(s) do not end in a numeric MIMIC-IV-ECG "
            f"study_id, e.g. {paths.loc[bad.index[0]]!r}. Load the cohort table with "
            f"load_cohort() and take '{ECG_STUDY_COLUMN}' from it instead."
        )
    return stems.astype("int64")


def load_split(
    data_path: Path | str,
    split: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Return one of the release's **own** splits, indexed by ``hadm_id``.

    An ``ecg_study_id`` column is added, recovered from ``ecg_path`` (see
    :func:`_derive_study_id`), because the split CSVs drop the shipped one — and
    without it there is no way to line the split up against MIMIC-IV-ECG.

    Args:
        data_path: the Symile-MIMIC release root.
        split: one of :data:`SPLIT_CSVS` — ``"train"``, ``"val"``,
            ``"val_retrieval"`` or ``"test"``.
        prefix: prepend this to every column name (the added ``ecg_study_id``
            included).

    ``val_retrieval`` and ``test`` are query x candidate expansions, so they hold
    10 rows per query and ``hadm_id`` repeats — those two are indexed by a
    ``(label_hadm_id, hadm_id)`` MultiIndex instead, positive candidate first. Use
    :func:`retrieval_queries` for one row per query.

    **This is not ECGBench's partition.** These splits are patient-disjoint among
    themselves but statistically independent of the ``mimic_iv_ecg`` folds: 75.6%
    of the test studies sit inside ECGBench's train split. Use one or the other,
    never both.

    ``val.csv`` additionally ships ``label`` and ``label_hadm_id`` columns that are
    constant (``label == 1``, ``label_hadm_id == hadm_id``) — a side effect of the
    release's split script mutating the frame in place before writing it. They
    carry no information; ``val`` is a plain validation set, and its ``data_npy``
    directory correctly has no label tensor.
    """
    if split not in SPLIT_CSVS:
        raise ValueError(f"split must be one of {sorted(SPLIT_CSVS)}, got {split!r}")

    path = Path(data_path) / SPLIT_CSVS[split]
    _require(path, f"the {split!r} split")

    df = pd.read_csv(path, low_memory=False)
    if "ecg_path" not in df.columns:
        raise ValueError(f"{path} has no 'ecg_path' column. Found: {list(df.columns)}")
    # concat rather than df[col] = ..., then copy to consolidate: these frames are
    # 110 columns wide, and leaving them fragmented makes every later reset_index
    # warn about it.
    studies = _derive_study_id(df["ecg_path"]).rename(ECG_STUDY_COLUMN)
    df = pd.concat([df, studies], axis=1).copy()

    if split in RETRIEVAL_SPLITS:
        df = _index_retrieval(df, path)
    else:
        df = _index_by_row_key(df, path)

    if prefix:
        df = df.add_prefix(prefix)

    expected = SPLIT_SIZES[split]
    if len(df) != expected:
        logger.warning("%s holds %d rows; the shipped release has %d", path.name, len(df), expected)
    logger.info("Loaded Symile-MIMIC %s: %d rows", split, len(df))
    return df


def _index_retrieval(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Index a retrieval split by ``(label_hadm_id, hadm_id)``, positive first."""
    for column in ("label_hadm_id", "label"):
        if column not in df.columns:
            raise ValueError(
                f"{source} has no '{column}' column, so its candidates cannot be "
                f"grouped by query. Found: {list(df.columns)}"
            )
    df = df.sort_values(["label_hadm_id", "label"], ascending=[True, False], kind="stable")
    return df.set_index(["label_hadm_id", ROW_KEY])


def by_study_id(
    df: pd.DataFrame,
    on_duplicate: str = "earliest_admission",
    prefix: str = "",
) -> pd.DataFrame:
    """Reindex a cohort or split frame by MIMIC-IV-ECG's ``study_id``.

    This is what makes the join onto MIMIC-IV-ECG's ECGBench folds possible::

        cohort = load_cohort(symile_path)
        joined = by_study_id(cohort).reindex(ds.metadata_df["study_id"].values)

    12 ECG studies serve two admissions each in the full table (10 of them inside
    the train split), so a ``study_id`` index is not unique as shipped and one row
    of each pair has to go. That choice is yours to make explicitly:

    Args:
        df: a frame from :func:`load_cohort` or :func:`load_split`.
        on_duplicate:
            - ``"earliest_admission"`` (default) — keep the row with the earliest
              ``admittime``, falling back to the lowest ``hadm_id`` when the frame
              has no ``admittime`` column (the split CSVs drop it). Deterministic
              either way, and never row order.
            - ``"raise"`` — refuse, naming the offending studies.
            - ``"keep_all"`` — return the duplicates. The index is then not unique
              and ``reindex`` will multiply rows, so only useful for inspection.
        prefix: the prefix given to :func:`load_cohort` / :func:`load_split`.

    The returned index is named ``study_id`` — MIMIC-IV-ECG's name for it — and the
    ``ecg_study_id`` column is consumed by it. ``hadm_id`` becomes a column, so
    nothing is lost; it picks up ``prefix`` on the way, since as an index name it
    had escaped the loader's ``add_prefix``.
    """
    if on_duplicate not in {"earliest_admission", "raise", "keep_all"}:
        raise ValueError(
            "on_duplicate must be 'earliest_admission', 'raise' or 'keep_all', "
            f"got {on_duplicate!r}"
        )

    key = _resolve(df, ECG_STUDY_COLUMN, prefix)
    out = df.reset_index()
    duplicated = out[key].duplicated(keep=False)
    n_dupes = int(duplicated.sum())

    if n_dupes and on_duplicate == "raise":
        studies = sorted(int(s) for s in out.loc[duplicated, key].unique())
        raise ValueError(
            f"{len(studies)} ECG study/studies serve more than one admission "
            f"({n_dupes} rows): {studies[:5]}{' ...' if len(studies) > 5 else ''}. "
            "Pass on_duplicate='earliest_admission' to keep one row each, or "
            "'keep_all' to inspect them."
        )

    if n_dupes and on_duplicate == "earliest_admission":
        n_studies = int(out.loc[duplicated, key].nunique())
        admittime = f"{prefix}admittime" if prefix else "admittime"
        if admittime in out.columns:
            order, by = pd.to_datetime(out[admittime], errors="coerce"), admittime
        else:
            order, by = out[ROW_KEY], ROW_KEY
        before = len(out)
        out = (
            out.assign(_order=order)
            .sort_values([key, "_order"], kind="stable")
            .drop_duplicates(subset=[key], keep="first")
            .drop(columns="_order")
            .sort_index()
        )
        logger.info(
            "%d ECG study/studies served more than one admission; dropped %d row(s), "
            "keeping the earliest by %s",
            n_studies,
            before - len(out),
            by,
        )

    out = out.set_index(key)
    out.index.name = JOIN_COLUMN
    # hadm_id was the index and so escaped add_prefix; prefix it now that it is a
    # column, or the frame would be 91 prefixed columns and one bare one.
    if prefix and ROW_KEY in out.columns and f"{prefix}{ROW_KEY}" not in out.columns:
        out = out.rename(columns={ROW_KEY: f"{prefix}{ROW_KEY}"})
    return out


def _resolve(df: pd.DataFrame, column: str, prefix: str = "") -> str:
    """Find ``column`` in ``df``, allowing for the ``prefix`` a loader added.

    Every helper takes the *logical* column name plus the ``prefix`` you passed to
    the loader. Guessing the prefix from the frame is not an option here: the 50
    lab columns and their 50 ``_percentile`` twins share suffixes, so a suffix
    match on a prefixed frame is ambiguous.
    """
    name = f"{prefix}{column}" if prefix and not column.startswith(prefix) else column
    if name in df.columns:
        return name
    raise ValueError(
        f"Column '{name}' not in frame. Found {len(df.columns)} columns "
        f"including {list(df.columns)[:8]}. Pass prefix= matching the one given to "
        "load_cohort()/load_split(), and load with columns=None (or include this "
        "column) so it is present."
    )


def labs_frame(
    df: pd.DataFrame,
    kind: str = "value",
    names: bool = False,
    prefix: str = "",
) -> pd.DataFrame:
    """The 50 blood labs as one frame, in the release's own itemid order.

    Args:
        df: a frame from :func:`load_cohort` or :func:`load_split`.
        kind: which representation —
            - ``"value"`` — the raw lab values, as measured. NaN where the lab was
              not drawn. Available in both the cohort table and the split CSVs.
            - ``"percentile"`` — the release's train-set NaN-aware ECDF
              percentiles, in [0, 1]. **Split CSVs only**; the cohort table has no
              ``_percentile`` columns.
            - ``"missingness"`` — 1 where the lab was measured, 0 where it was
              not, derived from ``"value"``. This is the second half of the
              100-dimensional labs vector the Symile paper feeds its model.
        names: label the columns with the lab names from :data:`LABS`
            (``"Hematocrit"``) instead of the itemids (``"51221"``).
        prefix: the prefix given to the loader.

    Coverage is uneven and worth looking at before you use these as features:
    Creatinine is present for 99.8% of admissions and the three single-letter
    itemids (``H``, ``L``, ``I``) for 10.5%. The mean admission has 35.0 of the 50,
    the minimum 1 and the maximum all 50 — every admission has at least one, which
    is what the release's ``labs_all_nan`` column records (0 for all 11,622 rows).

    Note the paper does **not** feed these percentiles to the model as-is: an
    unmeasured lab is replaced by that lab's train-set mean percentile from
    ``labs_means.json``, and the missingness indicator marks it. See
    :func:`load_split_tensors`, which returns the tensors that substitution
    produced.
    """
    if kind not in LAB_KINDS:
        raise ValueError(f"kind must be one of {list(LAB_KINDS)}, got {kind!r}")

    suffix = PERCENTILE_SUFFIX if kind == "percentile" else ""
    columns, missing = [], []
    for itemid in LABS:
        try:
            columns.append(_resolve(df, f"{itemid}{suffix}", prefix))
        except ValueError:
            missing.append(f"{itemid}{suffix}")
    # All 50 are present in every shipped file, so report the whole gap at once
    # rather than sending the caller round the loop 50 times.
    if missing:
        hint = (
            "the cohort table has raw values only — percentiles are derived per "
            "split, so load_split() is where they live"
            if kind == "percentile"
            else "pass columns=None to load_cohort()/load_split(), or prefix= matching "
            "the one you gave it"
        )
        raise ValueError(
            f"{len(missing)} of the {len(LABS)} lab columns are missing from this "
            f"frame, starting with {missing[:3]}: {hint}."
        )
    out = df[columns].copy()
    if kind == "missingness":
        out = out.notna().astype("int8")
    out.columns = list(LABS.values()) if names else list(LABS)
    return out


def chexpert_targets(
    df: pd.DataFrame,
    labels: list[str] | tuple[str, ...] | None = None,
    uncertain: str = "nan",
    not_mentioned: str = "negative",
    prefix: str = "",
) -> pd.DataFrame:
    """CheXpert findings as targets, with the two ambiguous states resolved by you.

    The shipped columns are CheXpert's four-state encoding — 1.0 positive, 0.0
    negative, -1.0 **uncertain**, NaN **not mentioned in the report** — and NaN is
    the majority state for most findings (64.2% for Atelectasis, 98.8% for
    Pleural Other). Treating -1.0 as negative, as positive, or as missing are
    three different modelling choices, as are the two readings of NaN, so this
    function has no default that hides them.

    Args:
        df: a frame from :func:`load_cohort` (all 14 findings) or
            :func:`load_split` (the 6 of :data:`SPLIT_CHEXPERT_LABELS`).
        labels: which findings to return. Defaults to whichever of
            :data:`CHEXPERT_LABELS` the frame actually carries.
        uncertain: what -1.0 becomes — ``"nan"`` (default), ``"positive"``,
            ``"negative"``, or ``"keep"`` to leave the -1.0 in place.
        not_mentioned: what NaN becomes — ``"negative"`` (default, the usual
            CheXpert convention) or ``"nan"`` to keep it missing.
        prefix: the prefix given to the loader.

    Returns:
        A float frame, one column per finding, on ``df``'s index. Float rather
        than int because both ``uncertain="nan"`` and ``not_mentioned="nan"`` can
        leave NaNs behind.

    ``"No Finding"`` is never 0.0 or -1.0 in this cohort — it is 1.0 for 1,606
    admissions and NaN for the other 10,016 — so with the defaults it becomes a
    plain binary column and its NaNs are the negatives.
    """
    if uncertain not in {"nan", "positive", "negative", "keep"}:
        raise ValueError(
            f"uncertain must be 'nan', 'positive', 'negative' or 'keep', got {uncertain!r}"
        )
    if not_mentioned not in {"negative", "nan"}:
        raise ValueError(f"not_mentioned must be 'negative' or 'nan', got {not_mentioned!r}")

    if labels is None:
        wanted = [c for c in CHEXPERT_LABELS if _has(df, c, prefix)]
        if not wanted:
            raise ValueError(
                "None of the 14 CheXpert findings are in this frame. load_cohort() "
                "carries all 14; the split CSVs carry only "
                f"{list(SPLIT_CHEXPERT_LABELS)}."
            )
    else:
        wanted = list(labels)

    columns = {}
    for label in wanted:
        series = df[_resolve(df, label, prefix)].astype("float64")
        # Which rows were *shipped* as NaN, before the uncertain mapping can add
        # more. Filling afterwards without this mask would quietly turn
        # uncertain="nan" into uncertain="negative".
        not_in_report = series.isna()
        if uncertain != "keep":
            replacement = {"nan": np.nan, "positive": 1.0, "negative": 0.0}[uncertain]
            series = series.mask(series == -1.0, replacement)
        if not_mentioned == "negative":
            series = series.mask(not_in_report, 0.0)
        columns[label] = series
    return pd.DataFrame(columns, index=df.index)


def _has(df: pd.DataFrame, column: str, prefix: str) -> bool:
    name = f"{prefix}{column}" if prefix and not column.startswith(prefix) else column
    return name in df.columns


def retrieval_queries(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """The positive row of each retrieval query — one row per query, not per candidate.

    ``test.csv`` and ``val_retrieval.csv`` hold :data:`RETRIEVAL_CANDIDATES` rows
    per query: the query itself (``label == 1``) plus 9 negatives sampled from the
    same split (``label == 0``). This returns the positives, so ``test`` collapses
    from 4,640 rows to its 464 real admissions and ``val_retrieval`` from 7,500 to
    750.

    The candidate pool is the split itself — every one of the 464 test admissions
    appears both as a query and as a negative candidate for other queries — so the
    negatives add no records, only pairings.

    Returns a frame indexed by ``hadm_id`` (the MultiIndex from
    :func:`load_split` is collapsed, since query and candidate coincide here).
    """
    label = _resolve(df, "label", prefix)
    positives = df[df[label] == 1]
    if isinstance(positives.index, pd.MultiIndex):
        positives = positives.reset_index(level=0, drop=True)
    if positives.index.has_duplicates:
        raise ValueError(
            "More than one positive candidate per query; this frame is not one of "
            "the release's retrieval splits."
        )
    return positives


def load_split_tensors(
    data_path: Path | str,
    split: str,
    modality: str,
    mmap: bool = True,
) -> tuple[np.ndarray, pd.Index]:
    """One preprocessed tensor from ``data_npy/``, with the ``hadm_id`` it is aligned to.

    Args:
        data_path: the Symile-MIMIC release root.
        split: one of :data:`SPLIT_CSVS`.
        modality: one of :data:`TENSOR_MODALITIES`.
        mmap: memory-map instead of reading into RAM. On by default — the four CXR
            tensors are 27 GB between them.

    Returns:
        ``(array, hadm_id)`` where ``hadm_id`` is a pandas Index read from the
        release's own ``hadm_id_<split>.npy``, so ``array[i]`` belongs to
        ``hadm_id[i]``. That alignment is the release's, not inferred: verified to
        equal the split CSV's row order in all four splits.

    **The ECG tensors are not in millivolts and the transform is not invertible.**
    Each record is min-max normalised to [-1, 1] over the whole 12-lead array at
    once — ``2 * (x - x.min()) / (x.max() - x.min()) - 1``, verified to float32
    precision against ``wfdb.rdrecord`` on the corresponding MIMIC-IV-ECG record —
    and the per-record min and max are not shipped, so millivolts cannot be
    recovered. They are also stored channel-last with a leading singleton:
    ``(n, 1, 5000, 12)``, not ECGBench's ``(12, 5000)``. Use
    :func:`as_leads_first` to reorient, or load ``ECGDataset("mimic_iv_ecg", ...)``
    for real millivolts on ECGBench's folds.

    ``label`` and ``label_hadm_id`` exist for :data:`RETRIEVAL_SPLITS` only; asking
    for either on ``train`` or ``val`` raises rather than returning an empty array.
    The release's own README describes these files as ``.pt`` (its script saves
    ``torch.save`` output); PhysioNet ships ``.npy``, which is what this reads.
    """
    if split not in SPLIT_CSVS:
        raise ValueError(f"split must be one of {sorted(SPLIT_CSVS)}, got {split!r}")
    if modality not in TENSOR_MODALITIES:
        raise ValueError(f"modality must be one of {list(TENSOR_MODALITIES)}, got {modality!r}")
    if modality.startswith("label") and split not in RETRIEVAL_SPLITS:
        raise ValueError(
            f"'{modality}' exists for {list(RETRIEVAL_SPLITS)} only — {split!r} is not "
            "a retrieval split, so it has no candidate labels."
        )

    directory = Path(data_path) / "data_npy" / split
    path = directory / f"{modality}_{split}.npy"
    _require(path, f"the {split!r} {modality} tensor")
    keys = directory / f"hadm_id_{split}.npy"
    _require(keys, f"the {split!r} row keys")

    array = np.load(path, mmap_mode="r" if mmap else None)
    index = pd.Index(np.load(keys), name=ROW_KEY)
    if len(array) != len(index):
        raise ValueError(
            f"{path.name} has {len(array)} rows but {keys.name} has {len(index)}; "
            "this copy is inconsistent and the alignment cannot be trusted."
        )
    return array, index


def as_leads_first(ecg: np.ndarray) -> np.ndarray:
    """Reorient a shipped ECG tensor to ECGBench's leads-first convention.

    ``(n, 1, 5000, 12) -> (n, 12, 5000)``, or ``(1, 5000, 12) -> (12, 5000)`` for a
    single record. A view where possible, so memory-mapped input stays lazy.

    The leads are in MIMIC-IV-ECG's order, :data:`ECG_LEAD_NAMES` — **aVF before
    aVL**, so ``result[4]`` is aVF. Values remain the unitless [-1, 1] of
    :func:`load_split_tensors`.
    """
    array = np.asarray(ecg)
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0].swapaxes(-1, -2)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0].swapaxes(-1, -2)
    raise ValueError(
        f"Expected (n, 1, samples, leads) or (1, samples, leads), got {array.shape}. "
        "These are the shapes load_split_tensors returns for 'ecg'."
    )
