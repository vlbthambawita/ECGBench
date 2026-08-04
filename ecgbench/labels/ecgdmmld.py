"""
ECGDMMLD labels: drug exposure, intervals and T-wave morphology.

**There is no diagnosis here.** All 22 participants were healthy volunteers
screened to exclude cardiac disease, so nothing in this dataset is a rhythm or
morphology class. What varies across records is which drug the subject had taken,
how long ago, what plasma concentration that produced, and what happened to
repolarisation as a result. Those are the labels.

Unlike its sibling ``ecgcipa``, this release ships a single well-formed table —
``SCR-003.Clinical.Data.csv``, one row per record, 4,211 rows and 33 columns, with
``EGREFID`` equal to the signal filename — so this module is mostly a rename, a
retype and two derivations rather than a four-way join. Everything below was
verified against the shipped v1.0.0, and all 21,059 files match the release's own
``SHA256SUMS.txt``, so what follows describes upstream properties and not download
damage.

**The one thing that will silently ruin a model: ``treatment`` is a randomisation
arm, not the drug in the blood.**

This was a 5-period crossover in which every subject received all five regimens,
and within a period the agents were staged hours apart — the late-sodium or
calcium blocker first, the hERG blocker later. So ``treatment`` describes the whole
period, while a given record sits at one timepoint inside it:

=====================================  =============================  ===============
``treatment``                          circulating at TPT 1.5-3 h     joins later
=====================================  =============================  ===============
``Dofetilide``                         dofetilide                     --
``Mexiletine + Dofetilide``            mexiletine only                dofetilide 6.5 h
``Lidocaine + Dofetilide``             lidocaine only                 dofetilide 6.5 h
``Moxifloxacin + Diltiazem``           moxifloxacin only              diltiazem 12 h
``Placebo``                            --                             --
=====================================  =============================  ===============

A record labelled ``Mexiletine + Dofetilide`` at ``timepoint_hours=2.0`` is a
mexiletine-only ECG, and there are hundreds of them. Train on the six
``plasma_*`` columns, or on ``treatment`` crossed with ``timepoint_hours``; never
on ``treatment`` alone. It is the stratification label because it is the only
patient-level categorical, not because it is a good target.

**The study's endpoint is computable here, which is the happy difference from
ecgcipa.** ``is_baseline`` flags the three pre-dose (``timepoint_hours == -0.5``)
records of each of the 101 (subject, period) pairs, so a baseline is just their
mean. :func:`load_baseline_deltas` returns the per-record change from it for every
interval and morphology measure. In ``ecgcipa`` the equivalent numbers exist only
on triplicate-average rows that carry no record id and cannot be joined to a
waveform at all; here nothing is lost. Placebo-correction is *not* done — it needs
the placebo arm's mean across subjects at the same timepoint, which is an analysis
decision rather than a label.

**Two derived columns, and one deliberately absent.**

- ``hr_bpm`` is ``60000 / rr_ms``: exact, and the release ships no heart rate.
- ``qtcf_ms`` is Fridericia, ``qt_ms / cbrt(rr_ms / 1000)``: the standard
  correction, applied because the release ships no corrected QT either. Observed
  range 353.7-499.1 ms against an uncorrected 288-522 ms.
- **``jtpeakc_ms`` is not provided.** Rate-correcting J-Tpeak needs the exponent
  the study's own authors fitted, not a textbook formula, and guessing one would
  produce a plausible column that reproduces nobody's analysis. Take the exponent
  from the paper (``doi:10.1002/cpt.205``) and apply it to ``jtpeak_ms`` yourself.

**Unit traps.**

- **Dofetilide plasma concentration is pg/mL. The other five analytes are ng/mL.**
  The shipped column description says so and nothing rescales it, so the columns
  are named for their own unit (``plasma_dofetilide_pg_ml`` against
  ``plasma_mexiletine_ng_ml``). Pooling them numerically is a 1000x error.
- **``twave_amplitude_uv`` is microvolts** while the waveforms are millivolts. The
  source measures it on the median beat's vector-magnitude lead; observed range
  81.7-1259.7 uV.
- A missing concentration is ``NA``, and it means either "not dosed / not drawn" or
  "below the limit of quantification" — the release does not distinguish them.
  Nothing in these six columns is 0, so unlike ``ecgcipa`` there is no
  zero-means-censored convention to unpick; there is simply no censoring flag.

**Three defects in the release.**

- ``TPEAKTPEAKP`` is **empty in all 4,211 rows** despite being documented, and no
  ``.atr`` file in the release marks a secondary T peak. The column is exposed as
  ``tpeak_tpeakp_ms`` so its absence is visible rather than inferred; see
  :data:`ALWAYS_EMPTY_COLUMNS`.
- **9 records have no QRS-offset annotation**, so ``qrs_ms`` and ``jtpeak_ms`` are
  NA for them and their ``.atr`` carries 4 marks instead of 5. ``rr_ms``,
  ``pr_ms``, ``qt_ms`` and ``tpeak_tend_ms`` are complete for all 4,211.
- **3 median-beat headers are corrupt** and raise ``IndexError`` from
  ``wfdb.rdrecord``: one channel's ``.dat`` filename has digits from the gain field
  spliced into it. The ``.dat`` and ``.atr`` files are intact. See
  :data:`MEDIAN_HEADER_CORRUPT` and the ``median_beat_readable`` column.

**The median beats are real, mostly readable, and deliberately not the signal.**
``medians/<subject>/<uuid>`` holds a 16-channel derived median beat — the 12 leads
plus the vector-magnitude lead ``VCGMAG`` and the Frank ``vx``, ``vy``, ``vz``
components — 1,200 samples at 1 kHz, in millivolts like the raw records. They get
no fold of their own: every median beat is a representation of a raw record
ECGBench already partitions, and generating a second partition of the same
recordings is the leakage trap ``ADD_DATASET_TODO.md`` warns about. So
:func:`load_labels` exposes ``median_beat_path`` and
:func:`load_median_beat_fiducials` reads their annotations, but ``signal_path``
always points at ``raw/``.

**The fiducials and the interval columns are not independent.** The ``.atr``
annotations on the median beat are what the intervals were measured from, and they
reproduce the CSV exactly: across all 4,211 records every one of PR, QT and
Tpeak-Tend recomputed from the fiducials equals the published value to the
millisecond, as do QRS and J-Tpeak for the 4,202 that have a QRS offset. Treat
agreement between them as a format check, never as corroboration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: The one shipped metadata table: one row per record, 33 columns.
CLINICAL_CSV = "SCR-003.Clinical.Data.csv"

#: Subdirectory holding the 10 s 12-lead acquisitions — the dataset's signal.
RAW_DIR = "raw"

#: Subdirectory holding the derived 16-channel median beats and their ``.atr``.
MEDIANS_DIR = "medians"

#: Annotator extension, per the shipped ANNOTATORS file ("atr  Reference ECG
#: Annotations").
ANNOTATOR = "atr"

#: Source column -> exposed column, for identity and design context.
CONTEXT_COLUMNS = {
    "RANDID": "patient_id",
    "TRTA": "treatment",
    "ARMCD": "treatment_sequence",
    "VISIT": "period_label",
    "TPT": "timepoint_hours",
}

#: Source column -> exposed column, for the interval measurements (all ms).
INTERVAL_COLUMNS = {
    "RR": "rr_ms",
    "PR": "pr_ms",
    "QRS": "qrs_ms",
    "QT": "qt_ms",
    "JTPEAK": "jtpeak_ms",
    "TPEAKTEND": "tpeak_tend_ms",
    "TPEAKTPEAKP": "tpeak_tpeakp_ms",  # empty in every row — see ALWAYS_EMPTY_COLUMNS
    "ERD_30": "erd_30_ms",  # 30% of early repolarisation duration
    "LRD_30": "lrd_30_ms",  # 30% of late repolarisation duration
}

#: Source column -> exposed column, for T-wave morphology. The amplitude is
#: MICROVOLTS (measured on the median beat's vector-magnitude lead); the other two
#: are dimensionless scores.
MORPHOLOGY_COLUMNS = {
    "Twave_amplitude": "twave_amplitude_uv",
    "Twave_asymmetry": "twave_asymmetry",
    "Twave_flatness": "twave_flatness",
}

#: Source column -> (exposed column, unit). Dofetilide is pg/mL and the other five
#: are ng/mL; the unit is in the column name so the difference cannot be lost in a
#: rename.
ANALYTE_COLUMNS = {
    "DOF": ("plasma_dofetilide_pg_ml", "pg/mL"),
    "LIDO": ("plasma_lidocaine_ng_ml", "ng/mL"),
    "MEXI": ("plasma_mexiletine_ng_ml", "ng/mL"),
    "MOXI": ("plasma_moxifloxacin_ng_ml", "ng/mL"),
    "MOXI.M2": ("plasma_moxifloxacin_m2_ng_ml", "ng/mL"),
    "DILT": ("plasma_diltiazem_ng_ml", "ng/mL"),
}

#: Source column -> exposed column, for subject-level attributes. Constant across
#: every record of a subject; height, weight and blood pressure are screening
#: measurements.
SUBJECT_COLUMNS = {
    "SEX": "sex",
    "AGE": "age_years",
    "HGHT": "height_cm",
    "WGHT": "weight_kg",
    "SYSBP": "systolic_bp_mmhg",
    "DIABP": "diastolic_bp_mmhg",
    "RACE": "race",
    "ETHNIC": "ethnicity",
}

#: ``ARMCD`` treatment code -> the ``TRTA`` it stands for, per the shipped
#: ``SCR-003.Clinical.Data.Description.txt``. ``treatment_sequence`` is the
#: subject's five-period order, e.g. ``E-A-B-D-C``; indexing it by ``period``
#: reproduces ``treatment`` for all 4,211 records, which is how this mapping was
#: checked.
ARM_CODE_TREATMENTS = {
    "A": "Dofetilide",
    "B": "Lidocaine + Dofetilide",
    "C": "Mexiletine + Dofetilide",
    "D": "Moxifloxacin + Diltiazem",
    "E": "Placebo",
}

#: Columns documented by the release that are NA in every one of the 4,211 rows.
#: Exposed anyway, so a user sees an empty column rather than wondering where the
#: documented measurement went.
ALWAYS_EMPTY_COLUMNS = ("tpeak_tpeakp_ms",)

#: The 3 records whose median-beat header is corrupt in v1.0.0, so that
#: ``wfdb.rdrecord`` on ``medians/<subject>/<record>`` raises ``IndexError``. In
#: each, one channel's ``.dat`` filename has digits from the gain field spliced
#: into it (``...FFDFD3526620008.dat`` for a file named ``...FFDFD3526628.dat``,
#: with gain ``62000``). Record id -> the channel whose line is broken. The
#: ``.dat`` payloads are intact and the right size, the ``.atr`` files parse, and
#: the corresponding ``raw/`` records are unaffected — only these three median
#: *signals* are unreachable without repairing the header by hand.
MEDIAN_HEADER_CORRUPT = {
    "9D7B03F2-8830-4BD2-A524-FFDFD3526628": "vy",
    "DCA7A8CC-230F-48DA-B639-48F31964B73D": "VCGMAG",
    "79B4DFED-9D6B-4FA7-B2F5-5B5C9D803D62": "VCGMAG",
}

#: Value of the ``BASELINE`` column marking a pre-dose record.
BASELINE_FLAG = "Y"

#: Nominal timepoint (hours) of the pre-dose triplicate. Every ``is_baseline``
#: record sits here and no other record does.
BASELINE_TIMEPOINT_HOURS = -0.5

#: Key identifying one crossover cell — the unit a baseline is defined over.
PERIOD_KEY = ["patient_id", "period"]

#: Measures :func:`load_baseline_deltas` differences against baseline. Excludes
#: ``tpeak_tpeakp_ms``, which has no values to difference.
DELTA_COLUMNS = (
    "rr_ms", "pr_ms", "qrs_ms", "qt_ms", "jtpeak_ms", "tpeak_tend_ms",
    "erd_30_ms", "lrd_30_ms", "hr_bpm", "qtcf_ms",
    "twave_amplitude_uv", "twave_asymmetry", "twave_flatness",
)

#: Prefix for the change-from-baseline columns :func:`load_baseline_deltas` adds.
DELTA_PREFIX = "delta_"

#: Annotation symbols in the median-beat ``.atr`` files: waveform onset, waveform
#: offset, and the T peak. Onsets and offsets are reused for P/QRS and QRS/T, so
#: :func:`_fiducials` resolves them positionally.
_ONSET, _OFFSET, _T_PEAK = "(", ")", "t"


def _read_clinical(data_path: Path, config: DatasetConfig) -> pd.DataFrame:
    """Read the one shipped metadata table, or explain what is missing."""
    from ecgbench.labels import LabelSourceMissingError

    csv_path = data_path / CLINICAL_CSV
    if not csv_path.exists():
        raise LabelSourceMissingError(
            f"ECGDMMLD labels come from {CLINICAL_CSV}, which is not in {data_path}. "
            f"ECGBench publishes fold CSVs only — labels stay with the source "
            f"dataset, so point data_path at a full local copy of the release, the "
            f"directory holding {CLINICAL_CSV}, RECORDS and raw/ (see {config.url})."
        )
    # 'NA' is the source's missing marker and is in pandas' default NA set.
    return pd.read_csv(csv_path)


def _signal_paths(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Attach ``signal_path`` and ``median_beat_path`` from subject + record id.

    The clinical table carries no path column at all. The layout is
    ``<dir>/<RANDID>/<EGREFID>``, verified for all 4,211 records: every
    ``EGREFID`` appears exactly once under ``raw/``, and the directory it sits in
    equals its ``RANDID`` in every case (0 mismatches).
    """
    signal_col = config.signal_path_columns[config.default_sampling_rate]
    stem = df["patient_id"].astype(str) + "/" + df[config.record_id_column].astype(str)
    df[signal_col] = RAW_DIR + "/" + stem
    # The derived median beat of the same acquisition. Its .atr holds the
    # fiducials; see load_median_beat_fiducials.
    df["median_beat_path"] = MEDIANS_DIR + "/" + stem
    return df


def _add_derived(df: pd.DataFrame, record_id_column: str) -> pd.DataFrame:
    """Add ``period``, ``is_baseline``, ``hr_bpm``, ``qtcf_ms`` and the flags.

    Heart rate and a rate-corrected QT are both absent from the release, and both
    are needed for anything comparable across subjects — resting RR here runs
    574-1330 ms, so an uncorrected QT is not interpretable. Fridericia is used
    because it is the standard correction; there is deliberately no ``jtpeakc_ms``
    (see the module docstring).
    """
    df["period"] = (
        df["period_label"].astype(str).str.extract(r"PERIOD-(\d+)-DOSING", expand=False)
    )
    unparsed = int(df["period"].isna().sum())
    if unparsed:
        logger.warning(
            "%d VISIT value(s) did not parse as 'PERIOD-<n>-DOSING'", unparsed
        )
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")

    df["is_baseline"] = df["is_baseline"].eq(BASELINE_FLAG)

    df["hr_bpm"] = 60_000.0 / df["rr_ms"]
    df["qtcf_ms"] = df["qt_ms"] / np.cbrt(df["rr_ms"] / 1000.0)

    # Named after the consequence rather than after the broken channel, because a
    # user filtering median beats cares only whether wfdb will read one.
    df["median_beat_readable"] = ~df[record_id_column].isin(MEDIAN_HEADER_CORRUPT)

    _check_arm_sequence(df)
    return df


def _check_arm_sequence(df: pd.DataFrame) -> None:
    """Warn if ``treatment_sequence`` indexed by ``period`` disagrees with ``treatment``.

    ``treatment_sequence`` (``ARMCD``) is the subject's randomised five-period
    order as treatment codes, e.g. ``E-A-B-D-C``; its ``period``-th element is the
    regimen for that period. In v1.0.0 this reproduces ``treatment`` for all 4,211
    records, which is what makes :data:`ARM_CODE_TREATMENTS` trustworthy — and
    means a future reissue that renumbered periods or relabelled arms would be
    caught here rather than silently mislabelling the stratification target.
    """
    codes = df["treatment_sequence"].astype(str).str.split("-")
    period = df["period"]
    expected = [
        ARM_CODE_TREATMENTS.get(seq[p - 1]) if pd.notna(p) and 1 <= p <= len(seq) else None
        for seq, p in zip(codes, period)
    ]
    disagree = int((pd.Series(expected, index=df.index) != df["treatment"]).sum())
    if disagree:
        logger.warning(
            "%d record(s) where treatment_sequence indexed by period disagrees with "
            "treatment — the arm codes or period numbering have changed", disagree,
        )


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return one row per record: drug exposure, intervals and demographics.

    Indexed by ``config.record_id_column`` (the ``EGREFID`` UUID). One row for
    each of the 4,211 records in the release.

    None of these columns is a classification target. ``treatment`` is the
    stratification label and the closest thing to a class — but it names the
    period's regimen rather than the drug circulating when the record was taken,
    so see the module docstring before using it as one. The interval and morphology
    columns are continuous measurements; the plasma concentrations are the study's
    predictor.
    """
    data_path = Path(data_path)
    raw = _read_clinical(data_path, config)

    renames = {
        "EGREFID": config.record_id_column,
        "BASELINE": "is_baseline",
        **CONTEXT_COLUMNS,
        **INTERVAL_COLUMNS,
        **MORPHOLOGY_COLUMNS,
        **SUBJECT_COLUMNS,
        **{code: name for code, (name, _) in ANALYTE_COLUMNS.items()},
    }
    unmapped = set(raw.columns) - set(renames)
    if unmapped:
        # A new release adding a column should be noticed, not silently dropped.
        logger.warning("%s has unmapped columns: %s", CLINICAL_CSV, sorted(unmapped))

    missing = [c for c in renames if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{CLINICAL_CSV} in {data_path} is missing expected columns {missing}. "
            f"Found: {list(raw.columns)}"
        )

    df = raw.rename(columns=renames)
    # Subject ids are integers in the CSV and directory names on disk; string
    # throughout, or the path join and every later merge silently misses.
    df["patient_id"] = df["patient_id"].astype(str)
    df = _signal_paths(df, config)
    df = _add_derived(df, config.record_id_column)

    for column in ("sex", "race", "ethnicity", "treatment", "treatment_sequence"):
        df[column] = df[column].astype(str).str.strip()

    empty = [c for c in ALWAYS_EMPTY_COLUMNS if c in df.columns and df[c].notna().any()]
    if empty:
        # These are empty in v1.0.0. If a reissue populates one, say so rather than
        # leaving the docstring wrong.
        logger.info(
            "Columns documented as always-empty now carry values: %s", sorted(empty)
        )

    df = df.set_index(config.record_id_column)
    logger.info(
        "ECGDMMLD labels: %d records, %d subjects, %d treatments, %d periods, "
        "%d pre-dose baseline records",
        len(df), df["patient_id"].nunique(), df["treatment"].nunique(),
        int(df["period"].nunique()), int(df["is_baseline"].sum()),
    )
    unreadable = int((~df["median_beat_readable"]).sum())
    if unreadable:
        logger.info(
            "%d median beat(s) have a corrupt header and cannot be read by wfdb; "
            "their raw/ records are unaffected", unreadable,
        )
    return df


def load_baseline_deltas(
    data_path: Path | str, config: DatasetConfig
) -> pd.DataFrame:
    """Per-record change from the subject's own pre-dose baseline for that period.

    The study's endpoint is the *change* in repolarisation, not the absolute
    interval, and this release makes it computable per record — which
    ``ecgcipa`` does not, because there the change columns live only on
    triplicate-average rows with no record id.

    The baseline for a (subject, period) pair is the **mean over its pre-dose
    triplicate** — the three ``timepoint_hours == -0.5`` records flagged
    ``is_baseline``. All 101 (subject, period) pairs in v1.0.0 have one, so no
    record is left without a reference.

    Returns the frame from :func:`load_labels` plus, for each measure in
    :data:`DELTA_COLUMNS`, a ``baseline_<measure>`` column and a
    ``delta_<measure>`` column (record minus baseline). The pre-dose records
    themselves are kept, with deltas near zero by construction — they are the
    triplicate's own deviation from its mean, not an error.

    **This is not the published analysis.** That is placebo-corrected change
    against plasma concentration, and placebo-correction needs the placebo arm's
    mean across subjects at the same nominal timepoint. Subtracting a
    time-matched placebo mean is an analysis decision, so it is left to the caller.
    """
    df = load_labels(data_path, config)

    measures = [c for c in DELTA_COLUMNS if c in df.columns]
    baselines = (
        df[df["is_baseline"]].groupby(PERIOD_KEY, dropna=False)[measures].mean()
    )
    logger.info(
        "Baselines from %d pre-dose records over %d (subject, period) pairs; "
        "%d pairs present in the data",
        int(df["is_baseline"].sum()), len(baselines),
        int(df.groupby(PERIOD_KEY, dropna=False).ngroups),
    )

    missing = df.groupby(PERIOD_KEY, dropna=False).ngroups - len(baselines)
    if missing:
        # Every pair has a baseline in v1.0.0; a reissue that drops one would
        # otherwise produce silently-NaN deltas.
        logger.warning(
            "%d (subject, period) pair(s) have no pre-dose record — their delta_* "
            "columns will be NaN", missing,
        )

    aligned = baselines.reindex(
        pd.MultiIndex.from_frame(df[PERIOD_KEY])
    ).set_index(df.index)
    out = df.copy()
    for measure in measures:
        out[f"baseline_{measure}"] = aligned[measure]
        out[f"{DELTA_PREFIX}{measure}"] = df[measure] - aligned[measure]
    return out


def load_median_beat_fiducials(
    data_path: Path | str, config: DatasetConfig
) -> pd.DataFrame:
    """Fiducial points from every median beat's ``.atr``, indexed by record ID.

    Reads 4,211 annotation files, so it is opt-in rather than part of
    :func:`load_labels`. Sample indices are milliseconds from the start of the
    1,200-sample median beat, because it is sampled at 1 kHz.

    Annotations are placed on the vector-magnitude lead in the order P onset, QRS
    onset, QRS offset, T peak, T offset. They are the source of the clinical
    table's intervals and reproduce them exactly, so this is not an independent
    measurement — see the module docstring.

    Missing values are real and match the table: 9 records carry 4 marks instead
    of 5 because no QRS offset could be annotated, which is exactly the 9 records
    whose ``qrs_ms`` and ``jtpeak_ms`` are NA. ``t_peak_secondary_ms`` is None for
    every record in v1.0.0 — nothing in the release marks one, matching the
    entirely-empty ``tpeak_tpeakp_ms``.

    All 4,211 ``.atr`` files parse, including those of the 3 records in
    :data:`MEDIAN_HEADER_CORRUPT` — that defect is in the header, not the
    annotations.
    """
    import wfdb

    data_path = Path(data_path)
    labels = load_labels(data_path, config)

    rows = []
    for record_id, subject in labels["patient_id"].items():
        stem = data_path / MEDIANS_DIR / str(subject) / str(record_id)
        row: dict[str, object] = {config.record_id_column: record_id}
        try:
            annotation = wfdb.rdann(str(stem), ANNOTATOR)
        except Exception as e:  # a missing .atr must not kill the scan
            logger.warning("Could not read %s.%s: %s", stem.name, ANNOTATOR, e)
            rows.append(row)
            continue
        row.update(_fiducials(list(zip(annotation.sample, annotation.symbol))))
        rows.append(row)

    df = pd.DataFrame(rows).set_index(config.record_id_column)
    logger.info(
        "Median-beat fiducials: %d records, %d without a QRS offset, %d without a "
        "T offset, %d with a secondary T peak",
        len(df), int(df["qrs_offset_ms"].isna().sum()),
        int(df["t_offset_ms"].isna().sum()),
        int(df["t_peak_secondary_ms"].notna().sum()),
    )
    return df


def _fiducials(marks: list[tuple[int, str]]) -> dict[str, object]:
    """Resolve one record's annotation marks into named fiducial points.

    Positional rules rather than distinct symbols, because the release reuses
    ``(`` for both the P and QRS onsets and ``)`` for both the QRS and T offsets,
    and there is no QRS-peak mark to anchor on (``ecgcipa`` has one; this release
    does not). The T peak is the anchor instead: it is present in all 4,211
    records, onsets before it are P then QRS in order, and offsets after it are the
    T offset. A ``)`` falling *between* the QRS onset and the T peak is the QRS
    offset — its absence in 9 records is what leaves ``qrs_ms`` NA.
    """
    out: dict[str, object] = {
        "p_onset_ms": None,
        "qrs_onset_ms": None,
        "qrs_offset_ms": None,
        "t_peak_ms": None,
        "t_peak_secondary_ms": None,
        "t_offset_ms": None,
        "n_annotations": len(marks),
    }
    marks = sorted(marks)
    t_peaks = [sample for sample, symbol in marks if symbol == _T_PEAK]
    if not t_peaks:
        return out
    t_peak = t_peaks[0]
    out["t_peak_ms"] = int(t_peak)
    if len(t_peaks) > 1:
        out["t_peak_secondary_ms"] = int(t_peaks[1])

    onsets = [sample for sample, symbol in marks if symbol == _ONSET and sample <= t_peak]
    if onsets:
        out["qrs_onset_ms"] = int(onsets[-1])
    if len(onsets) > 1:
        out["p_onset_ms"] = int(onsets[-2])

    # Between QRS onset and T peak -> QRS offset; after the T peak -> T offset.
    if out["qrs_onset_ms"] is not None:
        between = [
            sample
            for sample, symbol in marks
            if symbol == _OFFSET and out["qrs_onset_ms"] <= sample <= t_peak
        ]
        if between:
            out["qrs_offset_ms"] = int(between[0])
    after = [sample for sample, symbol in marks if symbol == _OFFSET and sample > t_peak]
    if after:
        out["t_offset_ms"] = int(after[-1])
    return out
