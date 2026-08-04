"""
ECGRDVQ labels: drug exposure, intervals and T-wave morphology.

**There is no diagnosis here.** All 22 participants were healthy volunteers
screened to exclude cardiac disease, so nothing in this dataset is a rhythm or
morphology class. What varies across records is which of four QT-prolonging drugs
the subject had taken, how long ago, what plasma concentration that produced, and
what happened to repolarisation as a result. Those are the labels.

The release ships one well-formed table — ``SCR-002.Clinical.Data.csv``, one row
per record, 5,232 rows and 32 columns, with ``EGREFID`` equal to the signal
filename — so this module is a rename, a retype and three derivations rather than
a join. Everything below was verified against the shipped v1.0.0, and all 26,137
files match the release's own ``SHA256SUMS.txt``, so what follows describes
upstream properties and not download damage.

**``treatment`` is usable here, unlike in the sibling ECGDMMLD.** This was a
5-period crossover in which every period administered a *single* agent, so
``treatment`` names both the randomisation arm and the drug on board:

=====================  =========  ===================================
``treatment``          Dose       Ion-channel profile
=====================  =========  ===================================
``Dofetilide``         500 µg     predominant hERG (positive control)
``Quinidine Sulph``    400 mg     hERG + peak/late sodium
``Ranolazine``         1500 mg    hERG + late sodium
``Verapamil HCL``      120 mg     hERG + L-type calcium
``Placebo``            --         control
=====================  =========  ===================================

ECGDMMLD's central trap — combination arms whose second agent was dosed hours
later, so the label names a drug that is not in the blood — **does not exist in
this release**. One caveat survives and it is much smaller: the **327 pre-dose
records** (``is_baseline``, ``timepoint_hours == -0.5``) carry their period's
``treatment`` while containing no drug at all, and at ``timepoint_hours == 0.5``
absorption is still incomplete. Filter on ``is_baseline`` or use
``plasma_concentration`` for exposure.

**Three derived columns, and one deliberately absent.**

- ``hr_bpm`` is ``60000 / rr_ms``: exact, and the release ships no heart rate.
- ``qtcf_ms`` is Fridericia, ``qt_ms / cbrt(rr_ms / 1000)``: the standard
  correction, applied because the release ships no corrected QT either. Observed
  range 338.6-563.2 ms against an uncorrected 325-579 ms.
- ``plasma_concentration_ng_ml`` rescales the one pg/mL analyte — see the unit
  trap below.
- **``jtpeakc_ms`` is not provided.** Rate-correcting J-Tpeak needs the exponent
  the study's own authors fitted, not a textbook formula, and guessing one would
  produce a plausible column that reproduces nobody's analysis. Take the exponent
  from the paper (``doi:10.1038/clpt.2014.155``) and apply it to ``jtpeak_ms``
  yourself.

**Unit traps, and this release has three.**

- **The pharmacokinetic table is long, not wide.** Each period dosed one agent, so
  each record carries at most one measurement: ``plasma_analyte`` names it,
  ``plasma_concentration`` is the value and ``plasma_concentration_unit`` its
  unit. **Dofetilide is reported in pg/mL and the other three analytes in ng/mL**,
  so ``groupby("treatment").plasma_concentration.mean()`` compares numbers 1000x
  apart in scale. ``plasma_concentration_ng_ml`` is the same quantity with the
  pg/mL rows divided by 1000, provided so that mistake is avoidable rather than
  merely documented.
- **``dose`` is 500 for dofetilide and 400-1500 for the rest, in different
  units** — ``dose_unit`` is ``ug`` for dofetilide and ``mg`` for quinidine,
  ranolazine and verapamil. Never sort or compare ``dose`` without it.
- **``twave_amplitude_uv`` is microvolts** while the waveforms are millivolts. The
  source measures it on the median beat's vector-magnitude lead; observed range
  66.6-1021.5 µV.

A missing concentration is ``NA``, and it means either "not dosed / not drawn" or
"below the limit of quantification" — the release does not distinguish them.
Nothing in the column is 0, so there is no zero-means-censored convention to
unpick; there is simply no censoring flag.

**Four defects in the release, and every one of them is a *missing* value rather
than a wrong one.** Verified: nothing here is download damage.

1. **9 records have no median beat at all.** ``medians/`` holds 5,223 of the 5,232
   records. Because every interval in the clinical table was measured *from* the
   median beat, those 9 rows have no ``pr_ms``, ``qrs_ms``, ``qt_ms``,
   ``jtpeak_ms`` or ``tpeak_tend_ms``. Their ``rr_ms`` is present (it comes from
   the raw rhythm strip) and, inconsistently, 2 of the 9 still carry ``erd_30_ms``,
   ``lrd_30_ms`` and ``twave_amplitude_uv``, and all 9 carry
   ``twave_asymmetry``/``twave_flatness`` — so the median beat clearly *was*
   computed upstream and simply was not published. See
   :data:`MEDIAN_BEAT_MISSING` and the ``median_beat_available`` column. The 9
   ``raw/`` records are intact, so this costs the split nothing.
2. **4 records have no T-offset annotation**, so ``qt_ms`` and ``tpeak_tend_ms``
   are NA for them and their ``.atr`` carries 4 marks instead of 5. All four are
   subject 1004 on quinidine at 2.5-3.5 h, whose T waves are the flattest in the
   release (``twave_amplitude_uv`` 69.7-78.5 µV against a median of ~400) —
   quinidine flattened the T wave until its end could not be marked. Combined with
   defect 1, ``qt_ms`` is NA for 13 records.
3. **2 records have a 32-bit integer wrap in ``PR``**, stored as
   ``-4294966951`` and ``-4294966972``. Both are subject 1007 on verapamil at
   1.0 h, and both are the two records whose ``.atr`` has **no P-onset mark**: the
   P onset fell *before* the start of the median-beat window, so an unsigned
   subtraction wrapped. Adding 2^32 recovers **345 ms** and **324 ms** — the only
   residues that are physiologic, corroborated by the third record of the same
   triplicate (PR 293 ms, the highest un-wrapped PR in the release) and by
   verapamil's expected AV-nodal PR prolongation. This module **repairs** them and
   flags each with ``pr_ms_repaired``; see :data:`PR_WRAP_MODULUS`.
4. **2 records have a dead V4 lead**, held at a constant -0.00625 mV for all
   10,000 samples. Both are subject 1019 on quinidine at 14.0 h and both are among
   the 9 of defect 1 — the flat lead is presumably why no median beat was derived.
   These are the only two records ``ecgbench splits`` excludes from ``clean/``.

**One genuine absence that is not a defect:** ``twave_asymmetry`` and
``twave_flatness`` are NA for **129 records** spread across all five treatments and
all 22 subjects with no pattern, and not concentrated on the flattest T waves. The
release does not say why.

**The median beats are real and deliberately not the signal.**
``medians/<subject>/<uuid>`` holds a 16-channel derived median beat — the 12 leads
plus the vector-magnitude lead ``VCGMAG`` and the Frank ``vx``, ``vy``, ``vz``
components — at 1 kHz in millivolts like the raw records. **Their length varies**,
968 to 1,876 samples across 667 distinct values, unlike ECGDMMLD's fixed 1,200, so
never assume a shape. They get no fold of their own: every median beat is a
representation of a raw record ECGBench already partitions, and generating a second
partition of the same recordings is the leakage trap ``ADD_DATASET_TODO.md`` warns
about. So :func:`load_labels` exposes ``median_beat_path`` and
:func:`load_median_beat_fiducials` reads their annotations, but ``signal_path``
always points at ``raw/``.

**The fiducials and the interval columns are not independent.** The ``.atr``
annotations on the median beat are what the intervals were measured from, and they
reproduce the CSV exactly: recomputed over every median beat in the release, PR
matches for all 5,221 un-wrapped records, QRS and J-Tpeak for all 5,223, QT and
Tpeak-Tend for the 5,219 with a T offset, and Tpeak-Tpeak' for all 42 records that
have a secondary T peak — every one to the millisecond. Treat agreement between
them as a format check, never as corroboration.

**Unlike ECGDMMLD, secondary T peaks exist here.** ``tpeak_tpeakp_ms`` is populated
for 42 records (40 quinidine, 2 dofetilide, in subjects 1007, 1009 and 1018), and
each is marked by a second ``t`` in the ``.atr``. In ECGDMMLD the same column is
empty in every row and no annotation marks one, so a pipeline written against that
release will silently ignore a real measurement here.
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

#: The one shipped metadata table: one row per record, 32 columns.
CLINICAL_CSV = "SCR-002.Clinical.Data.csv"

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
    "EXTRT": "treatment",
    "ARMCD": "treatment_sequence",
    "EXDOSE": "dose",
    "EXDOSU": "dose_unit",
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
    "TPEAKTPEAKP": "tpeak_tpeakp_ms",  # populated for 42 records — unlike ECGDMMLD
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

#: Source column -> exposed column, for the pharmacokinetic sample. **Long
#: format**, unlike ECGDMMLD's six wide analyte columns: each period dosed a single
#: agent, so a record carries at most one measurement and ``plasma_analyte`` says
#: which. The unit travels in its own column because it is not constant.
ANALYTE_COLUMNS = {
    "PCTEST": "plasma_analyte",
    "PCSTRESN": "plasma_concentration",
    "PCSTRESU": "plasma_concentration_unit",
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

#: ``ARMCD`` treatment code -> the ``EXTRT`` it stands for, per the shipped
#: ``SCR-002.Clinical.Data.Description.txt``. ``treatment_sequence`` is the
#: subject's period order as a **comma-separated** string, e.g. ``A,C,E,D,B``
#: (ECGDMMLD uses dashes); indexing it by ``period`` reproduces ``treatment`` for
#: all 5,232 records, which is how this mapping was checked.
ARM_CODE_TREATMENTS = {
    "A": "Ranolazine",
    "B": "Dofetilide",
    "C": "Verapamil HCL",
    "D": "Quinidine Sulph",
    "E": "Placebo",
}

#: Separator between treatment codes in ``ARMCD``.
ARM_CODE_SEPARATOR = ","

#: The 9 records with no ``medians/<subject>/<uuid>`` file in v1.0.0. Every
#: interval in the clinical table was measured from the median beat, so these rows
#: have no ``pr_ms``/``qrs_ms``/``qt_ms``/``jtpeak_ms``/``tpeak_tend_ms`` — though
#: ``rr_ms`` is present for all 9, and 2 of them inconsistently still carry
#: ``erd_30_ms``/``twave_amplitude_uv``. Their ``raw/`` records are intact.
#: Record id -> subject.
MEDIAN_BEAT_MISSING = {
    "89bc17e0-d8bd-46fa-a8bf-7f2643f0a1cb": "1001",
    "f1e4fa70-ef5c-4607-9ec4-7b0f927d2dff": "1003",
    "56168adc-57c9-4e41-b125-06d7bd7e70d0": "1005",
    "30cfccb3-ab8d-4fb3-a1f2-e6593523b3b3": "1005",
    "badfc6b3-c082-4fb8-97bc-5c683d99ea93": "1005",
    "8eacbe2b-51d3-441b-8a09-9f29f754decd": "1005",
    "6b53aa2e-6c4e-4b44-b52d-df136bb51ecb": "1019",
    "ec9ca3bf-f798-4262-b268-c2ded39fdcb6": "1019",
    "6f312ce9-46dc-4c73-b543-ec68b02dfdc0": "1022",
}

#: ``PR`` values below this are a 32-bit wrap rather than a measurement; adding
#: :data:`PR_WRAP_MODULUS` recovers the interval. See defect 3 in the module
#: docstring. The threshold is far below any physiologic PR and far above the two
#: observed wrapped values (-4294966951, -4294966972), so it cannot catch a real
#: one.
PR_WRAP_THRESHOLD = -1_000_000.0

#: 2^32 — the modulus of the unsigned subtraction that produced those values.
PR_WRAP_MODULUS = 2**32

#: Value of the ``BASELINE`` column marking a pre-dose record.
BASELINE_FLAG = "Y"

#: Nominal timepoint (hours) of the pre-dose triplicate. Every ``is_baseline``
#: record sits here and no other record does.
BASELINE_TIMEPOINT_HOURS = -0.5

#: Analyte whose concentration the release reports in pg/mL rather than ng/mL.
PG_ML_ANALYTE = "Dofetilide"

#: Unit strings used by ``PCSTRESU``.
UNIT_PG_ML, UNIT_NG_ML = "pg/mL", "ng/mL"

#: Key identifying one crossover cell — the unit a baseline is defined over.
PERIOD_KEY = ["patient_id", "period"]

#: Measures :func:`load_baseline_deltas` differences against baseline.
DELTA_COLUMNS = (
    "rr_ms", "pr_ms", "qrs_ms", "qt_ms", "jtpeak_ms", "tpeak_tend_ms",
    "tpeak_tpeakp_ms", "erd_30_ms", "lrd_30_ms", "hr_bpm", "qtcf_ms",
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
            f"ECGRDVQ labels come from {CLINICAL_CSV}, which is not in {data_path}. "
            f"ECGBench publishes fold CSVs only — labels stay with the source "
            f"dataset, so point data_path at a full local copy of the release, the "
            f"directory holding {CLINICAL_CSV}, RECORDS and raw/ (see {config.url})."
        )
    # 'NA' is the source's missing marker and is in pandas' default NA set.
    return pd.read_csv(csv_path)


def _signal_paths(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Attach ``signal_path`` and ``median_beat_path`` from subject + record id.

    The clinical table carries no path column at all. The layout is
    ``<dir>/<RANDID>/<EGREFID>``, verified for all 5,232 records: every
    ``EGREFID`` appears exactly once under ``raw/``, and the directory it sits in
    equals its ``RANDID`` in every case (0 mismatches). ``median_beat_path`` is
    built the same way but resolves for only 5,223 of them — see
    :data:`MEDIAN_BEAT_MISSING`.
    """
    signal_col = config.signal_path_columns[config.default_sampling_rate]
    stem = df["patient_id"].astype(str) + "/" + df[config.record_id_column].astype(str)
    df[signal_col] = RAW_DIR + "/" + stem
    df["median_beat_path"] = MEDIANS_DIR + "/" + stem
    return df


def _repair_pr(df: pd.DataFrame, record_id_column: str) -> pd.DataFrame:
    """Undo the 32-bit wrap in ``pr_ms`` for the records that carry one.

    Two records in v1.0.0 store ``PR`` as roughly -2^32 because the P onset fell
    before the start of the median-beat window and an unsigned subtraction wrapped.
    Adding 2^32 is the only recovery consistent with the annotations, and the only
    residue that is physiologic — see defect 3 in the module docstring. The
    ``pr_ms_repaired`` flag is what makes the repair auditable rather than silent.
    """
    wrapped = df["pr_ms"] < PR_WRAP_THRESHOLD
    df["pr_ms_repaired"] = wrapped.fillna(False)
    if wrapped.any():
        df.loc[wrapped, "pr_ms"] = df.loc[wrapped, "pr_ms"] + PR_WRAP_MODULUS
        logger.info(
            "Repaired a 32-bit wrap in pr_ms for %d record(s): %s",
            int(wrapped.sum()),
            ", ".join(
                f"{rid} -> {pr:.0f} ms"
                for rid, pr in zip(
                    df.loc[wrapped, record_id_column], df.loc[wrapped, "pr_ms"]
                )
            ),
        )
    return df


def _normalise_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``plasma_concentration_ng_ml``, the one analyte in pg/mL rescaled.

    The release reports dofetilide in pg/mL and quinidine, ranolazine and
    verapamil in ng/mL, so the raw column mixes two scales 1000x apart. This is
    the same quantity in one unit; ``plasma_concentration`` and
    ``plasma_concentration_unit`` are left exactly as shipped.
    """
    scale = pd.Series(1.0, index=df.index)
    is_pg = df["plasma_concentration_unit"].eq(UNIT_PG_ML)
    scale[is_pg] = 0.001
    df["plasma_concentration_ng_ml"] = df["plasma_concentration"] * scale

    unexpected = set(df["plasma_concentration_unit"].dropna().unique()) - {
        UNIT_PG_ML,
        UNIT_NG_ML,
    }
    if unexpected:
        # A reissue introducing a third unit must not be scaled by 1.0 in silence.
        logger.warning(
            "PCSTRESU has unrecognised unit(s) %s — plasma_concentration_ng_ml "
            "left unscaled for those rows",
            sorted(unexpected),
        )
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``period``, ``is_baseline``, ``hr_bpm``, ``qtcf_ms`` and the flags.

    Heart rate and a rate-corrected QT are both absent from the release, and both
    are needed for anything comparable across subjects — resting RR here runs
    618-1528 ms, so an uncorrected QT is not interpretable. Fridericia is used
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

    _check_arm_sequence(df)
    return df


def _check_arm_sequence(df: pd.DataFrame) -> None:
    """Warn if ``treatment_sequence`` indexed by ``period`` disagrees with ``treatment``.

    ``treatment_sequence`` (``ARMCD``) is the subject's randomised period order as
    comma-separated treatment codes, e.g. ``A,C,E,D,B``; its ``period``-th element
    is the drug for that period. In v1.0.0 this reproduces ``treatment`` for all
    5,232 records, which is what makes :data:`ARM_CODE_TREATMENTS` trustworthy —
    and means a future reissue that renumbered periods or relabelled arms would be
    caught here rather than silently mislabelling the stratification target.

    One subject (1002) withdrew after 4 of the 5 periods and carries a 4-code
    sequence, so a period index past the end of the sequence is expected rather
    than a fault.
    """
    codes = df["treatment_sequence"].astype(str).str.split(ARM_CODE_SEPARATOR)
    expected = [
        ARM_CODE_TREATMENTS.get(seq[p - 1]) if pd.notna(p) and 1 <= p <= len(seq) else None
        for seq, p in zip(codes, df["period"])
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
    each of the 5,232 records in the release.

    ``treatment`` is the stratification label and, unlike in the sibling
    ECGDMMLD, it does name the drug that was administered — each crossover period
    dosed a single agent. It is still not a clean training target: the 327
    ``is_baseline`` records were taken *before* dosing, so they carry a drug name
    with no drug in the blood. The interval and morphology columns are continuous
    measurements; ``plasma_concentration`` is the study's predictor.
    """
    data_path = Path(data_path)
    raw = _read_clinical(data_path, config)

    renames = {
        "EGREFID": config.record_id_column,
        "BASELINE": "is_baseline",
        **CONTEXT_COLUMNS,
        **INTERVAL_COLUMNS,
        **MORPHOLOGY_COLUMNS,
        **ANALYTE_COLUMNS,
        **SUBJECT_COLUMNS,
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
    df = _add_derived(df)
    df = _repair_pr(df, config.record_id_column)
    df = _normalise_concentration(df)

    for column in (
        "sex", "race", "ethnicity", "treatment", "treatment_sequence", "dose_unit",
        "plasma_analyte", "plasma_concentration_unit",
    ):
        df[column] = df[column].where(df[column].isna(), df[column].astype(str).str.strip())

    # Named after the consequence rather than after the cause: a user filtering
    # median beats cares only whether there is a file to read.
    df["median_beat_available"] = ~df[config.record_id_column].isin(MEDIAN_BEAT_MISSING)

    df = df.set_index(config.record_id_column)
    logger.info(
        "ECGRDVQ labels: %d records, %d subjects, %d treatments, %d periods, "
        "%d pre-dose baseline records",
        len(df), df["patient_id"].nunique(), df["treatment"].nunique(),
        int(df["period"].nunique()), int(df["is_baseline"].sum()),
    )
    absent = int((~df["median_beat_available"]).sum())
    if absent:
        logger.info(
            "%d record(s) have no median beat in the release, so their PR, QRS, QT, "
            "J-Tpeak and Tpeak-Tend are NA; their raw/ records are unaffected", absent,
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
    ``is_baseline``. All 109 (subject, period) pairs in v1.0.0 have one, so no
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

    Reads 5,223 annotation files, so it is opt-in rather than part of
    :func:`load_labels`. Sample indices are milliseconds from the start of the
    median beat, because it is sampled at 1 kHz. **The beat's length varies**
    (968-1,876 samples), so a fiducial cannot be interpreted as a fraction of it.

    Annotations are placed on the vector-magnitude lead in the order P onset, QRS
    onset, QRS offset, T peak, (secondary T peak), T offset. They are the source of
    the clinical table's intervals and reproduce them exactly, so this is not an
    independent measurement — see the module docstring.

    Missing values are real and match the table:

    - The **9 records with no median beat** get an all-None row and
      ``n_annotations`` 0 — there is no ``.atr`` to read.
    - **4 records have no T offset** (subject 1004 on quinidine), which is exactly
      the 4 whose ``qt_ms`` and ``tpeak_tend_ms`` are NA beyond those 9.
    - **2 records have no P onset** (subject 1007 on verapamil) — the two whose
      ``PR`` wrapped through 2^32, because the P onset lies before the start of the
      window.
    - **42 records have a secondary T peak** in ``t_peak_secondary_ms``, matching
      the 42 populated ``tpeak_tpeakp_ms`` values. This is unlike ECGDMMLD, where
      no annotation marks one at all.
    """
    import wfdb

    data_path = Path(data_path)
    labels = load_labels(data_path, config)

    rows = []
    for record_id, subject in labels["patient_id"].items():
        stem = data_path / MEDIANS_DIR / str(subject) / str(record_id)
        row: dict[str, object] = {config.record_id_column: record_id}
        if not stem.with_suffix(f".{ANNOTATOR}").exists():
            # One of the 9 records the release ships no median beat for.
            row.update(_fiducials([]))
            rows.append(row)
            continue
        try:
            annotation = wfdb.rdann(str(stem), ANNOTATOR)
        except Exception as e:  # an unreadable .atr must not kill the scan
            logger.warning("Could not read %s.%s: %s", stem.name, ANNOTATOR, e)
            row.update(_fiducials([]))
            rows.append(row)
            continue
        row.update(_fiducials(list(zip(annotation.sample, annotation.symbol))))
        rows.append(row)

    df = pd.DataFrame(rows).set_index(config.record_id_column)
    logger.info(
        "Median-beat fiducials: %d records, %d with no annotation file, %d without a "
        "P onset, %d without a T offset, %d with a secondary T peak",
        len(df), int((df["n_annotations"] == 0).sum()),
        int(df["p_onset_ms"].isna().sum() - (df["n_annotations"] == 0).sum()),
        int(df["t_offset_ms"].isna().sum() - (df["n_annotations"] == 0).sum()),
        int(df["t_peak_secondary_ms"].notna().sum()),
    )
    return df


def _fiducials(marks: list[tuple[int, str]]) -> dict[str, object]:
    """Resolve one record's annotation marks into named fiducial points.

    Positional rules rather than distinct symbols, because the release reuses
    ``(`` for both the P and QRS onsets and ``)`` for both the QRS and T offsets,
    and there is no QRS-peak mark to anchor on (``ecgcipa`` has one; this release
    does not). The T peak is the anchor instead: it is present in all 5,223 median
    beats, onsets before it are P then QRS in order, and offsets after it are the T
    offset. A ``)`` falling *between* the QRS onset and the T peak is the QRS
    offset.

    Two patterns short of the usual five marks occur, and each drops exactly the
    interval the clinical table also leaves NA: ``()t)`` has no P onset (2 records,
    so no PR) and ``(()t`` has no T offset (4 records, so no QT or Tpeak-Tend).
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
