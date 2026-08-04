"""
CiPA ECG Validation Study labels: drug exposure and interval measurements.

**There is no diagnosis here.** All 60 participants were healthy volunteers
screened to exclude cardiac disease, so nothing in this dataset is a rhythm or
morphology class. What varies across records is which drug the subject had taken,
how long ago, what plasma concentration that produced, and what happened to the
repolarisation intervals as a result. Those are the labels.

The release ships four CDISC-style analysis datasets and no table joining records
to files, so this module assembles one frame per record out of all four:

======================  ==========================================================
``adeg.csv``            9 interval measurements per record, long format
``adpc.csv``            plasma concentration per analyte per subject/timepoint
``adsl.csv``            one row per subject: age, sex, race, randomised arm
``addm.csv``            5 vital signs per subject, long format
======================  ==========================================================

Everything below was verified against the shipped v1.0.0, and all 28,752 files
match the release's own ``SHA256SUMS.txt`` — so these are upstream properties, not
download damage.

**The study's real endpoints cannot be attached to a waveform.**

``adeg.csv`` has 69,556 rows. Only 51,686 belong to a record; the other 17,870
carry ``DTYPE=AVERAGE`` and a **blank** ``EGREFID``. They are the mean over each
triplicate, and they are the only rows where ``BASE`` (baseline), ``CHG`` (change
from baseline), ``CCOMPCHG`` (placebo-corrected change) and ``ABLFL`` are
populated — on the per-record rows all four are 100% null. The published
analysis (delta-QTcF, delta-J-Tpeakc against concentration) therefore lives one
level of aggregation above the signals, and no join can bring it down.

:func:`load_labels` returns the per-record **absolute** intervals.
:func:`load_triplicate_averages` returns those AVERAGE rows for anyone who wants
the endpoints; it is keyed by ``(patient_id, period, timepoint_n)``, not by record.

**Three unit and sentinel traps.**

- **Dofetilide plasma concentration is pg/mL. The other six analytes are ng/mL.**
  The source says so in ``PCSTRESU`` and nothing rescales it, so the columns are
  named for their own unit (``plasma_dofetilide_pg_ml`` against
  ``plasma_ranolazine_ng_ml``). Pooling them numerically is a 1000x error.
- **A plasma concentration of 0 means "below the lower limit of quantification",
  not "no drug".** 263 of the 1,934 ``adpc.csv`` rows carry ``LLOQFL=Y`` and
  ``AVAL=0``; every unflagged value is >= 1.24. ``plasma_below_lloq`` names the
  analytes that were censored for that record, so a 0 can be told from a measured
  0. Nothing else in the file is 0. **Filter on ``plasma_any_below_lloq``, not on
  ``plasma_below_lloq != ""``** — the list column is empty for uncensored records
  and pandas reads that empty string back from a CSV as ``NaN``, so the string
  comparison matches every row of a re-read frame.
- **The signals are microvolts** (header gain ``0.26595744680851063(0)/uV``). That
  is the config's ``signal_unit_scale``, not this module's business, but it is the
  same class of trap.

**Interval measurements are missing for 19 records, for two different reasons.**
10 records have no ``pr_ms`` because no P onset could be annotated, and 9 have no
``qt_ms``, ``qtcf_ms``, ``jtpeak_ms``, ``jtpeakc_ms`` or ``tpeak_tend_ms`` because
no T annotation could be placed. ``hr_bpm``, ``rr_ms`` and ``qrs_ms`` are complete
for all 5,749. This is why the columns are float rather than int.

**``replicate_number`` disagrees with itself in 4 records.** ``EGREPNUM`` is
constant across the 9 parameters of a record in 5,745 of them, but in
``18F39342-6619-4868-97C8-8CE5D833FB0C``,
``454D8FEA-5B86-4321-ABCD-FA4F8E1700E2``,
``4E877663-5922-47E7-A1AD-4CB53897B64F`` and
``A543C581-383C-43E0-8249-939A1108B4CB`` the interval parameters are numbered one
apart from HR/RR/QRS. ``ADTM`` is identical within each, so the record is not
ambiguous — only its index within the triplicate is. This module anchors
``replicate_number`` on the HR row (HR is present for all 5,749) and sets
``replicate_number_inconsistent`` for those four rather than picking silently.

**The median beats are real, readable, and deliberately not the dataset's signal.**
``medians/<subject>/<uuid>`` holds a 16-channel derived median beat — the 12 leads
plus the vector-magnitude lead ``VCGMAG`` and the Frank ``X``, ``Y``, ``Z``
components — 1,200 samples at 1 kHz, in microvolts like the raw records (spot
checked at 0.87-0.93x the raw peak-to-peak of lead II, as median-beat averaging
predicts). Its header gains look corrupt at a glance (``6276255.687397709
(-1227133513)/uV``) but are not: each channel is scaled to fill the int32 range,
and ``wfdb.rdrecord`` recovers physiologic microvolts.

They still get no fold of their own — every median beat is a representation of a
raw record ECGBench already partitions, and generating a second partition of the
same recordings is the leakage trap ``ADD_DATASET_TODO.md`` warns about. So
``load_labels`` exposes ``median_beat_path`` and :func:`load_median_beat_fiducials`
reads their annotations, but ``signal_path`` always points at ``raw/``.

**The fiducials and the interval table are not independent.** The ``.atr``
annotations on the median beat are what the intervals were measured from, and they
reproduce ``adeg.csv`` exactly: across all 5,749 records every one of PR, QRS, QT,
J-Tpeak and Tpeak-Tend recomputed from the fiducials equals the published value to
the millisecond. Treat agreement between them as a format check, never as
corroboration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Newline-delimited list of record stems in the dataset root. Holds both the
#: ``raw/<subject>/<uuid>`` and ``medians/<subject>/<uuid>`` families, 5,749 each.
RECORDS_FILE = "RECORDS"

#: Subdirectory holding the 10 s 12-lead acquisitions — the dataset's signal.
RAW_DIR = "raw"

#: Subdirectory holding the derived 16-channel median beats and their ``.atr``.
MEDIANS_DIR = "medians"

#: Annotator extension, per the shipped ANNOTATORS file ("atr  semi-automatic
#: annotations of each median beat in the vector magnitude lead").
ANNOTATOR = "atr"

#: The four clinical analysis datasets, all read relative to the dataset root.
ADEG_CSV = "adeg.csv"
ADPC_CSV = "adpc.csv"
ADSL_CSV = "adsl.csv"
ADDM_CSV = "addm.csv"

#: adeg ``PARAMCD`` -> exposed column. Units are ms except heart rate; they are in
#: the names because ``EGSTRESU`` is dropped in the pivot.
INTERVAL_COLUMNS = {
    "HR": "hr_bpm",
    "RR": "rr_ms",
    "PR": "pr_ms",
    "QRS": "qrs_ms",
    "QT": "qt_ms",
    "QTCF": "qtcf_ms",  # Fridericia-corrected QT
    "JTP": "jtpeak_ms",  # J point (QRS offset) to T peak
    "JTPC": "jtpeakc_ms",  # rate-corrected J-Tpeak
    "TPTE": "tpeak_tend_ms",
}

#: adeg column -> exposed column, for the per-record design context. Every one of
#: these is constant across a record's 9 parameter rows (``EGREPNUM`` is handled
#: separately — see the module docstring).
CONTEXT_COLUMNS = {
    "STUDYID": "study_id",
    "USUBJID": "patient_id",
    "TRTA": "treatment",
    "TRTP": "planned_treatment",
    "TRTSEQA": "treatment_sequence",
    "APERIOD": "period",
    "APERIODC": "period_label",
    # Two different clocks, and they disagree by design — see _add_timepoint_hours.
    "ATPT": "timepoint",
    "ATPTN": "timepoint_n",
    "NRRLT": "nominal_hours_from_reference",
    "ARRLT": "actual_hours_from_reference",
    "ADTM": "acquisition_datetime",
    "ADY": "study_day",
    "APERDAY": "period_day",
}

#: Parameter whose ``EGREPNUM`` defines ``replicate_number``. HR is one of the
#: three parameters present for all 5,749 records.
REPLICATE_ANCHOR_PARAM = "HR"

#: Reference timepoint every ``ARRLT``/``NRRLT`` is measured from, per the source's
#: ``ATPTREF`` column (one value throughout the release).
TIMEPOINT_REFERENCE = "Morning fasted dose"

#: adpc ``PARAMCD`` -> (exposed column, source unit). Dofetilide is pg/mL and the
#: other six are ng/mL; the unit is in the column name so the difference cannot be
#: lost in a rename.
ANALYTE_COLUMNS = {
    "RAN": ("plasma_ranolazine_ng_ml", "ng/mL"),
    "VER": ("plasma_verapamil_ng_ml", "ng/mL"),
    "LOP": ("plasma_lopinavir_ng_ml", "ng/mL"),
    "RIT": ("plasma_ritonavir_ng_ml", "ng/mL"),
    "CHL": ("plasma_chloroquine_ng_ml", "ng/mL"),
    "DIL": ("plasma_diltiazem_ng_ml", "ng/mL"),
    "DOF": ("plasma_dofetilide_pg_ml", "pg/mL"),
}

#: Key joining a record to its pharmacokinetic sample. There is no record id in
#: adpc.csv — blood was drawn per subject per nominal timepoint, not per ECG, so
#: all three replicates of a timepoint share one concentration.
PK_KEY = ["patient_id", "period", "timepoint_n"]

#: Separator for the ``plasma_below_lloq`` analyte list. Records in the crossover
#: arm can have two analytes measured at once.
LIST_SEPARATOR = ";"

#: adsl column -> exposed column.
SUBJECT_COLUMNS = {
    "AGE": "age_years",
    "SEX": "sex",
    "RACE": "race",
    "ETHNIC": "ethnicity",
    "ARM": "planned_arm",
    "ACTARM": "actual_arm",
}

#: addm ``PARAMCD`` -> exposed column, one screening measurement per subject.
VITAL_COLUMNS = {
    "HEIGHT": "height_cm",
    "WEIGHT": "weight_kg",
    "BMI": "bmi_kg_m2",
    "SYSBP": "systolic_bp_mmhg",
    "DIABP": "diastolic_bp_mmhg",
}

#: Annotation symbols in the median-beat ``.atr`` files: waveform onset, waveform
#: offset, the QRS peak, and the T peak.
_ONSET, _OFFSET, _QRS, _T_PEAK = "(", ")", "N", "t"


def _read_csv(data_path: Path, name: str, config: DatasetConfig) -> pd.DataFrame:
    """Read one of the four analysis datasets, or explain what is missing."""
    from ecgbench.labels import LabelSourceMissingError

    csv_path = data_path / name
    if not csv_path.exists():
        raise LabelSourceMissingError(
            f"CiPA labels come from {name}, which is not in {data_path}. ECGBench "
            f"publishes fold CSVs only — labels stay with the source dataset, so "
            f"point data_path at a full local copy of the release, the directory "
            f"holding adeg.csv, RECORDS and raw/ (see {config.url})."
        )
    return pd.read_csv(csv_path, low_memory=False)


def scan_records(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """List every raw record with its subject and both signal paths.

    Built from the shipped ``RECORDS`` file rather than by reading 5,749 headers:
    the path itself carries both identifiers (``raw/1001/00689D31-...``), and that
    was verified against all 5,749 header comment blocks — the header's ``EGREFID``
    equals the filename and its ``USUBJID`` equals the directory in every record.
    The set is cross-checked against ``adeg.csv`` in :func:`load_labels`, which
    catches a partial download without the header reads.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / RECORDS_FILE

    if records_file.exists():
        stems = [
            line.strip()
            for line in records_file.read_text().splitlines()
            if line.strip().startswith(f"{RAW_DIR}/")
        ]
        stems = [stem[len(RAW_DIR) + 1 :] for stem in stems]
    else:
        logger.warning("%s not found — falling back to globbing %s/*/*.hea",
                       records_file, RAW_DIR)
        stems = sorted(
            f"{path.parent.name}/{path.stem}"
            for path in (data_path / RAW_DIR).glob("*/*.hea")
        )

    if not stems:
        raise LabelSourceMissingError(
            f"No raw records found under {data_path / RAW_DIR}. Point data_path at "
            f"the dataset root — the directory holding raw/, medians/, RECORDS and "
            f"adeg.csv (see {config.url})."
        )

    signal_col = config.signal_path_columns[config.default_sampling_rate]
    rows = []
    for stem in stems:
        subject, _, record_id = stem.partition("/")
        rows.append(
            {
                config.record_id_column: record_id,
                # From the directory, and identical to adeg's USUBJID.
                config.patient_id_column: subject,
                signal_col: f"{RAW_DIR}/{stem}",
                # The derived median beat of the same acquisition. Its .atr holds
                # the fiducials; see load_median_beat_fiducials.
                "median_beat_path": f"{MEDIANS_DIR}/{stem}",
            }
        )
    return pd.DataFrame(rows)


def load_intervals(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Per-record interval measurements and design context, indexed by record ID.

    Drops ``adeg.csv``'s 17,870 ``DTYPE=AVERAGE`` rows, which have no record ID —
    see :func:`load_triplicate_averages`.
    """
    adeg = _read_csv(Path(data_path), ADEG_CSV, config)
    per_record = adeg[adeg["EGREFID"].notna()].copy()
    logger.info(
        "adeg.csv: %d rows, %d with a record ID (%d records), %d triplicate averages",
        len(adeg), len(per_record), per_record["EGREFID"].nunique(),
        len(adeg) - len(per_record),
    )

    unknown = set(per_record["PARAMCD"]) - set(INTERVAL_COLUMNS)
    if unknown:
        # A new release adding a parameter should be noticed, not silently dropped.
        logger.warning("adeg.csv has unmapped PARAMCD values: %s", sorted(unknown))

    intervals = (
        per_record.pivot_table(
            index="EGREFID", columns="PARAMCD", values="AVAL", aggfunc="first"
        )
        .rename(columns=INTERVAL_COLUMNS)
        .reindex(columns=list(INTERVAL_COLUMNS.values()))
    )

    context = (
        per_record.drop_duplicates(subset=["EGREFID"])
        .set_index("EGREFID")[list(CONTEXT_COLUMNS)]
        .rename(columns=CONTEXT_COLUMNS)
    )
    # AEGBLFL flags the pre-dose records averaged into each subject's baseline;
    # ECGPCFL flags records with a matching pharmacokinetic sample. Both are 'Y'
    # or blank in the source, so a bare read gives object dtype and NaN.
    flags = per_record.drop_duplicates(subset=["EGREFID"]).set_index("EGREFID")
    context["used_for_baseline"] = flags["AEGBLFL"].eq("Y")
    context["has_matching_pk"] = flags["ECGPCFL"].eq("Y")

    # EGREPNUM is anchored on HR rather than taken from an arbitrary row: it
    # disagrees across parameters in 4 records. See the module docstring.
    anchor = per_record[per_record["PARAMCD"] == REPLICATE_ANCHOR_PARAM]
    context["replicate_number"] = anchor.set_index("EGREFID")["EGREPNUM"]
    inconsistent = per_record.groupby("EGREFID")["EGREPNUM"].nunique(dropna=False) > 1
    context["replicate_number_inconsistent"] = inconsistent.reindex(
        context.index, fill_value=False
    )
    if int(inconsistent.sum()):
        logger.info(
            "%d record(s) have EGREPNUM disagreeing across parameters; "
            "replicate_number taken from the %s row",
            int(inconsistent.sum()), REPLICATE_ANCHOR_PARAM,
        )

    _add_timepoint_hours(context)

    df = context.join(intervals)
    df.index.name = config.record_id_column
    return df


def _add_timepoint_hours(context: pd.DataFrame) -> None:
    """Add ``nominal_hours_from_period_start`` and tidy ``replicate_number``.

    The release carries two clocks and they are not the same number.
    ``timepoint`` ("54 hrs") counts from the **period's first dose**, while
    ``nominal_hours_from_reference`` (``NRRLT``) counts from **that day's**
    reference dose (:data:`TIMEPOINT_REFERENCE`) — so a record on study day 3 reads
    ``timepoint`` 54 and ``nominal_hours_from_reference`` 6, because 48 + 6 = 54.
    Neither is wrong; picking the wrong one silently collapses three dosing days
    onto one time axis. The numeric form of the first is parsed out here so nobody
    has to strip " hrs" by hand.
    """
    context["nominal_hours_from_period_start"] = pd.to_numeric(
        context["timepoint"].astype(str).str.removesuffix(" hrs"), errors="coerce"
    )
    unparsed = int(context["nominal_hours_from_period_start"].isna().sum())
    if unparsed:
        logger.warning("%d timepoint label(s) did not parse as '<hours> hrs'", unparsed)
    # No NaN in v1.0.0 (HR is measured for every record), but Int64 keeps a future
    # gap as a gap instead of turning the whole column into floats.
    context["replicate_number"] = context["replicate_number"].astype("Int64")


def load_triplicate_averages(
    data_path: Path | str, config: DatasetConfig
) -> pd.DataFrame:
    """The ``DTYPE=AVERAGE`` rows of ``adeg.csv``, long format.

    These are the study's actual analysis units: the mean of each triplicate, with
    ``BASE``, ``CHG`` and ``CCOMPCHG`` (baseline, change from baseline and
    placebo-corrected change) populated. They carry **no** ``EGREFID``, so they
    cannot be joined to a waveform — the key is
    ``(patient_id, period, timepoint_n, parameter)``.

    Returned close to the source shape on purpose: this is an escape hatch for
    reproducing the published exposure-response analysis, not a label frame.
    """
    adeg = _read_csv(Path(data_path), ADEG_CSV, config)
    averages = adeg[adeg["DTYPE"] == "AVERAGE"].copy()
    averages = averages.rename(
        columns={
            "USUBJID": "patient_id",
            "APERIOD": "period",
            "ATPTN": "timepoint_n",
            "PARAMCD": "parameter",
        }
    )
    logger.info("adeg.csv: %d triplicate-average rows", len(averages))
    return averages.reset_index(drop=True)


def load_pharmacokinetics(
    data_path: Path | str, config: DatasetConfig
) -> pd.DataFrame:
    """Plasma concentration per analyte, keyed by :data:`PK_KEY`.

    One row per (subject, period, nominal timepoint). Concentrations of 0 are
    below-LLOQ censoring rather than measurements; ``plasma_below_lloq`` lists
    which analytes those were.
    """
    adpc = _read_csv(Path(data_path), ADPC_CSV, config)
    adpc = adpc.rename(
        columns={"USUBJID": "patient_id", "APERIOD": "period", "ATPTN": "timepoint_n"}
    )

    unknown = set(adpc["PARAMCD"]) - set(ANALYTE_COLUMNS)
    if unknown:
        logger.warning("adpc.csv has unmapped analyte codes: %s", sorted(unknown))

    concentrations = (
        adpc.pivot_table(index=PK_KEY, columns="PARAMCD", values="AVAL", aggfunc="first")
        .rename(columns={code: name for code, (name, _) in ANALYTE_COLUMNS.items()})
        .reindex(columns=[name for name, _ in ANALYTE_COLUMNS.values()])
    )

    censored = adpc[adpc["LLOQFL"] == "Y"]
    below = (
        censored.groupby(PK_KEY)["PARAMCD"]
        .apply(lambda codes: LIST_SEPARATOR.join(sorted(set(codes))))
        .rename("plasma_below_lloq")
    )
    out = concentrations.join(below)
    out["plasma_below_lloq"] = out["plasma_below_lloq"].fillna("")
    # The boolean exists because the list column does NOT survive a CSV round trip:
    # pandas writes "" and reads it back as NaN, so `!= ""` on a re-read frame
    # matches every row. Filter on this; read the list for which analyte.
    out["plasma_any_below_lloq"] = out["plasma_below_lloq"].ne("")
    logger.info(
        "adpc.csv: %d timepoint samples, %d with an analyte below the LLOQ",
        len(out), int((out["plasma_below_lloq"] != "").sum()),
    )
    return out.reset_index()


def load_subjects(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Subject demographics and screening vitals, indexed by ``patient_id``."""
    data_path = Path(data_path)
    adsl = _read_csv(data_path, ADSL_CSV, config)
    addm = _read_csv(data_path, ADDM_CSV, config)

    subjects = adsl.rename(columns={"USUBJID": "patient_id"}).set_index("patient_id")
    subjects = subjects[list(SUBJECT_COLUMNS)].rename(columns=SUBJECT_COLUMNS)
    # RACE ships with leading spaces ('  WHITE', ' ASIAN'), so a groupby on the raw
    # value produces categories that look identical and are not.
    for column in ("sex", "race", "ethnicity", "planned_arm", "actual_arm"):
        subjects[column] = subjects[column].astype(str).str.strip()

    vitals = (
        addm.rename(columns={"USUBJID": "patient_id"})
        .pivot_table(index="patient_id", columns="PARAMCD", values="AVAL", aggfunc="first")
        .rename(columns=VITAL_COLUMNS)
        .reindex(columns=list(VITAL_COLUMNS.values()))
    )
    return subjects.join(vitals)


def load_median_beat_fiducials(
    data_path: Path | str, config: DatasetConfig
) -> pd.DataFrame:
    """Fiducial points from every median beat's ``.atr``, indexed by record ID.

    Reads 5,749 annotation files, so it is opt-in rather than part of
    :func:`load_labels`. Sample indices are milliseconds from the start of the
    1,200-sample median beat, because it is sampled at 1 kHz.

    Annotations are placed on the vector-magnitude lead in the documented order
    P onset, QRS onset, QRS peak, QRS offset, T peak (plus a secondary T peak in
    30 records) and T offset. They are the source of ``adeg.csv``'s intervals and
    reproduce them exactly, so this is not an independent measurement — see the
    module docstring.

    Missing values are real: 10 records have no P onset and 9 have no T
    annotations at all, matching ``adeg.csv``'s 10 absent ``pr_ms`` and 9 absent
    ``qt_ms``.
    """
    import wfdb

    data_path = Path(data_path)
    rows = []
    for record_id, subject in (
        scan_records(data_path, config)[
            [config.record_id_column, config.patient_id_column]
        ].itertuples(index=False)
    ):
        stem = data_path / MEDIANS_DIR / subject / record_id
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
        "Median-beat fiducials: %d records, %d without a P onset, %d without a T offset",
        len(df), int(df["p_onset_ms"].isna().sum()), int(df["t_offset_ms"].isna().sum()),
    )
    return df


def _fiducials(marks: list[tuple[int, str]]) -> dict[str, object]:
    """Resolve one record's annotation marks into named fiducial points.

    Positional rules rather than distinct symbols, because the release reuses
    ``(`` for both the P and QRS onsets and ``)`` for both the QRS and T offsets:
    the QRS peak ``N`` anchors everything, onsets before it are QRS then P
    working backwards, and offsets after it are QRS then T working forwards.
    """
    out: dict[str, object] = {
        "p_onset_ms": None,
        "qrs_onset_ms": None,
        "qrs_peak_ms": None,
        "qrs_offset_ms": None,
        "t_peak_ms": None,
        "t_peak_secondary_ms": None,
        "t_offset_ms": None,
        "n_annotations": len(marks),
    }
    marks = sorted(marks)
    peaks = [sample for sample, symbol in marks if symbol == _QRS]
    if not peaks:
        return out
    qrs_peak = peaks[0]
    out["qrs_peak_ms"] = int(qrs_peak)

    onsets = [sample for sample, symbol in marks if symbol == _ONSET and sample <= qrs_peak]
    if onsets:
        out["qrs_onset_ms"] = int(onsets[-1])
    if len(onsets) > 1:
        out["p_onset_ms"] = int(onsets[-2])

    offsets = [sample for sample, symbol in marks if symbol == _OFFSET and sample >= qrs_peak]
    if offsets:
        out["qrs_offset_ms"] = int(offsets[0])
    if len(offsets) > 1:
        out["t_offset_ms"] = int(offsets[-1])

    t_peaks = [sample for sample, symbol in marks if symbol == _T_PEAK]
    if t_peaks:
        out["t_peak_ms"] = int(t_peaks[0])
    if len(t_peaks) > 1:
        out["t_peak_secondary_ms"] = int(t_peaks[1])
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return one row per record: drug exposure, intervals and demographics.

    Joins :func:`load_intervals`, :func:`load_pharmacokinetics` and
    :func:`load_subjects` onto :func:`scan_records`, indexed by
    ``config.record_id_column``.

    None of these columns is a classification target. ``treatment`` is the
    stratification label and the closest thing to a class; the interval columns
    are continuous measurements; the plasma concentrations are the study's
    predictor. See the module docstring for what is deliberately absent.
    """
    data_path = Path(data_path)
    records = scan_records(data_path, config)
    intervals = load_intervals(data_path, config)

    on_disk = set(records[config.record_id_column])
    measured = set(intervals.index)
    if on_disk != measured:
        # A partial download is the likely cause, and it must not become a silent
        # inner join that drops records from the split.
        logger.warning(
            "Record IDs disagree: %d on disk, %d in %s (%d only on disk, %d only "
            "in the table)",
            len(on_disk), len(measured), ADEG_CSV,
            len(on_disk - measured), len(measured - on_disk),
        )

    df = records.merge(
        intervals.reset_index().drop(columns=[config.patient_id_column]),
        on=config.record_id_column,
        how="left",
    )

    pk = load_pharmacokinetics(data_path, config)
    # patient_id is a string in the frame built from directory names and an int in
    # the CSVs; align before joining or every key misses.
    for frame in (df, pk):
        frame["patient_id"] = frame["patient_id"].astype(str)
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df = df.merge(pk, on=PK_KEY, how="left")
    df["plasma_below_lloq"] = df["plasma_below_lloq"].fillna("")
    df["plasma_any_below_lloq"] = df["plasma_any_below_lloq"].fillna(False).astype(bool)

    subjects = load_subjects(data_path, config)
    subjects.index = subjects.index.astype(str)
    df = df.merge(subjects, left_on="patient_id", right_index=True, how="left")

    logger.info(
        "CiPA labels: %d records, %d subjects, %d treatments, %d with a plasma "
        "concentration",
        len(df), df["patient_id"].nunique(), df["treatment"].nunique(),
        int(df["has_matching_pk"].sum()),
    )
    return df.set_index(config.record_id_column)
