"""
Eye Tracking Dataset for 12-Lead ECG Interpretation — gaze behaviour, not signals.

This release ships **no ECG waveforms at all**. It holds ten *rendered ECG images*
(JPG/PNG) that were shown to 63 clinicians, and aggregate eye-tracking metrics
describing where each clinician looked while interpreting each image:

- ``Datasets/Grid_Anonymized.csv`` — one row per respondent x image x grid AOI,
  where the areas of interest are the individual leads and rhythm-strip segments
  (16-25 AOIs per image).
- ``Datasets/Long_Short_Anonymized.csv`` — the same sessions scored against a
  two-region split of each image instead: ``Long`` (the rhythm strip) versus
  ``Short`` (the twelve short lead traces).
- ``ECGs/ECG_Images/`` — the ten stimulus images themselves.
- ``AOI_Distributions/`` — two figures showing how the AOIs were laid out.

**There is deliberately no ``eye_tracking_ecg`` dataset config, and no
``ecgbench splits --dataset eye_tracking_ecg``.** Nothing here is an ECG recording
in ECGBench's sense: there is no sampled signal to load, no sampling rate, no
patient behind a record, and ``signal_format`` has nothing to name. The unit of
observation is a *reader session* (one clinician looking at one image), so a
ten-fold partition over "records" would be partitioning ten pictures. What the
dataset supports is modelling human interpretation behaviour — which lead draws
attention first, how expertise changes the scan path — so ECGBench exposes it as
a table provider and leaves splitting to whoever defines a task on it.

**The grid AOI vocabulary is not self-explanatory**, and the figure at
``AOI_Distributions/Grid_AOIs.png`` is what decodes it. A full 25-AOI image is
twelve lead boxes, three four-segment rhythm strips, and the printed footer:

- ``1``, ``2``, ``3`` are **leads I, II and III** — numbered, not named, so they
  read like indices rather than leads.
- ``aVR``, ``aVL``, ``aVF``, ``V1``-``V6`` are the remaining lead boxes.
- ``II-1``..``II-4``, ``V1-1``..``V1-4``, ``V5-1``..``V5-4`` are quarters of the
  three rhythm strips running along the bottom.
- ``Information`` is the footer strip (``25mm/s 10mm/mV 150Hz 12SL …``).

Not every image carries the full grid: atrial fibrillation and hyperkalemia have a
single rhythm strip, four images have no ``Information`` box, atrial flutter names
two of its lead boxes ``V1 short`` and ``V5 short``, and the hyperkalemia image is
a **16-lead** trace with ``V3R``, ``V4R``, ``V7`` and ``V8`` on it. So AOI counts
run 16-25 per image. :func:`load_aoi_metrics` derives ``aoi_lead`` and
``aoi_kind`` so you can group across images without re-deriving any of this.

Six properties of the shipped tables that will bite a naive read. All were
confirmed against the release's own ``SHA256SUMS.txt``, so they are upstream
rather than download damage:

1. **Grid AOI labels are scoped to their image.** The same lead is ``V1 NSR`` on
   one image and ``V1 AFib`` on another, giving 233 distinct labels for what is
   really ~25 regions. Grouping by ``Label`` therefore compares nothing across
   images. :func:`load_aoi_metrics` adds ``aoi_area`` — the label with its
   stimulus suffix removed — which is what you want to group on.
2. **Three ``Complete heart block`` labels carry authoring-tool "copy" suffixes**
   (``II-2 CompleteHeartBlock copy``, ``II-3 ... copy copy``,
   ``II-4 ... copy copy copy``), so splitting a label on whitespace does not
   recover the area. ``aoi_area`` strips these too.
3. **``II-3 VTach`` names two different regions.** Ventricular tachycardia has 25
   AOI rows per respondent but only 24 distinct labels: one name was reused for
   two regions, with different metrics in each row (all 63 respondents affected).
   Aggregating by name silently merges them, and indexing by
   ``(respondent, stimulus, label)`` is not unique. :func:`load_aoi_metrics` adds
   ``aoi_occurrence`` (0 or 1, in file order) so the key is.
4. **"Never happened" is encoded as ``-1``, not as a blank.** ``Hit_time_G`` is
   ``-1`` for the 2,086 AOIs a respondent never gazed at, and
   ``First_Fixation_Duration`` / ``Average_Fixations_Duration`` are ``-1`` for the
   3,654 with no fixation. ``notna()`` reports 100% populated and every mean is
   silently wrong. Converted to ``NaN`` by default; pass
   ``sentinels_to_nan=False`` for the raw values.
5. **``Age`` is ``0`` for 54 of the 63 respondents** — an anonymisation artefact,
   not a newborn cardiologist. Also converted to ``NaN``.
6. **The normal-sinus-rhythm image labels one strip quarter ``V3-3``** where the
   sequence is otherwise ``V1-1``, ``V1-2``, ``V1-4``: that image has no ``V1-3``
   and no other V3 strip, so it is upstream a typo for ``V1-3``. ECGBench reports
   it as it is written — ``aoi_lead`` is ``"V3"`` for that one region — rather than
   silently rewriting data on a guess. Filter it out if you aggregate the V1 strip.

One further caveat that is *not* auto-corrected, because fixing it would be
inference rather than decoding: ``TTFF_F`` (time to first fixation) carries a
plausible-looking value even on the 3,654 rows where ``Fixations_Count`` is 0 and
there was no first fixation to time. Mask it on ``Fixations_Count > 0`` yourself.

Reference: Tahri Sqalli et al., *Understanding Cardiology Practitioners'
Interpretations of Electrocardiograms: An Eye-Tracking Study*, JMIR Human Factors
2022. https://doi.org/10.2196/34058
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: The two AOI scorings of the same 630 sessions, relative to the dataset root.
AOI_TABLES = {
    "grid": "Datasets/Grid_Anonymized.csv",
    "long_short": "Datasets/Long_Short_Anonymized.csv",
}

#: Stimulus name as the CSVs spell it -> its image, relative to the dataset root.
#: Not derivable from the name: extensions differ, and "ST elevation MI" is filed
#: as STEMI.png.
STIMULUS_IMAGES = {
    "Atrial fibrillation": "ECGs/ECG_Images/Atrial_Fibrillation.png",
    "Atrial flutter": "ECGs/ECG_Images/Atrial_Flutter.jpg",
    "Complete heart block": "ECGs/ECG_Images/Complete_Heart_Block.jpg",
    "Left bundle branch block": "ECGs/ECG_Images/Left_Bundle_Branch_Block.jpg",
    "Normal sinus rhythm": "ECGs/ECG_Images/Normal_Sinus_Rhythm.jpg",
    "ST elevation MI": "ECGs/ECG_Images/STEMI.png",
    "Ventricular paced rhythm": "ECGs/ECG_Images/Ventricular_Paced_Rhythm.jpg",
    "Ventricular tachycardia": "ECGs/ECG_Images/Ventricular_Tachycardia.jpg",
    "Wolf Parkinson White syndrome": "ECGs/ECG_Images/Wolf_Parkinson_White_Syndrome.jpg",
    "hyperkalemia": "ECGs/ECG_Images/Hyperkalemia.jpg",
}

#: Stimulus name -> the abbreviation its grid AOI labels are suffixed with.
#: Also not derivable: "Atrial flutter" becomes AFlutter but "Atrial fibrillation"
#: becomes AFib, and "hyperkalemia" is capitalised in the suffix but not the name.
STIMULUS_AOI_SUFFIXES = {
    "Atrial fibrillation": "AFib",
    "Atrial flutter": "AFlutter",
    "Complete heart block": "CompleteHeartBlock",
    "Left bundle branch block": "LBBB",
    "Normal sinus rhythm": "NSR",
    "ST elevation MI": "STEMI",
    "Ventricular paced rhythm": "VentPaced",
    "Ventricular tachycardia": "VTach",
    "Wolf Parkinson White syndrome": "WPW",
    "hyperkalemia": "Hyperkalemia",
}

#: Columns where -1 means "this never happened", not "minus one millisecond".
MINUS_ONE_SENTINEL_COLUMNS = (
    "Hit_time_G",
    "First_Fixation_Duration",
    "Average_Fixations_Duration",
)

#: Respondents whose age was not recorded carry 0 rather than a blank.
AGE_SENTINEL = 0

#: Per-respondent columns, constant across that respondent's 244 rows.
RESPONDENT_COLUMNS = ("Group", "Gender", "Age")

#: Rows describing a whole session rather than one AOI within it.
STIMULUS_ROW_TYPE = "Stimulus"
AOI_ROW_TYPE = "Static AOI"

#: The limb-lead boxes are numbered rather than named in the AOI grid. Without
#: this, ``1``/``2``/``3`` read as indices and the three limb leads go missing.
NUMBERED_LIMB_LEADS = {"1": "I", "2": "II", "3": "III"}

#: The footer AOI covering the printed acquisition settings, not a trace.
INFORMATION_AREA = "Information"

_COPY_SUFFIX = re.compile(r"(?:\s+copy)+$", re.IGNORECASE)
#: ``II-3`` -> rhythm-strip quarter 3 of lead II.
_STRIP_SEGMENT = re.compile(r"^(?P<lead>.+?)-(?P<segment>\d+)$")
#: A few areas spell out the short-trace/long-strip distinction the long/short
#: table is built on: ``V1 short`` and ``V5 short`` on atrial flutter,
#: ``V1-4 long`` on Wolff-Parkinson-White. It is redundant with ``aoi_kind``.
_SIZE_QUALIFIER = re.compile(r"\s+(?:short|long)$", re.IGNORECASE)


def _require(path: Path, what: str) -> None:
    from ecgbench.labels import LabelSourceMissingError

    if not path.exists():
        raise LabelSourceMissingError(
            f"Eye-tracking ECG {what} not found at {path}. Point data_path at the "
            "dataset root — the directory holding Datasets/, ECGs/ and "
            "AOI_Distributions/ — from "
            "https://physionet.org/content/eye-tracking-ecg/1.0.0/ ."
        )


def _read_table(data_path: Path | str, table: str) -> pd.DataFrame:
    """Read one raw AOI CSV, dropping its unnamed row-number column."""
    if table not in AOI_TABLES:
        raise ValueError(f"table must be one of {sorted(AOI_TABLES)}, got {table!r}")
    path = Path(data_path) / AOI_TABLES[table]
    _require(path, f"{table} table")
    return pd.read_csv(path, index_col=0)


def aoi_area(label: str, stimulus: str) -> str:
    """Strip a grid AOI label down to the region it names.

    ``"V1-2 NSR" -> "V1-2"``, and ``"II-3 CompleteHeartBlock copy copy" -> "II-3"``.
    Labels in the long/short table carry no suffix and come back unchanged.
    """
    text = _COPY_SUFFIX.sub("", str(label)).strip()
    suffix = STIMULUS_AOI_SUFFIXES.get(stimulus)
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)].strip()
    return text or str(label)


def classify_area(area: str) -> tuple[str, str | None]:
    """Map an :func:`aoi_area` string to ``(kind, lead)``.

    ``kind`` is ``"lead"`` (one of the printed lead boxes), ``"rhythm_strip"`` (a
    quarter of one of the strips along the bottom) or ``"information"`` (the
    printed settings footer, which is not a trace at all). ``lead`` is the lead
    that region shows, in standard spelling, or ``None`` for the footer.

    ``"1" -> ("lead", "I")``, ``"V5-3" -> ("rhythm_strip", "V5")``,
    ``"V1 short" -> ("lead", "V1")``, ``"V1-4 long" -> ("rhythm_strip", "V1")``,
    ``"Information" -> ("information", None)``.
    """
    text = str(area).strip()
    if text.casefold() == INFORMATION_AREA.casefold():
        return "information", None

    # "short"/"long" restates what the segment suffix already says, so drop it
    # before parsing rather than letting "V1-4 long" look like a lead box.
    text = _SIZE_QUALIFIER.sub("", text).strip()

    match = _STRIP_SEGMENT.match(text)
    if match:
        lead = match.group("lead").strip()
        return "rhythm_strip", NUMBERED_LIMB_LEADS.get(lead, lead)

    return "lead", NUMBERED_LIMB_LEADS.get(text, text)


def load_respondents(data_path: Path | str, table: str = "grid") -> pd.DataFrame:
    """One row per clinician, indexed by the anonymised ``Respondent_Name``.

    ``Age`` is ``NaN`` wherever the source stored the ``0`` sentinel (54 of 63).
    """
    df = _read_table(data_path, table)
    out = (
        df.groupby("Respondent_Name")[list(RESPONDENT_COLUMNS)]
        .first()
        .assign(n_sessions=df[df.Type == STIMULUS_ROW_TYPE].groupby("Respondent_Name").size())
    )
    out["Age"] = out["Age"].replace(AGE_SENTINEL, pd.NA).astype("Float64")
    logger.info("Loaded %d respondents from the %s table", len(out), table)
    return out


def load_sessions(data_path: Path | str, table: str = "grid") -> pd.DataFrame:
    """One row per respondent x image — the 630 reader sessions.

    These are the source's ``Type == "Stimulus"`` rows, whose ``Label`` holds the
    image name. Renamed to ``stimulus`` here so it matches the ``ParentStimulus``
    the AOI rows join on.

    ``Duration`` is nominally 30 s but really ranges 2,839-30,392 ms: 139 of the
    630 sessions were cut short. Any ``Time_spent_*`` percentage is relative to the
    session's own duration, so compare percentages rather than absolute times.
    """
    df = _read_table(data_path, table)
    out = (
        df[df.Type == STIMULUS_ROW_TYPE]
        .rename(columns={"Label": "stimulus"})
        .loc[:, ["Respondent_Name", "stimulus", *RESPONDENT_COLUMNS, "Start", "Duration"]]
        .reset_index(drop=True)
    )
    out["Age"] = out["Age"].replace(AGE_SENTINEL, pd.NA).astype("Float64")
    if out.duplicated(["Respondent_Name", "stimulus"]).any():
        raise ValueError(
            "Duplicate (respondent, stimulus) sessions in "
            f"{AOI_TABLES[table]}; the file is not the shipped one."
        )
    logger.info("Loaded %d sessions from the %s table", len(out), table)
    return out


def load_aoi_metrics(
    data_path: Path | str,
    table: str = "grid",
    sentinels_to_nan: bool = True,
) -> pd.DataFrame:
    """Per-AOI gaze and fixation metrics, one row per respondent x image x AOI.

    Args:
        data_path: the dataset root (the directory holding ``Datasets/``).
        table: ``"grid"`` (16-25 lead-level AOIs per image) or ``"long_short"``
            (rhythm strip versus short leads).
        sentinels_to_nan: convert the ``-1`` "never happened" codes in
            :data:`MINUS_ONE_SENTINEL_COLUMNS` and the ``0`` age code to ``NaN``.
            Leave the raw values in place with ``False``.

    Returns:
        The source columns plus four derived ones:

        - ``aoi_area`` — the label with its per-image suffix and any authoring
          ``copy`` suffix removed, so the same region is comparable across images.
        - ``aoi_kind`` — ``"lead"``, ``"rhythm_strip"`` or ``"information"``.
        - ``aoi_lead`` — the lead that region shows, in standard spelling
          (``1`` becomes ``I``, ``V5-3`` becomes ``V5``), or ``None`` for the
          printed-settings footer.
        - ``aoi_occurrence`` — 0, or 1 for the second of the two regions the source
          named ``II-3 VTach``. With it,
          ``(Respondent_Name, ParentStimulus, Label, aoi_occurrence)`` is unique.

        The long/short table has no lead structure, so there ``aoi_kind`` is
        ``"region"`` and ``aoi_lead`` is ``None``.
    """
    df = _read_table(data_path, table)
    out = df[df.Type == AOI_ROW_TYPE].reset_index(drop=True).copy()

    out["aoi_area"] = [
        aoi_area(label, stimulus) for label, stimulus in zip(out["Label"], out["ParentStimulus"])
    ]
    if table == "long_short":
        out["aoi_kind"] = "region"
        out["aoi_lead"] = None
    else:
        classified = [classify_area(area) for area in out["aoi_area"]]
        out["aoi_kind"] = [kind for kind, _ in classified]
        out["aoi_lead"] = [lead for _, lead in classified]

    key = ["Respondent_Name", "ParentStimulus", "Label"]
    out["aoi_occurrence"] = out.groupby(key).cumcount()

    reused = out.loc[out["aoi_occurrence"] > 0, "Label"].unique()
    if len(reused):
        logger.info(
            "%d AOI label(s) name more than one region (%s); disambiguated by " "aoi_occurrence",
            len(reused),
            ", ".join(sorted(map(str, reused))),
        )

    if sentinels_to_nan:
        for column in MINUS_ONE_SENTINEL_COLUMNS:
            if column in out.columns:
                out[column] = out[column].mask(out[column] == -1)
        out["Age"] = out["Age"].replace(AGE_SENTINEL, pd.NA).astype("Float64")

    if out.duplicated([*key, "aoi_occurrence"]).any():
        raise ValueError(
            "(respondent, stimulus, label, occurrence) is not unique in "
            f"{AOI_TABLES[table]}; the file is not the shipped one."
        )
    logger.info(
        "Loaded %d AOI rows x %d columns from the %s table",
        len(out),
        out.shape[1],
        table,
    )
    return out


def stimulus_image_path(data_path: Path | str, stimulus: str) -> Path:
    """Path to the rendered ECG image a session's ``ParentStimulus`` refers to.

    Returned as a path and never decoded: these are pictures of ECGs, not signals,
    so there is no array with units for ECGBench to hand back.
    """
    if stimulus not in STIMULUS_IMAGES:
        raise ValueError(
            f"Unknown stimulus {stimulus!r}. Expected one of {sorted(STIMULUS_IMAGES)}"
        )
    path = Path(data_path) / STIMULUS_IMAGES[stimulus]
    _require(path, f"stimulus image for {stimulus!r}")
    return path


def load_eye_tracking_ecg(
    data_path: Path | str,
    table: str = "grid",
    sentinels_to_nan: bool = True,
) -> pd.DataFrame:
    """AOI metrics with the session's duration and the stimulus image attached.

    The convenience entry point: :func:`load_aoi_metrics` plus the parent session's
    ``Duration``, and the image path for each row's stimulus.

    Example:
        >>> df = load_eye_tracking_ecg("/data/eye-tracking-ecg/1.0.0/")
        >>> df.groupby(["Group", "aoi_area"])["Time_spent_G"].mean()
    """
    aoi = load_aoi_metrics(data_path, table, sentinels_to_nan=sentinels_to_nan)
    sessions = load_sessions(data_path, table).rename(
        columns={"stimulus": "ParentStimulus", "Duration": "session_duration_ms"}
    )
    out = aoi.merge(
        sessions[["Respondent_Name", "ParentStimulus", "session_duration_ms"]],
        on=["Respondent_Name", "ParentStimulus"],
        how="left",
        validate="many_to_one",
    )
    out["stimulus_image"] = out["ParentStimulus"].map(STIMULUS_IMAGES)
    return out
