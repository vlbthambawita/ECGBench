"""
VitalDB Arrhythmia Database — beat and rhythm annotations **for VitalDB's cases**.

This release ships **no ECG waveforms**. It is one ``metadata.csv`` and 482
per-case annotation CSVs holding anesthesiologist-validated beat and rhythm
labels for intraoperative Lead II recordings that live in the public VitalDB
project, keyed by VitalDB's own ``case_id``. The waveforms are fetched
separately, and not as files — ``vitaldb.load_case(case_id, ['SNUADC/ECG_II'],
1/500)`` pulls them over the network.

**There is deliberately no ``vitaldb_arrhythmia`` dataset config, and no
``ecgbench splits --dataset vitaldb_arrhythmia``.** Nothing in the package is a
signal file: ``signal_format`` has nothing to name, ``signal_path_columns`` has
nothing to resolve against ``data_path``, and ``validate_dataset`` — which reads
every record off disk — has nothing to read. So ECGBench exposes this as an
**annotation provider**: you load the tables here, and fetch the waveform for a
case from VitalDB when you need samples. Any split is the task's to define, and
the note under "Grouping" below says what it must group on.

**Grouping: split on ``subjectid``, never on ``case_id``.** The 482 cases come
from **473 distinct patients** — eight patients contributed two cases each and
one contributed three. Case-level folds put the same patient on both sides of
the boundary. :func:`load_cases` carries ``subjectid`` for exactly this.

**The annotations are windows, not whole cases.** Each case contributes a single
contiguous screened window averaging 1,109 s (median 1,198 s, range 139-2,991 s),
starting anywhere from 2 s to 33,628 s into the surgery — so ``time_second`` is
an offset into the *VitalDB recording*, not into the annotated segment, and a
naive ``waveform[:n]`` reads a stretch that carries no labels at all.
:func:`case_window` returns the ``(start, end)`` a case's labels actually cover.

Seven properties of the shipped files that change what you can do with them, all
recomputed from the release (every one of its 485 files matches the shipped
``SHA256SUMS.txt``, so these are upstream, not download damage):

1. **``total_beats`` counts annotation rows, not beats.** It matches the row
   count of the case's annotation file exactly for all 482 cases, and those rows
   sum to 676,250 — but 17,364 ship with an empty ``beat_type``, and a further 12
   are case 2453's misfiled boundary markers (point 2), so **17,376 rows annotate
   no heartbeat at all**. Only **658,874** classify a beat: 439,458 N, 184,203 S,
   18,972 U, 16,234 V and 7 P. The abstract's "over 660,000 individually
   annotated heartbeats" is therefore the row count, not the beat count — the
   beat count falls just under it. :func:`load_beats` adds ``is_beat`` so the
   distinction survives.
2. **Three cases are written with different columns.** 480 files use
   ``time_second, beat_type, rhythm_label, bad_signal_quality,
   bad_signal_quality_label``; case **3828** emits the last two in the opposite
   order (harmless by name, fatal positionally); and case **2453** has no
   ``bad_signal_quality_label`` column at all, carries an extra ``caseid``
   column, and writes its twelve segment boundary markers into ``beat_type`` as
   the literal strings ``Start`` and ``End``. A release-wide
   ``value_counts("beat_type")`` therefore reports ``Start`` and ``End`` as beat
   classes. :func:`load_annotations` normalises all three to one schema.
3. **``beat_type`` has an undocumented fifth value.** The paper names four
   classes (N, S, V, U); ``P`` also occurs, on 7 beats across cases 708, 1018,
   3433 and 3631. It is left as written rather than folded into ``U`` — see
   :data:`BEAT_TYPES`, where it maps to ``"Unknown (undocumented)"``.
4. **``(case_id, time_second)`` is not unique.** 111 cases contain repeated
   timestamps, 250 rows in all. Index on position, or de-duplicate deliberately;
   a ``set_index("time_second")`` silently misaligns.
5. **4,258 rows in 333 cases have an empty ``rhythm_label``** — usually a single
   beat inside an otherwise labelled run, not a segment boundary. They are gaps
   in the rhythm annotation and are kept as NaN, never forward-filled.
6. **``bad_signal_quality_label`` markers do not always pair up.** They run
   ``Start1``/``End1`` … up to index 57, but in **17 cases** the Start and End
   counts differ. :func:`bad_signal_intervals` closes an unterminated interval at
   the last annotated sample and logs it, rather than dropping or guessing.
7. **``age`` is a string, capped at ``">89"``** for two patients — a
   de-identification artefact. Reading the column as a number yields NaN for
   them; :func:`load_cases` adds numeric ``age_years`` plus ``age_censored``.

**The per-rhythm durations in the README are not reproducible from these files,
and the beat counts are — exactly.** Every rhythm's beat count and case count
matches the published table to the unit (408,420 Normal Sinus Rhythm beats in
370 cases, 163,270 AFIB/AFL beats in 111 cases, and so on down to 199
Unclassifiable beats in 6 cases). The duration column does not: it sums to
734,525 s — the abstract's "734,528 seconds of continuous ECG" — while the
annotated windows shipped here span **534,398 s** in total. Since the beat
annotations agree exactly, the durations were measured over intervals that
overlap each other rather than partitioning the windows; treat the published
seconds-per-rhythm as an upper bound and derive durations from
:func:`rhythm_segments` if you need figures that add up.

Reference: Eun et al., *An Anesthesiologist-Validated Large-Scale Intraoperative
Arrhythmia Dataset with Beat and Rhythm Labels*, Scientific Data 2026.
https://doi.org/10.1038/s41597-026-07076-8
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Case-level table at the dataset root: one row per annotated case, plus the
#: surgical and preoperative columns VitalDB publishes for it.
METADATA_CSV = "metadata.csv"

#: Directory holding the per-case annotation CSVs.
ANNOTATION_DIR = "Annotation_Files"

#: Filename pattern inside :data:`ANNOTATION_DIR`.
ANNOTATION_FILENAME = "Annotation_file_{case_id}.csv"

#: The VitalDB track the annotations describe, and the rate the authors read it
#: at. Not shipped here — ``vitaldb.load_case(case_id, [WAVEFORM_TRACK],
#: 1 / SAMPLING_RATE_HZ)`` is what returns samples.
WAVEFORM_TRACK = "SNUADC/ECG_II"
SAMPLING_RATE_HZ = 500
LEAD_NAME = "II"

#: ``beat_type`` value -> what it means. ``P`` is undocumented: the paper names
#: four classes, but 7 beats across four cases carry a fifth. It is surfaced as
#: written rather than folded into ``U``, because guessing "paced" from one
#: letter would put an invented label on real data.
BEAT_TYPES = {
    "N": "Normal",
    "S": "Supraventricular",
    "V": "Ventricular",
    "U": "Unclassifiable",
    "P": "Unknown (undocumented)",
}

#: ``rhythm_label`` value -> the name the README's summary table gives it. The
#: files mix abbreviations with spelled-out names, and two abbreviations are
#: wider than the README's wording suggests: ``AFIB/AFL`` covers atrial flutter
#: as well as fibrillation, and ``N`` is normal *sinus* rhythm.
RHYTHM_LABELS = {
    "N": "Normal Sinus Rhythm",
    "AFIB/AFL": "Atrial Fibrillation / Flutter",
    "Patterned Ventricular Ectopy": "Patterned Ventricular Ectopy",
    "SND": "Sinus Node Dysfunction",
    "Patterned Atrial Ectopy": "Patterned Atrial Ectopy",
    "WAP/MAT": "Wandering Atrial Pacemaker / Multifocal Atrial Rhythm",
    "Noise": "Noise",
    "SVTA": "Supraventricular Tachyarrhythmia",
    "AVB": "Atrioventricular Block",
    "VT": "Ventricular Tachyarrhythmia",
    "Unclassifiable": "Unclassifiable",
}

#: ``Noise`` is a signal-quality verdict rather than a rhythm: its segments carry
#: no beat classification at all (all 10,098 of its rows have an empty
#: ``beat_type``). Excluding it is what makes "10 rhythm categories" come out at
#: ten, and it is the reason ``rhythm_label`` and ``beat_type`` disagree on which
#: rows are beats.
NOISE_LABEL = "Noise"

#: Age above this is published as ``">89"`` rather than a number, for 2 of 482
#: cases.
AGE_CENSOR_VALUE = ">89"
AGE_CENSOR_THRESHOLD = 89

#: The one case that writes its boundary markers into ``beat_type`` instead of
#: ``bad_signal_quality_label``, and ships a ``caseid`` column no other case has.
ODD_LAYOUT_CASE = 2453
_BOUNDARY_MARKERS = ("Start", "End")

#: The canonical column order :func:`load_annotations` normalises every case to.
ANNOTATION_COLUMNS = (
    "time_second",
    "beat_type",
    "rhythm_label",
    "bad_signal_quality",
    "bad_signal_quality_label",
)

_MARKER = re.compile(r"^(?P<kind>Start|End)(?P<index>\d+)$")

#: Stands in for a missing ``rhythm_label`` while grouping runs, so that gaps form
#: their own segments instead of each NaN row becoming one. Not a shipped value.
_RHYTHM_GAP_SENTINEL = "\x00<gap>"


def _require(path: Path, what: str) -> None:
    from ecgbench.labels import LabelSourceMissingError

    if not path.exists():
        raise LabelSourceMissingError(
            f"VitalDB Arrhythmia {what} not found at {path}. Point data_path at the "
            f"dataset root — the directory holding {METADATA_CSV} and "
            f"{ANNOTATION_DIR}/ — from "
            "https://physionet.org/content/vitaldb-arrhythmia/1.0.0/ ."
        )


def _split_rhythm_classes(value: object) -> list[str]:
    """``"N, Noise, SVTA"`` -> ``["N", "Noise", "SVTA"]``."""
    if pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(",") if token.strip()]


def load_cases(data_path: Path | str) -> pd.DataFrame:
    """The case-level table, indexed by ``case_id``.

    Adds four derived columns and changes nothing else:

    - ``rhythm_class_list`` — ``rhythm_classes`` split into a list, so membership
      is a test rather than a substring search (``"N"`` is a substring of
      ``"Noise"``, so ``str.contains`` gets this wrong for 250 cases).
    - ``rhythm_class_names`` — the same list under :data:`RHYTHM_LABELS`.
    - ``age_years`` — ``age`` as a number, NaN for the two ``">89"`` patients.
    - ``age_censored`` — True for those two.

    ``subjectid`` is the patient. It is **not** unique: 482 cases come from 473
    patients, so group on it before splitting.
    """
    path = Path(data_path) / METADATA_CSV
    _require(path, "case metadata")
    df = pd.read_csv(path)

    df["rhythm_class_list"] = df["rhythm_classes"].map(_split_rhythm_classes)
    df["rhythm_class_names"] = df["rhythm_class_list"].map(
        lambda labels: [RHYTHM_LABELS.get(label, label) for label in labels]
    )
    age = df["age"].astype("string").str.strip()
    df["age_censored"] = age.eq(AGE_CENSOR_VALUE).fillna(False)
    df["age_years"] = pd.to_numeric(age.where(~df["age_censored"]), errors="coerce")

    df = df.set_index("case_id").sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"case_id is not unique in {path}; the file is not the shipped one.")
    logger.info(
        "Loaded %d VitalDB Arrhythmia cases from %d patients",
        len(df),
        df["subjectid"].nunique(),
    )
    return df


def annotation_path(data_path: Path | str, case_id: int) -> Path:
    """Path to one case's annotation CSV."""
    path = Path(data_path) / ANNOTATION_DIR / ANNOTATION_FILENAME.format(case_id=case_id)
    _require(path, f"annotations for case {case_id}")
    return path


def _normalise_odd_layout(df: pd.DataFrame) -> pd.DataFrame:
    """Fold case 2453's ``Start``/``End`` beat_type rows into the normal schema.

    That case carries its twelve segment boundaries in ``beat_type`` and has no
    ``bad_signal_quality_label`` column. The markers are moved into the column
    every other case uses and numbered in file order — ``Start1``, ``End1``,
    ``Start2`` … — which is how the other 481 cases number theirs. The rows stay:
    unlike elsewhere, they are marker-only rows that annotate no beat, so their
    ``beat_type`` becomes NaN and ``is_beat`` is False.
    """
    out = df.copy()
    out = out.drop(columns=["caseid"], errors="ignore")
    marker = out["beat_type"].isin(_BOUNDARY_MARKERS)
    if "bad_signal_quality_label" not in out.columns:
        out["bad_signal_quality_label"] = pd.Series(pd.NA, index=out.index, dtype="object")
    counts = {kind: 0 for kind in _BOUNDARY_MARKERS}
    for position in out.index[marker]:
        kind = out.at[position, "beat_type"]
        counts[kind] += 1
        out.at[position, "bad_signal_quality_label"] = f"{kind}{counts[kind]}"
    out.loc[marker, "beat_type"] = pd.NA
    return out


def load_annotations(data_path: Path | str, case_id: int) -> pd.DataFrame:
    """Beat and rhythm annotations for one case, normalised to one schema.

    Returns a frame with :data:`ANNOTATION_COLUMNS` in a fixed order plus three
    derived columns, in file order (``time_second`` is ascending in every case,
    but is **not** unique in 111 of them, so the positional index is the key):

    - ``is_beat`` — the row classifies a heartbeat. False for the 17,364 rows
      release-wide whose ``beat_type`` is empty, which include every ``Noise``
      row and the twelve boundary markers in case 2453.
    - ``beat_type_name`` / ``rhythm_label_name`` — :data:`BEAT_TYPES` and
      :data:`RHYTHM_LABELS` applied, NaN preserved.

    The three shipped column layouts (see the module docstring) all come back
    identical here.
    """
    path = annotation_path(data_path, case_id)
    df = pd.read_csv(path)

    if "caseid" in df.columns or "bad_signal_quality_label" not in df.columns:
        df = _normalise_odd_layout(df)

    missing = [column for column in ANNOTATION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Annotation file for case {case_id} is missing {missing}; expected "
            f"{list(ANNOTATION_COLUMNS)} in some order. The file is not the shipped one."
        )

    out = df.loc[:, list(ANNOTATION_COLUMNS)].reset_index(drop=True)
    out["beat_type"] = out["beat_type"].astype("string")
    out["rhythm_label"] = out["rhythm_label"].astype("string")
    out["bad_signal_quality_label"] = out["bad_signal_quality_label"].astype("string")
    out["bad_signal_quality"] = out["bad_signal_quality"].astype(bool)
    out["is_beat"] = out["beat_type"].notna()
    out["beat_type_name"] = out["beat_type"].map(BEAT_TYPES).astype("string")
    out["rhythm_label_name"] = out["rhythm_label"].map(RHYTHM_LABELS).astype("string")
    return out


def load_beats(
    data_path: Path | str,
    case_ids: list[int] | None = None,
    beats_only: bool = False,
) -> pd.DataFrame:
    """Annotations for many cases at once, with a ``case_id`` column prepended.

    ``case_ids`` defaults to every case in :data:`METADATA_CSV`. Reading all 482
    files yields 676,250 rows, of which 658,874 classify a beat — pass
    ``beats_only=True`` to keep only those.

    Example:
        >>> beats = load_beats("/data/vitaldb-arrhythmia/1.0.0/", beats_only=True)
        >>> beats.groupby("rhythm_label_name")["beat_type"].value_counts()
    """
    if case_ids is None:
        case_ids = load_cases(data_path).index.tolist()

    frames = []
    for case_id in case_ids:
        frame = load_annotations(data_path, case_id)
        frame.insert(0, "case_id", case_id)
        frames.append(frame)
    if not frames:
        raise ValueError("case_ids is empty; nothing to load.")

    out = pd.concat(frames, ignore_index=True)
    if beats_only:
        out = out[out["is_beat"]].reset_index(drop=True)
    logger.info("Loaded %d annotation rows from %d cases", len(out), len(case_ids))
    return out


def case_window(data_path: Path | str, case_id: int) -> tuple[float, float]:
    """``(start, end)`` in seconds that a case's annotations cover.

    Read from ``metadata.csv`` rather than from the annotations, because it is the
    screened window the anesthesiologists reviewed — annotations never fall
    outside it, and both ends are what you pass to VitalDB. Offsets are into the
    **whole surgical recording**, and start as late as 33,628 s, so slicing a
    waveform from 0 gets you unlabelled signal.

    This re-reads ``metadata.csv`` on every call. For more than a handful of
    cases take the two columns off :func:`load_cases` directly:
    ``load_cases(path)[["analysis_start_time_sec", "analysis_end_time_sec"]]``.
    """
    cases = load_cases(data_path)
    if case_id not in cases.index:
        raise KeyError(f"case_id {case_id} is not in {METADATA_CSV}")
    row = cases.loc[case_id]
    return float(row["analysis_start_time_sec"]), float(row["analysis_end_time_sec"])


def rhythm_segments(data_path: Path | str, case_id: int) -> pd.DataFrame:
    """Consecutive runs of one ``rhythm_label``, collapsed into segments.

    One row per run, with ``start_second`` / ``end_second`` taken from the first
    and last annotated sample in it, ``duration_second`` their difference, and
    ``n_rows`` / ``n_beats`` counting rows and classified beats. Runs of NaN
    ``rhythm_label`` are kept as their own segments rather than merged into a
    neighbour — they are gaps in the annotation, and which side they belong to is
    not recorded.

    These durations partition the case's annotated span and therefore sum to less
    than the README's per-rhythm seconds, which overlap; see the module docstring.
    """
    df = load_annotations(data_path, case_id)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "rhythm_label",
                "rhythm_label_name",
                "start_second",
                "end_second",
                "duration_second",
                "n_rows",
                "n_beats",
            ]
        )

    label = df["rhythm_label"]
    # Two traps in one line. NaN != NaN, so an unfilled compare makes every gap row
    # its own run; and on the nullable "string" dtype the compare against the
    # shifted first element is pd.NA rather than True, which cumsum propagates —
    # silently dropping the first row of the first segment of every case.
    filled = label.fillna(_RHYTHM_GAP_SENTINEL)
    runs = filled.ne(filled.shift()).fillna(True).astype(int).cumsum()
    grouped = df.groupby(runs, sort=True)
    out = pd.DataFrame(
        {
            "case_id": case_id,
            "rhythm_label": grouped["rhythm_label"].first(),
            "rhythm_label_name": grouped["rhythm_label_name"].first(),
            "start_second": grouped["time_second"].first(),
            "end_second": grouped["time_second"].last(),
            "n_rows": grouped.size(),
            "n_beats": grouped["is_beat"].sum(),
        }
    ).reset_index(drop=True)
    out["duration_second"] = out["end_second"] - out["start_second"]
    return out.loc[
        :,
        [
            "case_id",
            "rhythm_label",
            "rhythm_label_name",
            "start_second",
            "end_second",
            "duration_second",
            "n_rows",
            "n_beats",
        ],
    ]


def bad_signal_intervals(data_path: Path | str, case_id: int) -> pd.DataFrame:
    """The ``StartN``/``EndN`` marker pairs, decoded into intervals.

    One row per interval with ``marker_index``, ``start_second``, ``end_second``,
    ``duration_second`` and ``closed`` — False where the file gives a ``Start``
    with no matching ``End``, which happens in 17 of the 482 cases. An unclosed
    interval is closed at the case's last annotated sample and logged at WARNING;
    dropping it would silently hide a bad-quality stretch, and inventing an end
    time elsewhere would be a guess.

    ``bad_signal_quality`` is the per-row flag and is the one to filter beats on;
    these intervals are the segment boundaries the annotators marked, which is
    what you want when cutting the waveform.
    """
    df = load_annotations(data_path, case_id)
    markers = df[df["bad_signal_quality_label"].notna()]

    starts: dict[int, float] = {}
    rows = []
    for _, marker in markers.iterrows():
        parsed = _MARKER.match(str(marker["bad_signal_quality_label"]))
        if parsed is None:
            raise ValueError(
                f"Case {case_id} has bad_signal_quality_label "
                f"{marker['bad_signal_quality_label']!r}, which is neither StartN nor "
                "EndN. The file is not the shipped one."
            )
        index = int(parsed.group("index"))
        if parsed.group("kind") == "Start":
            starts[index] = float(marker["time_second"])
        else:
            rows.append(
                {
                    "marker_index": index,
                    "start_second": starts.pop(index, float("nan")),
                    "end_second": float(marker["time_second"]),
                    "closed": True,
                }
            )

    last = float(df["time_second"].iloc[-1]) if len(df) else float("nan")
    for index, start in sorted(starts.items()):
        logger.warning(
            "Case %d: bad-signal interval Start%d has no End%d; closing it at the "
            "last annotated sample (%.3f s).",
            case_id,
            index,
            index,
            last,
        )
        rows.append(
            {"marker_index": index, "start_second": start, "end_second": last, "closed": False}
        )

    out = pd.DataFrame(
        rows, columns=["marker_index", "start_second", "end_second", "closed"]
    ).sort_values("marker_index", ignore_index=True)
    out.insert(0, "case_id", case_id)
    out["duration_second"] = out["end_second"] - out["start_second"]
    return out


def load_vitaldb_arrhythmia(data_path: Path | str) -> pd.DataFrame:
    """Case table with per-case annotation counts attached, indexed by ``case_id``.

    The convenience entry point: :func:`load_cases` plus ``n_rows``, ``n_beats``
    and one ``beats_<X>`` column per :data:`BEAT_TYPES` letter, computed from the
    annotation files rather than read from ``total_beats`` — which counts rows,
    not beats.

    Reads all 482 annotation files, so it takes a few seconds.

    Example:
        >>> cases = load_vitaldb_arrhythmia("/data/vitaldb-arrhythmia/1.0.0/")
        >>> cases.groupby("subjectid")["n_beats"].sum().describe()
    """
    cases = load_cases(data_path)
    beats = load_beats(data_path, cases.index.tolist())

    counts = beats.pivot_table(
        index="case_id", columns="beat_type", values="time_second", aggfunc="size"
    )
    counts = counts.reindex(columns=list(BEAT_TYPES), fill_value=0).fillna(0).astype(int)
    counts.columns = [f"beats_{letter}" for letter in counts.columns]

    out = cases.join(
        pd.DataFrame(
            {
                "n_rows": beats.groupby("case_id").size(),
                "n_beats": beats.groupby("case_id")["is_beat"].sum(),
            }
        )
    ).join(counts)

    mismatched = out.index[out["n_rows"] != out["total_beats"]].tolist()
    if mismatched:
        logger.warning(
            "total_beats disagrees with the annotation row count for %d cases: %s",
            len(mismatched),
            mismatched[:10],
        )
    return out
