"""
Post-Ictal Heart Rate Oscillations in Partial Epilepsy labels: seizures, beats, ST.

Nothing tabular ships with this release. The metadata is three things — the WFDB
headers, the ``.ari`` annotation files, and the 10-line ``times.seize`` text file
that marks when each seizure began and ended — and this module is what turns them
into one row per record.

**1. THE SUBJECT GROUPING IS RECONSTRUCTED BY ECGBENCH, NOT SHIPPED, AND IT IS
THE MOST CONSEQUENTIAL THING IN THIS FILE.** The release contains 7 records with
no subject identifier anywhere: not in the headers, not in ``RECORDS``, not in
the annotations. The paper it accompanies describes **five** patients. Taking
``patient_id_column: null`` at face value — the natural reading, since there is
no column to point at — would put three recordings of the *same woman* into
different folds and leak her across train and test. :data:`SUBJECT_IDS` is the
recovered mapping, and :func:`verify_subject_grouping` recomputes the evidence
from the signals. See :data:`SUBJECT_IDS` for the derivation and how to re-check
it.

**2. The seizure times are the point of the database, and they are not
annotations.** They live in ``times.seize``, a plain text file of ``<record>
<hh:mm:ss> <hh:mm:ss>`` lines, not in the ``.ari`` files, so nothing in the WFDB
annotation stream marks a seizure. The columns here (``n_seizures``,
``seizure_starts_secs``, ``seizure_ends_secs``, …) are the only machine-readable
form of them. Onsets and offsets were read from *simultaneous EEG* by an
electroencephalographer blinded to the heart-rate analysis; the EEG itself was
never released, so the times cannot be checked against their own source.

**3. ``times.seize`` documents 10 seizures; the paper describes 11.** The paper
reports "11 partial seizures recorded in five women patients" lasting "15-110
seconds". The shipped file lists 10 intervals, the shortest 25 s and the longest
110 s. The 110 s matches; the 15 s seizure is simply not in the file. So one
seizure of the eleven has no released interval, and any per-seizure count derived
from this database is a count of 10. Nothing in the release explains the
omission — there is no changelog of any kind here.

**4. The beat annotations are unaudited detector output, and the first 50 of
every record are its warm-up.** ``ANNOTATORS`` says "unaudited beat annotations
from an automated detector", and the ``.ari`` extension names the detector
(ARISTOTLE). Every record opens with exactly 50 ``?`` (WFDB "Learning")
annotations — 350 in all, always annotation 0 through 49, spanning the first 42 s
to 51 s — which are QRS detections the detector had not yet classified rather
than beats it found unclassifiable. They are counted in ``n_beats`` and reported
separately as ``n_learning_beats``; :data:`SZDB_AAMI_CLASSES` folds them into
AAMI ``Q``. All of it is machine output: do not train a beat classifier on it.

**5. ``s`` is not a beat symbol — it is ST change, and it carries the only
episode layer here.** 74 ``s`` annotations across the release delimit 37 ST
episodes through their ``aux_note`` (``(ST0+``/``ST0+)`` for elevation,
``(ST0-``/``ST0-)`` for depression, channel 0 being the only channel). Counting
``s`` as a beat would inflate the beat total and put a non-beat into the AAMI
reduction. The burden is very uneven: sz02 holds 24 depression episodes totalling
335.6 s, sz06 one lasting 826.4 s, and sz04 none at all.

**6. Rhythm annotation exists in exactly one record, and it sits 25 s before a
seizure.** sz02 carries the release's only two ``+`` markers: ``(AFIB`` at
10,508.6 s and ``(N`` at 10,526.0 s — 17.4 s of annotated atrial fibrillation,
ending 25 s before that record's second seizure begins at 10,551 s. The other six
records carry no ``+`` at all, so their ``af_secs`` of 0.0 means "never
assessed", not "no atrial fibrillation"; ``has_rhythm_annotation`` is the column
that tells them apart.

**7. Beat annotation covers essentially the whole of every record.** The first
beat falls 0.1 s to 0.9 s in and the last within 0.6 s of the end, in all 7
records, so any window has annotation behind it — unlike ``nsrdb``, whose beats
stop hours early.

There are **no demographics at all**. The paper says five women aged 31 to 48
without clinical evidence of cardiac disease, with partial seizures of frontal or
temporal origin, and gives an age for three of them in its figure legends (37,
48, 46) without saying which record each belongs to. None of that is per-record
and none of it is in the files, so there is no ``age`` or ``sex`` column here:
inventing one from the figure legends would attach a real person's age to the
wrong recording.

``mean_hr_bpm``, ``sdnn_ms`` and ``rmssd_ms`` are whole-record summaries over RR
intervals in :data:`RR_RANGE_SECS`. They span pre-ictal, ictal and post-ictal
time in one number, which is exactly what this database exists to separate — the
paper's finding is a transient 0.01-0.1 Hz oscillation lasting two to six minutes
after a seizure. Use the seizure times to segment; do not use these summaries as
a substitute for that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ecgbench.labels.svdb import AAMI_ORDER

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Annotator extension, per the shipped ANNOTATORS file: "unaudited beat
#: annotations from an automated detector". It is the only one.
ANNOTATOR = "ari"

#: The one clinical fact the release asserts about every record, and the value of
#: ``cohort_label`` for all 7: a patient with partial epilepsy monitored for
#: seizures. It is a constant, so it cannot balance a fold — see
#: :func:`attach_stratify_class`.
COHORT_LABEL = "partial_epilepsy"

#: Sampling rate, uniform across all 7 records.
SAMPLING_RATE = 200

#: The seizure interval file, and the only machine-readable record of what this
#: database is about. Lines are ``<record> <hh:mm:ss> <hh:mm:ss>``.
SEIZURE_FILE = "times.seize"

#: **Record to subject, reconstructed by ECGBench.** The release ships no subject
#: identifier; these five groups were recovered from the waveforms and validated
#: against the two counts the paper states.
#:
#: The evidence, in the order it was established — :func:`verify_subject_grouping`
#: recomputes the first point from the data files:
#:
#: 1. **Beat morphology puts sz02, sz03 and sz04 together and nothing else.**
#:    Median beats over a P-through-T window (-0.30 s to +0.50 s about each ``N``
#:    annotation, baseline-removed and divided by QRS peak amplitude, so the
#:    T-to-QRS ratio survives) correlate at 0.9989 between sz03 and sz04, 0.9844
#:    between sz02 and sz04 and 0.9806 between sz02 and sz03. The self-control is
#:    each record's own first-half against its own second-half, which is the
#:    ceiling a same-subject pair can reach: 0.9987 for sz03 and 0.9984 for sz04.
#:    **sz03 and sz04 resemble each other more closely than either resembles its
#:    own other half.** No other pair comes near: the best is sz01-sz07 at 0.8527
#:    against sz01's own ceiling of 0.9980, and four pairs correlate *negatively*
#:    because their T waves are inverted relative to one another. A beat-level
#:    nearest-template assignment says the same thing — 40.8% of sz04's beats
#:    match sz03's template better than their own, against 3.7% for sz01 to sz07.
#: 2. **The resulting subject count is exactly the five the paper states**, and
#:    that is the check that licenses the grouping rather than merely permitting
#:    it: sz01, {sz02, sz03, sz04}, sz05, sz06, sz07.
#: 3. **The seizures then fall the way the paper says they do.** It reports "Two
#:    of the subjects had multiple recorded seizures". Under this grouping
#:    subject 2 has 5 and subject 4 has 2 while subjects 1, 3 and 5 have one each
#:    — exactly two multi-seizure subjects. The arithmetic is discriminating:
#:    of the groupings that give five subjects, {sz05, sz06} + {sz02, sz03} is the
#:    only other one that also gives two multi-seizure subjects, and morphology
#:    rejects it outright (sz05 to sz06 correlates at 0.7278 against sz05's
#:    ceiling of 0.9970). Every other five-subject grouping — {sz01, sz07} with
#:    {sz02, sz03}, with {sz03, sz04}, and so on — implies three or four
#:    multi-seizure subjects and contradicts the paper.
#:
#: A weaker fourth signal agrees and is worth recording because it is independent
#: of the signals: the ``.dat`` files of sz02, sz03 and sz04 were digitised
#: within five minutes of one another (26 Mar 1998, 17:35 / 17:37 / 17:40) and
#: share a declared gain of 25 adu/mV.
#:
#: The names are deliberately not ``1``-``5`` or ``patient_01``: they are
#: ECGBench's, not PhysioNet's, and nothing in the release corroborates them. If
#: a future release publishes real subject ids, replace this map rather than
#: reconciling the two.
SUBJECT_IDS = {
    "sz01": "szdb_subj_1",
    "sz02": "szdb_subj_2",
    "sz03": "szdb_subj_2",
    "sz04": "szdb_subj_2",
    "sz05": "szdb_subj_3",
    "sz06": "szdb_subj_4",
    "sz07": "szdb_subj_5",
}

#: Beat symbols occurring in this release, descending by frequency. These seven
#: sum to 73,843 annotations; the remaining 76 are the 74 ``s`` ST markers and
#: the 2 ``+`` rhythm markers, neither of which is a beat.
BEAT_SYMBOLS = ("N", "?", "Q", "S", "V", "r")

BEAT_NAMES = {
    "N": "normal beat",
    "?": "learning-phase detection (detector warm-up, first 50 of every record)",
    "Q": "unclassifiable beat",
    "S": "supraventricular premature or ectopic beat",
    "V": "premature ventricular contraction",
    "r": "R-on-T premature ventricular contraction",
}

#: AAMI EC57 reduction for the symbols that occur here. It extends the shared map
#: in ``ecgbench.labels.svdb`` with one entry: ``?``, WFDB's "Learning", which no
#: audited MIT-BIH database emits and which the shared map therefore does not
#: carry. Folding it into ``Q`` is the honest reduction — a learning-phase
#: detection is a QRS the detector located and did not classify — but it means
#: ``aami_Q`` here is at least 50 per record before any genuinely unclassifiable
#: beat is counted. ``n_learning_beats`` is reported separately so the two can be
#: separated again.
SZDB_AAMI_CLASSES = {
    "N": "N",
    "?": "Q",
    "Q": "Q",
    "S": "S",
    "V": "V",
    "r": "V",
}

#: Non-beat annotation symbols, mapped to the column counting them. ``s`` (ST
#: change) and ``+`` (rhythm change) both occur; ``~`` and ``|`` do not, and are
#: listed so a re-release adding a signal-quality layer is counted rather than
#: warned about. Never add any of these to ``n_beats``.
NON_BEAT_COLUMNS = {
    "s": "n_st_markers",
    "+": "n_rhythm_changes",
    "~": "n_quality_changes",
    "|": "n_isolated_artifacts",
}

#: ``aux_note`` of an ``s`` annotation. ``0`` is the channel — there is only one.
ST_OPEN = {"(ST0+": "elevation", "(ST0-": "depression"}
ST_CLOSE = {"ST0+)": "elevation", "ST0-)": "depression"}

#: ``aux_note`` of a ``+`` annotation, as this release spells them. Note ``(AFIB``
#: and not the ``(AF`` that ``chfdb`` uses — the same rhythm, spelled differently
#: by a different annotator, which is why this is a per-dataset constant.
AF_RHYTHM = "(AFIB"
NORMAL_RHYTHM = "(N"

#: RR intervals outside this range are dropped before any HRV summary — double
#: detections below, and gaps spanning artefact above.
RR_RANGE_SECS = (0.3, 2.0)


def _parse_hms(value: str) -> float:
    """``hh:mm:ss`` from ``times.seize`` to seconds from the start of the record."""
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def load_seizure_times(data_path: Path | str) -> dict[str, list[tuple[float, float]]]:
    """Parse ``times.seize`` into ``{record: [(start_secs, end_secs), ...]}``.

    Offsets are from the start of the record. The file is the only released form
    of the seizure times and covers 10 of the paper's 11 seizures; a record that
    appears in ``RECORDS`` but not here has an empty list, which for this release
    never happens — all 7 records carry at least one seizure.
    """
    from ecgbench.labels import LabelSourceMissingError

    path = Path(data_path) / SEIZURE_FILE
    if not path.exists():
        raise LabelSourceMissingError(
            f"No {SEIZURE_FILE} under {data_path}. The seizure onset and offset times "
            "are the point of this database and ship only in that file — they are not "
            "in the WFDB annotations. Point data_path at the dataset root, the flat "
            "directory holding sz01.hea, RECORDS, ANNOTATORS and times.seize. Get it "
            "from https://physionet.org/content/szdb/1.0.0/"
        )

    out: dict[str, list[tuple[float, float]]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            logger.warning("Unparsed %s line, skipped: %r", SEIZURE_FILE, line)
            continue
        record, start, end = fields
        try:
            interval = (_parse_hms(start), _parse_hms(end))
        except ValueError:
            logger.warning("Unparsed %s timestamp, skipped: %r", SEIZURE_FILE, line)
            continue
        if interval[1] <= interval[0]:
            logger.warning("Non-positive seizure interval in %s: %r", SEIZURE_FILE, line)
        out.setdefault(record, []).append(interval)

    for intervals in out.values():
        intervals.sort()
    return out


def _seizure_columns(
    intervals: list[tuple[float, float]], duration_secs: float
) -> dict[str, object]:
    """Summarise one record's seizure intervals.

    ``seizure_starts_secs`` and ``seizure_ends_secs`` are pipe-joined so a record
    with two seizures survives the CSV round trip as one row; parse them with
    ``[float(v) for v in value.split("|")]``.
    """
    out: dict[str, object] = {
        "n_seizures": len(intervals),
        "seizure_starts_secs": "|".join(f"{s:g}" for s, _ in intervals),
        "seizure_ends_secs": "|".join(f"{e:g}" for _, e in intervals),
        "seizure_durations_secs": "|".join(f"{e - s:g}" for s, e in intervals),
        "seizure_secs": float(sum(e - s for s, e in intervals)),
        "first_seizure_start_secs": intervals[0][0] if intervals else np.nan,
        "longest_seizure_secs": max((e - s for s, e in intervals), default=np.nan),
        "shortest_seizure_secs": min((e - s for s, e in intervals), default=np.nan),
        # How much record there is after the last seizure ends. The post-ictal
        # oscillation this database was published for runs two to six minutes, so
        # a record with less than that left cannot show one; all 7 have hours.
        "post_ictal_tail_secs": (duration_secs - intervals[-1][1]) if intervals else np.nan,
    }
    return out


def _rhythm_seconds(
    events: list[tuple[int, str]], sig_len: int, fs: float
) -> dict[str, float | int]:
    """Turn ``+`` rhythm markers into seconds of annotated atrial fibrillation.

    Each marker opens an interval running to the next marker, or to the end of the
    record. Only sz02 has any: ``(AFIB`` then ``(N``, so the AF episode is closed
    and the span before the first marker is non-AF by implication.
    ``rhythm_head_unasserted_secs`` stays 0.0 here and exists for the opposite
    case — a first marker of ``(N``, which would imply an unmarked AF span before
    it, as happens in ``chfdb``.
    """
    out: dict[str, float | int] = {
        "af_secs": 0.0,
        "n_af_episodes": 0,
        "rhythm_asserted_secs": 0.0,
        "rhythm_head_unasserted_secs": 0.0,
    }
    if not events:
        return out

    if events[0][1] == NORMAL_RHYTHM:
        out["rhythm_head_unasserted_secs"] = events[0][0] / fs
    out["rhythm_asserted_secs"] = (sig_len - events[0][0]) / fs

    unexpected: set[str] = set()
    for i, (start, note) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else sig_len
        if note == AF_RHYTHM:
            out["af_secs"] = float(out["af_secs"]) + (end - start) / fs
            out["n_af_episodes"] = int(out["n_af_episodes"]) + 1
        elif note != NORMAL_RHYTHM:
            unexpected.add(note)
    if unexpected:
        logger.warning(
            "Rhythm notes outside {%s, %s}, not counted: %s",
            AF_RHYTHM,
            NORMAL_RHYTHM,
            sorted(unexpected),
        )
    return out


def _st_episodes(
    markers: list[tuple[int, str]], fs: float
) -> dict[str, object]:
    """Pair ``(ST0+``/``ST0+)`` and ``(ST0-``/``ST0-)`` markers into episodes.

    An episode is only counted once its matching close arrives, and a close of a
    different polarity than the open is logged rather than silently paired. All 37
    episodes in this release are well formed and none dangles at the end of a
    record, so a warning from here means a re-release changed the convention.
    """
    out: dict[str, object] = {
        "n_st_episodes": 0,
        "n_st_elevation_episodes": 0,
        "n_st_depression_episodes": 0,
        "st_secs": 0.0,
        "st_elevation_secs": 0.0,
        "st_depression_secs": 0.0,
        "longest_st_episode_secs": np.nan,
        "n_st_unclosed": 0,
    }
    open_kind: str | None = None
    open_sample = 0
    longest = 0.0
    for sample, note in markers:
        if note in ST_OPEN:
            if open_kind is not None:
                out["n_st_unclosed"] = int(out["n_st_unclosed"]) + 1
                logger.warning("ST episode %r reopened before closing at sample %d", note, sample)
            open_kind, open_sample = ST_OPEN[note], sample
        elif note in ST_CLOSE:
            kind = ST_CLOSE[note]
            if open_kind is None:
                logger.warning("ST close %r with no matching open at sample %d", note, sample)
                continue
            if kind != open_kind:
                logger.warning(
                    "ST close %r does not match open %r at sample %d", note, open_kind, sample
                )
            secs = (sample - open_sample) / fs
            out["n_st_episodes"] = int(out["n_st_episodes"]) + 1
            out[f"n_st_{open_kind}_episodes"] = int(out[f"n_st_{open_kind}_episodes"]) + 1
            out["st_secs"] = float(out["st_secs"]) + secs
            out[f"st_{open_kind}_secs"] = float(out[f"st_{open_kind}_secs"]) + secs
            longest = max(longest, secs)
            open_kind = None
        else:
            logger.warning("ST marker with unrecognised note %r at sample %d", note, sample)
    if open_kind is not None:
        out["n_st_unclosed"] = int(out["n_st_unclosed"]) + 1
        logger.warning("ST episode %r never closed", open_kind)
    if int(out["n_st_episodes"]) > 0:
        out["longest_st_episode_secs"] = longest
    return out


def summarise_annotations(record_path: Path, sig_len: int) -> dict[str, object]:
    """Summarise one record's ``.ari`` annotations.

    Returns per-symbol beat counts, the AAMI five-class reduction, ectopy burden,
    ST episodes, annotated atrial fibrillation, the annotated span against the
    record length, and whole-record HRV summaries. All of it is unaudited detector
    output — see the module docstring.
    """
    import wfdb

    counts: dict[str, object] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS if symbol != "?"}
    counts.update({f"aami_{cls}": 0 for cls in AAMI_ORDER})
    counts.update({column: 0 for column in NON_BEAT_COLUMNS.values()})
    counts.update(
        {
            "n_learning_beats": 0,
            "n_beats": 0,
            "n_annotations": 0,
            "n_veb": 0,
            "veb_fraction": np.nan,
            "veb_per_hour": np.nan,
            "n_sveb": 0,
            "sveb_fraction": np.nan,
            "n_ectopic_beats": 0,
            "ectopic_fraction": np.nan,
            "n_st_episodes": 0,
            "n_st_elevation_episodes": 0,
            "n_st_depression_episodes": 0,
            "st_secs": 0.0,
            "st_elevation_secs": 0.0,
            "st_depression_secs": 0.0,
            "longest_st_episode_secs": np.nan,
            "n_st_unclosed": 0,
            "af_secs": 0.0,
            "af_fraction": np.nan,
            "n_af_episodes": 0,
            "has_rhythm_annotation": False,
            "rhythm_asserted_secs": 0.0,
            "rhythm_head_unasserted_secs": 0.0,
            "annotated_secs": np.nan,
            "unannotated_head_secs": np.nan,
            "unannotated_tail_secs": np.nan,
            "annotated_fraction": np.nan,
            "mean_hr_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "n_rr_rejected": 0,
        }
    )

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .ari must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    fs = float(getattr(annotation, "fs", SAMPLING_RATE) or SAMPLING_RATE)
    beat_set = set(BEAT_SYMBOLS)
    unexpected: set[str] = set()
    rhythm_events: list[tuple[int, str]] = []
    st_markers: list[tuple[int, str]] = []
    beat_samples: list[int] = []

    counts["n_annotations"] = len(annotation.symbol)
    for symbol, sample, note in zip(annotation.symbol, annotation.sample, annotation.aux_note):
        if symbol in beat_set:
            # "?" is a QRS the detector located during its warm-up, so it counts
            # towards n_beats and the RR series but gets its own column rather
            # than a "beat_?" one, which would be an awkward CSV header and would
            # read as a beat class it is not.
            if symbol == "?":
                counts["n_learning_beats"] = int(counts["n_learning_beats"]) + 1
            else:
                counts[f"beat_{symbol}"] = int(counts[f"beat_{symbol}"]) + 1
            counts["n_beats"] = int(counts["n_beats"]) + 1
            beat_samples.append(int(sample))
            aami = SZDB_AAMI_CLASSES.get(symbol)
            if aami is None:
                # A beat symbol with no AAMI class would vanish from the reduction
                # while still counting in n_beats — the drift the aami_* columns
                # exist to prevent.
                logger.warning("Beat symbol %r has no AAMI class, not reduced", symbol)
            else:
                counts[f"aami_{aami}"] = int(counts[f"aami_{aami}"]) + 1
        elif symbol in NON_BEAT_COLUMNS:
            column = NON_BEAT_COLUMNS[symbol]
            counts[column] = int(counts[column]) + 1
            if symbol == "+":
                rhythm_events.append((int(sample), str(note or "").strip()))
            elif symbol == "s":
                st_markers.append((int(sample), str(note or "").strip()))
        else:
            unexpected.add(symbol)

    if unexpected:
        logger.warning(
            "%s: annotation symbols outside BEAT_SYMBOLS and NON_BEAT_COLUMNS, not counted: %s",
            record_path.name,
            sorted(unexpected),
        )

    n_beats = int(counts["n_beats"])
    counts["n_veb"] = int(counts["aami_V"])
    counts["n_sveb"] = int(counts["aami_S"])
    counts["n_ectopic_beats"] = int(counts["n_veb"]) + int(counts["n_sveb"])
    if n_beats > 0:
        counts["veb_fraction"] = int(counts["n_veb"]) / n_beats
        counts["sveb_fraction"] = int(counts["n_sveb"]) / n_beats
        counts["ectopic_fraction"] = int(counts["n_ectopic_beats"]) / n_beats
    if sig_len > 0:
        counts["veb_per_hour"] = int(counts["n_veb"]) / (sig_len / fs / 3600.0)

    for key, value in _st_episodes(st_markers, fs).items():
        counts[key] = value
    for key, value in _rhythm_seconds(rhythm_events, sig_len, fs).items():
        counts[key] = value
    counts["has_rhythm_annotation"] = bool(rhythm_events)
    if sig_len > 0:
        counts["af_fraction"] = float(counts["af_secs"]) / (sig_len / fs)

    if beat_samples:
        first, last = beat_samples[0], beat_samples[-1]
        counts["annotated_secs"] = (last - first) / fs
        counts["unannotated_head_secs"] = first / fs
        counts["unannotated_tail_secs"] = (sig_len - last) / fs
        if sig_len > 0:
            counts["annotated_fraction"] = (last - first) / sig_len

    if len(beat_samples) > 2:
        rr = np.diff(np.asarray(beat_samples, dtype=np.int64)) / fs
        low, high = RR_RANGE_SECS
        keep = (rr >= low) & (rr <= high)
        counts["n_rr_rejected"] = int((~keep).sum())
        rr = rr[keep]
        if rr.size > 1:
            counts["mean_hr_bpm"] = float(60.0 / rr.mean())
            counts["sdnn_ms"] = float(1000.0 * rr.std(ddof=1))
            counts["rmssd_ms"] = float(1000.0 * np.sqrt(np.mean(np.diff(rr) ** 2)))

    return counts


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header, annotation file and ``times.seize`` into one frame.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so
    the 7 superseded ``.hea-`` copies PhysioNet keeps beside the current headers
    cannot enter the partition. They differ substantively, not cosmetically: the
    ``.hea-`` files describe the single channel as ``column 1`` where the current
    ``.hea`` files name it ``ECG``, so a reader pointed at the old copies gets an
    unnamed channel and ``ECGDataset(leads=["ECG"])`` stops working.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. Post-Ictal Heart Rate Oscillations "
            "labels live in the record headers, the .ari annotation files and "
            "times.seize, so point data_path at the dataset root — the flat directory "
            "holding sz01.hea, RECORDS, ANNOTATORS and times.seize. Get it from "
            "https://physionet.org/content/szdb/1.0.0/"
        )

    import wfdb

    seizures = load_seizure_times(data_path)
    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue

        header = wfdb.rdheader(str(hea.with_suffix("")))
        sig_len = int(header.sig_len)
        duration_secs = sig_len / float(header.fs)

        row: dict[str, object] = {"record_name": name}
        # Reconstructed, not shipped — see SUBJECT_IDS. A record missing from the
        # map would silently become its own subject and quietly leak, so it is an
        # error rather than a default.
        if name not in SUBJECT_IDS:
            raise KeyError(
                f"Record {name!r} has no entry in ecgbench.labels.szdb.SUBJECT_IDS. That "
                "map is ECGBench's reconstruction of the 5 subjects behind these 7 "
                "records; a new record needs the reconstruction redone (see "
                "verify_subject_grouping) rather than a guessed group."
            )
        row["subject_id"] = SUBJECT_IDS[name]
        row["subject_id_is_reconstructed"] = True
        row["n_samples"] = sig_len
        row["duration_secs"] = duration_secs
        row["sampling_rate"] = int(header.fs)
        # 25 adu/mV for five records and 10 for sz05 and sz06. It sets the
        # clipping rail in millivolts, which is why it is exposed: the 8-bit
        # samples span [-100, 155] adu whatever the gain.
        row["adc_gain"] = float(header.adc_gain[0])
        row["lead_names"] = "|".join(header.sig_name or [])

        row.update(_seizure_columns(seizures.get(name, []), duration_secs))
        row.update(summarise_annotations(hea.with_suffix(""), sig_len))
        row["cohort_label"] = COHORT_LABEL
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d records from %d reconstructed subjects: %.2f h of signal, "
        "%d unaudited beat detections (%d learning-phase), %d seizures totalling %.0f s, "
        "%d ST episodes, %d record(s) with rhythm annotation",
        len(df),
        df["subject_id"].nunique(),
        df["duration_secs"].sum() / 3600,
        int(df["n_beats"].sum()),
        int(df["n_learning_beats"].sum()),
        int(df["n_seizures"].sum()),
        df["seizure_secs"].sum(),
        int(df["n_st_episodes"].sum()),
        int(df["has_rhythm_annotation"].sum()),
    )
    return df


def verify_subject_grouping(
    data_path: Path | str, annotator: str = ANNOTATOR
) -> pd.DataFrame:
    """Recompute the morphology evidence behind :data:`SUBJECT_IDS`.

    Returns one row per record pair with the cross-record median-beat correlation
    and each record's own split-half correlation, which is the ceiling a
    same-subject pair can reach. A pair whose cross-correlation approaches or
    exceeds both self-controls is the same subject.

    This is **not** run at load time and no test calls it — it needs the real
    signal files and reads all 16.8 h of them. It exists so the grouping is
    re-checkable rather than merely asserted, and so a re-release can be tested
    against it::

        from ecgbench.labels.szdb import verify_subject_grouping
        print(verify_subject_grouping("/path/to/szdb/1.0.0").head(10))

    Expect sz03-sz04 at ~0.999 — above both records' self-controls — sz02 to each
    of them at ~0.98, and the best cross-subject pair (sz01-sz07) at ~0.85.
    """
    import itertools

    import wfdb

    data_path = Path(data_path)
    names = [line.strip() for line in (data_path / "RECORDS").read_text().split() if line.strip()]
    pre, post = int(0.30 * SAMPLING_RATE), int(0.50 * SAMPLING_RATE)

    def median_beat(signal: np.ndarray, samples: np.ndarray) -> np.ndarray:
        beats = []
        for index in samples:
            window = signal[index - pre : index + post] - np.median(
                signal[index - pre : index + post]
            )
            # Normalise by QRS peak rather than by standard deviation, so the
            # T-wave-to-QRS ratio — the person-specific part — survives.
            amplitude = np.abs(window[pre - 10 : pre + 10]).max()
            if amplitude > 0:
                beats.append(window / amplitude)
        return np.median(np.asarray(beats), axis=0)

    def correlate(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    templates: dict[str, np.ndarray] = {}
    self_control: dict[str, float] = {}
    for name in names:
        signal = wfdb.rdrecord(str(data_path / name)).p_signal[:, 0].astype(np.float64)
        annotation = wfdb.rdann(str(data_path / name), annotator)
        samples = np.asarray(annotation.sample)[np.asarray(annotation.symbol) == "N"]
        samples = samples[(samples > pre) & (samples < len(signal) - post)]
        templates[name] = median_beat(signal, samples)
        half = len(samples) // 2
        self_control[name] = correlate(
            median_beat(signal, samples[:half]), median_beat(signal, samples[half:])
        )

    rows = []
    for a, b in itertools.combinations(names, 2):
        rows.append(
            {
                "record_a": a,
                "record_b": b,
                "same_subject_claimed": SUBJECT_IDS.get(a) == SUBJECT_IDS.get(b),
                "cross_correlation": correlate(templates[a], templates[b]),
                "self_control_a": self_control[a],
                "self_control_b": self_control[b],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("cross_correlation", ascending=False)
        .reset_index(drop=True)
    )


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class`` — a constant — and explain why nothing else fits.

    This is the **only** derivation of the stratification label; ``SZDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **Every fold here is one subject, and there is nothing left to balance.** The
    release is 7 records from 5 reconstructed subjects, so ``n_folds`` is 5 (see
    ``DatasetConfig.n_folds``) and ``StratifiedGroupKFold`` puts one subject in
    each fold. ``StratifiedKFold`` and ``StratifiedGroupKFold`` both raise when
    *every* class holds fewer records than there are folds, which over 7 records
    and 5 folds means a usable split needs one class of 5 or more. Measured
    against the candidate axes:

    - **seizure count**, 1 against >1: 3 subjects against 2 — *raises*, and it is
      the axis a reader would reach for first;
    - **atrial fibrillation annotated**: 1 subject against 4 — *raises*, and it
      would be a class of one record besides;
    - **ST episode burden**, at any cut: the counts per subject are 4, 25, 5, 1, 2
      episodes, so no cut gives a class of 5;
    - **record length**, 1.5 h against longer: 3 records against 4 — this one does
      not raise, but it is not constant within subject 2 (1.5 h, 3.5 h and 3.8 h),
      and an axis that varies inside a group cannot balance a grouped split.

    So ``cohort_label`` — ``"partial_epilepsy"`` for all 7 records — is what is
    left. A constant is a legitimate stratification label only because the split
    is grouped: it reduces ``StratifiedGroupKFold`` to a plain partition of the 5
    subjects, which is the leave-one-subject-out structure this database wants
    anyway. Do not read the fold layout as balanced on anything.

    Use ``n_seizures``, ``seizure_secs``, ``veb_fraction``, ``mean_hr_bpm``,
    ``sdnn_ms`` or the ``aami_*`` counts as targets; never ``stratify_class``.
    """
    out = df.copy()
    out["stratify_class"] = out["cohort_label"]
    logger.info(
        "Stratification classes (constant): %s", out["stratify_class"].value_counts().to_dict()
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Post-Ictal Heart Rate Oscillations labels indexed by record name.

    **Every annotation-derived column is unaudited machine output** — the ``.ari``
    files are an automated detector's uncorrected output, and the first 50
    detections of every record are its learning phase. Only the seizure times, the
    subject grouping and the record geometry are independent of it, and the
    subject grouping is ECGBench's reconstruction rather than a shipped field.

    Columns:

    - ``subject_id`` — one of five ``szdb_subj_*`` values, **reconstructed by
      ECGBench**, with ``subject_id_is_reconstructed`` True on every row to say
      so. sz02, sz03 and sz04 are one woman; the other four records are one each.
      See :data:`SUBJECT_IDS` for the derivation and
      :func:`verify_subject_grouping` to recompute it.
    - ``cohort_label`` — ``"partial_epilepsy"`` for all 7 records. The release
      asserts it of the cohort; it is a constant and carries no information.
    - ``n_seizures``, ``seizure_starts_secs``, ``seizure_ends_secs``,
      ``seizure_durations_secs``, ``seizure_secs``, ``first_seizure_start_secs``,
      ``longest_seizure_secs``, ``shortest_seizure_secs``,
      ``post_ictal_tail_secs`` — from ``times.seize``, the only released form of
      the seizure times. The three ``*_secs`` list columns are pipe-joined so a
      two-seizure record stays one row: split on ``"|"`` and cast to float.
      **These cover 10 seizures; the paper describes 11.**
    - ``beat_N``, ``beat_Q``, ``beat_S``, ``beat_V``, ``beat_r``,
      ``n_learning_beats``, ``n_beats``, ``n_annotations`` — detector output.
      ``n_learning_beats`` is 50 in every record by construction.
    - ``aami_N``, ``aami_S``, ``aami_V``, ``aami_F``, ``aami_Q`` — the EC57
      reduction, with ``?`` folded into ``Q`` (see :data:`SZDB_AAMI_CLASSES`), so
      ``aami_Q`` starts at 50 per record. ``aami_F`` is 0 everywhere: the detector
      emitted no fusion beats at all.
    - ``n_veb``, ``veb_fraction``, ``veb_per_hour``, ``n_sveb``,
      ``sveb_fraction``, ``n_ectopic_beats``, ``ectopic_fraction`` — ectopy
      burden. It is light: 183 ventricular and 196 supraventricular detections in
      73,843 beats, and sz06 holds a third of the ventricular ones.
    - ``n_st_episodes``, ``n_st_elevation_episodes``,
      ``n_st_depression_episodes``, ``st_secs``, ``st_elevation_secs``,
      ``st_depression_secs``, ``longest_st_episode_secs``, ``n_st_markers``,
      ``n_st_unclosed`` — the ``s`` annotation layer, 37 episodes over the
      release. Depression dominates (31 of 37).
    - ``af_secs``, ``af_fraction``, ``n_af_episodes``, ``has_rhythm_annotation``,
      ``rhythm_asserted_secs``, ``rhythm_head_unasserted_secs``,
      ``n_rhythm_changes`` — **only sz02 carries any rhythm annotation.** For the
      other six records ``af_secs`` of 0.0 means "never assessed"; read
      ``has_rhythm_annotation`` first.
    - ``annotated_secs``, ``unannotated_head_secs``, ``unannotated_tail_secs``,
      ``annotated_fraction`` — beat annotation covers ~100% of every record.
    - ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, ``n_rr_rejected`` — whole-record
      RR summaries, spanning pre-ictal, ictal and post-ictal time in one figure.
      The transient this database exists to show is invisible in them.
    - ``n_samples``, ``duration_secs``, ``sampling_rate``, ``adc_gain``,
      ``lead_names``, ``signal_path`` — record geometry. Lengths vary from 1.5 h
      to 3.77 h; ``adc_gain`` is 25 adu/mV except in sz05 and sz06, where it is 10
      and the clipping rail is therefore ±10.0/+15.5 mV rather than ±4.0/+6.2.
    - ``stratify_class`` — a constant; see :func:`attach_stratify_class`. Not a
      target.

    ``n_quality_changes`` and ``n_isolated_artifacts`` are 0 for all 7 records:
    this release has **no** signal-quality annotation layer, where ``mitdb``,
    ``svdb`` and ``nsrdb`` all do. They are exposed so a re-release adding one is
    visible, and there are deliberately no clean/noisy second counts — inventing
    them would assert 16.8 hours of assessed quality that nobody assessed.
    """
    df = attach_stratify_class(scan_records(data_path))
    return df.set_index("record_name")
