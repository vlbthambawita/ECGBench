"""
Sudden Cardiac Death Holter labels: clinical table, VF onset, two annotators.

23 complete Holter recordings of patients who experienced sudden cardiac death
*during the recording*. Every subject had a sustained ventricular
tachyarrhythmia and most had an actual cardiac arrest, so unlike every other
long-term database in this catalogue the interesting event is not a rhythm the
subject lives with — it is the terminal one, and it happens once.

**Nothing clinical ships inside the files.** Every header is signal
specification plus at most two comment lines::

    30 2 250 22099250 12:00:00
    30.dat 212 800 12 0 51 -24065 0 ECG
    30.dat 212 800 12 0 145 21051 0 ECG
    #Produced by xform_new from record 30, beginning at 26:35.000
    #vfon: 07:54:33

So the age, sex, history, medication and underlying rhythm come from
:data:`CLINICAL_TABLE`, transcribed from the "Clinical information" table on the
PhysioNet landing page — see that constant for why it is a literal here and what
that means for trusting it. The ``#vfon:`` comment is the one clinical fact the
files themselves carry.

**1. THE ``(AFIB`` MARKERS ARE NOT AN ATRIAL FIBRILLATION LABEL, and this is the
trap most likely to cost someone a paper.** The ``.ari`` files carry 1,019 ``+``
rhythm markers spelling ``(AFIB`` and ``(N``, in 22 of the 23 records. Held
against the landing page's own "Underlying Cardiac Rhythm" column they are wrong
in both directions: the detector finds 72.9%–99.4% atrial fibrillation in three of
the four clinically-AF subjects (35, 36, 50) but only **0.95% in record 37**,
which the page also calls atrial fibrillation, and it flags six clinically-*sinus*
records at 22%–36% (33, 38, 39, 41, 44, 46). These are unaudited detector output,
and the ``.ari`` annotator is documented as such. ``ari_afib_secs`` and its
fraction are exposed because they describe the annotation file, and named with the
``ari_`` prefix for the same reason. **Use ``underlying_rhythm``/``rhythm_class``
from the clinical table for an AF label; never the ``ari_afib_*`` columns.**

**2. There are TWO annotators covering DIFFERENT records, and neither is "the"
annotator.** The shipped ``ANNOTATORS`` file names both::

    ari     unaudited beat annotations
    atr     reference beat annotations

``.ari`` exists for all 23 records and is machine output. ``.atr`` is the audited
reference and exists for only **12** (30, 31, 32, 34, 35, 36, 41, 45, 46, 49, 51,
52) — PhysioNet calls it "an incomplete set of audited annotation files" and
invites contributions to finish it. Every beat statistic here is therefore
prefixed ``ari_`` or ``atr_``: an unprefixed ``n_beats`` would mean one thing for
half the release and another for the rest. ``has_audited_annotation`` is the
column to filter on, and the two disagree substantially where both exist — record
30 is 129,970 ``ari`` normal beats against 126,565 ``atr`` ones.

**3. The audited annotation STOPS EARLY, and in four records it stops at exactly
24 hours.** ``.atr`` beats end at 86,398.6–86,399.4 s in records 30, 32, 35 and
51 — a hard 24-hour cutoff on recordings that run to 25.1 h — leaving 1,998 s to
4,111 s unannotated at the tail. Record 49 loses 4,993.7 s and record 41 2,700.0
s. Record 51 is additionally unannotated for its first **1,078.7 s**. The
``.ari`` files behave the opposite way: they start 29.7–65.2 s in (the detector's
learning phase, see point 5) and run to within 0.2–452.7 s of the end. Read
``atr_unannotated_tail_secs`` before windowing against the audited beats — a
window past the cutoff has no reference behind it and no error says so.

**4. The two annotators use disjoint symbol vocabularies, so the AAMI reduction
is not optional.** ``.atr`` carries 54,725 ``B`` (bundle branch block, all in
record 36), 23,123 ``/`` (paced) and 412 ``f`` (paced/normal fusion) — and no
``r`` at all. ``.ari`` carries 58,820 ``r`` (R-on-T premature ventricular
contraction) and no ``B``, ``/`` or ``f``. Counting ``beat_V`` alone therefore
undercounts ventricular ectopy by a factor of three in the ``.ari`` files, and
counting ``beat_N`` alone misses all of record 36's atr beats. The
``ari_aami_*``/``atr_aami_*`` columns collapse both vocabularies onto AAMI EC57
and are the only cross-annotator-comparable counts here.

**5. ``.ari`` carries two extra layers that are not beats and must not be counted
as such.** Exactly **50** ``?`` (LEARN) annotations open every one of the 23
records, all inside the first 30–65 s, which is what the detector emits while
training; they are counted in ``ari_n_learning``. And 3,577 ``s`` (STCH)
annotations mark ST-segment episodes, with ``aux_note`` of the form ``(ST0+`` /
``ST0+)`` — channel and direction, opening and closing — in 22 of 23 records
(record 46 has none, record 47 has 363 episodes). They are counted in
``ari_n_st_episodes``. Both are unaudited, and neither is in
:data:`ecgbench.labels.svdb.AAMI_CLASSES`, so a scanner that assumed every symbol
was a beat would silently inflate ``n_beats`` by 4,727.

**6. The ``~`` quality subtype in this release is 51, which is not a valid
channel bitmask, so no per-channel noise is reported.** WFDB defines a ``~``
subtype as a bitmask over the signals — 0 clean, 1 first noisy, 2 second noisy, 3
both — and the whole of ``svdb`` and ``nsrdb`` uses exactly that. Here the only
values present in all 83 ``~`` annotations are **0 and 51**, strictly alternating
51, 0, 51, 0, so 51 plainly means "noisy" and 0 "clean", but 51 cannot be read as
a two-channel mask and this module does not pretend otherwise: it reports
``atr_noisy_secs`` and no ``noisy_ECG1``/``noisy_ECG2`` split. The annotated noise
is in any case slight — 0.161 h over the whole release, at most 0.95% of a record
(41).

**7. 20 of the 23 records contain NaN samples, and the reason is not corruption.**
Digital ``-2048`` is WFDB's invalid-sample marker in format 212, and ``wfdb``
turns it into NaN on read: 201,708 samples across the release, in every record
except 31, 33 and 46. They are short scattered analog-tape dropouts — at most
1.79 s in a run, median 4–84 ms, 26 to 900 runs per channel, worst 0.93% of a
channel (39) — not gaps. This module does **not** scan the signals to count them,
because that is 1.6 GB of packed samples and would make ``load_labels`` a
ten-minute call; :func:`scan_invalid_samples` does it on request, and
``ecgbench splits`` records the per-record count in ``original/`` under
``quality_issues``. It is also why ``clean/`` holds 3 records and ``original/`` is
the version to use — see the config.

``mean_hr_bpm``, ``sdnn_ms`` and ``rmssd_ms`` come from the ``.ari`` RR intervals,
because that is the annotator covering all 23 records, and they are whole-record
summaries spanning a terminal ventricular arrhythmia. Averaging heart rate across
the moment a subject died is not an HRV measurement of anything; take them as a
description of the annotation file and segment around ``vf_onset_secs`` if you
want physiology.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ecgbench.labels.svdb import AAMI_CLASSES, AAMI_ORDER

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: The audited reference annotator, per the shipped ``ANNOTATORS`` file
#: ("reference beat annotations"). Present for only 12 of the 23 records.
REFERENCE_ANNOTATOR = "atr"

#: The unaudited annotator ("unaudited beat annotations"). Present for all 23.
DETECTOR_ANNOTATOR = "ari"

#: The one clinical fact the release asserts about every subject, and the value of
#: ``cohort_label`` for all 23 records: "All patients had a sustained ventricular
#: tachyarrhythmia, and most had an actual cardiac arrest."
COHORT_LABEL = "sudden_cardiac_death"

#: Sampling rate, uniform across all 23 records.
SAMPLING_RATE = 250

#: Beat symbols occurring in the ``.atr`` files, descending by frequency. These 11
#: sum to 849,831; the remaining 16,486 annotations are ``|`` and ``~``. Note the
#: absence of ``r`` and the presence of ``B``, ``/`` and ``f`` — the opposite of
#: :data:`DETECTOR_BEAT_SYMBOLS`.
REFERENCE_BEAT_SYMBOLS = ("N", "B", "V", "/", "J", "f", "S", "F", "Q", "E", "a")

#: Beat symbols occurring in the ``.ari`` files, descending by frequency. These six
#: sum to 1,888,495; the remaining 5,746 annotations are ``+``, ``?`` and ``s``.
DETECTOR_BEAT_SYMBOLS = ("N", "r", "S", "V", "E", "Q")

BEAT_NAMES = {
    "N": "normal beat",
    "B": "bundle branch block beat (unspecified)",
    "V": "premature ventricular contraction",
    "/": "paced beat",
    "J": "nodal (junctional) premature beat",
    "f": "fusion of paced and normal beat",
    "S": "supraventricular premature or ectopic beat",
    "F": "fusion of ventricular and normal beat",
    "Q": "unclassifiable beat",
    "E": "ventricular escape beat",
    "a": "aberrated atrial premature beat",
    "r": "R-on-T premature ventricular contraction",
}

#: Non-beat annotation symbols, mapped to the un-prefixed column counting them.
#: ``?`` (LEARN) and ``s`` (STCH) are the two this release adds to the usual set,
#: and leaving them out would fold 4,727 non-beats into ``n_beats``. Never add any
#: of these to a beat count.
NON_BEAT_COLUMNS = {
    "+": "n_rhythm_changes",
    "~": "n_quality_changes",
    "|": "n_isolated_artifacts",
    "?": "n_learning",
    "s": "n_st_markers",
}

#: ``aux_note`` of a ``+`` annotation, as this release spells them. Both occur, in
#: near-equal numbers (513 and 506), and **neither is a trustworthy AF label** —
#: see point 1 of the module docstring.
AFIB_RHYTHM = "(AFIB"
NORMAL_RHYTHM = "(N"

#: ``aux_note`` of an ``s`` (STCH) annotation: ``(ST<channel><direction>`` opens an
#: ST episode and ``ST<channel><direction>)`` closes it. All eight combinations of
#: channel 0/1 and direction +/- occur. Episodes are counted from the openings,
#: because a few run to the end of the record and are never closed.
_ST_OPEN_RE = re.compile(r"^\(ST(?P<channel>[01])(?P<direction>[+-])$")

#: ``#vfon: HH:MM:SS`` — the onset of the terminal ventricular tachyarrhythmia,
#: **elapsed from the start of the record**, not a time of day. Present in 20 of
#: the 23 headers; absent in 40, 42 and 49, for which the landing page says
#: "(paced, no VF)", "(no VF)" and "(paced, no VF)".
_VFON_RE = re.compile(r"^#\s*vfon:\s*(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\s*$")

#: RR intervals outside this range are dropped before any HRV summary — double
#: detections below, and the gaps spanning artefact and asystole above.
RR_RANGE_SECS = (0.3, 2.0)

#: WFDB's invalid-sample marker in format 212, which ``wfdb`` reads back as NaN.
#: See point 7 of the module docstring and :func:`scan_invalid_samples`.
INVALID_SAMPLE_ADU = -2048

#: The "Clinical information" table from https://physionet.org/content/sddb/1.0.0/,
#: keyed by record name, as ``(gender, age, history, medication, underlying_rhythm)``.
#:
#: **THIS IS TRANSCRIBED FROM THE LANDING PAGE, NOT DERIVED FROM THE FILES**, and
#: that is a real caveat rather than a formality: unlike ``chfdb`` (a header
#: comment) or ``mitdb`` (a shipped directory), nothing in this release's 109 files
#: carries a subject's age, sex, history, medication or underlying rhythm. The page
#: is the only source, so this table cannot be recomputed and cannot be checked
#: against the data — it can only be checked for *internal* consistency, which it
#: passes: the ``rhythm_class`` reduction below reproduces PhysioNet's own summary
#: of the cohort ("18 patients with underlying sinus rhythm (4 with intermittent
#: pacing), 1 who was continuously paced, and 4 with atrial fibrillation") exactly.
#:
#: ``"Unknown"`` is the page's own word for a missing value and is preserved
#: verbatim rather than blanked, as is record 52's ``"None listed"`` — which means
#: no medication was *recorded*, not that the subject was on none. PhysioNet is
#: explicit about why so much is missing: "Because of the retrospective nature of
#: this collection, there are important limitations. Patient information is
#: limited, and sometimes completely unavailable, including data regarding drug
#: regimens and drug dosages."
CLINICAL_TABLE: dict[str, tuple[str, str, str, str, str]] = {
    "30": ("Male", "43", "Unknown", "Unknown", "Sinus"),
    "31": ("Female", "72", "Heart failure", "digoxin; quinidine gluconate", "Sinus"),
    "32": (
        "Unknown",
        "62",
        "Coronary bypass grafting; history of arrhythmia",
        "Procan SR; beta-blocker",
        "Sinus with intermittent demand ventricular pacing; CPR at time of cardiac arrest",
    ),
    "33": ("Female", "30", "Unknown", "Unknown", "Sinus"),
    "34": ("Male", "34", "Unknown", "Unknown", "Sinus"),
    "35": ("Female", "72", "Mitral valve replacement", "digoxin", "Atrial fibrillation"),
    "36": ("Male", "75", "Cardiac surgery", "digoxin; quinidine", "Atrial fibrillation"),
    "37": ("Female", "89", "Unknown", "Unknown", "Atrial fibrillation"),
    "38": ("Unknown", "Unknown", "Unknown", "Unknown", "Sinus"),
    "39": ("Male", "66", "Acute myelogenous leukemia", "digoxin; quinidine", "Sinus"),
    "40": ("Male", "79", "Unknown", "Unknown", "Paced"),
    "41": ("Male", "Unknown", "Unknown", "Unknown", "Sinus"),
    "42": (
        "Male",
        "17",
        "Hypertrophic cardiomyopathy; positive family history of sudden death",
        "Unknown",
        "Sinus",
    ),
    "43": ("Male", "35", "Coronary artery disease", "Unknown", "Intermittent ventricular pacing"),
    "44": ("Male", "Unknown", "Unknown", "Unknown", "Sinus"),
    "45": (
        "Male",
        "68",
        "History of ventricular ectopy",
        "digoxin; quinidine gluconate",
        "Sinus",
    ),
    "46": ("Female", "Unknown", "Unknown", "Unknown", "Sinus"),
    "47": ("Male", "34", "Unknown", "Unknown", "Sinus"),
    "48": ("Male", "80", "Unknown", "Unknown", "Sinus"),
    "49": (
        "Male",
        "73",
        "Coronary artery s/p myocardial infarction; history of ventricular tachycardia",
        "Unknown",
        "Sinus with intermittent pacing",
    ),
    "50": (
        "Female",
        "68",
        "Coronary artery bypass graft; mitral valve replacement",
        "digoxin; quinidine; propranolol; potassium; diuretics",
        "Atrial fibrillation",
    ),
    "51": ("Female", "67", "Unknown", "Unknown", "Sinus with intermittent pacing"),
    "52": ("Female", "82", "Heart failure", "None listed", "Sinus"),
}

#: Values of the page's "Underlying Cardiac Rhythm" column that mean the subject's
#: own rhythm was sinus, whatever pacing rode on top of it. Record 43's
#: "Intermittent ventricular pacing" is in here rather than under ``paced``, and
#: that assignment is what makes the reduction reproduce PhysioNet's "18 patients
#: with underlying sinus rhythm (4 with intermittent pacing)": the other three
#: intermittently-paced subjects are 32, 49 and 51, and 14 + 4 = 18. Record 40,
#: "Paced" with no qualifier, is the one continuously-paced subject.
_SINUS_RHYTHMS = frozenset(
    {
        "Sinus",
        "Sinus with intermittent demand ventricular pacing; CPR at time of cardiac arrest",
        "Sinus with intermittent pacing",
        "Intermittent ventricular pacing",
    }
)

#: The page's spelling for atrial fibrillation, the one non-sinus non-paced value.
_AFIB_RHYTHM_TEXT = "Atrial fibrillation"

#: Substrings of "Underlying Cardiac Rhythm" that indicate a pacemaker was active
#: at some point, whether or not the underlying rhythm was sinus.
_PACING_MARKERS = ("pacing", "Paced")

#: The three ``rhythm_class`` values and the counts they must produce, straight
#: from PhysioNet's description of the cohort. Asserted in :func:`_rhythm_class`
#: so a typo in :data:`CLINICAL_TABLE` fails loudly instead of quietly rebalancing
#: every fold.
EXPECTED_RHYTHM_COUNTS = {"sinus": 18, "afib": 4, "paced": 1}


def _rhythm_class(rhythm: str) -> str:
    """Reduce the page's free-text rhythm to ``sinus`` / ``afib`` / ``paced``."""
    if rhythm in _SINUS_RHYTHMS:
        return "sinus"
    if rhythm == _AFIB_RHYTHM_TEXT:
        return "afib"
    if rhythm == "Paced":
        return "paced"
    # Not a fallback worth having: an unrecognised value would land in a class of
    # its own and silently change every fold, so say what happened.
    raise ValueError(
        f"Unrecognised underlying rhythm {rhythm!r}. Add it to _SINUS_RHYTHMS or "
        "extend _rhythm_class in ecgbench/labels/sddb.py — do not let it fall "
        "through into a class of its own."
    )


def clinical_frame() -> pd.DataFrame:
    """Return :data:`CLINICAL_TABLE` as a frame indexed by record name.

    Adds the derived columns: ``age`` as a float with the page's ``"Unknown"``
    becoming NaN (4 of 23 subjects: 38, 41, 44, 46), ``sex`` as ``M``/``F``/``""``
    (13 / 8 / 2 — 32 and 38 are unstated), ``rhythm_class``, ``has_pacing``, and
    ``has_history``/``has_medication`` flags that are False for both of the page's
    ways of saying nothing was recorded.

    The class counts are checked against :data:`EXPECTED_RHYTHM_COUNTS`, which is
    PhysioNet's own summary of the cohort. That check is the only external
    validation this table admits — see :data:`CLINICAL_TABLE`.
    """
    rows = []
    for name, (gender, age, history, medication, rhythm) in CLINICAL_TABLE.items():
        unstated = {"Unknown", "None listed"}
        rows.append(
            {
                "record_name": name,
                "sex": {"Male": "M", "Female": "F"}.get(gender, ""),
                "age": np.nan if age == "Unknown" else float(age),
                "history": history,
                "has_history": history not in unstated,
                "medication": medication,
                "has_medication": medication not in unstated,
                "underlying_rhythm": rhythm,
                "rhythm_class": _rhythm_class(rhythm),
                "has_pacing": any(marker in rhythm for marker in _PACING_MARKERS),
            }
        )
    df = pd.DataFrame(rows)

    counts = df["rhythm_class"].value_counts().to_dict()
    if counts != EXPECTED_RHYTHM_COUNTS:
        raise ValueError(
            f"rhythm_class counts {counts} do not match PhysioNet's description of "
            f"the cohort {EXPECTED_RHYTHM_COUNTS}. CLINICAL_TABLE or _SINUS_RHYTHMS "
            "has been edited into disagreement with the landing page."
        )
    return df


def parse_header(hea_path: Path) -> dict[str, object]:
    """Read geometry, per-record gain and the VF-onset comment out of one header.

    Two things here are per-record rather than per-release. The ADC **gain is 800
    adu/mV for 21 records and 200 for records 39 and 47**; ``wfdb`` applies each
    record's own gain so nothing downstream needs to care, but the 12-bit rail
    moves with it — ±2.55875 mV at 800 and ±10.235 mV at 200 — which is why the
    config's ``amplitude_range_mv`` is the union of the two and not the first one
    measured. And ``#vfon:`` is present in only 20 of the 23 headers.
    """
    lines = [line for line in hea_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    comments = [line for line in lines if line.startswith("#")]

    out: dict[str, object] = {"vf_onset_secs": np.nan}
    for comment in comments:
        match = _VFON_RE.match(comment)
        if match:
            # Elapsed from the start of the record, and it can exceed 24 h — record
            # 35's onset is at 24:34:56 of a 24.87 h recording.
            out["vf_onset_secs"] = float(
                int(match.group("h")) * 3600 + int(match.group("m")) * 60 + int(match.group("s"))
            )
            break
    return out


def _rhythm_seconds(
    events: list[tuple[int, str]], sig_len: int, fs: float
) -> dict[str, float | int]:
    """Turn ``.ari`` ``+`` markers into seconds the DETECTOR called AF.

    **Not an atrial fibrillation label** — see point 1 of the module docstring.
    Each marker opens an interval running to the next marker or to the end of the
    record. The span before the first marker is not counted as anything: it is
    ``(AFIB`` in every one of the 22 annotated records, so what preceded it was
    non-AF by implication, and it is returned as ``rhythm_head_unasserted_secs``
    only for the case where a re-release changes that.
    """
    out: dict[str, float | int] = {
        "afib_secs": 0.0,
        "n_afib_episodes": 0,
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
        if note == AFIB_RHYTHM:
            out["afib_secs"] = float(out["afib_secs"]) + (end - start) / fs
            out["n_afib_episodes"] = int(out["n_afib_episodes"]) + 1
        elif note != NORMAL_RHYTHM:
            unexpected.add(note)
    if unexpected:
        logger.warning("Rhythm notes outside {(AFIB, (N}, not counted: %s", sorted(unexpected))
    return out


def _quality_seconds(
    events: list[tuple[int, int]], sig_len: int, fs: float
) -> dict[str, float | int]:
    """Turn ``.atr`` ``~`` markers into noisy seconds, WITHOUT a per-channel split.

    The subtype values in this release are 0 and 51 only, strictly alternating, so
    a non-zero subtype is read as "noisy" and 0 as "clean". 51 is **not** a valid
    two-channel bitmask — WFDB would spell "both channels noisy" as 3 — so no
    ``noisy_ECG1``/``noisy_ECG2`` breakdown is offered; see point 6 of the module
    docstring. The distinct subtypes seen are returned so a re-release that starts
    using the documented encoding is visible rather than silently reinterpreted.

    Where the first marker is a transition *into* noise the leading span is clean
    by implication, which is 8 of the 9 annotated records. Record 35 is the
    exception: its single ``~`` has subtype 0, a transition into clean, so the
    12,600.3 s before it was never asserted to be anything. That span is reported
    as ``quality_head_unasserted_secs`` and counted as clean, which is what WFDB
    itself does.
    """
    out: dict[str, float | int] = {
        "noisy_secs": 0.0,
        "clean_secs": float(sig_len) / fs,
        "quality_head_unasserted_secs": 0.0,
        "quality_subtypes": "",
    }
    if not events:
        return out

    if events[0][1] == 0:
        out["quality_head_unasserted_secs"] = events[0][0] / fs

    noisy = 0.0
    for i, (start, subtype) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else sig_len
        if subtype != 0:
            noisy += (end - start) / fs
    out["noisy_secs"] = noisy
    out["clean_secs"] = sig_len / fs - noisy
    out["quality_subtypes"] = "|".join(str(s) for s in sorted({s for _, s in events}))
    return out


def summarise_annotations(
    record_path: Path, annotator: str, beat_symbols: tuple[str, ...], sig_len: int
) -> dict[str, object]:
    """Summarise one record's annotations for one annotator, columns un-prefixed.

    :func:`scan_records` prefixes every key with ``atr_`` or ``ari_``, because the
    two annotators cover different records and different symbol vocabularies — see
    points 2 and 4 of the module docstring. A missing annotation file returns the
    zero/NaN row rather than raising, which is the normal case for the 11 records
    with no ``.atr``.
    """
    import wfdb

    counts: dict[str, object] = {f"beat_{symbol}": 0 for symbol in beat_symbols}
    counts.update({f"aami_{cls}": 0 for cls in AAMI_ORDER})
    counts.update({column: 0 for column in NON_BEAT_COLUMNS.values()})
    counts.update(
        {
            "available": False,
            "n_annotations": 0,
            "n_beats": 0,
            "n_veb": 0,
            "veb_fraction": np.nan,
            "veb_per_hour": np.nan,
            "n_sveb": 0,
            "sveb_fraction": np.nan,
            "n_paced_beats": 0,
            "paced_fraction": np.nan,
            "n_ectopic_beats": 0,
            "ectopic_fraction": np.nan,
            "n_st_episodes": 0,
            "afib_secs": 0.0,
            "afib_fraction": np.nan,
            "n_afib_episodes": 0,
            "has_rhythm_annotation": False,
            "rhythm_asserted_secs": 0.0,
            "rhythm_head_unasserted_secs": 0.0,
            "noisy_secs": 0.0,
            "clean_secs": np.nan,
            "quality_head_unasserted_secs": 0.0,
            "quality_subtypes": "",
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

    if not record_path.with_suffix(f".{annotator}").exists():
        return counts

    try:
        annotation = wfdb.rdann(str(record_path), annotator)
    except Exception as e:  # one unreadable file must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, annotator, e)
        return counts

    counts["available"] = True
    fs = float(getattr(annotation, "fs", SAMPLING_RATE) or SAMPLING_RATE)
    beat_set = set(beat_symbols)
    unexpected: set[str] = set()
    rhythm_events: list[tuple[int, str]] = []
    quality_events: list[tuple[int, int]] = []
    beat_samples: list[int] = []

    counts["n_annotations"] = len(annotation.symbol)
    for symbol, sample, subtype, note in zip(
        annotation.symbol, annotation.sample, annotation.subtype, annotation.aux_note
    ):
        clean_note = str(note or "").strip().rstrip("\x00")
        if symbol in beat_set:
            counts[f"beat_{symbol}"] = int(counts[f"beat_{symbol}"]) + 1
            counts["n_beats"] = int(counts["n_beats"]) + 1
            beat_samples.append(int(sample))
            aami = AAMI_CLASSES.get(symbol)
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
                rhythm_events.append((int(sample), clean_note))
            elif symbol == "~":
                quality_events.append((int(sample), int(subtype)))
            elif symbol == "s" and _ST_OPEN_RE.match(clean_note):
                # Openings only: a few episodes run to the end of the record and
                # are never closed, so counting all markers and halving would be
                # wrong (record 44 has two openings and no closings at all).
                counts["n_st_episodes"] = int(counts["n_st_episodes"]) + 1
        else:
            unexpected.add(symbol)

    if unexpected:
        logger.warning(
            "%s.%s: symbols outside the beat and non-beat vocabularies, not counted: %s",
            record_path.name,
            annotator,
            sorted(unexpected),
        )

    n_beats = int(counts["n_beats"])
    counts["n_veb"] = int(counts["aami_V"])
    counts["n_sveb"] = int(counts["aami_S"])
    counts["n_paced_beats"] = int(counts.get("beat_/", 0)) + int(counts.get("beat_f", 0))
    counts["n_ectopic_beats"] = n_beats - int(counts["aami_N"])
    if n_beats > 0:
        counts["veb_fraction"] = int(counts["n_veb"]) / n_beats
        counts["sveb_fraction"] = int(counts["n_sveb"]) / n_beats
        counts["paced_fraction"] = int(counts["n_paced_beats"]) / n_beats
        counts["ectopic_fraction"] = int(counts["n_ectopic_beats"]) / n_beats
    if sig_len > 0:
        counts["veb_per_hour"] = int(counts["n_veb"]) / (sig_len / fs / 3600.0)

    for key, value in _rhythm_seconds(rhythm_events, sig_len, fs).items():
        counts[key] = value
    counts["has_rhythm_annotation"] = bool(rhythm_events)
    if sig_len > 0:
        counts["afib_fraction"] = float(counts["afib_secs"]) / (sig_len / fs)

    for key, value in _quality_seconds(quality_events, sig_len, fs).items():
        counts[key] = value

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


def scan_invalid_samples(data_path: Path | str) -> pd.DataFrame:
    """Count WFDB invalid samples (digital −2048, read back as NaN) per channel.

    **Not called by :func:`load_labels`, on purpose.** It reads all 1.6 GB of
    packed samples and takes minutes, where everything else in this module reads
    headers and annotation files. Call it directly when you need the per-record
    figures, or read them out of ``original/`` after ``ecgbench splits`` — the
    ``quality_issues`` column carries the count that ``nan_values`` reported.

    Measured on v1.0.0 (all 109 files verified against the release's own
    ``SHA256SUMS.txt``): 201,708 invalid samples over 20 of the 23 records, absent
    only from 31, 33 and 46. Runs are short and scattered rather than gaps — at
    most 1.79 s, median 4–84 ms, 26 to 900 runs per channel — and the worst
    affected channel is 0.93% invalid (record 39). See point 7 of the module
    docstring.

    Returns a frame indexed by record name with ``n_invalid_<lead>`` and
    ``invalid_fraction_<lead>`` per channel, plus totals.
    """
    import wfdb

    data_path = Path(data_path)
    names = _record_names(data_path)
    chunk = 4_000_000
    rows = []
    for name in names:
        header = wfdb.rdheader(str(data_path / name))
        sig_len = int(header.sig_len)
        leads = list(header.sig_name or [])
        counts = np.zeros(len(leads), dtype=np.int64)
        longest = np.zeros(len(leads), dtype=np.int64)
        for start in range(0, sig_len, chunk):
            stop = min(start + chunk, sig_len)
            digital = wfdb.rdrecord(
                str(data_path / name), sampfrom=start, sampto=stop, physical=False, return_res=16
            ).d_signal
            for i in range(len(leads)):
                mask = digital[:, i] == INVALID_SAMPLE_ADU
                counts[i] += int(mask.sum())
                if mask.any():
                    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
                    longest[i] = max(int(longest[i]), int((edges[1::2] - edges[0::2]).max()))
        row: dict[str, object] = {"record_name": name, "n_samples": sig_len}
        for i, lead in enumerate(leads):
            row[f"n_invalid_{lead}"] = int(counts[i])
            row[f"invalid_fraction_{lead}"] = float(counts[i]) / sig_len if sig_len else np.nan
            row[f"longest_invalid_run_secs_{lead}"] = float(longest[i]) / float(header.fs)
        row["n_invalid_total"] = int(counts.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("record_name")


def _record_names(data_path: Path) -> list[str]:
    """Read the shipped ``RECORDS`` file, or explain what is missing."""
    from ecgbench.labels import LabelSourceMissingError

    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. Sudden Cardiac Death Holter labels "
            "come from the record headers, the .ari and .atr annotation files and the "
            "clinical table on the landing page, so point data_path at the dataset "
            "root — the flat directory holding 30.hea, RECORDS and ANNOTATORS. Get it "
            "from https://physionet.org/content/sddb/1.0.0/"
        )
    return [line.strip() for line in records_file.read_text().split() if line.strip()]


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and both annotators into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so
    the 23 superseded ``.hea-`` backup headers PhysioNet keeps beside the current
    revisions cannot enter the partition. They are not merely stale copies: they
    predate the 2008 revision that renamed both signal descriptions to ``ECG``, so
    a reader pointed at them sees "record 30, signal 0" instead — and ``49.hea-``
    additionally declares 22,525,000 samples where the current ``49.hea`` declares
    22,380,957. Both are listed in the release's ``SHA256SUMS.txt``.
    """
    import wfdb

    data_path = Path(data_path)
    names = _record_names(data_path)

    clinical = clinical_frame().set_index("record_name")
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue

        row: dict[str, object] = {"record_name": name}
        if name in clinical.index:
            row.update(clinical.loc[name].to_dict())
        else:
            # The clinical table is a literal, so a record it does not cover means
            # a re-release added records. Better visible than silently blank.
            logger.warning("No clinical-table row for record %s; demographics blank", name)

        header = wfdb.rdheader(str(hea.with_suffix("")))
        sig_len = int(header.sig_len)
        row["n_samples"] = sig_len
        row["duration_secs"] = sig_len / float(header.fs)
        row["sampling_rate"] = int(header.fs)
        row["lead_names"] = "|".join(header.sig_name or [])
        # 800 adu/mV for 21 records, 200 for 39 and 47. See parse_header.
        row["adc_gain"] = "|".join(f"{float(g):g}" for g in (header.adc_gain or []))
        # Time of day the Holter tape started. No date ships anywhere in the release.
        row["start_time"] = str(header.base_time) if header.base_time else ""
        row.update(parse_header(hea))

        vf_onset = float(row["vf_onset_secs"])  # NaN for 40, 42 and 49
        row["has_vf_onset"] = bool(np.isfinite(vf_onset))
        row["vf_onset_fraction"] = (
            vf_onset / row["duration_secs"] if np.isfinite(vf_onset) else np.nan
        )
        row["secs_after_vf_onset"] = (
            row["duration_secs"] - vf_onset if np.isfinite(vf_onset) else np.nan
        )

        stem = hea.with_suffix("")
        reference = summarise_annotations(
            stem, REFERENCE_ANNOTATOR, REFERENCE_BEAT_SYMBOLS, sig_len
        )
        detector = summarise_annotations(stem, DETECTOR_ANNOTATOR, DETECTOR_BEAT_SYMBOLS, sig_len)
        row["has_audited_annotation"] = bool(reference.pop("available"))
        detector.pop("available")
        row.update({f"atr_{k}": v for k, v in reference.items()})
        row.update({f"ari_{k}": v for k, v in detector.items()})

        # HRV from the .ari intervals, the only annotator covering all 23 records.
        for key in ("mean_hr_bpm", "sdnn_ms", "rmssd_ms", "n_rr_rejected"):
            row[key] = row[f"ari_{key}"]

        row["cohort_label"] = COHORT_LABEL
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d records: %.1f h of signal; %d unaudited beats over 23 records and "
        "%d audited beats over %d; %d records carry a VF-onset comment",
        len(df),
        df["duration_secs"].sum() / 3600,
        int(df["ari_n_beats"].sum()),
        int(df["atr_n_beats"].sum()),
        int(df["has_audited_annotation"].sum()),
        int(df["has_vf_onset"].sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``, the underlying cardiac rhythm, and say why it is that.

    This is the **only** derivation of the stratification label — ``SDDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **There is no diagnostic contrast to stratify on.** Every one of the 23
    subjects had a sustained ventricular tachyarrhythmia and most had a cardiac
    arrest, so ``cohort_label`` is one value across the release. What differs is
    the rhythm underneath the terminal event, and PhysioNet describes the cohort in
    exactly those terms: 18 subjects with underlying sinus rhythm, 4 with atrial
    fibrillation, 1 continuously paced. ``rhythm_class`` is that reduction, and
    :func:`clinical_frame` asserts those three counts.

    **Why not the finer axes.** ``StratifiedKFold`` raises only when *every* class
    holds fewer members than there are folds, so the 18-record sinus class carries
    the split on its own and the singleton ``paced`` class is tolerated (sklearn
    warns; 23 records over 10 folds gives 2–3 records a fold regardless). The
    alternatives are all worse:

    - **ventricular ectopy burden**, which is what ``svdb`` and ``chfdb`` reach
      for, rests here on the ``.ari`` detector for 11 of the 23 records because
      they have no audited annotation at all — so the bands would mean two
      different things in one column;
    - **VF onset present**, 20 against 3, puts almost everything in one class and
      splits on whether a comment was written rather than on the signal;
    - **sex**, 13 M / 8 F / 2 unstated, is balanced but says nothing about the
      recording, and the two unstated subjects would form a third class;
    - **audited-annotation availability**, 12 against 11, is the best-balanced
      column in the release and is a fact about PhysioNet's annotation backlog,
      not about the patients.

    Use ``rhythm_class`` as a covariate, ``vf_onset_secs`` and the ``atr_*``/
    ``ari_*`` ectopy columns as targets, and never ``stratify_class``.
    """
    out = df.copy()
    # Missing rhythm would be its own class and would rebalance folds silently;
    # clinical_frame() already raises on an unrecognised value, so this only
    # catches a record absent from CLINICAL_TABLE entirely.
    out["stratify_class"] = out["rhythm_class"].fillna("unknown").replace("", "unknown")
    logger.info(
        "Stratification classes (underlying rhythm): %s",
        out["stratify_class"].value_counts().to_dict(),
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Sudden Cardiac Death Holter labels indexed by record name.

    **Two provenance boundaries run through these columns and both matter.**
    ``sex``, ``age``, ``history``, ``medication``, ``underlying_rhythm``,
    ``rhythm_class`` and ``has_pacing`` are transcribed from the PhysioNet landing
    page and appear nowhere in the 109 shipped files (:data:`CLINICAL_TABLE`).
    Everything prefixed ``ari_`` is **unaudited detector output**; everything
    prefixed ``atr_`` is the audited reference but exists for only 12 of the 23
    records. Only the record geometry and ``vf_onset_secs`` come from the files
    themselves.

    Columns:

    - ``cohort_label`` — ``"sudden_cardiac_death"`` for **all 23 records**. This
      database is a positive class, not a classification task in itself.
    - ``sex``, ``age``, ``history``, ``medication``, ``has_history``,
      ``has_medication`` — the landing-page clinical table. 13 men, 8 women, 2
      unstated; ages 17–89 for the 19 subjects who have one. ``"Unknown"`` and
      ``"None listed"`` are kept verbatim and flagged False, because the page's
      "None listed" means no medication was *recorded*, not that there was none.
      Record 37 is 89, so a pipeline applying the usual ≥89 age-ceiling
      convention should know that value is real and at the boundary.
    - ``underlying_rhythm`` (verbatim), ``rhythm_class`` (``sinus`` 18 / ``afib``
      4 / ``paced`` 1), ``has_pacing`` (5 records: 32, 40, 43, 49, 51). The
      reduction reproduces PhysioNet's own summary of the cohort and
      :func:`clinical_frame` fails if it stops doing so.
    - ``vf_onset_secs``, ``has_vf_onset``, ``vf_onset_fraction``,
      ``secs_after_vf_onset`` — the ``#vfon:`` header comment, **elapsed from the
      start of the record**. Present in 20 of 23; absent for 40, 42 and 49, which
      the page marks "(paced, no VF)", "(no VF)" and "(paced, no VF)". Onset falls
      anywhere from 6.1% (record 37) to 98.9% (record 35) of the way through, so
      ``secs_after_vf_onset`` runs from 976 s (record 31) to 85,007 s (record 37):
      in some records there is barely a quarter-hour of signal after the terminal
      event and in others most of a day before it. **This is the column
      that makes the database what it is** — window relative to it rather than to
      the record start.
    - ``atr_*`` — the audited reference annotations, for the 12 records where
      ``has_audited_annotation`` is True: ``atr_beat_N``, ``atr_beat_B``,
      ``atr_beat_V``, ``atr_beat_/``, ``atr_beat_J``, ``atr_beat_f``,
      ``atr_beat_S``, ``atr_beat_F``, ``atr_beat_Q``, ``atr_beat_E``,
      ``atr_beat_a`` (see :data:`BEAT_NAMES`), the ``atr_aami_*`` reduction,
      ``atr_n_veb``/``atr_veb_fraction``/``atr_veb_per_hour``,
      ``atr_n_sveb``/``atr_sveb_fraction``,
      ``atr_n_paced_beats``/``atr_paced_fraction``,
      ``atr_n_isolated_artifacts`` (16,403 ``|`` markers release-wide),
      ``atr_n_quality_changes``, ``atr_noisy_secs``, ``atr_clean_secs``,
      ``atr_quality_head_unasserted_secs``, ``atr_quality_subtypes``, and the
      coverage columns below. There are **no** ``atr_*`` rhythm markers: the
      audited files contain not one ``+``.
    - ``ari_*`` — the same shape for the unaudited annotator, over all 23 records,
      with ``ari_beat_r`` instead of ``B``/``/``/``f``. Additionally
      ``ari_n_learning`` (exactly 50 per record, the detector's start-up phase),
      ``ari_n_st_markers``/``ari_n_st_episodes`` (unaudited ST-segment episodes,
      22 of 23 records), and ``ari_afib_secs``/``ari_afib_fraction``/
      ``ari_n_afib_episodes``/``ari_has_rhythm_annotation``. **The AF columns are
      not an AF label** — see point 1 of the module docstring; they disagree with
      the clinical table in both directions.
    - ``atr_annotated_secs``, ``atr_unannotated_head_secs``,
      ``atr_unannotated_tail_secs``, ``atr_annotated_fraction`` and their ``ari_``
      counterparts — where each annotator's beats actually start and stop. **Not a
      formality here:** the audited files stop at exactly 24 h in records 30, 32,
      35 and 51 and up to 4,993.7 s short in record 49, and record 51 is
      unannotated for its first 1,078.7 s. The unaudited files start 29.7–65.2 s
      in and reach within 0.2–452.7 s of the end.
    - ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, ``n_rr_rejected`` — copies of the
      ``ari_`` values, since that is the annotator covering every record. Whole-
      record summaries spanning a terminal ventricular arrhythmia: descriptive
      only, and see the module docstring on why they are not HRV results.
    - ``n_samples``, ``duration_secs``, ``sampling_rate``, ``lead_names``,
      ``adc_gain``, ``start_time``, ``signal_path`` — record geometry. All 23
      lengths differ, 14,160 s to 90,510 s, and ``adc_gain`` is 200 rather than 800
      for records 39 and 47.
    - ``stratify_class`` — the underlying rhythm, **for fold construction only**.
      See :func:`attach_stratify_class`.

    There is no patient identifier column, and that is not an omission: the
    release's own subject identifier *is* the record name — the landing page
    labels the clinical table's key "Subject Number" and its values are 30–52 —
    with one record per subject, so a patient column would duplicate the index.

    Per-record NaN counts are deliberately absent; they need a full signal read.
    Use :func:`scan_invalid_samples`, or read them from ``original/`` after
    ``ecgbench splits``.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
