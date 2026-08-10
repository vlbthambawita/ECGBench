"""
European ST-T Database labels: ST/T change episodes, beats, rhythm, and header clinical text.

Nothing machine-readable ships with this dataset. Each record's ``.hea`` carries the
subject's age, sex, clinical findings, medications and the recorder model, and the
expert-adjudicated ground truth lives in a companion ``.atr``::

    e0103 2 250 1800000
    e0103.dat 212 200 12 0 91 56457 0 V4
    e0103.dat 212 200 12 0 751 48959 0 MLIII

    #Age: 62  Sex: M
    #Mixed angina
    #1-vessel disease (RCA)
    #Medications: nitrates, diltiazem
    #Recorder type: ICR 7200

This database's product is **ST and T episodes**, not a record-level diagnosis: two
cardiologists marked the onset, extremum and end of every interval in which the ST
segment or T wave deviated from that subject's own reference waveform, working on each
of the two signals independently, and a third resolved their disagreements. The
episodes, the beats, the rhythm spans, the per-channel signal quality and the whole of
the header text are exposed here.

Six things about this release shape the module, all verified against the files:

**1. An episode is three annotations, and the aux text is the only thing that says
which.** ``(ST0+`` opens an episode of ST elevation in signal 0, ``AST0+600`` marks its
extremum at +600 uV relative to the subject's reference, and ``ST0+)`` closes it. The
symbol (``s`` for ST, ``T`` for T) does not distinguish onset from peak from end, so
:data:`EPISODE_RE` parses the text.

**2. ``aux_note`` carries bytes past its NUL terminator, and comparing the raw string
silently splits the episode counts.** Seven of the 401 T-episode onsets read
``'(T0+\\x00\\x13'``, ``'(T0-\\x00N'``, ``'(T1-\\x00\\x1a'`` and so on. ``.strip()`` and
``.rstrip('\\x00')`` both leave the trailing garbage in place, so those seven land in
their own categories and the T-episode total comes out at 394 rather than 401.
:func:`_aux` truncates at the first NUL, which is what the WFDB format means.

**3. ``++`` and ``--`` are not episodes.** Within a T episode whose deviation exceeds
400 uV, extra ``T`` annotations mark where it crosses that threshold. There are 166 of
them and counting them as episodes would inflate the T total by 41%. They are reported
separately as ``n_extreme_t_markers``.

**4. Lower-case episode text is an artefact, not a finding.** In six records (e0161,
e0509, e0601, e0611, e0613, e0615) a positional change shifts the axis and *looks* like
an ST or T change; those spans are annotated with ``"`` comment annotations spelled
``(st0+``/``(t1-`` in lower case precisely so they can be told apart. There are 21 of
them and they are **excluded** from the episode counts, under
``n_axis_shift_episodes``. Case-folding the aux text would fold real ischaemia together
with recognised artefact.

**5. Twelve episodes are never closed.** Eight T and four ST episodes have an onset and
an extremum but no end annotation. Ten of the twelve open in the last four minutes of
the record and plainly run past its end, but record e0409's two ST depressions open at
8.4 min and 17.2 min and are simply unterminated. This module closes an open episode at
the end of the record and counts them in ``n_unterminated_episodes`` rather than
dropping them, which would lose e0409's ischaemia entirely.

**6. The ``~`` subtype bitmask does not match the shipped documentation.**
``annotations.shtml`` tabulates nine values, and three of them disagree with the files:
it gives ``un`` as ``0x12``, ``cu`` as ``0x20`` and ``nu`` as ``0x21``, but the release
contains ``0x13``, ``0x22`` and ``0x23`` and none of the documented three. The table is
internally inconsistent — its own ``uc`` is ``0x11``, which already sets the
"noisy" bit for signal 0 — so :func:`decode_quality` reads it as the bitmask it is
(bit 0/1 noisy for signal 0/1, bit 4/5 unreadable for signal 0/1, unreadable also
setting noisy) rather than as a lookup of nine literals, and no value in the release
falls outside it.

Two further notes on coverage:

- **Rhythm is annotated from the start of every record.** All 90 open with a ``+``
  between sample 2 and 264 — ``(N`` in 88, ``(SBR`` in e0303 and e0611 — so
  ``rhythm_secs_*`` covers the whole recording bar under a second.
- **Signal quality is not.** 89 of 90 records carry ``~`` annotations but none has one
  at sample 0; the first falls at a median of 9.3 min. A ``~`` marks a *change*, so the
  span before the first one is clean by implication, which is how WFDB reads it and what
  this module counts. ``quality_head_unasserted_secs`` reports that span so the
  assumption is visible. e0121 carries no ``~`` at all.

**Subject identity is reconstructed, not released.** See
:func:`reconstruct_patient_ids`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

#: AAMI EC57 five-class beat reduction, shared with the other MIT-BIH-family databases
#: in this catalogue rather than copied, so the reductions cannot drift apart. ``n``
#: (supraventricular escape) occurs only here, which is why it is in that table.
from ecgbench.labels.svdb import AAMI_CLASSES, AAMI_ORDER

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Annotator extension, per the shipped ANNOTATORS file. It is the only one.
ANNOTATOR = "atr"

#: Samples per record and sampling rate. Uniform across all 90 (1,800,000 at 250 Hz =
#: 7200 s exactly), and used to close the rhythm, quality and episode spans that have
#: no annotation after them.
RECORD_SAMPLES = 1800000
SAMPLING_RATE = 250
RECORD_SECONDS = RECORD_SAMPLES / SAMPLING_RATE

#: Beat symbols occurring in this release, descending by frequency. These eight sum to
#: 790,565 of the 802,909 annotations; the rest are the non-beat markers below.
BEAT_SYMBOLS = ("N", "V", "S", "F", "Q", "n", "J", "a")

BEAT_NAMES = {
    "N": "normal beat",
    "V": "premature ventricular contraction",
    "S": "supraventricular premature or ectopic beat",
    "F": "fusion of ventricular and normal beat",
    "Q": "unclassifiable beat",
    "n": "supraventricular escape beat",
    "J": "nodal (junctional) premature beat",
    "a": "aberrated atrial premature beat",
}

#: Non-beat annotation symbols, mapped to the column counting them. None of these may
#: be added to ``n_beats``.
NON_BEAT_COLUMNS = {
    "+": "n_rhythm_changes",
    "~": "n_quality_changes",
    "s": "n_st_annotations",
    "T": "n_t_annotations",
    '"': "n_comment_annotations",
    "|": "n_isolated_artifacts",
}

#: Rhythm codes carried in the ``aux_note`` of a ``+`` annotation. All 765 in the
#: release carry one of these ten.
RHYTHM_NAMES = {
    "N": "normal sinus rhythm",
    "VT": "ventricular tachycardia",
    "T": "ventricular trigeminy",
    "B": "ventricular bigeminy",
    "AB": "atrial bigeminy",
    "SVTA": "supraventricular tachyarrhythmia",
    "SAB": "sino-atrial block",
    "SBR": "sinus bradycardia",
    "AFIB": "atrial fibrillation",
    "B3": "third degree heart block",
}

#: Free-text ``aux_note`` values on ``"`` comment annotations that are not axis-shift
#: episode markers.
COMMENT_NOTE_NAMES = {
    "BUTTON": "patient-activated event button pressed",
    "TS": "tape slippage",
}

#: ST/T episode annotation text. Groups: ``(`` onset, ``A`` extremum, ``ST``/``T`` kind,
#: signal ``0``/``1``, direction (``+``/``-`` for an episode, doubled for an extreme-T
#: threshold crossing), the extremum magnitude in microvolts, and ``)`` for the end.
#: Case matters: the lower-case spellings are recognised axis-shift artefact, not
#: findings, and are matched by :data:`AXIS_SHIFT_RE` instead.
EPISODE_RE = re.compile(r"^(?:(\()|(A))?(ST|T)([01])([+-]{1,2})(\d*)(\))?$")

#: Axis-shift pseudo-episodes: the same grammar in lower case, on ``"`` annotations.
AXIS_SHIFT_RE = re.compile(r"^(?:\(|a)?(st|t)[01][+-]{1,2}\d*\)?$")

#: Episode kinds and the two signals, for the per-kind count columns.
EPISODE_KINDS = ("ST", "T")
SIGNALS = ("0", "1")

#: ``~`` subtype bitmask. The shipped documentation tabulates nine literal values and
#: three of them do not occur in the release; these bits do, for all 8,918
#: annotations. Unreadable also sets the channel's noisy bit, which is why the
#: documented ``uc`` is ``0x11`` and not ``0x10``.
QUALITY_BITS = {
    "sig0_noisy": 0x01,
    "sig1_noisy": 0x02,
    "sig0_unreadable": 0x10,
    "sig1_unreadable": 0x20,
}

#: Quality states per channel, worst last — ``decode_quality`` returns one per signal.
QUALITY_STATES = ("clean", "noisy", "unreadable")

#: ST-episode count bands, the axis folds are stratified on. Fixed edges rather than
#: quantiles: a quantile boundary moves when the input moves, so a re-release with one
#: extra episode would silently relabel records that had not changed. See
#: :func:`attach_stratify_class`.
ST_BURDEN_EDGES = (1, 3, 6)
ST_BURDEN_BANDS = ("none", "1-2", "3-5", "6+")

#: Angina descriptions in the header, mapped to ``angina_type``. "Angina pectoris" and
#: "Chest pain" state the symptom without the pattern, so they map to ``unspecified``
#: rather than being forced into one of the three.
ANGINA_TYPES = {
    "resting angina": "resting",
    "mixed angina": "mixed",
    "effort angina": "effort",
    "angina pectoris": "unspecified",
    "chest pain": "unspecified",
}

#: Myocardial-infarction descriptions, mapped to ``mi_location``.
MI_LOCATIONS = {
    "myocardial infarction": "unspecified",
    "inferior myocardial infarction": "inferior",
    "anterior myocardial infarction": "anterior",
    "infero-lateral myocardial infarction": "infero-lateral",
    "non-q myocardial infarction": "non-Q",
}

#: Coronary-angiography findings that assert normal vessels. The release spells this
#: three ways.
NORMAL_CORONARY = {
    "normal coronary arteries",
    "normal coronary vessels",
    "normal coronary artery vessels",
    "no coronary artery disease",
}

#: Drug-name spelling fixes. The release contains one typo, in e0404.
MEDICATION_FIXES = {"nidefipine": "nifedipine"}

_AGE_SEX_RE = re.compile(r"^Age:\s*(?P<age>\S+)\s+Sex:\s*(?P<sex>\S+)\s*$")
_RECORDER_RE = re.compile(r"^Recorder type:\s*(?P<recorder>.*)$")
_MEDICATIONS_RE = re.compile(r"^Medications:\s*(?P<medications>.*)$")
_VESSELS_RE = re.compile(r"^(?P<n>\d)-vessel disease(?:\s*\((?P<which>[^)]*)\))?")


def _aux(value: object) -> str:
    """``aux_note`` truncated at its NUL terminator.

    The WFDB aux field is a length-prefixed byte string and the bytes after the NUL are
    not part of it, but ``wfdb.rdann`` hands the whole thing back. Seven T-episode
    onsets in this release carry trailing garbage, so ``.strip()`` is not enough — see
    note 2 in the module docstring.
    """
    if value is None:
        return ""
    return str(value).split("\x00", 1)[0].strip()


def decode_quality(subtype: int) -> tuple[str, str]:
    """Signal-quality state of each channel, from a ``~`` annotation's subtype.

    Returns ``(signal_0, signal_1)`` drawn from :data:`QUALITY_STATES`. Read as a
    bitmask rather than as the nine-value lookup table in ``annotations.shtml``,
    because three of that table's values do not occur in the release and three that do
    are absent from it — see note 6 in the module docstring.
    """
    out = []
    for signal in ("sig0", "sig1"):
        if subtype & QUALITY_BITS[f"{signal}_unreadable"]:
            out.append("unreadable")
        elif subtype & QUALITY_BITS[f"{signal}_noisy"]:
            out.append("noisy")
        else:
            out.append("clean")
    return out[0], out[1]


def parse_header_comments(hea_path: Path) -> dict[str, object]:
    """Parse one EDB header's comment block into demographics, clinical fields and text.

    Missing pieces come back empty or NaN rather than raising, so one malformed header
    cannot fail the whole scan; genuinely broken records are flagged by the validation
    engine's ``corrupt_header`` check.

    Age and sex are the *subject's*; everything from the clinical lines is subject-level
    too. Only the recorder model describes this particular recording.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    comments = [line.lstrip("#").strip() for line in lines if line.startswith("#")]

    out: dict[str, object] = {
        "age": np.nan,
        "sex": "",
        "recorder_type": "",
        "medications": "",
        "angina_type": "",
        "myocardial_infarction": False,
        "mi_location": "",
        "n_diseased_vessels": np.nan,
        "diseased_vessels": "",
        "lca_main_stem": False,
        "normal_coronary_arteries": False,
        "coronary_angiography": True,
        "hypertension": False,
        "bypass_graft": False,
        "other_findings": "",
        "clinical_findings": "",
    }

    findings: list[str] = []
    other: list[str] = []

    for comment in comments:
        match = _AGE_SEX_RE.match(comment)
        if match:
            age = match.group("age")
            # "-" is the release's "unknown" for the one subject whose information is
            # missing (e0166) and for e0418's age. Keeping it as a number is not an
            # option and keeping it as 0 would drag every mean down.
            out["age"] = float(age) if age.isdigit() else np.nan
            sex = match.group("sex")
            out["sex"] = sex if sex in ("M", "F") else ""
            continue
        match = _RECORDER_RE.match(comment)
        if match:
            out["recorder_type"] = match.group("recorder").strip()
            continue
        match = _MEDICATIONS_RE.match(comment)
        if match:
            drugs = [d.strip().lower() for d in match.group("medications").split(",")]
            drugs = [MEDICATION_FIXES.get(d, d) for d in drugs if d]
            out["medications"] = "|".join(drugs)
            continue

        # Anything else is a clinical finding. Kept verbatim as well as parsed, because
        # the parse is lossy by design and the raw text is the source of record.
        findings.append(comment)
        low = comment.lower()
        if low in ANGINA_TYPES:
            # A record naming both a pattern and bare "angina pectoris" keeps the
            # pattern: it is the more specific statement.
            if not out["angina_type"] or out["angina_type"] == "unspecified":
                out["angina_type"] = ANGINA_TYPES[low]
        elif low in MI_LOCATIONS:
            out["myocardial_infarction"] = True
            if not out["mi_location"] or out["mi_location"] == "unspecified":
                out["mi_location"] = MI_LOCATIONS[low]
        elif low in NORMAL_CORONARY:
            out["normal_coronary_arteries"] = True
            out["n_diseased_vessels"] = 0.0
        elif low == "no coronary angiography":
            out["coronary_angiography"] = False
        elif low == "arterial hypertension":
            out["hypertension"] = True
        elif "by-pass graft" in low:
            out["bypass_graft"] = True
        else:
            match = _VESSELS_RE.match(comment)
            if match:
                out["n_diseased_vessels"] = float(match.group("n"))
                which = match.group("which") or ""
                out["diseased_vessels"] = "|".join(
                    v.strip().upper() for v in which.split(",") if v.strip()
                )
                if "lca main stem" in low:
                    out["lca_main_stem"] = True
            elif low == "lca main stem":
                out["lca_main_stem"] = True
            else:
                # "Coronary artery disease" with no vessel count, "Hyperkalemia",
                # "Chronic renal failure", "Aortic valvular regurgitation", ...
                other.append(comment)

    out["other_findings"] = "|".join(other)
    out["clinical_findings"] = "|".join(findings)
    return out


def parse_lead_names(hea_path: Path) -> list[str]:
    """Lead names in the order *this* record stores them.

    Read per record because the layout is not constant and cannot be inferred: all 90
    records store two leads, but they use **fifteen different orderings of eleven
    different lead pairs**. V5 appears in 51 records and MLIII in 47, and no lead
    appears in all 90. ``config.record_lead_layouts`` is what makes
    ``ECGDataset(leads=...)`` honour that; this column is what lets a user see it
    without opening a header.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    names = []
    for line in lines[1:]:
        if line.startswith("#") or not line.strip():
            continue
        names.append(line.split()[-1])
    return names


def _span_seconds(
    transitions: list[tuple[int, tuple[str, str]]],
) -> tuple[dict[str, float], float]:
    """Seconds each channel spends in each quality state, plus the unasserted head.

    ``transitions`` is ``(sample, (state_sig0, state_sig1))`` in time order. A ``~``
    marks a *change*, so the span before the first one was never asserted; it is
    counted as clean (which is what WFDB does) and returned separately so the
    assumption stays visible.
    """
    secs = {f"sig{i}_{state}_secs": 0.0 for i in (0, 1) for state in QUALITY_STATES}
    head = transitions[0][0] / SAMPLING_RATE if transitions else RECORD_SECONDS
    secs["sig0_clean_secs"] += head
    secs["sig1_clean_secs"] += head
    for i, (sample, states) in enumerate(transitions):
        end = transitions[i + 1][0] if i + 1 < len(transitions) else RECORD_SAMPLES
        duration = (end - sample) / SAMPLING_RATE
        for channel, state in enumerate(states):
            secs[f"sig{channel}_{state}_secs"] += duration
    return secs, head


def _union_seconds(spans: list[tuple[int, int]]) -> float:
    """Seconds covered by the union of ``(start, end)`` sample spans.

    The two signals are annotated independently, so ST episodes in signal 0 and signal 1
    overlap in time and summing their durations can exceed the record length — e0607
    reaches 131.5 min of ST in a 120-min recording. This is the bounded companion
    measure.
    """
    if not spans:
        return 0.0
    # Sort BEFORE seeding: a span is appended when its episode closes, and the two
    # signals close independently, so the first span appended is often not the
    # earliest one. Seeding from spans[0] understates the union whenever a longer
    # episode in one signal encloses a shorter one in the other.
    ordered = sorted(spans)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total / SAMPLING_RATE


def summarise_annotations(record_path: Path) -> dict[str, object]:
    """Summarise one record's reference annotations.

    Returns beat counts per symbol and per AAMI class, the non-beat marker counts, the
    ST/T episode inventory, seconds in each annotated rhythm, and per-channel signal
    quality seconds.
    """
    import wfdb

    counts: dict[str, object] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
    counts.update({f"aami_{cls}": 0 for cls in AAMI_ORDER})
    counts.update({column: 0 for column in NON_BEAT_COLUMNS.values()})
    counts.update({f"note_{note}": 0 for note in COMMENT_NOTE_NAMES})
    counts.update({f"rhythm_secs_{code}": 0.0 for code in RHYTHM_NAMES})
    for kind in EPISODE_KINDS:
        low = kind.lower()
        counts[f"n_{low}_episodes"] = 0
        for signal in SIGNALS:
            counts[f"n_{low}_episodes_sig{signal}"] = 0
        counts[f"n_{low}_up"] = 0
        counts[f"n_{low}_down"] = 0
        counts[f"{low}_episode_secs"] = 0.0
        counts[f"{low}_secs_any_signal"] = 0.0
        counts[f"peak_{low}_deviation_uv"] = 0
    counts.update(
        {
            "n_beats": 0,
            "n_annotations": 0,
            "n_extreme_t_markers": 0,
            "n_axis_shift_episodes": 0,
            "n_unterminated_episodes": 0,
            "rhythms": "",
            "dominant_rhythm": "",
            "dominant_rhythm_fraction": np.nan,
            "quality_head_unasserted_secs": np.nan,
            "first_beat_secs": np.nan,
            "last_beat_secs": np.nan,
        }
    )
    for i in (0, 1):
        for state in QUALITY_STATES:
            counts[f"sig{i}_{state}_secs"] = np.nan

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .atr must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    beat_set = set(BEAT_SYMBOLS)
    unexpected: set[str] = set()
    unparsed: set[str] = set()
    rhythm_episodes: list[tuple[str, int]] = []
    quality: list[tuple[int, tuple[str, str]]] = []
    # (kind, signal, direction) -> onset sample, for episodes currently open
    open_episodes: dict[tuple[str, str, str], int] = {}
    spans: dict[str, list[tuple[int, int]]] = {kind: [] for kind in EPISODE_KINDS}
    beat_samples: list[int] = []

    counts["n_annotations"] = len(annotation.symbol)
    n_ann = len(annotation.symbol)
    notes = annotation.aux_note if annotation.aux_note is not None else [""] * n_ann
    # subtype carries the signal-quality bitmask, and it is indexed by ANNOTATION, not
    # by position among the '~' ones -- zipped in here so it cannot be mis-indexed.
    subtypes = annotation.subtype if annotation.subtype is not None else [0] * n_ann

    for symbol, sample, raw, subtype in zip(annotation.symbol, annotation.sample, notes, subtypes):
        note = _aux(raw)
        sample = int(sample)
        if symbol in beat_set:
            counts[f"beat_{symbol}"] = int(counts[f"beat_{symbol}"]) + 1
            counts["n_beats"] = int(counts["n_beats"]) + 1
            beat_samples.append(sample)
            cls = AAMI_CLASSES.get(symbol)
            if cls:
                counts[f"aami_{cls}"] = int(counts[f"aami_{cls}"]) + 1
            continue
        if symbol in NON_BEAT_COLUMNS:
            column = NON_BEAT_COLUMNS[symbol]
            counts[column] = int(counts[column]) + 1
        else:
            unexpected.add(symbol)
            continue

        if symbol == "+":
            if note.startswith("("):
                rhythm_episodes.append((note[1:], sample))
            continue
        if symbol == "~":
            quality.append((sample, decode_quality(int(subtype))))
            continue
        if symbol == '"':
            if note in COMMENT_NOTE_NAMES:
                counts[f"note_{note}"] = int(counts[f"note_{note}"]) + 1
            elif AXIS_SHIFT_RE.match(note):
                # A recognised positional artefact that mimics ischaemia. Counted, and
                # deliberately kept out of the episode totals.
                if note.startswith("(") and "++" not in note and "--" not in note:
                    counts["n_axis_shift_episodes"] = int(counts["n_axis_shift_episodes"]) + 1
            else:
                unparsed.add(note)
            continue

        if symbol not in ("s", "T"):
            # '|' isolated-artifact markers reach here, and they carry no aux text. Only
            # ST and T change annotations get parsed as episodes.
            continue

        match = EPISODE_RE.match(note)
        if not match:
            unparsed.add(note)
            continue
        onset, extremum, kind, signal, direction, magnitude, end = match.groups()
        low = kind.lower()
        if len(direction) == 2:
            # A threshold crossing inside a T episode, not an episode of its own.
            counts["n_extreme_t_markers"] = int(counts["n_extreme_t_markers"]) + 1
            continue
        key = (kind, signal, direction)
        if onset:
            counts[f"n_{low}_episodes"] = int(counts[f"n_{low}_episodes"]) + 1
            counts[f"n_{low}_episodes_sig{signal}"] = (
                int(counts[f"n_{low}_episodes_sig{signal}"]) + 1
            )
            field = f"n_{low}_up" if direction == "+" else f"n_{low}_down"
            counts[field] = int(counts[field]) + 1
            open_episodes[key] = sample
        elif end:
            start = open_episodes.pop(key, None)
            if start is None:
                # No start for this end: would mean the file is inconsistent. There are
                # none in this release, so seeing it means something changed.
                logger.warning(
                    "%s: %r closes an episode that was never opened", record_path.name, note
                )
                continue
            counts[f"{low}_episode_secs"] = (
                float(counts[f"{low}_episode_secs"]) + (sample - start) / SAMPLING_RATE
            )
            spans[kind].append((start, sample))
        elif extremum and magnitude:
            peak = f"peak_{low}_deviation_uv"
            counts[peak] = max(int(counts[peak]), int(magnitude))

    # Twelve episodes across the release never close. Ten run past the end of the
    # record; e0409's two ST depressions simply stop being annotated. Closing them at
    # the record end keeps their ischaemia in the totals -- dropping them would zero
    # e0409's ST seconds entirely.
    for key, start in open_episodes.items():
        kind = key[0]
        low = kind.lower()
        counts["n_unterminated_episodes"] = int(counts["n_unterminated_episodes"]) + 1
        counts[f"{low}_episode_secs"] = (
            float(counts[f"{low}_episode_secs"]) + (RECORD_SAMPLES - start) / SAMPLING_RATE
        )
        spans[kind].append((start, RECORD_SAMPLES))

    for kind in EPISODE_KINDS:
        counts[f"{kind.lower()}_secs_any_signal"] = _union_seconds(spans[kind])

    if unexpected:
        logger.warning(
            "%s: annotation symbols outside BEAT_SYMBOLS and NON_BEAT_COLUMNS, not counted: %s",
            record_path.name,
            sorted(unexpected),
        )
    if unparsed:
        logger.warning("%s: unparsed aux_note values: %s", record_path.name, sorted(unparsed))

    if beat_samples:
        counts["first_beat_secs"] = beat_samples[0] / SAMPLING_RATE
        counts["last_beat_secs"] = beat_samples[-1] / SAMPLING_RATE

    # A rhythm annotation marks where an episode STARTS; it runs until the next one, and
    # the last runs to the end of the record. Every record opens with one inside the
    # first second, so this covers the whole recording.
    for i, (code, start) in enumerate(rhythm_episodes):
        end_sample = rhythm_episodes[i + 1][1] if i + 1 < len(rhythm_episodes) else RECORD_SAMPLES
        if code not in RHYTHM_NAMES:
            logger.warning("%s: unknown rhythm code %r", record_path.name, code)
            continue
        counts[f"rhythm_secs_{code}"] = (
            float(counts[f"rhythm_secs_{code}"]) + (end_sample - start) / SAMPLING_RATE
        )

    seconds = {
        code: float(counts[f"rhythm_secs_{code}"])
        for code in RHYTHM_NAMES
        if float(counts[f"rhythm_secs_{code}"]) > 0
    }
    if seconds:
        ordered = sorted(seconds, key=lambda c: -seconds[c])
        counts["rhythms"] = "|".join(ordered)
        counts["dominant_rhythm"] = ordered[0]
        counts["dominant_rhythm_fraction"] = seconds[ordered[0]] / sum(seconds.values())

    quality_secs, head = _span_seconds(quality)
    counts.update(quality_secs)
    counts["quality_head_unasserted_secs"] = head if quality else RECORD_SECONDS

    return counts


def _subject_key(row: pd.Series) -> tuple:
    """Header fields that identify a subject, order-insensitive over the findings.

    The findings are compared as a **set**: e0126 lists exactly the same five clinical
    lines as e0123-e0125 but with "Aortic valvular regurgitation" and "1-vessel disease
    (LAD)" swapped, so an order-sensitive comparison drops it from its own subject's
    group.
    """
    return (
        row["age"] if pd.notna(row["age"]) else -1.0,
        row["sex"],
        row["recorder_type"],
        frozenset(str(row["clinical_findings"]).split("|")),
        row["medications"],
    )


def reconstruct_patient_ids(df: pd.DataFrame) -> pd.Series:
    """Group the 90 records into the 79 subjects they came from.

    **The release ships no subject identifier.** ``edb.txt`` states that the 90 records
    come from 79 subjects, and nothing in the files says which. Left ungrouped, the six
    subjects who contributed more than one record would put the same person in train and
    test — e0118, e0119, e0121 and e0122 are one 51-year-old man recorded four times.

    So subject identity is *reconstructed* from the header: records agreeing on age,
    sex, recorder model, medications and the set of clinical findings are taken to be
    one subject. That yields 80 groups. The remaining merge is **e0206 and e0210**,
    which agree on everything except age — 55 against 53, two recordings of the same
    3-vessel-disease man on the same Oxford Medilog MR-20 in the same V5/MLI placement,
    two years apart — and merging them gives 79.

    That the reconstruction lands on the published subject count is the check that it is
    not merely plausible, and it reproduces the published demographics as well: 70 men
    aged 30-84, against ``edb.txt``'s "70 men aged 30 to 84", 8 women, and exactly one
    subject whose information is missing (e0166).

    **It is a reconstruction, and it is conservative in one direction only.** Two
    genuinely different subjects who happen to share age, sex, recorder, medication and
    findings would be merged, which costs a little fold flexibility and no correctness;
    the reverse — splitting one subject across folds — is the error that matters, and
    the count agreement bounds it. An attempt to confirm the grouping from the signals
    was inconclusive and is *not* the basis for it: normalised median QRST complexes
    compared across the same lead pair do not separate subjects in this cohort, with a
    different-sex control pair (e0203/e0206) scoring 0.978 against a within-group
    minimum of 0.758.
    """
    keys = df.apply(_subject_key, axis=1)
    groups: dict[tuple, str] = {}
    ids = []
    for record, key in zip(df["record_name"], keys):
        if key not in groups:
            groups[key] = record  # first record of the subject names them
        ids.append(groups[key])

    ids = pd.Series(ids, index=df.index, name="patient_id")

    # The one merge the header cannot make on its own: identical clinical text, same
    # recorder, same lead placement, age two years apart.
    same_subject = ("e0206", "e0210")
    if set(same_subject) <= set(df["record_name"].values):
        target = ids[df["record_name"] == same_subject[0]].iloc[0]
        ids[df["record_name"].isin(same_subject)] = target

    n = ids.nunique()
    if n != 79 and len(df) == 90:
        # Not fatal -- a user may be running against a partial copy -- but the whole
        # justification for this grouping is that it reproduces the published count.
        logger.warning(
            "Reconstructed %d subjects from %d records; edb.txt states 79 from 90. "
            "The grouping no longer reproduces the published count.",
            n,
            len(df),
        )
    else:
        logger.info("Reconstructed %d subjects from %d records", n, len(df))
    return ids


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``st_burden_band`` and ``st_t_class``, and stratify folds on the former.

    This is the **only** derivation of the stratification label — ``EDBSplitter`` reads
    the column rather than recomputing it, so the exposed label and the fold label
    cannot drift.

    Folds are balanced on **how many ST episodes a record holds**, banded at fixed
    edges of 1, 3 and 6 episodes (:data:`ST_BURDEN_EDGES`), giving 4 / 32 / 29 / 25
    records. ST change is what this database was built to evaluate, and the burden is
    wildly uneven: e0604 holds 20 episodes and four records hold none at all. Balancing
    on it is what stops a fold being all quiet records or all busy ones.

    The edges are fixed rather than quantiles so that a re-release adding one episode
    cannot silently relabel records that did not change.

    The ``none`` band holds 4 records, fewer than the 10 folds ECGBench generates, so it
    cannot appear in every fold. It is kept as its own band anyway: a record with no ST
    change at all is the negative control an ST detector is scored against, and hiding
    those four inside the "1-2" band would make them invisible.

    ``st_t_class`` is a record-level summary of which change types occur — 68 records
    have both, 18 ST only, 2 T only, 2 neither. It is **not** the fold label: with two
    classes of size 2 it cannot be spread over 10 folds.
    """
    out = df.copy()
    band_index = np.digitize(out["n_st_episodes"].to_numpy(), ST_BURDEN_EDGES)
    out["st_burden_band"] = [ST_BURDEN_BANDS[i] for i in band_index]
    out["stratify_class"] = out["st_burden_band"]

    has_st = out["n_st_episodes"] > 0
    has_t = out["n_t_episodes"] > 0
    out["st_t_class"] = np.select(
        [has_st & has_t, has_st & ~has_t, ~has_st & has_t],
        ["st_and_t", "st_only", "t_only"],
        default="none",
    )

    logger.info("ST burden bands: %s", out["st_burden_band"].value_counts().to_dict())
    logger.info("ST/T classes: %s", out["st_t_class"].value_counts().to_dict())
    return out


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so a
    stray copy in the directory cannot enter the partition.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. European ST-T labels live in the record "
            "headers and .atr annotation files, so point data_path at the dataset "
            "root — the flat directory holding e0103.hea, RECORDS and ANNOTATORS. "
            "Get it from https://physionet.org/content/edb/1.0.0/"
        )

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue
        row: dict[str, object] = {"record_name": name}
        row.update(parse_header_comments(hea))
        row["lead_names"] = "|".join(parse_lead_names(hea))
        row.update(summarise_annotations(hea.with_suffix("")))
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    df["patient_id"] = reconstruct_patient_ids(df)
    df["pvc_fraction"] = (df["beat_V"] / df["n_beats"]).where(df["n_beats"] > 0)
    df["sveb_fraction"] = (df["aami_S"] / df["n_beats"]).where(df["n_beats"] > 0)
    df["ischaemic_fraction"] = df["st_secs_any_signal"] / RECORD_SECONDS
    df["usable_fraction"] = 1.0 - (
        df[["sig0_unreadable_secs", "sig1_unreadable_secs"]].min(axis=1) / RECORD_SECONDS
    )
    logger.info(
        "Parsed %d European ST-T records from %d subjects; %d beats, "
        "%d ST and %d T episodes, %d lead layouts",
        len(df),
        df["patient_id"].nunique(),
        int(df["n_beats"].sum()),
        int(df["n_st_episodes"].sum()),
        int(df["n_t_episodes"].sum()),
        df["lead_names"].nunique(),
    )
    return df


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return European ST-T Database labels indexed by record name.

    Columns:

    - **ST/T episodes — the database's reason for existing.** ``n_st_episodes`` and
      ``n_t_episodes`` (368 and 401 across the release), split per signal
      (``n_st_episodes_sig0``/``_sig1``) and per direction (``n_st_up`` for elevation,
      ``n_st_down`` for depression; ``n_t_up``/``n_t_down`` for T amplitude).
      ``st_episode_secs`` sums the episodes of both signals and so can exceed the
      record's 7,200 s — the signals are annotated independently — while
      ``st_secs_any_signal`` is the bounded union, and ``ischaemic_fraction`` is that
      over the record. ``peak_st_deviation_uv`` / ``peak_t_deviation_uv`` are the
      largest annotated extremum in microvolts, measured against **that subject's own
      reference waveform from the first 30 s**, not against an absolute isoelectric
      line. ``n_extreme_t_markers`` counts the 400 uV threshold crossings inside T
      episodes, ``n_axis_shift_episodes`` the 21 spans the annotators identified as
      positional artefact mimicking ischaemia (**not** findings), and
      ``n_unterminated_episodes`` the 12 with no end annotation, which this module
      closes at the record end.
    - ``st_t_class`` — which change types a record holds (``st_and_t``, ``st_only``,
      ``t_only``, ``none``). A record-level summary, not the fold label.
    - ``st_burden_band`` / ``stratify_class`` — ST-episode-count band, **for fold
      construction**. Not a clinical label.
    - ``beat_N`` … ``beat_a`` — reference beat counts per symbol (see
      :data:`BEAT_NAMES`), with ``n_beats``, ``pvc_fraction`` and ``sveb_fraction``.
      ``aami_N``/``S``/``V``/``F``/``Q`` are the same beats under the AAMI EC57
      reduction, directly comparable with ``mitdb``, ``svdb`` and ``incartdb``.
      **Multi-class per record**: most records carry several beat types, so there is no
      single record-level beat label.
    - ``dominant_rhythm``, ``rhythms``, ``dominant_rhythm_fraction`` and
      ``rhythm_secs_<CODE>`` — seconds in each annotated rhythm (see
      :data:`RHYTHM_NAMES`). Every record opens with a rhythm annotation inside its
      first second, so these cover the whole recording.
    - ``sig0_clean_secs`` … ``sig1_unreadable_secs`` and ``usable_fraction`` — per
      channel signal quality from the ``~`` annotations, decoded as a bitmask (see
      :func:`decode_quality`). ``quality_head_unasserted_secs`` is the leading span
      before the first ``~``, counted as clean by implication.
    - ``note_BUTTON``, ``note_TS``, ``n_isolated_artifacts`` — the patient event button
      (54 presses), tape slippage (1) and QRS-like artefact markers.
    - ``age``, ``sex``, ``angina_type``, ``mi_location``, ``myocardial_infarction``,
      ``n_diseased_vessels``, ``diseased_vessels``, ``lca_main_stem``,
      ``normal_coronary_arteries``, ``coronary_angiography``, ``hypertension``,
      ``bypass_graft``, ``medications``, ``other_findings`` — parsed from the header
      comments, with the verbatim text kept in ``clinical_findings``. All
      subject-level, and all describing why the subject was recorded rather than what
      the recording shows. Age is NaN for e0166 and e0418.
    - ``patient_id`` — **reconstructed**, not released: 79 subjects over 90 records.
      See :func:`reconstruct_patient_ids` before relying on it.
    - ``lead_names`` — the two leads *this* record stores, pipe-separated. The release
      uses fifteen orderings of eleven lead pairs and no lead is present in all 90
      records, so this is not decoration.
    - ``recorder_type`` — one of ten Holter models; recorder-specific artefact is a real
      confounder in a 1980s ambulatory collection.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
