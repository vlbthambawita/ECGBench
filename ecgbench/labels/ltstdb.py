"""
Long-Term ST Database labels: ST episodes, beats, and the header's clinical record.

Nothing machine-readable ships with this dataset. Each record's ``.hea`` carries a
full clinical summary of the subject, and the expert-adjudicated ground truth lives
in three companion annotation files::

    s20021 2 250 18975000 11:00:00 28/02/1984
    s20021.dat 212 200/mV 12 0 6 -10121 0 MLIII
    s20021.dat 212 200/mV 12 0 -2 -17799 0 V4
    #Age: 55  Sex: M
    #Comments:
    #  The patient demonstrates three salvos of ST elevation with
    ...
    #Diagnoses:
    #  Prinzmetal's angina

This database's product is **ST episodes over a whole day**, not a record-level
diagnosis. Three expert groups (Ljubljana, Pisa, Cambridge) annotated every record
independently with the SEMIA tool and met to agree consensus references. What they
marked, per signal, is the onset, extremum and end of every interval in which the
ST *deviation* function left the normal range — where the deviation is measured
against a piecewise-linear baseline the annotators themselves placed, so that
positional drift and other non-ischaemic wander are subtracted out before an
episode is declared.

Seven things about this release shape the module, all verified against the files.

**1. The release counts EXTREMA, not onsets, and the difference is 22 episodes.**
An episode is three annotations — ``(st0-120`` opens it, ``ast0-160`` marks the
extremum at -160 uV, ``st0-90)`` closes it — and the natural thing to count is the
onset. That undercounts: 22 episodes were already in progress when the recording
started, so they have an extremum and an end but no onset. Counting extrema
reproduces the shipped ``.cnt`` summaries **exactly, in all 258 blocks** (86 records
x 3 criteria x 6 quantities), which is the check that this module's episode counts
are the release's own and not a re-derivation that happens to look plausible.
:func:`summarise_episodes` therefore counts extrema, and reports the 22 head-open
episodes separately as ``n_episodes_open_at_start``.

**2. There are THREE sets of ST episode annotations, not one, and they disagree by
a factor of two.** The same recordings are annotated under three detection criteria,
which differ only in how big and how long a deviation has to be:

===========  ============  ==========  ==============  ==================
Annotator    Threshold     Duration    Ischaemic eps.  Rate-related eps.
===========  ============  ==========  ==============  ==================
``.sta``     75 uV         30 s        1,795           516
``.stb``     100 uV        30 s        1,130           234
``.stc``     100 uV        60 s        857             116
===========  ============  ==========  ==============  ==================

None is more correct than the others; the release ships all three because the right
criterion depends on the application. This module makes ``.sta`` the unsuffixed
default because it is the most inclusive, and exposes the other two under ``_b`` and
``_c`` suffixes. **Quote the criterion with any figure from this database** — a paper
reporting "1,795 ischaemic episodes" and one reporting "857" can both be right.

The four non-episode quantities — axis shifts, conduction-change shifts, noise events
and unreadable intervals — are *identical* in all three files (1,493 / 895 / 31 / 60),
because they are marks rather than threshold crossings. They get no suffix.

**3. Ischaemic and rate-related episodes are different findings, and both are
here.** The database exists to tell them apart: ``(st0-120`` is an ischaemic
episode, ``(rtst0-120`` is the same ST deviation attributed to a heart-rate rise.
1,795 of the 2,311 criterion-A episodes are ischaemic and 516 rate-related, and the
split is very uneven across records — s20011's 20 episodes are all rate-related and
its header says so in as many words ("all episodes ... are compatible with
heart-rate induced non-ischemic changes ... It is recognized that this is an
arbitrary decision"). Never sum the two into "ST episodes" without saying so.

**4. 22 of the 86 records do not name their leads.** Their headers describe both
signals only as ``ECG`` and state plainly why: "Electrode locations were not
recorded." All 22 say it, and no record says it while naming its leads, so this is
a documented gap rather than a parsing failure. ``leads_named`` flags them.
The consequence for ``ECGDataset(leads=[...])`` is in the config; the consequence
here is that ``lead_names`` reads ``ECG|ECG`` for those records and no per-lead
column can be attributed to a physical lead.

**5. Subject identity is IN the record name, and the release says so.** "For each
recording, the first digit in the record name (2 or 3) indicates the number of ECG
signals ... Records obtained from the same subject have names that differ in the
last digit only." So ``s20271``-``s20274`` are one subject and ``s30731``/``s30732``
another: 80 subjects over 86 records, which is exactly the published figure.
:func:`parse_record_name` reads it off, and four of the six multi-record subjects
have it confirmed in their own header text ("Records s20271, s20272, s20273 and
s20274 are from the same patient."). This is *not* the reconstruction ``edb``
needs — it is the release's own naming rule.

**6. Ten records are the same tapes as ten European ST-T Database records, and
their headers name the partner.** "An excerpt of this recording is included in the
European ST-T Database (record e0113)." The pairs are exposed as ``edb_record``.
The tapes were **redigitised** for this database and rescaled, so the sample values
are not comparable and PhysioNet says annotations are not either — but the two hours
of heart are the same two hours of heart. See the catalogue entry for the overlap
check; the short version is that eight of the ten pairs are confirmed from the beat
annotations and two are not.

**7. ``.atr`` holds beats and nothing else.** No rhythm (``+``) annotation, no
signal-quality (``~``) annotation, no ``aux_note`` anywhere in the 86 files — unlike
every other MIT-BIH-family database in this catalogue. Noise and unreadable spans
are annotated instead in the ST files, as ``noi`` and ``(urd``/``urd)``, and are
reported here as ``n_noise_events`` and ``unreadable_secs``. Beat coverage is
otherwise complete: the first beat falls 0.02-9.18 s in and the last 0.13-1.96 s
before the end, so every record is annotated over at least 99.98% of its length and
there is no unannotated tail to window around.

The remaining shipped files are derived products this module does not read: ``.16a``
(ST measurements at eight points of every beat), ``.stf`` (the ST level, reference
and deviation functions as ASCII), ``.klt.zip`` (Karhunen-Loeve coefficients),
``.tsr.zip`` (SEMIA viewer input), ``.ari`` (the *uncorrected* automatic beat
detector output, superseded by ``.atr``) and 26 ``.hea-`` superseded headers.
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

#: The three ST episode annotators, in the order the shipped ``.cnt`` files print
#: them, mapped to the column suffix this module gives their counts. ``.sta`` is the
#: default and gets no suffix — see the module docstring, point 2.
ST_ANNOTATORS: tuple[tuple[str, str, str], ...] = (
    ("sta", "", "75 uV / 30 s"),
    ("stb", "_b", "100 uV / 30 s"),
    ("stc", "_c", "100 uV / 60 s"),
)

#: Beat annotation extension. ``.ari`` is the same detector's *uncorrected* output
#: and is deliberately not read.
BEAT_ANNOTATOR = "atr"

#: The aux-note grammar of the ``.sta``/``.stb``/``.stc`` files, transcribed from
#: ``tables/acodes.png`` in the release. Every one of the 109,670 ST annotations in
#: the 258 files matches exactly one of these, and the lead digit inside the text
#: always agrees with the annotation's own ``chan`` field.
#:
#: - ``GRST n`` global reference for lead n
#: - ``LRST n +/- ll`` local reference, ST level ll uV
#: - ``s [cc] st n`` significant ST shift: axis shift, or conduction change if ``cc``
#: - ``( [rt] st n +/- dd`` episode onset; ``rt`` means heart-rate related
#: - ``a [rt] st n +/- dd`` episode extremum, deviation dd uV
#: - ``[rt] st n +/- dd )`` episode end
#: - ``noi n +/- dd`` noise
#: - ``( urd n`` / ``urd n )`` unreadable interval
ST_AUX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("global_ref", re.compile(r"^GRST(?P<lead>\d)$")),
    ("local_ref", re.compile(r"^LRST(?P<lead>\d)(?P<value>[+-]\d+)$")),
    ("shift", re.compile(r"^s(?P<cc>cc)?st(?P<lead>\d)$")),
    ("episode_start", re.compile(r"^\((?P<rt>rt)?st(?P<lead>\d)(?P<value>[+-]\d+)$")),
    ("episode_extremum", re.compile(r"^a(?P<rt>rt)?st(?P<lead>\d)(?P<value>[+-]\d+)$")),
    ("episode_end", re.compile(r"^(?P<rt>rt)?st(?P<lead>\d)(?P<value>[+-]\d+)\)$")),
    ("noise", re.compile(r"^noi(?P<lead>\d)(?P<value>[+-]\d+)$")),
    ("unreadable_start", re.compile(r"^\(urd(?P<lead>\d)$")),
    ("unreadable_end", re.compile(r"^urd(?P<lead>\d)\)$")),
)

#: PhysioBank beat symbols. Thirteen occur in this release; the rest are listed so a
#: re-release using them is counted rather than warned about.
BEAT_SYMBOLS = (
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F",
    "e", "j", "n", "E", "/", "f", "Q",
)

#: Human-readable names for the symbols this release actually uses, for the docs.
BEAT_NAMES = {
    "N": "normal", "B": "bundle branch block (unspecified)", "V": "premature ventricular",
    "S": "supraventricular premature or ectopic", "A": "atrial premature",
    "a": "aberrated atrial premature", "F": "fusion of ventricular and normal",
    "E": "ventricular escape", "e": "atrial escape", "j": "nodal (junctional) escape",
    "J": "nodal (junctional) premature", "/": "paced", "Q": "unclassifiable",
}

#: Header fields whose value is a short free-text answer that is *usually* yes/no but
#: not always ("Yes, 1986", "Septum 13 mm", "Borderline"). Each yields two columns:
#: the verbatim text, and a nullable boolean. Mapping the text to the boolean is the
#: job of :func:`_yes_no`; the verbatim column is what to read when the detail matters.
YES_NO_FIELDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("Treatment", "Balloon Angioplasty"), "balloon_angioplasty"),
    (("Treatment", "Coronary Artery bypass Grafting"), "bypass_grafting"),
    (("History", "Hypertension"), "hypertension"),
    (("History", "Left ventricular hypertrophy"), "lv_hypertrophy"),
    (("History", "Cardiomyopathy"), "cardiomyopathy"),
    (("History", "Valve disease"), "valve_disease"),
    (("History", "Electrolyte abnormalities"), "electrolyte_abnormalities"),
    (
        ("History", "Hypercapnia, anemia, hypotension, hyperventilation"),
        "hypercapnia_anemia_hypotension_hyperventilation",
    ),
    (("History", "Atrioventricular nodal conduction delay"), "av_nodal_conduction_delay"),
    (("History", "Intraventricular conduction block"), "intraventricular_conduction_block"),
    (("History", "Previous Myocardial Infarction"), "previous_mi"),
)

#: Header fields kept as free text only. Path in the comment tree -> column name.
TEXT_FIELDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("Comments",), "comments"),
    (("Symptoms during Holter recording",), "symptoms"),
    (("Diagnoses",), "diagnoses"),
    (("Treatment", "Medications"), "medications"),
    (("History",), "history"),
    (("History", "Previous tests", "ECG stress test"), "ecg_stress_test"),
    (("History", "Previous tests", "Thallium/Stress echo"), "thallium_stress_echo"),
    (("History", "Previous tests", "Left ventricular function"), "lv_function"),
    (("History", "Previous tests", "Echocardiogram"), "echocardiogram"),
    (("History", "Previous tests", "Coronary Arteriography"), "coronary_arteriography"),
    (("History", "Previous tests", "Baseline ECG"), "baseline_ecg"),
    (("Holter Recording", "Recorder"), "recorder"),
    (("Holter Recording", "Date"), "holter_date"),
)

#: Sub-keys of the comment tree. A ``Key: value`` line is treated as a field only if
#: its key is one of these (or it sits at the top level); anything else is free text.
#: Without the whitelist, prose like "At 12:30 the patient reported chest pain"
#: becomes a field called "At 12".
_FIELD_KEYS = frozenset(
    key for path, _ in (*YES_NO_FIELDS, *TEXT_FIELDS) for key in path
) | {"Previous tests", "Age"}

#: Text meaning "not recorded", case-folded. Kept verbatim in the ``*_text`` column
#: and mapped to NA in the derived one.
_NO_DATA = frozenset({"no data", "not recorded", "", "-", "none"})

#: Cross-references the headers themselves make. ``s20021`` says its tape also
#: produced European ST-T's ``e0113``; eleven records name the record they were
#: called in the 1995-98 pilot collection.
_EDB_RE = re.compile(r"European\s+ST-T\s+Database\s+\(record\s+(e\d+)\)", re.S)
_PILOT_RE = re.compile(r"initial\s+Long-Term\s+ST\s+Database\s+of.*?\(record\s+(s\d+)\)", re.S)
_NO_ELECTRODES = "Electrode locations were not recorded"

#: Ischaemic-episode-count edges for the fold label. See :func:`attach_stratify_class`.
ISCHEMIC_BURDEN_EDGES = (1, 6, 21)
ISCHEMIC_BURDEN_NAMES = ("none", "1-5", "6-20", "21+")


def parse_record_name(name: str) -> dict[str, object]:
    """Split ``sXYYYZ`` into its lead count, subject number and record number.

    The release documents this naming rule on its landing page: X is the number of
    ECG signals, YYY the subject and Z that subject's record number. It is the only
    subject identifier published, and it is the reason 86 records come from 80
    subjects.
    """
    m = re.fullmatch(r"s([23])(\d{3})(\d)", name)
    if not m:
        raise ValueError(
            f"Record name {name!r} does not follow the Long-Term ST Database's "
            "sXYYYZ convention (X = lead count, YYY = subject, Z = record number). "
            "Patient grouping is derived from it, so an unrecognised name cannot be "
            "grouped."
        )
    return {
        "patient_id": m.group(2),
        "subject_number": int(m.group(2)),
        "record_number": int(m.group(3)),
        "name_lead_count": int(m.group(1)),
    }


def _aux(value: object) -> str:
    """The aux note up to its NUL terminator, stripped.

    WFDB aux notes are length-prefixed and may carry bytes past the text; comparing
    the raw string splits identical annotations into separate categories. ``edb`` was
    bitten by exactly this.
    """
    return str(value or "").split("\x00", 1)[0].strip()


def classify_st_aux(value: object) -> tuple[str | None, re.Match[str] | None]:
    """Match one ST aux note against :data:`ST_AUX_PATTERNS`."""
    text = _aux(value)
    for kind, pattern in ST_AUX_PATTERNS:
        m = pattern.match(text)
        if m:
            return kind, m
    return None, None


def _yes_no(text: str) -> bool | None:
    """Map a header answer to a tri-state.

    ``No`` and ``No data`` are the common cases, but the field is free text and the
    interesting answers are not booleans: "Yes, 1986", "Septum 13 mm", "Mild",
    "Right bundle branch block", "Borderline". Anything that is neither a negation
    nor a not-recorded marker is a positive finding, and the verbatim text is kept
    alongside so the detail is never lost.
    """
    t = text.strip().casefold().rstrip(".")
    if t in _NO_DATA:
        return None
    if t.startswith("no data") or t.startswith("not recorded"):
        return None
    if t == "no" or t.startswith("no,") or t.startswith("no -") or t.startswith("no("):
        return False
    return True


def parse_header_comments(hea_path: Path) -> dict[str, object]:
    """Parse one ``.hea``'s ``#`` comment block into a flat dict of columns.

    The comments are an indentation tree — two spaces per level, ``Key: value`` for
    fields and bare lines for prose — and it is completely regular: all 28 field keys
    appear in all 86 headers. The tree is read generically and the fields named in
    :data:`YES_NO_FIELDS` and :data:`TEXT_FIELDS` are pulled out of it, so a field
    that gains sub-structure in a re-release still parses.
    """
    tree = _comment_tree(hea_path)
    out: dict[str, object] = {}

    # "#Age: 58  Sex: M" — two fields on one line, and both read "No data" in the
    # five records where the subject's demographics were not recorded.
    age_line = tree.get(("Age",), ("", []))[0]
    m = re.match(r"^\s*(?P<age>\S+)(?:\s+Sex:\s*(?P<sex>\S+))?", age_line)
    age_text = (m.group("age") if m else "") or ""
    sex_text = (m.group("sex") if m and m.group("sex") else "") or ""
    out["age"] = float(age_text) if age_text.isdigit() else np.nan
    out["sex"] = sex_text.upper() if sex_text.upper() in {"M", "F"} else ""

    for path, column in TEXT_FIELDS:
        out[column] = _flatten(tree.get(path))
    for path, column in YES_NO_FIELDS:
        text = _flatten(tree.get(path))
        out[f"{column}_text"] = text
        out[column] = _yes_no(text)

    comments = out["comments"]
    assert isinstance(comments, str)
    # The cross-references are wrapped over two header lines, so they only exist as
    # a sentence once the line joins are undone: _flatten glues the lines with " | "
    # and "European | ST-T Database (record e0113)" matches no regex written for the
    # prose. Search the squashed text instead.
    squashed = re.sub(r"\s+", " ", comments.replace("|", " "))
    edb = _EDB_RE.search(squashed)
    pilot = _PILOT_RE.search(squashed)
    out["edb_record"] = edb.group(1) if edb else ""
    out["pilot_record"] = pilot.group(1) if pilot else ""
    out["leads_named"] = _NO_ELECTRODES not in comments

    meds = out["medications"]
    assert isinstance(meds, str)
    out["n_medications"] = (
        0 if _yes_no(meds) in (False, None) else len([p for p in meds.split(" | ") if p])
    )
    return out


def _comment_tree(hea_path: Path) -> dict[tuple[str, ...], tuple[str, list[str]]]:
    """``#`` comment block -> {path: (value, [free-text lines])}."""
    out: dict[tuple[str, ...], tuple[str, list[str]]] = {}
    stack: list[str] = []
    for raw in hea_path.read_text(errors="replace").splitlines():
        if not raw.startswith("#"):
            continue
        body = raw[1:].rstrip()
        if not body.strip():
            continue
        depth = (len(body) - len(body.lstrip(" "))) // 2
        text = body.strip()
        stack = stack[:depth]
        m = re.match(r"^([A-Z][^:]*):\s*(.*)$", text)
        if m and (not stack or m.group(1).strip() in _FIELD_KEYS):
            path = (*stack, m.group(1).strip())
            value, free = out.get(path, ("", []))
            out[path] = (m.group(2).strip() or value, free)
            stack = list(path)
        elif stack:
            value, free = out.get(tuple(stack), ("", []))
            out[tuple(stack)] = (value, [*free, text])
    return out


def _flatten(node: tuple[str, list[str]] | None) -> str:
    """A field's value plus its free-text children, as one pipe-separated string."""
    if node is None:
        return ""
    value, free = node
    parts = ([value] if value else []) + list(free)
    return " | ".join(p.strip() for p in parts if p.strip())


def summarise_episodes(record_path: Path, n_sig: int, sig_len: int, fs: float) -> dict[str, object]:
    """Counts, durations and peak deviations from all three ST annotation files.

    Episodes are counted at their **extremum**, which is what the shipped ``.cnt``
    files count and what reproduces them exactly — see the module docstring, point 1.
    Durations are a separate matter and come from the onset/end pair: an episode with
    no onset (already running at sample 0) is measured from 0, and one with no end
    (still running at the last sample) to ``sig_len``, so neither is silently dropped.
    """
    import wfdb

    out: dict[str, object] = {}
    unmatched: dict[str, int] = {}

    for ext, suffix, _criterion in ST_ANNOTATORS:
        ann = wfdb.rdann(str(record_path), ext)
        counts = {"ischemic": 0, "rate_related": 0}
        per_lead = {lead: 0 for lead in range(n_sig)}
        deviations: list[int] = []
        elevation = depression = 0
        spans: dict[str, list[tuple[int, int]]] = {"ischemic": [], "rate_related": []}
        unreadable: list[tuple[int, int]] = []
        open_start: dict[tuple[int, bool], int] = {}
        open_urd: dict[int, int] = {}
        head_open = unterminated = 0
        axis = conduction = noise = urd_count = 0

        for sample, aux in zip(ann.sample, ann.aux_note):
            kind, m = classify_st_aux(aux)
            if kind is None:
                unmatched[_aux(aux)] = unmatched.get(_aux(aux), 0) + 1
                continue
            if kind == "shift":
                if m.group("cc"):
                    conduction += 1
                else:
                    axis += 1
                continue
            if kind == "noise":
                noise += 1
                continue
            if kind == "unreadable_start":
                open_urd[int(m.group("lead"))] = int(sample)
                urd_count += 1
                continue
            if kind == "unreadable_end":
                lead = int(m.group("lead"))
                unreadable.append((open_urd.pop(lead, 0), int(sample)))
                continue
            if kind not in ("episode_start", "episode_extremum", "episode_end"):
                continue

            lead = int(m.group("lead"))
            rate = bool(m.group("rt"))
            key = (lead, rate)
            group = "rate_related" if rate else "ischemic"

            if kind == "episode_start":
                open_start[key] = int(sample)
            elif kind == "episode_extremum":
                counts[group] += 1
                if lead in per_lead and not rate:
                    per_lead[lead] += 1
                dev = int(m.group("value"))
                deviations.append(abs(dev))
                if dev > 0:
                    elevation += 1
                else:
                    depression += 1
            else:  # episode_end
                start = open_start.pop(key, None)
                if start is None:
                    head_open += 1
                    start = 0
                spans[group].append((start, int(sample)))

        for (_lead, rate), start in open_start.items():
            unterminated += 1
            spans["rate_related" if rate else "ischemic"].append((start, sig_len))

        out[f"n_ischemic_episodes{suffix}"] = counts["ischemic"]
        out[f"n_rate_related_episodes{suffix}"] = counts["rate_related"]
        out[f"ischemic_secs{suffix}"] = _sum_seconds(spans["ischemic"], fs)
        out[f"rate_related_secs{suffix}"] = _sum_seconds(spans["rate_related"], fs)
        out[f"ischemic_secs_any_lead{suffix}"] = _union_seconds(spans["ischemic"], fs)
        out[f"peak_st_deviation_uv{suffix}"] = max(deviations) if deviations else 0
        out[f"n_st_elevation_episodes{suffix}"] = elevation
        out[f"n_st_depression_episodes{suffix}"] = depression
        out[f"n_episodes_open_at_start{suffix}"] = head_open
        out[f"n_unterminated_episodes{suffix}"] = unterminated

        if not suffix:
            # Identical in all three files — these are marks, not threshold
            # crossings — so they are reported once, without a suffix.
            out["n_axis_shifts"] = axis
            out["n_conduction_change_shifts"] = conduction
            out["n_noise_events"] = noise
            out["n_unreadable_intervals"] = urd_count
            out["unreadable_secs"] = _union_seconds(unreadable, fs)
            for lead in range(3):
                out[f"n_ischemic_episodes_lead{lead}"] = (
                    per_lead[lead] if lead in per_lead else np.nan
                )

    if unmatched:
        logger.warning(
            "%s: ST annotation text outside the documented grammar, not counted: %s",
            record_path.name,
            unmatched,
        )
    return out


def _sum_seconds(spans: list[tuple[int, int]], fs: float) -> float:
    """Total annotated seconds, counting every lead's episodes separately."""
    return round(sum(max(0, end - start) for start, end in spans) / fs, 3)


def _union_seconds(spans: list[tuple[int, int]], fs: float) -> float:
    """Seconds covered by at least one span — the bounded version of the above.

    The leads are annotated independently, so an episode seen in two leads is two
    episodes and its seconds are counted twice by :func:`_sum_seconds`. That sum can
    exceed the record's own length; this cannot.
    """
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return round(sum(end - start for start, end in merged) / fs, 3)


def summarise_beats(record_path: Path, sig_len: int, fs: float) -> dict[str, object]:
    """Beat counts, their AAMI reduction, and coverage, from the ``.atr`` file."""
    import wfdb

    ann = wfdb.rdann(str(record_path), BEAT_ANNOTATOR)
    counts: dict[str, object] = {f"beat_{s}": 0 for s in BEAT_SYMBOLS}
    counts.update({f"aami_{c}": 0 for c in AAMI_ORDER})

    beat_set = set(BEAT_SYMBOLS)
    samples: list[int] = []
    other: dict[str, int] = {}
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in beat_set:
            counts[f"beat_{symbol}"] = int(counts[f"beat_{symbol}"]) + 1
            aami = AAMI_CLASSES.get(symbol)
            if aami is not None:
                counts[f"aami_{aami}"] = int(counts[f"aami_{aami}"]) + 1
            samples.append(int(sample))
        else:
            other[symbol] = other.get(symbol, 0) + 1
    if other:
        # The 86 .atr files hold beats and nothing else — no '+' rhythm marks and no
        # '~' quality marks. If that ever changes, say so rather than silently
        # dropping annotations out of n_beats.
        logger.warning(
            "%s: non-beat annotations in .atr, not counted: %s", record_path.name, other
        )

    counts["n_beats"] = len(samples)
    counts["n_ectopic_beats"] = len(samples) - int(counts["beat_N"])
    counts["n_non_beat_annotations"] = sum(other.values())
    if len(samples) > 2:
        counts["annotated_fraction"] = round((samples[-1] - samples[0]) / sig_len, 6)
        counts["mean_hr_bpm"] = round(
            60.0 * (len(samples) - 1) / ((samples[-1] - samples[0]) / fs), 2
        )
        rr = np.diff(np.asarray(samples, dtype=np.float64)) / fs
        rr = rr[(rr > 0.25) & (rr < 2.5)]
        counts["sdnn_ms"] = round(float(np.std(rr) * 1000.0), 2) if rr.size else np.nan
    else:
        counts["annotated_fraction"] = np.nan
        counts["mean_hr_bpm"] = np.nan
        counts["sdnn_ms"] = np.nan
    return counts


def _st_class(row: pd.Series) -> str:
    """Which kinds of ST event a record holds, under criterion A."""
    if row["n_ischemic_episodes"] > 0:
        return "ischemic" if row["n_rate_related_episodes"] == 0 else "ischemic_and_rate_related"
    if row["n_rate_related_episodes"] > 0:
        return "rate_related_only"
    if row["n_axis_shifts"] > 0 or row["n_conduction_change_shifts"] > 0:
        return "shift_only"
    return "none"


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ischemic_burden_band`` and ``stratify_class``.

    Ischaemic-episode burden under criterion A, banded at 1, 6 and 21 episodes:
    18 records with none, 14 with 1-5, 25 with 6-20 and 29 with 21 or more. Ischaemic
    ST change is what this database was built to evaluate and the burden is
    enormously uneven — s20274 holds 143 episodes and 18 records hold none — so a
    fold drawn without regard to it can easily be all quiet records or all busy ones.

    Every band holds at least 14 records, more than the 10 folds ECGBench generates,
    so unlike ``edb`` no band is forced to skip folds. The edges are wide because the
    distribution is heavy-tailed, not because the classes were tuned: 1 separates the
    records with no ischaemia at all, and 6 and 21 are near the tertiles of the rest.

    Nothing clinical works better as a fold label. The header findings describe the
    subject rather than the recording, ``diagnoses`` is free text with 60-odd distinct
    values, and ``st_class`` puts 68 of the 86 records in one class.
    """
    counts = df["n_ischemic_episodes"].fillna(0).to_numpy()
    bands = np.digitize(counts, ISCHEMIC_BURDEN_EDGES)
    df = df.copy()
    df["ischemic_burden_band"] = [ISCHEMIC_BURDEN_NAMES[b] for b in bands]
    df["stratify_class"] = df["ischemic_burden_band"]
    return df


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so
    neither a stray copy nor the 26 superseded ``.hea-`` headers can enter the
    partition.
    """
    import wfdb

    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. Long-Term ST labels live in the "
            "record headers and the .atr/.sta/.stb/.stc annotation files, so point "
            "data_path at the dataset root — the flat directory holding s20011.hea, "
            "RECORDS and ANNOTATORS. Get it from "
            "https://physionet.org/content/ltstdb/1.0.0/"
        )

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows: list[dict[str, object]] = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue
        header = wfdb.rdheader(str(hea.with_suffix("")))
        fs = float(header.fs)
        row: dict[str, object] = {"record_name": name}
        row.update(parse_record_name(name))
        row["n_leads"] = int(header.n_sig)
        row["lead_names"] = "|".join(header.sig_name or [])
        row["sig_len"] = int(header.sig_len)
        row["duration_secs"] = round(header.sig_len / fs, 3)
        row["duration_hours"] = round(header.sig_len / fs / 3600.0, 4)
        row["start_time"] = str(header.base_time or "")
        row["recording_date"] = str(header.base_date or "")
        row.update(parse_header_comments(hea))
        row.update(summarise_episodes(hea.with_suffix(""), header.n_sig, header.sig_len, fs))
        row.update(summarise_beats(hea.with_suffix(""), header.sig_len, fs))
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)

    mismatched = df.loc[df["n_leads"] != df["name_lead_count"], "record_name"].tolist()
    if mismatched:
        # The first digit of the record name is supposed to BE the signal count.
        # If it is not, patient grouping is derived from a name nobody can trust.
        raise ValueError(
            "Record name and header disagree on the number of signals for "
            f"{mismatched}. The Long-Term ST naming rule (sXYYYZ, X = lead count) is "
            "what patient_id is derived from, so this must be resolved before the "
            "records can be grouped."
        )
    df = df.drop(columns=["name_lead_count"])

    df["st_class"] = df.apply(_st_class, axis=1)
    df["ischemic_fraction"] = (df["ischemic_secs_any_lead"] / df["duration_secs"]).round(6)
    df["ectopic_fraction"] = (df["n_ectopic_beats"] / df["n_beats"]).where(df["n_beats"] > 0)

    logger.info(
        "Parsed %d Long-Term ST records from %d subjects; %.0f h of signal, "
        "%d beats, %d ischaemic and %d rate-related ST episodes (criterion A), "
        "%d lead layouts, %d records with unnamed leads",
        len(df),
        df["patient_id"].nunique(),
        df["sig_len"].sum() / 250.0 / 3600.0,
        int(df["n_beats"].sum()),
        int(df["n_ischemic_episodes"].sum()),
        int(df["n_rate_related_episodes"].sum()),
        df["lead_names"].nunique(),
        int((~df["leads_named"]).sum()),
    )
    return df


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Long-Term ST Database labels indexed by record name.

    Columns:

    - **ST episodes — the database's reason for existing, under three criteria.**
      ``n_ischemic_episodes`` and ``n_rate_related_episodes`` (1,795 and 516 across
      the release) come from ``.sta``, 75 uV / 30 s; the same columns suffixed ``_b``
      (``.stb``, 100 uV / 30 s: 1,130 and 234) and ``_c`` (``.stc``, 100 uV / 60 s:
      857 and 116) are the stricter criteria. **Always say which criterion a figure
      uses.** ``n_ischemic_episodes_lead0``/``_lead1``/``_lead2`` split criterion A by
      signal (``lead2`` is NaN for the 68 two-lead records), and
      ``n_st_elevation_episodes``/``n_st_depression_episodes`` by the sign of the
      annotated extremum.
    - ``ischemic_secs`` sums every lead's episodes and so can exceed the record's own
      length — the leads are annotated independently — while
      ``ischemic_secs_any_lead`` is the bounded union and ``ischemic_fraction`` is
      that over the record. ``peak_st_deviation_uv`` is the largest annotated
      extremum in microvolts over episodes of **either** kind — the release's
      biggest, 1,495 uV in s20621, belongs to a rate-related episode in a record
      with no ischaemia at all — measured against the **annotator-placed baseline
      ST level function** for that record, not against an absolute isoelectric line.
      ``n_episodes_open_at_start`` and ``n_unterminated_episodes`` count the episodes
      running at the first and last sample, which this module measures from 0 and to
      the record end rather than dropping.
    - ``n_axis_shifts``, ``n_conduction_change_shifts``, ``n_noise_events``,
      ``n_unreadable_intervals``, ``unreadable_secs`` — the marks that are **not**
      threshold crossings, and so are identical in all three annotation files
      (1,493 / 895 / 31 / 60 across the release). Axis shifts are positional artefact
      that mimics ischaemia; they are findings about the recording, not the heart.
    - ``st_class`` — which kinds of ST event a record holds
      (``ischemic``, ``ischemic_and_rate_related``, ``rate_related_only``,
      ``shift_only``, ``none``). A record-level summary, not the fold label.
    - ``ischemic_burden_band`` / ``stratify_class`` — banded ischaemic episode count,
      **for fold construction only**. Not a clinical label.
    - ``beat_N`` … ``beat_Q`` — reference beat counts per symbol (see
      :data:`BEAT_NAMES`); 8,897,780 beats over the release, the largest annotated
      beat collection in this catalogue. ``aami_N``/``S``/``V``/``F``/``Q`` are the
      AAMI EC57 reduction, directly comparable with ``mitdb``, ``svdb``, ``incartdb``
      and ``edb``. With ``n_beats``, ``n_ectopic_beats``, ``ectopic_fraction``,
      ``mean_hr_bpm``, ``sdnn_ms`` and ``annotated_fraction`` (never below 0.9998 —
      every record is annotated end to end). **Multi-class per record**: most records
      carry several beat types, so there is no single record-level beat label.
    - ``age``, ``sex``, ``diagnoses``, ``symptoms``, ``medications``,
      ``n_medications``, ``comments``, ``history``, ``ecg_stress_test``,
      ``thallium_stress_echo``, ``lv_function``, ``echocardiogram``,
      ``coronary_arteriography``, ``baseline_ecg`` — the header's clinical record,
      verbatim, pipe-separated where the source had several lines. Age and sex read
      "No data" in five records and are NaN / "" there.
    - ``hypertension``, ``previous_mi``, ``lv_hypertrophy``, ``cardiomyopathy``,
      ``valve_disease``, ``electrolyte_abnormalities``,
      ``hypercapnia_anemia_hypotension_hyperventilation``,
      ``av_nodal_conduction_delay``, ``intraventricular_conduction_block``,
      ``balloon_angioplasty``, ``bypass_grafting`` — nullable booleans, each with the
      verbatim answer beside it in ``<name>_text``. Read the text when the detail
      matters: "Yes, 1986", "Septum 13 mm" and "Right bundle branch block" all
      reduce to True. NA means the header said "No data", which is 11-45 records
      depending on the field.
    - ``patient_id``, ``subject_number``, ``record_number`` — read off the record
      name, which is where the release puts subject identity (see
      :func:`parse_record_name`). 80 subjects over 86 records.
    - ``lead_names``, ``n_leads``, ``leads_named`` — the leads *this* record stores,
      pipe-separated. 68 records hold two signals and 18 hold three, in twelve
      layouts, and 22 records name neither of theirs (``leads_named`` False).
    - ``edb_record`` — the European ST-T Database record cut from the same tape, for
      the ten records whose headers say so. **Do not train on one and evaluate on the
      other.** ``pilot_record`` is the name eleven records carried in the 1995-98
      pilot collection, which was never published — **do not read it as a record id
      in this release**, because one of the eleven collides: ``s20071``'s pilot name
      is ``s20511``, and ``s20511`` is also a record here, belonging to a different
      subject.
    - ``recorder`` — one of five Holter models, or "No data" for 36 records.
    - ``duration_hours``, ``sig_len``, ``start_time``, ``recording_date``.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
