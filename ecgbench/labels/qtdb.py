"""
QT Database labels: the manual P/QRS/T/U boundary annotations, and where every
excerpt came from.

This database exists to evaluate **waveform delineation**, so its ground truth is
not a record-level class at all — it is 3,623 individually annotated beats, each
carrying up to eleven fiducial points placed by a cardiologist. ``load_labels``
therefore returns a per-record *summary* of that annotation (how many beats, which
waves, the interval medians), and :func:`load_beat_annotations` returns the beats
themselves, one row per beat. Anything training a delineator wants the second
function; the first is what stratifies folds and what a dataset page can tabulate.

**Every record is an excerpt of another database's recording**, and that is the
single most important thing to know before using it. Nine things, all verified
against the shipped files:

1. **105 records, 15 minutes each, drawn from seven sources** (paper Table 1):
   European ST-T 33, BIH sudden-death 24, MIT-BIH Arrhythmia 15, MIT-BIH
   Supraventricular 13, MIT-BIH NSR 10, MIT-BIH ST Change 6, MIT-BIH Long-Term 4.
   ``source_database`` carries it and it is also the stratification class. Six of
   the seven sources are datasets ECGBench already partitions, so **training on
   qtdb and testing on mitdb/edb/svdb/nsrdb/sddb — or the reverse — is testing on
   training data.** The overlap is declared in the catalogue and quantified below.

2. **The overlap is verified from the waveforms, not inferred from the names.**
   Of the 33 European ST-T excerpts, 30 are *bit-identical* to ``edb`` 1.0.0 at the
   offset their own header states; the other three (``sele0112``, ``sele0116``,
   ``sele0136``) are the same waveform with a DC pedestal added and one channel
   rescaled to match a re-declared gain (r >= 0.99985, slope equal to the gain
   ratio to four figures). Of the 23 sudden-death excerpts, 22 reproduce ``sddb``
   1.0.0 exactly — ``sel39`` and ``sel47`` bit-identical, the other 20 as
   ``trunc(sddb_digital / 4)``, which is exact, not approximate. The remaining
   four sources were resampled to 250 Hz and so match in waveform but not in
   samples.

3. **``sel32`` is not in ``sddb``.** Its 4,096-sample opening does not occur
   anywhere in sddb record 32, in either channel, raw or divided by four. The
   paper places it at 20:52:20 of record 32 and the header agrees, but the samples
   do not. ``source_record_verified`` is False for it and True for the other 22.
   Treat its provenance as unconfirmed rather than assuming a lost tape.

4. **Two subjects contributed two records each**, so this is 105 records from
   **103 subjects at most**. European ST-T records e0121+e0122 are one 51-year-old
   man and e0124+e0126 are one subject — both pairs are in qtdb. ``patient_id``
   groups them. qtdb's own header text catches only the first pair (the second
   pair's coarse clinical blocks differ), which is why ``EDB_SHARED_SUBJECTS``
   below is a literal rather than a derivation. The 13 Supraventricular and 6 ST
   Change records carry no subject information in any release, so 103 is an upper
   bound, not a count.

5. **The annotated beats live in the last five minutes and nowhere else.** The
   paper says annotation began after 10 minutes to leave learning time, and the
   files agree: the earliest manual annotation in any record is at 600.464 s and
   the latest at 896.916 s. ``window=(150000, 74993)`` is the whole annotated
   region and fits every record; a window from sample 0 contains no ground truth
   at all.

6. **3,623 beats, not the published 3,622.** ``sel223`` carries 31 annotated beats
   where Table 2 says 30; every other record matches. ``PUBLISHED_ANNOTATED_BEATS``
   is the paper's column and ``annotated_beats_matches_published`` is the check.

7. **The two annotators do not cover the same beats, and one record is nearly
   empty.** Annotator 2 annotated 11 records (all from MIT-BIH Arrhythmia) and 404
   of the 487 beats annotator 1 annotated in those same records. In ``sel102`` the
   audit cut annotator 2 from 97 annotations to **13** — three beats against
   annotator 1's 85. Any inter-observer study has to weight by
   ``n_annotated_beats_annotator2``, not assume parity.

8. **Not every record annotates every wave, and two annotate no T wave at all.**
   3,542 of 3,623 beats have a measurable QT; ``sel35`` (atrial fibrillation) and
   ``sel37`` have QRS boundaries only, so their ``median_qt_ms`` is NaN rather than
   zero. P waves are annotated in 3,194 beats and are absent from seven records
   entirely; U waves in 821. ``waveform_pattern`` recomputes Table 2's notation
   from the files and agrees for 101 of the 105 records. The four disagreements
   are the paper's column being inconsistent with itself, not a parse failure:
   ``sel117`` and ``sel14157`` are listed with ``u)`` on the strength of 11 and 8
   U waves in 30 beats, while ``sele0704`` is listed *without* a T onset it
   carries in 20 of 30; and ``sel37``'s beats are N:24, B:20, Q:6, so its modal
   symbol is ``N`` where Table 2 writes ``(Q)``.

9. **Amplitude is not trustworthy for 34 records and four carry a +5.12 mV
   pedestal.** The paper states the sudden-death Holters "are not calibrated with
   respect to amplitude; thus the signal gains recorded in the header files for
   these records are only estimates" — that is the group of 24, the 23 sddb
   excerpts plus ``sel17152``. Ten further headers declare a gain of ``0``, which
   *means* uncalibrated and which wfdb silently replaces with 200 adu/mV: all four
   Long-Term records and six Supraventricular ones. Separately ``sel100``,
   ``sel102``, ``sel103`` and ``sel104`` are the only four records declaring an
   explicit baseline of 0 alongside an ``adc_zero`` of 1024, so wfdb returns them
   offset by a constant +5.12 mV relative to mitdb's copy of the same recording —
   their signals never go negative. ``amplitude_calibrated`` and
   ``dc_pedestal_mv`` flag both cases.

Two further release-level notes. The paper describes ``record.ari`` files holding
ARISTOTLE's automatic QRS annotations; **no ``.ari`` file ships**, and the
``ANNOTATORS`` file does not list one. And the clinical text in the 33 European
ST-T headers is an earlier, coarser vintage than ``edb`` 1.0.0's — "Coronary
artery disease" in place of the angina type, "unspecified medication" in place of
the drug list — which for ``sele0116``, ``sele0121`` and ``sele0122`` **contradicts**
edb, whose headers record normal coronary arteries for those subjects. Prefer edb
for those 33; ``clinical_source`` says which vintage a row came from.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Uniform across all 105 records. Records not originally sampled at 250 Hz were
#: converted with ``xform``, which recorded the original rate as the header's
#: counter frequency — that is where ``source_sampling_rate`` comes from.
SAMPLING_RATE = 250

#: The manual boundary annotators, second (audited) pass. ``qt1``/``qt2`` are the
#: unaudited first pass and carry ``|`` in place of the wave symbols, so they are
#: counted but never parsed for boundaries.
MANUAL_ANNOTATORS = ("q1c", "q2c")
FIRST_PASS_ANNOTATORS = ("qt1", "qt2")
#: ``ecgpuwave`` output: both signals, signal 0 alone, signal 1 alone.
AUTO_ANNOTATORS = ("pu", "pu0", "pu1")
#: Beat locations only, the file each annotator started from.
BEAT_ANNOTATOR = "man"
#: Reference annotations inherited from the source database. Absent for the 23
#: sudden-death records, which had none to inherit.
SOURCE_ANNOTATOR = "atr"

#: ``num`` field of a ``(`` or ``)`` annotation, per the paper.
WAVE_TYPES = {0: "p", 1: "qrs", 2: "t", 3: "u"}

#: ``num`` field of a ``t`` annotation in the ``pu*`` files: ecgpuwave's T-wave
#: morphology class. Manual annotations do not carry it.
T_MORPHOLOGY = {
    0: "normal",
    1: "inverted",
    2: "only_upwards",
    3: "only_downwards",
    4: "biphasic_neg_pos",
    5: "biphasic_pos_neg",
}

#: Beat symbols occurring in the manual annotation files. Only "normal" beats were
#: annotated (paper, section "The QT Database"), so this is short by design: the
#: five here are what the annotators actually left after the audit reclassified 65
#: of ``man``'s 3,593 ``N`` beats.
BEAT_SYMBOLS = ("N", "A", "B", "V", "Q")
BEAT_NAMES = {
    "N": "normal beat",
    "A": "atrial premature beat",
    "B": "bundle branch block beat",
    "V": "premature ventricular contraction",
    "Q": "unclassifiable beat",
}

#: Which records came from which source, transcribed from Table 1 and Table 2 of
#: Laguna et al. 1997. It cannot be derived from the record name: ``sel17152`` and
#: ``sel17453`` differ only in the last three digits and come from different
#: sources, and three headers name their source record ``mqt2`` rather than the
#: real one.
SOURCE_DATABASE_RECORDS = {
    "mitdb": (
        "sel100", "sel102", "sel103", "sel104", "sel114", "sel116", "sel117",
        "sel123", "sel213", "sel221", "sel223", "sel230", "sel231", "sel232",
        "sel233",
    ),
    "stdb": ("sel301", "sel302", "sel306", "sel307", "sel308", "sel310"),
    "svdb": (
        "sel803", "sel808", "sel811", "sel820", "sel821", "sel840", "sel847",
        "sel853", "sel871", "sel872", "sel873", "sel883", "sel891",
    ),
    "ltdb": ("sel14046", "sel14157", "sel14172", "sel15814"),
    "edb": (
        "sele0104", "sele0106", "sele0107", "sele0110", "sele0111", "sele0112",
        "sele0114", "sele0116", "sele0121", "sele0122", "sele0124", "sele0126",
        "sele0129", "sele0133", "sele0136", "sele0166", "sele0170", "sele0203",
        "sele0210", "sele0211", "sele0303", "sele0405", "sele0406", "sele0409",
        "sele0411", "sele0509", "sele0603", "sele0604", "sele0606", "sele0607",
        "sele0609", "sele0612", "sele0704",
    ),
    "nsrdb": (
        "sel16265", "sel16272", "sel16273", "sel16420", "sel16483", "sel16539",
        "sel16773", "sel16786", "sel16795", "sel17453",
    ),
    "sddb": (
        "sel30", "sel31", "sel32", "sel33", "sel34", "sel35", "sel36", "sel37",
        "sel38", "sel39", "sel40", "sel41", "sel42", "sel43", "sel44", "sel45",
        "sel46", "sel47", "sel48", "sel49", "sel50", "sel51", "sel52",
    ),
    # The paper counts this record in its sudden-death group of 24, but it is
    # neither in sddb nor in nsrdb: the authors gathered "age-and-gender matched
    # patients without diagnosed cardiac disease" alongside the sudden-death
    # Holters, and this is the one such control that made it into the database.
    # It is the only record here with no published source recording anywhere.
    "bih_control": ("sel17152",),
}

SOURCE_DATABASE_NAMES = {
    "mitdb": "MIT-BIH Arrhythmia Database",
    "stdb": "MIT-BIH ST Change Database",
    "svdb": "MIT-BIH Supraventricular Arrhythmia Database",
    "ltdb": "MIT-BIH Long-Term ECG Database",
    "edb": "European ST-T Database",
    "nsrdb": "MIT-BIH Normal Sinus Rhythm Database",
    "sddb": "BIH sudden-death Holter recordings (published as sddb)",
    "bih_control": "BIH matched control Holter recording (unpublished elsewhere)",
}

#: Catalogue slug of the ECGBench dataset holding the same recordings, or None
#: where ECGBench does not carry the source. ``ltdb`` and ``bih_control`` have no
#: entry; ``stdb`` does. This is what the ``related:`` block in
#: ``docs/_datasets/qt-database-qtdb.md`` is built from.
SOURCE_CATALOGUE_SLUG = {
    "mitdb": "mit-bih-arrhythmia-database",
    "stdb": "mit-bih-st-change-database",
    "svdb": "mit-bih-supraventricular-arrhythmia-database",
    "ltdb": None,
    "edb": "european-st-t-database-edb",
    "nsrdb": "mit-bih-normal-sinus-rhythm-database",
    "sddb": "sudden-cardiac-death-holter-database",
    "bih_control": None,
}

#: Original sampling rate of each source, in Hz. Cross-checked against the counter
#: frequency every resampled header declares (``250/360``, ``250/128``); the two
#: agree for all 49 records that declare one, and the sources at 250 Hz declare
#: none because no conversion happened.
SOURCE_SAMPLING_RATE = {
    "mitdb": 360,
    "stdb": 360,
    "svdb": 128,
    "ltdb": 128,
    "edb": 250,
    "nsrdb": 128,
    "sddb": 250,
    "bih_control": 128,
}

#: The paper's Table 2 "annot beats" column. Kept so the recomputed count can be
#: checked against it — see ``annotated_beats_matches_published``, which is False
#: for ``sel223`` alone.
PUBLISHED_ANNOTATED_BEATS = {
    "sel100": 30, "sel102": 85, "sel103": 30, "sel104": 77, "sel114": 50,
    "sel116": 50, "sel117": 30, "sel123": 30, "sel14046": 31, "sel14157": 30,
    "sel14172": 50, "sel15814": 30, "sel16265": 30, "sel16272": 30,
    "sel16273": 30, "sel16420": 30, "sel16483": 30, "sel16539": 30,
    "sel16773": 30, "sel16786": 30, "sel16795": 30, "sel17152": 30,
    "sel17453": 30, "sel213": 71, "sel221": 30, "sel223": 30, "sel230": 50,
    "sel231": 50, "sel232": 30, "sel233": 30, "sel30": 30, "sel301": 30,
    "sel302": 30, "sel306": 36, "sel307": 30, "sel308": 50, "sel31": 30,
    "sel310": 30, "sel32": 30, "sel33": 30, "sel34": 30, "sel35": 31,
    "sel36": 31, "sel37": 50, "sel38": 30, "sel39": 30, "sel40": 30,
    "sel41": 30, "sel42": 30, "sel43": 30, "sel44": 30, "sel45": 30,
    "sel46": 30, "sel47": 30, "sel48": 30, "sel49": 30, "sel50": 32,
    "sel51": 30, "sel52": 30, "sel803": 30, "sel808": 30, "sel811": 30,
    "sel820": 30, "sel821": 30, "sel840": 70, "sel847": 33, "sel853": 30,
    "sel871": 70, "sel872": 30, "sel873": 33, "sel883": 30, "sel891": 71,
    "sele0104": 30, "sele0106": 30, "sele0107": 34, "sele0110": 30,
    "sele0111": 30, "sele0112": 50, "sele0114": 30, "sele0116": 30,
    "sele0121": 30, "sele0122": 30, "sele0124": 50, "sele0126": 30,
    "sele0129": 30, "sele0133": 30, "sele0136": 30, "sele0166": 36,
    "sele0170": 30, "sele0203": 30, "sele0210": 30, "sele0211": 30,
    "sele0303": 30, "sele0405": 30, "sele0406": 31, "sele0409": 30,
    "sele0411": 30, "sele0509": 30, "sele0603": 30, "sele0604": 30,
    "sele0606": 30, "sele0607": 30, "sele0609": 30, "sele0612": 30,
    "sele0704": 30,
}

#: Table 2's "Waveform Pattern" column with the spaces removed. ``(p)(N)t)`` means
#: the annotators marked P onset, peak and end, QRS onset, the beat, QRS end, T
#: peak and T end — but not T onset. Recomputed as ``waveform_pattern`` and
#: compared, because it is the compact statement of what ground truth a record has.
PUBLISHED_WAVEFORM_PATTERN = {
    "sel100": "(p)(N)t)", "sel102": "(N)t)u)", "sel103": "(p)(N)t)u)",
    "sel104": "(N)t)", "sel114": "(p)(N)t)", "sel116": "(p)(N)t)u)",
    "sel117": "(p)(N)t)u)", "sel123": "(p)(N)t)u)", "sel14046": "(p)(N)t)",
    "sel14157": "(p)(N)t)u)", "sel14172": "(p)(N)(t)(u)", "sel15814": "(p)(N)t)",
    "sel16265": "(p)(N)t)", "sel16272": "(p)(N)t)", "sel16273": "(p)(N)t)",
    "sel16420": "(p)(N)t)", "sel16483": "(p)(N)t)", "sel16539": "(p)(N)(t)",
    "sel16773": "(p)(N)t)u)", "sel16786": "(p)(N)(t)", "sel16795": "(p)(N)(t)",
    "sel17152": "(p)(N)(t)", "sel17453": "(p)(N)(t)u)", "sel213": "(p)(N)t)",
    "sel221": "(N)t)", "sel223": "(p)(N)(t)", "sel230": "(p)(N)t)",
    "sel231": "(p)(N)t)", "sel232": "(A)t)", "sel233": "(p)(N)t)",
    "sel30": "(p)(N)(t)", "sel301": "(p)(N)(t)", "sel302": "(p)(N)t)",
    "sel306": "(p)(N)t)", "sel307": "(p)(N)t)", "sel308": "(p)(N)t)",
    "sel31": "(p)(N)(t)", "sel310": "(N)t)u)", "sel32": "(p)(N)(t)",
    "sel33": "(p)(N)(t)", "sel34": "(p)(N)(t)", "sel35": "(N)",
    "sel36": "(B)(t)u)", "sel37": "(Q)", "sel38": "(p)(N)(t)",
    "sel39": "(p)(N)(t)", "sel40": "(p)(N)(t)", "sel41": "(p)(N)(t)",
    "sel42": "(p)(N)(t)", "sel43": "(p)(N)(t)", "sel44": "(p)(N)(t)",
    "sel45": "(p)(N)(t)u)", "sel46": "(p)(N)(t)", "sel47": "(p)(N)(t)(u)",
    "sel48": "(p)(N)(t)", "sel49": "(p)(N)(t)", "sel50": "(N)(t)u)",
    "sel51": "(p)(N)(t)", "sel52": "(p)(N)(t)", "sel803": "(p)(N)t)",
    "sel808": "(p)(N)t)u)", "sel811": "(p)(N)t)", "sel820": "(p)(N)t)",
    "sel821": "(p)(N)(t)", "sel840": "(p)(N)(t)", "sel847": "(p)(N)t)",
    "sel853": "(p)(N)t)", "sel871": "(p)(N)t)", "sel872": "(p)(N)t)",
    "sel873": "(p)(N)t)", "sel883": "(p)(N)t)", "sel891": "(p)(N)t)",
    "sele0104": "(p)(N)t)", "sele0106": "(p)(N)(t)u)", "sele0107": "(p)(N)t)u)",
    "sele0110": "(p)(N)t)", "sele0111": "(p)(N)t)u)", "sele0112": "(p)(N)t)u)",
    "sele0114": "(p)(N)(t)(u)", "sele0116": "(p)(N)(t)", "sele0121": "(p)(N)t)",
    "sele0122": "(p)(N)t)u)", "sele0124": "(p)(N)(t)", "sele0126": "(p)(N)t)u)",
    "sele0129": "(p)(N)t)", "sele0133": "(p)(N)t)", "sele0136": "(p)(N)(t)",
    "sele0166": "(p)(N)(t)", "sele0170": "(p)(N)t)", "sele0203": "(p)(N)t)",
    "sele0210": "(p)(N)(t)", "sele0211": "(p)(N)(t)", "sele0303": "(p)(N)(t)",
    "sele0405": "(p)(N)t)", "sele0406": "(p)(N)t)", "sele0409": "(p)(N)(t)",
    "sele0411": "(p)(N)t)", "sele0509": "(p)(N)t)", "sele0603": "(p)(N)t)u)",
    "sele0604": "(p)(N)t)u)", "sele0606": "(p)(N)t)", "sele0607": "(p)(N)t)",
    "sele0609": "(p)(N)(t)", "sele0612": "(p)(N)(t)", "sele0704": "(p)(N)t)u)",
}

#: qtdb records that share a subject, keyed by the record that names the group.
#: Computed by running ``ecgbench.labels.edb.reconstruct_patient_ids`` over edb
#: 1.0.0 and intersecting its 7 multi-record subjects with qtdb's 33-record
#: selection: edb subject "e0118" holds e0118, e0119, e0121, e0122 and subject
#: "e0123" holds e0123, e0124, e0125, e0126. Hard-coded because qtdb's own header
#: text is too coarse to recover the second pair, and because a qtdb user has no
#: reason to have edb on disk. Nothing groups the Supraventricular or ST Change
#: records — neither release identifies its subjects at all.
EDB_SHARED_SUBJECTS = {
    "sele0121": ("sele0121", "sele0122"),
    "sele0124": ("sele0124", "sele0126"),
}

#: Records whose opening 4,096 samples were located in the source release, exactly.
#: Everything in the edb and sddb groups except ``sel32``; see point 3 of the module
#: docstring. Resampled sources are not in scope — they cannot match sample-wise —
#: so they are reported as pd.NA rather than False.
UNVERIFIED_SOURCE_RECORDS = ("sel32",)

#: The four records declaring an explicit baseline of 0 against an ``adc_zero`` of
#: 1024, which is what puts a constant pedestal on their physical signal. The value
#: is ``adc_zero / gain`` = 1024 / 200 mV and is identical on both channels.
DC_PEDESTAL_MV = 1024 / 200.0
DC_PEDESTAL_RECORDS = ("sel100", "sel102", "sel103", "sel104")

#: AAMI EC57 classes, for reducing the inherited ``.atr`` beat symbols. Same
#: mapping as ecgbench.labels.mitdb uses, so the two are comparable.
AAMI_CLASSES = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

#: Rhythm codes appearing in the ``aux_note`` of a ``+`` annotation in the
#: inherited ``.atr`` files.
RHYTHM_NAMES = {
    "(N": "sinus rhythm",
    "(PREX": "pre-excitation (WPW)",
    "(B": "ventricular bigeminy",
    "(VT": "ventricular tachycardia",
    "(AFIB": "atrial fibrillation",
    "(T": "ventricular trigeminy",
    "(SAB": "sino-atrial block",
    "(P": "paced rhythm",
    "(BII": "second-degree heart block",
    "(SBR": "sinus bradycardia",
    "(NOD": "nodal (AV junctional) rhythm",
    "(SVTA": "supraventricular tachyarrhythmia",
}

#: First sample any manual annotation may occupy, and the window that covers all of
#: them. The paper's rule was "only during the final 5 minutes"; measured, the
#: earliest annotation sits at 600.5 s and the latest at 896.9 s. The length is
#: set by the shortest record (224,993 samples), so this window fits all 105.
ANNOTATED_WINDOW = (150000, 74993)

_AGE_SEX_ESC_RE = re.compile(r"^Age:\s*(?P<age>\S+)\s+Sex:\s*(?P<sex>\S+)\s*$")
_AGE_SEX_MIT_RE = re.compile(
    r"^(?P<age>-?\d+)\s+(?P<sex>[MF])\s+(?P<tape>\S+)\s+(?P<recorder>\S+)\s+"
    r"x(?P<speed>\d+)\s*$"
)
_RECORDER_RE = re.compile(r"^Recorder type:\s*(?P<recorder>.*)$")
_PROVENANCE_RE = re.compile(
    r"^Produced by (?:xform|xform_new) from record (?P<record>\S+), "
    r"beginning at (?P<offset>\S+)\s*$"
)
_DELAY_RE = re.compile(r"^The signal (?P<signal>\d+) was delayed with a delay=(?P<n>\d+) samples")
_MITDB_MEDICATION_HINT = re.compile(r"^[A-Z][A-Za-z0-9]*(,\s*[A-Za-z0-9]+)*$")


def _record_source() -> dict[str, str]:
    """Record name -> source database key."""
    return {
        record: database
        for database, records in SOURCE_DATABASE_RECORDS.items()
        for record in records
    }


def _hms_to_seconds(text: str) -> float:
    """Parse the ``H:MM:SS.mmm`` / ``MM:SS.mmm`` / ``0`` offsets xform writes."""
    parts = [float(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_header(hea_path: Path) -> dict[str, object]:
    """Read one ``.hea``: signal description, declared calibration and provenance.

    Reads the text rather than going through ``wfdb.rdheader`` for two fields wfdb
    normalises away: a declared gain of ``0`` (which wfdb silently replaces with
    its 200 adu/mV fallback, hiding that the record is uncalibrated) and an
    explicit baseline, which only four records carry and which is what puts a
    +5.12 mV pedestal on them.
    """
    lines = hea_path.read_text().splitlines()
    fields = lines[0].split()
    n_sig = int(fields[1])
    # "250" or "250/360" — the second number is xform's record of the original rate.
    rate_field = fields[2]
    counter_freq = float(rate_field.split("/")[1]) if "/" in rate_field else None

    lead_names: list[str] = []
    declared_gains: list[float] = []
    declared_baselines: list[int | None] = []
    adc_zeros: list[int] = []
    adc_res: list[int] = []
    for line in lines[1 : 1 + n_sig]:
        parts = line.split()
        gain_field = parts[2]
        if "(" in gain_field:
            gain_text, baseline_text = gain_field.rstrip(")").split("(")
            declared_baselines.append(int(baseline_text))
        else:
            gain_text = gain_field
            declared_baselines.append(None)
        declared_gains.append(float(gain_text.split("/")[0]))
        adc_res.append(int(parts[3]))
        adc_zeros.append(int(parts[4]))
        lead_names.append(parts[-1])

    provenance: list[tuple[str, float]] = []
    delays: dict[int, int] = {}
    clinical: list[str] = []
    for line in lines:
        if not line.startswith("#"):
            continue
        text = line[1:].strip()
        if not text or set(text) <= {"-"}:
            continue
        match = _PROVENANCE_RE.match(text)
        if match:
            provenance.append((match["record"], _hms_to_seconds(match["offset"])))
            continue
        match = _DELAY_RE.match(text)
        if match:
            delays[int(match["signal"])] = int(match["n"])
            continue
        if text.startswith('Produced by "'):
            continue
        clinical.append(text)

    return {
        "n_sig": n_sig,
        "n_samples": int(fields[3]),
        "counter_freq": counter_freq,
        "lead_names": lead_names,
        "declared_gains": declared_gains,
        "declared_baselines": declared_baselines,
        "adc_zeros": adc_zeros,
        "adc_res": adc_res,
        "provenance": provenance,
        "delays": delays,
        "clinical": clinical,
    }


def parse_lead_names(hea_path: Path) -> list[str]:
    """Signal descriptions in file order, as the release spells them."""
    return list(parse_header(hea_path)["lead_names"])  # type: ignore[arg-type]


def _parse_clinical(clinical: list[str], source_database: str) -> dict[str, object]:
    """Split the header's clinical block into columns, per its vintage.

    Two formats and one absence:

    - **MIT-BIH Arrhythmia** (15 records) — ``69 M 1085 1629 x1`` then medications
      then free text, copied verbatim from mitdb, where field 3 is the analog tape
      (and so the subject) and field 4 the recorder. See ``ecgbench.labels.mitdb``,
      which decoded them.
    - **European ST-T** (33 records) — ``Age: 47  Sex: M``, then one line per
      clinical finding, then ``Recorder type: ...``. This is an earlier, coarser
      vintage of the same text edb 1.0.0 carries; see the module docstring.
    - **Nothing at all** (57 records) — the Supraventricular, NSR, Long-Term, ST
      Change and sudden-death headers carry no clinical line whatsoever.
    """
    out: dict[str, object] = {
        "age": np.nan,
        "sex": "",
        "recorder": "",
        "medications": "",
        "clinical_findings": "",
        "clinical_notes": "",
        "analog_tape": "",
        "playback_speed": "",
        "clinical_source": "none",
    }
    if not clinical:
        return out

    if source_database == "mitdb":
        out["clinical_source"] = "mitdb_header"
        match = _AGE_SEX_MIT_RE.match(clinical[0])
        if match:
            age = int(match["age"])
            # mitdb writes -1 for an unrecorded age; sel103 is the one such record.
            out["age"] = float(age) if age >= 0 else np.nan
            out["sex"] = match["sex"]
            out["analog_tape"] = match["tape"]
            out["recorder"] = match["recorder"]
            out["playback_speed"] = f"x{match['speed']}"
            rest = clinical[1:]
        else:
            rest = clinical
        if rest and _MITDB_MEDICATION_HINT.match(rest[0]):
            out["medications"] = "" if rest[0] == "None" else rest[0]
            rest = rest[1:]
        out["clinical_notes"] = " ".join(rest)
        return out

    out["clinical_source"] = "esc_header"
    findings: list[str] = []
    for line in clinical:
        match = _AGE_SEX_ESC_RE.match(line)
        if match:
            # sele0166 records both as "-", meaning unknown.
            out["age"] = pd.to_numeric(match["age"], errors="coerce")
            out["sex"] = "" if match["sex"] == "-" else match["sex"]
            continue
        match = _RECORDER_RE.match(line)
        if match:
            # "ICR model 7200", "ICR  model 7200", "OXFORD model MEDILOG1",
            # "OXFORD Medilog 1" — the same handful of recorders spelled several
            # ways. Collapse the whitespace and leave the spelling alone; edb's
            # own loader is the place to go for a normalised value.
            out["recorder"] = " ".join(match["recorder"].split())
            continue
        if "medication" in line.lower():
            out["medications"] = line
            continue
        findings.append(line)
    out["clinical_findings"] = ";".join(findings)
    return out


def load_beat_annotations(
    data_path: Path | str,
    records: list[str] | None = None,
    annotator: str = "q1c",
) -> pd.DataFrame:
    """Return the manual boundary annotations, one row per annotated beat.

    This is the dataset's actual ground truth. ``load_labels`` summarises it; a
    delineation model wants this.

    Args:
        data_path: Root of a local qtdb copy (the directory holding ``sel100.hea``).
        records: Restrict to these record names. Defaults to the shipped ``RECORDS``.
        annotator: ``"q1c"`` (annotator 1, all 105 records) or ``"q2c"``
            (annotator 2, 11 records). The unaudited first pass ``qt1``/``qt2``
            replaces every boundary symbol with ``|`` and so cannot be parsed for
            wave types — passing it raises.

    Returns:
        One row per beat, with sample indices (``NaN`` where the annotator did not
        mark that point) and the intervals they imply:

        ============================ ======================================
        record_name, beat_index      identity within the record
        symbol, morphology_group     beat class, and its ARISTOTLE cluster
        qrs_onset/peak/offset        samples; peak is the annotation itself
        p_onset/peak/offset          samples
        t_onset/peak/offset          samples
        u_onset/peak/offset          samples
        qrs_ms, pr_ms, p_ms, qt_ms   intervals in milliseconds
        rr_ms, qtc_bazett_ms         NaN unless the previous beat is adjacent
        ============================ ======================================

    Marks are assigned to beats by position: P marks and the QRS onset belong to
    the next beat, everything from the QRS offset onwards to the previous one.
    That is unambiguous for this annotation style and needs no tolerance
    parameter, because the annotators worked beat by beat in a graphic editor.

    ``rr_ms`` — and therefore Bazett's ``qtc_bazett_ms`` — is only defined where
    the preceding annotated beat is the preceding *actual* beat. Records were
    annotated in runs of 30 consecutive beats plus up to 20 of each non-dominant
    morphology, so runs are separated by gaps of arbitrary length. Any RR longer
    than ``_MAX_PLAUSIBLE_RR_S`` is treated as a gap rather than a 3-second pause.
    """
    import wfdb

    if annotator not in MANUAL_ANNOTATORS:
        raise ValueError(
            f"annotator must be one of {MANUAL_ANNOTATORS}; {annotator!r} is not a "
            "second-pass file. The first-pass files qt1/qt2 mark every boundary "
            "with '|' and do not say which wave it belongs to."
        )

    data_path = Path(data_path)
    names = records if records is not None else read_record_list(data_path)

    frames = []
    for record in names:
        ann_path = data_path / f"{record}.{annotator}"
        if not ann_path.exists():
            continue
        frames.append(
            _beats_from_annotation(
                record, wfdb.rdann(str(data_path / record), annotator)
            )
        )
    if not frames:
        # An empty frame with the real columns, not a shapeless one. ``q2c`` covers
        # 11 of the 105 records, so a subset holding none of them is ordinary — and
        # scan_records groups this result by record_name, which a columnless frame
        # cannot do.
        return pd.DataFrame(columns=list(_BEAT_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    logger.info(
        "qtdb %s: %d beats over %d records", annotator, len(out), out.record_name.nunique()
    )
    return out


#: Longest RR treated as consecutive. The slowest annotated beat-to-beat interval
#: in the release is under 2 s; anything longer is a gap between annotated runs,
#: not a pause, and must not become an RR value.
_MAX_PLAUSIBLE_RR_S = 3.0

_MARK_FIELDS = (
    "p_onset", "p_peak", "p_offset",
    "qrs_onset", "qrs_peak", "qrs_offset",
    "t_onset", "t_peak", "t_offset",
    "u_onset", "u_peak", "u_offset",
)

_INTERVAL_FIELDS = ("qrs_ms", "p_ms", "pr_ms", "qt_ms", "rr_ms", "qtc_bazett_ms")

#: Full column set of a per-beat frame, so an empty result still has a shape.
_BEAT_COLUMNS = (
    ("record_name", "beat_index", "symbol", "morphology_group")
    + _MARK_FIELDS
    + _INTERVAL_FIELDS
)


def _beats_from_annotation(record: str, ann) -> pd.DataFrame:  # noqa: ANN001
    """Turn one record's annotation stream into a per-beat frame."""
    symbols = list(ann.symbol)
    samples = np.asarray(ann.sample, dtype=np.int64)
    nums = np.asarray(ann.num, dtype=np.int64)

    beat_positions = [i for i, s in enumerate(symbols) if s in BEAT_SYMBOLS]
    rows: list[dict[str, object]] = []
    for order, position in enumerate(beat_positions):
        row: dict[str, object] = {
            "record_name": record,
            "beat_index": order,
            "symbol": symbols[position],
            "morphology_group": int(nums[position]),
            "qrs_peak": int(samples[position]),
        }
        for field in _MARK_FIELDS:
            row.setdefault(field, np.nan)
        row["qrs_peak"] = int(samples[position])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(_BEAT_COLUMNS))

    # Marks before a beat (P wave, QRS onset) attach to the next beat; marks after
    # it (QRS offset, T, U) to the previous one.
    next_beat = np.searchsorted(beat_positions, np.arange(len(symbols)), side="left")
    previous_beat = next_beat - 1
    for index, (symbol, num) in enumerate(zip(symbols, nums)):
        if symbol in BEAT_SYMBOLS:
            continue
        wave = WAVE_TYPES.get(int(num))
        if wave is None:
            continue
        if symbol == "(":
            role, before = f"{wave}_onset", wave in ("p", "qrs")
        elif symbol == ")":
            role, before = f"{wave}_offset", wave == "p"
        elif symbol in ("p", "t", "u"):
            role, before = f"{symbol}_peak", symbol == "p"
        else:
            continue
        target = next_beat[index] if before else previous_beat[index]
        if not 0 <= target < len(rows):
            continue
        if pd.isna(rows[target][role]):
            rows[target][role] = int(samples[index])

    out = pd.DataFrame(rows)
    per_ms = 1000.0 / SAMPLING_RATE
    out["qrs_ms"] = (out["qrs_offset"] - out["qrs_onset"]) * per_ms
    out["p_ms"] = (out["p_offset"] - out["p_onset"]) * per_ms
    out["pr_ms"] = (out["qrs_onset"] - out["p_onset"]) * per_ms
    out["qt_ms"] = (out["t_offset"] - out["qrs_onset"]) * per_ms

    rr = out["qrs_peak"].diff() * per_ms
    rr[rr > _MAX_PLAUSIBLE_RR_S * 1000] = np.nan
    out["rr_ms"] = rr
    out["qtc_bazett_ms"] = out["qt_ms"] / np.sqrt(out["rr_ms"] / 1000.0)
    return out


def read_record_list(data_path: Path | str) -> list[str]:
    """Record names from the shipped ``RECORDS`` file.

    Read from the file rather than globbed, so the 57 superseded ``.hea-`` headers
    and the 105 ``.xws`` display-settings files cannot enter the partition.
    """
    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        from ecgbench.labels import LabelSourceMissingError

        raise LabelSourceMissingError(
            f"No RECORDS file in {data_path}. QTDB's record list ships with the "
            "data; ECGBench publishes fold CSVs only, so point data_path at a "
            "full local copy (see https://physionet.org/content/qtdb/1.0.0/)."
        )
    return records_file.read_text().split()


def _waveform_pattern(beats: pd.DataFrame) -> str:
    """Rebuild Table 2's pattern string from the annotations actually present.

    A mark counts as part of the pattern when at least half the record's beats
    carry it, which is how the paper's own column behaves: ``sel36``'s pattern is
    ``(B)(t)u)`` because its beats are bundle-branch-block beats with no P wave.
    """
    if beats.empty:
        return ""
    half = len(beats) / 2.0
    modal_symbol = beats["symbol"].mode().iat[0]
    tokens = [
        ("p_onset", "("), ("p_peak", "p"), ("p_offset", ")"),
        ("qrs_onset", "("), ("qrs_peak", modal_symbol), ("qrs_offset", ")"),
        ("t_onset", "("), ("t_peak", "t"), ("t_offset", ")"),
        ("u_onset", "("), ("u_peak", "u"), ("u_offset", ")"),
    ]
    return "".join(
        token for field, token in tokens if beats[field].notna().sum() >= half
    )


def _summarise_manual(record: str, beats: pd.DataFrame) -> dict[str, object]:
    """Per-record summary of one annotator's manual boundary annotations."""
    if beats.empty:
        return {"n_annotated_beats": 0}
    out: dict[str, object] = {
        "n_annotated_beats": int(len(beats)),
        "n_p_waves": int(beats["p_peak"].notna().sum()),
        "n_qrs_onsets": int(beats["qrs_onset"].notna().sum()),
        "n_qrs_offsets": int(beats["qrs_offset"].notna().sum()),
        "n_t_onsets": int(beats["t_onset"].notna().sum()),
        "n_t_ends": int(beats["t_offset"].notna().sum()),
        "n_u_waves": int(beats["u_peak"].notna().sum()),
        "annotates_p": bool(beats["p_peak"].notna().sum() >= len(beats) / 2),
        "annotates_t": bool(beats["t_offset"].notna().sum() >= len(beats) / 2),
        "annotates_u": bool(beats["u_peak"].notna().sum() >= len(beats) / 2),
        "beat_symbols": ";".join(
            f"{s}:{n}" for s, n in beats["symbol"].value_counts().items()
        ),
        "n_morphology_groups": int(beats["morphology_group"].nunique()),
        "waveform_pattern": _waveform_pattern(beats),
        "annotation_start_secs": round(
            float(beats[["p_onset", "qrs_onset"]].min().min()) / SAMPLING_RATE, 3
        ),
        "annotation_end_secs": round(
            float(beats[["t_offset", "u_offset", "qrs_offset"]].max().max())
            / SAMPLING_RATE,
            3,
        ),
    }
    for column, name in (
        ("qt_ms", "median_qt_ms"),
        ("qtc_bazett_ms", "median_qtc_bazett_ms"),
        ("rr_ms", "median_rr_ms"),
        ("qrs_ms", "median_qrs_ms"),
        ("pr_ms", "median_pr_ms"),
        ("p_ms", "median_p_ms"),
    ):
        values = beats[column].dropna()
        out[name] = round(float(values.median()), 1) if len(values) else np.nan
    rr = beats["rr_ms"].dropna()
    out["median_heart_rate_bpm"] = (
        round(60000.0 / float(rr.median()), 1) if len(rr) else np.nan
    )
    published = PUBLISHED_ANNOTATED_BEATS.get(record)
    out["n_annotated_beats_published"] = published if published is not None else -1
    out["annotated_beats_matches_published"] = bool(published == len(beats))
    out["waveform_pattern_matches_published"] = bool(
        PUBLISHED_WAVEFORM_PATTERN.get(record) == out["waveform_pattern"]
    )
    return out


def _summarise_source_annotations(record_path: Path) -> dict[str, object]:
    """Summarise the ``.atr`` annotations inherited from the source database.

    Present for 82 of 105 records. The 23 sudden-death excerpts have none — the
    paper says so explicitly ("The record.atr files do not exist for the 24 sudden
    death records"), and it is 23 rather than 24 because ``sel17152``, the matched
    control the paper counts in that group, does have one.
    """
    import wfdb

    out: dict[str, object] = {
        "has_source_annotations": False,
        "n_source_annotations": 0,
        "n_source_beats": 0,
        "n_source_rhythm_changes": 0,
        "n_source_missed_beats": 0,
        "n_source_st_markers": 0,
        "n_source_t_markers": 0,
        "n_source_artifacts": 0,
        "n_source_quality_changes": 0,
        "source_beat_symbols": "",
        "source_rhythms": "",
        "dominant_source_rhythm": "",
    }
    for aami in ("N", "S", "V", "F", "Q"):
        out[f"n_source_aami_{aami}"] = 0
    if not record_path.with_suffix(f".{SOURCE_ANNOTATOR}").exists():
        return out

    ann = wfdb.rdann(str(record_path), SOURCE_ANNOTATOR)
    symbols = pd.Series(ann.symbol)
    out["has_source_annotations"] = True
    out["n_source_annotations"] = int(len(symbols))

    beats = symbols[symbols.isin(AAMI_CLASSES)]
    out["n_source_beats"] = int(len(beats))
    out["source_beat_symbols"] = ";".join(
        f"{s}:{n}" for s, n in beats.value_counts().items()
    )
    aami = beats.map(AAMI_CLASSES).value_counts()
    for name, count in aami.items():
        out[f"n_source_aami_{name}"] = int(count)

    out["n_source_rhythm_changes"] = int((symbols == "+").sum())
    out["n_source_artifacts"] = int((symbols == "|").sum())
    out["n_source_quality_changes"] = int((symbols == "~").sum())
    out["n_source_st_markers"] = int((symbols == "s").sum())
    out["n_source_t_markers"] = int((symbols == "T").sum())

    notes = [str(a).strip("\x00").strip() for a in ann.aux_note]
    out["n_source_missed_beats"] = sum(
        1 for s, n in zip(ann.symbol, notes) if s == '"' and n == "MISSB"
    )

    # Time in each rhythm, so "dominant" means what it does in the mitdb, afdb and
    # ltafdb loaders: the rhythm holding the most seconds, not the most markers.
    rhythm_spans: dict[str, float] = {}
    current: str | None = None
    start = 0
    end_sample = int(ann.sample[-1]) if len(ann.sample) else 0
    for symbol, sample, note in zip(ann.symbol, ann.sample, notes):
        if symbol != "+" or not note.startswith("("):
            continue
        if current is not None:
            rhythm_spans[current] = rhythm_spans.get(current, 0.0) + (
                int(sample) - start
            ) / SAMPLING_RATE
        current = note
        start = int(sample)
    if current is not None:
        rhythm_spans[current] = rhythm_spans.get(current, 0.0) + (
            end_sample - start
        ) / SAMPLING_RATE
    if rhythm_spans:
        out["source_rhythms"] = ";".join(
            f"{RHYTHM_NAMES.get(code, code)}:{seconds:.0f}s"
            for code, seconds in sorted(
                rhythm_spans.items(), key=lambda kv: -kv[1]
            )
        )
        dominant = max(rhythm_spans, key=lambda k: rhythm_spans[k])
        out["dominant_source_rhythm"] = RHYTHM_NAMES.get(dominant, dominant)
    return out


def _summarise_auto(record_path: Path) -> dict[str, object]:
    """Summarise the three ``ecgpuwave`` layers, including T-wave morphology.

    The automatic annotations cover the whole 15 minutes, unlike the manual ones,
    so they are ~1,000x more numerous and ~0 percent audited. They are useful as a
    weak label or a baseline to beat, never as ground truth.
    """
    import wfdb

    out: dict[str, object] = {}
    for annotator in AUTO_ANNOTATORS:
        out[f"n_auto_beats_{annotator}"] = 0
    out["dominant_t_morphology"] = ""
    out["t_morphology_counts"] = ""

    for annotator in AUTO_ANNOTATORS:
        if not record_path.with_suffix(f".{annotator}").exists():
            continue
        ann = wfdb.rdann(str(record_path), annotator)
        symbols = pd.Series(ann.symbol)
        out[f"n_auto_beats_{annotator}"] = int(symbols.isin(BEAT_SYMBOLS).sum())
        if annotator != "pu":
            continue
        t_nums = pd.Series(ann.num)[symbols == "t"]
        if t_nums.empty:
            continue
        counts = t_nums.value_counts()
        out["t_morphology_counts"] = ";".join(
            f"{T_MORPHOLOGY.get(int(k), k)}:{v}" for k, v in counts.items()
        )
        out["dominant_t_morphology"] = T_MORPHOLOGY.get(
            int(counts.idxmax()), str(counts.idxmax())
        )
    return out


def reconstruct_patient_ids(df: pd.DataFrame) -> pd.Series:
    """Return a subject id per record, collapsing the two known shared subjects.

    Every record here comes from a different source recording, so the record name
    is a subject id for 101 of the 105. The exceptions are the two European ST-T
    subjects who each contributed two recordings — see ``EDB_SHARED_SUBJECTS`` for
    where that comes from and why it is a literal.

    **This is an upper bound on distinctness, not a verified subject count.** The
    13 Supraventricular and 6 ST Change records inherit nothing that identifies
    their subjects, in qtdb or in their own releases, so if two of them share a
    person nothing here can tell.
    """
    group_of = {
        member: naming
        for naming, members in EDB_SHARED_SUBJECTS.items()
        for member in members
    }
    ids = df["record_name"].map(lambda r: group_of.get(r, r))
    shared = ids.value_counts()
    shared = shared[shared > 1]
    if len(shared):
        logger.info(
            "qtdb: %d records over %d subjects (%d subject(s) contributed more "
            "than one record: %s)",
            len(df),
            ids.nunique(),
            len(shared),
            shared.to_dict(),
        )
    return ids.rename("patient_id")


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``stratify_class``: the source database each excerpt came from.

    This is the release's own stratum — Table 1 of the paper is exactly this
    breakdown — and it is the axis everything else follows from: original sampling
    rate, electrode placement, whether reference annotations exist, whether the
    amplitude is calibrated, and what pathology the cohort has. edb 33, sddb 23,
    mitdb 15, svdb 13, nsrdb 10, stdb 6, ltdb 4, bih_control 1.

    Three alternatives were weaker:

    - **Annotated-beat count.** 79 of 105 records have exactly 30, so the bands are
      one huge class and a tail — it stratifies almost nothing.
    - **Waveform pattern** (which waves are annotated). 26 distinct values over 105
      records, most of them singletons, and it is a property of the annotation
      rather than of the signal.
    - **A clinical label.** There is no record-level diagnosis. 57 of the 105
      headers carry no clinical line at all, and the ``.atr`` rhythm markers exist
      for only 82 and are the source databases' annotations, not qtdb's.

    ``ltdb`` (4) and ``bih_control`` (1) are smaller than the 10 folds ECGBench
    generates, so neither can appear in every fold. They are kept separate rather
    than pooled because pooling four Long-Term Holter records into "other" would
    make the column stop meaning what Table 1 means.
    """
    return df.assign(stratify_class=df["source_database"])


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Build the per-record frame by reading every header and annotation file."""
    data_path = Path(data_path)
    records = read_record_list(data_path)
    source_of = _record_source()

    unknown = [r for r in records if r not in source_of]
    if unknown:
        raise ValueError(
            f"{len(unknown)} record(s) in {data_path / 'RECORDS'} are not in "
            f"SOURCE_DATABASE_RECORDS: {unknown[:5]}. That table is transcribed "
            "from Table 1 of Laguna et al. 1997 and covers qtdb 1.0.0's 105 "
            "records exactly, so a mismatch means a different release."
        )

    beats_by_record = {
        record: frame
        for record, frame in load_beat_annotations(data_path, records).groupby(
            "record_name"
        )
    }
    beats2_by_record = {
        record: frame
        for record, frame in load_beat_annotations(
            data_path, records, annotator="q2c"
        ).groupby("record_name")
    }

    rows: list[dict[str, object]] = []
    for record in records:
        source = source_of[record]
        header = parse_header(data_path / f"{record}.hea")
        lead_names = list(header["lead_names"])  # type: ignore[arg-type]
        gains = list(header["declared_gains"])  # type: ignore[arg-type]
        baselines = list(header["declared_baselines"])  # type: ignore[arg-type]
        adc_zeros = list(header["adc_zeros"])  # type: ignore[arg-type]
        provenance = list(header["provenance"])  # type: ignore[arg-type]
        delays: dict[int, int] = header["delays"]  # type: ignore[assignment]

        # The last provenance line is the offset within the source record. The 23
        # sudden-death headers carry two, written by two xform_new passes, and it
        # is the second that locates the excerpt in sddb — verified for 22 of them.
        source_record, source_offset = (
            provenance[-1] if provenance else ("", float("nan"))
        )
        if source_record == "mqt2":
            # sel16265, sel16272 and sel17152 name an intermediate file rather than
            # the source recording. The record name carries the real id.
            source_record = record[3:] if source == "nsrdb" else ""

        row: dict[str, object] = {
            "record_name": record,
            "signal_path": record,
            "source_database": source,
            "source_database_name": SOURCE_DATABASE_NAMES[source],
            "source_catalogue_slug": SOURCE_CATALOGUE_SLUG[source] or "",
            "source_record": source_record,
            "source_offset_secs": round(source_offset, 3),
            "source_sampling_rate": SOURCE_SAMPLING_RATE[source],
            "resampled_from_source": header["counter_freq"] is not None,
            "n_samples": header["n_samples"],
            "duration_secs": round(
                int(header["n_samples"]) / SAMPLING_RATE, 3  # type: ignore[arg-type]
            ),
            "lead_names": ";".join(lead_names),
            "lead_0": lead_names[0],
            "lead_1": lead_names[1] if len(lead_names) > 1 else "",
            "positional_lead_names": lead_names == ["ECG1", "ECG2"],
            "declared_gain_0": gains[0],
            "declared_gain_1": gains[1] if len(gains) > 1 else np.nan,
            # A declared gain of 0 means "uncalibrated"; wfdb quietly substitutes
            # 200 adu/mV, so p_signal looks like millivolts and is not. The paper
            # additionally says the sudden-death gains are estimates.
            "amplitude_calibrated": bool(
                0.0 not in gains and source not in ("sddb", "bih_control")
            ),
            "dc_pedestal_mv": _dc_pedestal_mv(record, gains, baselines, adc_zeros),
            "signal_0_delay_samples": delays.get(0, 0),
            "n_first_pass_annotations": _count_annotations(
                data_path / record, FIRST_PASS_ANNOTATORS[0]
            ),
            "has_second_annotator": record in beats2_by_record,
            "n_annotated_beats_annotator2": len(beats2_by_record.get(record, [])),
            "source_record_verified": (
                pd.NA
                if source not in ("edb", "sddb")
                else record not in UNVERIFIED_SOURCE_RECORDS
            ),
        }
        if header["counter_freq"] is not None:
            declared = int(header["counter_freq"])  # type: ignore[arg-type]
            if declared != SOURCE_SAMPLING_RATE[source]:
                logger.warning(
                    "%s: header counter frequency %d Hz disagrees with the %d Hz "
                    "expected for source database %s",
                    record,
                    declared,
                    SOURCE_SAMPLING_RATE[source],
                    source,
                )
        row.update(_parse_clinical(list(header["clinical"]), source))  # type: ignore[arg-type]
        row.update(_summarise_manual(record, beats_by_record.get(record, pd.DataFrame())))
        row.update(_summarise_source_annotations(data_path / record))
        row.update(_summarise_auto(data_path / record))
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.assign(patient_id=reconstruct_patient_ids(df).to_numpy())
    return attach_stratify_class(df)


def _dc_pedestal_mv(
    record: str,
    gains: list[float],
    baselines: list[int | None],
    adc_zeros: list[int],
) -> float:
    """Constant offset wfdb will add, in millivolts, from the declared calibration.

    wfdb computes ``physical = (digital - baseline) / gain`` and falls back to
    ``baseline = adc_zero`` when the header declares none. So a header that declares
    a baseline *different* from its ``adc_zero`` shifts the whole signal by
    ``(adc_zero - baseline) / gain``. Four records do — ``sel100``, ``sel102``,
    ``sel103``, ``sel104``, all declaring ``200(0)`` against an ``adc_zero`` of 1024
    — and the shift is +5.12 mV on both channels.

    Derived rather than looked up, so a re-release that fixes or spreads the problem
    is reported rather than mis-described. ``DC_PEDESTAL_RECORDS`` records which
    records it is in 1.0.0, and this raises if the two stop agreeing.
    """
    offsets = []
    for gain, baseline, adc_zero in zip(gains, baselines, adc_zeros):
        if baseline is None or baseline == adc_zero or not gain:
            offsets.append(0.0)
        else:
            offsets.append((adc_zero - baseline) / gain)
    pedestal = max(offsets, key=abs)
    expected = record in DC_PEDESTAL_RECORDS
    if bool(pedestal) != expected:
        logger.warning(
            "%s: declared calibration implies a DC pedestal of %.3f mV, but "
            "DC_PEDESTAL_RECORDS %s it. This is a different release from qtdb 1.0.0, "
            "or the header changed.",
            record,
            pedestal,
            "expects" if expected else "does not list",
        )
    return round(pedestal, 3)


def _count_annotations(record_path: Path, annotator: str) -> int:
    """Number of annotations in one file, 0 when it is absent."""
    import wfdb

    if not record_path.with_suffix(f".{annotator}").exists():
        return 0
    return int(len(wfdb.rdann(str(record_path), annotator).sample))


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return QTDB labels indexed by record name, one row per record.

    Everything is recomputed from the shipped files; nothing is copied from the
    paper except ``PUBLISHED_ANNOTATED_BEATS`` and ``PUBLISHED_WAVEFORM_PATTERN``,
    which are there to be checked against.

    Column groups:

    ``source_database`` … ``source_record_verified``
        Where the excerpt came from. **Read this before combining qtdb with any
        other ECGBench dataset** — six of the seven sources are datasets ECGBench
        also partitions, so ``source_catalogue_slug`` names the leakage partner.
    ``n_annotated_beats`` … ``median_p_ms``
        The manual boundary annotations, summarised. Use
        :func:`load_beat_annotations` for the beats themselves.
    ``has_second_annotator``, ``n_annotated_beats_annotator2``
        Inter-observer material. 11 records, and one of them has 2 beats.
    ``n_source_*``, ``dominant_source_rhythm``
        The source database's own reference annotations, inherited via ``.atr``.
        Absent for the 23 sudden-death records. These are *not* qtdb annotations.
    ``n_auto_beats_*``, ``dominant_t_morphology``
        ``ecgpuwave`` output over the whole record. Unaudited; a baseline, not
        ground truth.
    ``age``, ``sex``, ``clinical_*``, ``medications``, ``recorder``
        Populated for 48 records — 15 from mitdb headers, 33 from an older ESC
        vintage. ``clinical_source`` says which. Empty for the other 57.
    ``lead_names``, ``lead_0``, ``lead_1``, ``positional_lead_names``
        20 distinct layouts. ``ECG1``/``ECG2`` in 57 records are channel positions
        the release declines to name, not leads — ``positional_lead_names`` is
        True there. The 33 European ST-T records use the ESC's own electrode
        nomenclature (``D3``, ``CM5``, ``CC5``, ``ML5``), which does **not** match
        edb 1.0.0's relabelling for 31 of them.
    ``amplitude_calibrated``, ``declared_gain_*``, ``dc_pedestal_mv``
        False for 33 records; see point 9 of the module docstring.
    ``patient_id``, ``stratify_class``
        Fold construction.

    There is no diagnostic label, single- or multi-label, because the release has
    none: it is a delineation reference. ``stratify_class`` is provenance, not
    pathology — do not train on it.
    """
    data_path = Path(data_path)
    if not (data_path / "RECORDS").exists():
        from ecgbench.labels import LabelSourceMissingError

        raise LabelSourceMissingError(
            f"QTDB labels are built from the headers and annotation files in "
            f"{data_path}, which does not look like a qtdb copy (no RECORDS "
            f"file). ECGBench publishes fold CSVs only — point data_path at a "
            f"full local copy (see {config.url})."
        )

    df = scan_records(data_path)
    out = df.set_index(config.record_id_column)

    mismatched = df.loc[~df["annotated_beats_matches_published"], "record_name"]
    if len(mismatched):
        logger.info(
            "qtdb: %d beats manually annotated, against the paper's %d. %d record(s) "
            "disagree with Table 2: %s",
            int(df["n_annotated_beats"].sum()),
            sum(PUBLISHED_ANNOTATED_BEATS.values()),
            len(mismatched),
            ", ".join(
                f"{r} ({df.loc[df.record_name == r, 'n_annotated_beats'].iat[0]} vs "
                f"{PUBLISHED_ANNOTATED_BEATS[r]})"
                for r in mismatched
            ),
        )
    uncalibrated = df.loc[~df["amplitude_calibrated"], "record_name"]
    if len(uncalibrated):
        logger.warning(
            "qtdb: %d of %d records have unreliable amplitude calibration (declared "
            "gain 0, or a sudden-death Holter whose gain the paper calls an "
            "estimate). Do not compare millivolt amplitudes across source "
            "databases without checking amplitude_calibrated.",
            len(uncalibrated),
            len(df),
        )
    logger.info(
        "Loaded QTDB labels: %d records, %d subjects, %d annotated beats over "
        "%d source databases",
        len(out),
        df["patient_id"].nunique(),
        int(df["n_annotated_beats"].sum()),
        df["source_database"].nunique(),
    )
    return out
