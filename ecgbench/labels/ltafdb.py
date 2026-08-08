"""
Long Term AF Database labels: the reference rhythm episodes and beats, per record.

Nothing machine-readable ships with this dataset. Every ``.hea`` is three lines of
pure signal specification with **no comment lines at all** — no age, no sex, no
medications, no clinical description::

    00 2 128 9661440  9:30:00 31/01/2003
    00.dat 16 166.945/mV 0 0 -1 -8202 0 ECG
    00.dat 16 173.01/mV 0 0 3 6311 0 ECG

So, as with ``afdb``, **the labels are the annotations**. Unlike ``afdb``, they are
very good ones: the ``.atr`` files hold over 9 million beat annotations and 60,817
rhythm episodes produced by MEDICALgorithmics' PocketECG algorithm and then
manually verified by their ECG technicians — reference annotations in the full
sense, contributed to PhysioNet in 2012, four years after the signals.

Two annotators ship, per the release's own ``ANNOTATORS`` file:

- ``.atr`` — **reference beat and rhythm annotations**. Beats are typed (``N``,
  ``A``, ``V``, ``Q``), rhythm changes are ``+`` annotations whose ``aux_note``
  names one of nine codes, and ``"`` comment annotations mark missed beats and
  pauses. This is the annotator everything below is derived from.
- ``.qrs`` — **unaudited** ``sqrs`` detections. Every beat is labelled ``N``
  whatever it actually is, artifacts are ``|``, and a handful of ``T`` markers
  flag spontaneous AF terminations. Summarised, never mixed into the ``.atr``
  counts.

Everything below was verified against the files.

**There is no atrial flutter code here.** ``afdb`` annotates four rhythms
(``N``/``AFIB``/``AFL``/``J``); this release annotates nine, and ``AFL`` is not
among them while ventricular bigeminy, trigeminy, atrial bigeminy, sinus
bradycardia, SVT, VT and idioventricular rhythm are. So ``af_burden`` here is
AFIB alone — there is no flutter to fold in — and the two databases' burden
figures are computed from different code sets even though they carry the same
name. See :data:`RHYTHM_NAMES` and :data:`AF_CODES`.

**The signal runs past the annotations, by hours in some records.** Both
annotators stop at the same place, so this is a property of the recording rather
than of one annotator. The median record's beat annotations stop 4.9 s from the
end, but 35 of the 84 stop more than ten minutes early, 17 stop more than an hour
early, and record 117 stops **8.05 hours** early — a third of the record.
``unannotated_tail_secs`` reports it per record, and ``window=`` users need it: a
window into that tail returns signal with no reference behind it.

**Rhythm durations nonetheless run to the end of the signal**, because the last
episode has no annotation after it to close it and PhysioNet's own summary tables
close it at the record end. This module follows that convention, and the result
was checked against ``tables.shtml`` row by row: **all 336 beat counts (84 records
x N/A/V/Q) and all 756 rhythm cells (84 x nine codes, episode count and duration)
agree**, with one exception given below. It does mean a record like 117
attributes its final 8 unannotated hours to whatever rhythm was running at 15.9 h.

**Record 20's published AFIB duration is 1.1 s short, and the release says why.**
``tables.shtml`` gives 24:19:08 (87,548 s) where the shipped header yields
87,549.1. The landing page thanks Mariano Llamedo Soria "for reporting an error
in the original version of ``20.hea``, and for providing a correction incorporated
in the current version": the table was generated against the pre-correction
header, which was 144 samples shorter. The current header is the authority, so
this module reports 87,549.1 and the difference is recorded rather than papered
over.

**``af_burden``'s denominator is annotated rhythm time, not record time.** The
first rhythm annotation sits a little way into each record — 47.5 s into record
20 — and nothing classifies what precedes it. Across the release that lead-in is
25.7 of the 1,960.6 recorded hours, and ``rhythm_annotated_secs`` states the
denominator explicitly rather than leaving it to be inferred.

**One annotation lies past the end of its record**, and it is not an error.
Record 30's ``.atr`` ends with a ``"`` comment carrying the WFDB ``\\x01 Aux``
marker at sample 4,198,064, while the record holds 2,826,240. That marker is the
annotation file's own terminator, present once in all 84 files; its position is
not a claim about the signal. :func:`summarise_annotations` excludes it from
every measurement, which is why 30 does not appear as a 3-hour-negative outlier
in ``unannotated_tail_secs``.

**The headers name both channels ``ECG``.** Not ``ECG1``/``ECG2`` as in ``afdb``
— literally the same string twice, in all 84 headers. Two identically named
channels cannot be told apart by name, so the config declares the positional
names ``ECG1``/``ECG2`` instead and says so; see ``ltafdb.yaml``. Nothing in the
release states an electrode placement, so they are channel positions either way.

**The ADC gains are real, and they vary per record and per channel.** ``afdb``
declares a gain of ``0`` (WFDB's "uncalibrated") on every signal and leans on
wfdb's 200 adu/mV fallback; here the headers carry 50 distinct measured gains,
sometimes different for the two channels of one record (record 100 is 88.968 and
131.062). wfdb applies them, so the samples arrive in genuine millivolts and
``signal_unit_scale`` stays 1.0. ``adc_gains`` exposes the pair per record, which
is worth looking at before comparing amplitudes across records — see next.

**Record 62's ECG1 gain is anomalous, and this module does not correct it.** It
declares 1123.6 adu/mV where the other 167 signal lines run 75.0188 to 222.222,
and its raw swing is entirely normal — 2,756 adu peak-to-peak against a release
median of 1,220 and a maximum of 2,808. The declared gain therefore turns the
largest raw excursion in the release into its smallest calibrated one, 2.45 mV
peak-to-peak against a median of 6.65. ECGBench reports what the header declares,
because a silently corrected record would disagree with every other tool reading
the same file, but anything comparing absolute amplitudes across records should
exclude record 62's ECG1 or rescale it. Its ECG2 (202.429) is unaffected.
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

#: The reference annotator: beats *and* rhythms, manually verified.
REFERENCE_ANNOTATOR = "atr"

#: The unaudited automatic detector, kept strictly separate from the above.
DETECTOR_ANNOTATOR = "qrs"

#: Beat symbols in the ``.atr`` files. Verified exhaustive across all 84 records:
#: every annotation is one of these four, a ``+`` rhythm change, or a ``"``
#: comment. There is no paced, bundle-branch-block or fusion beat in this release
#: — the PocketECG taxonomy used here is coarser than mitdb's fifteen symbols.
BEAT_SYMBOLS = ("N", "A", "V", "Q")

BEAT_NAMES = {
    "N": "normal beat",
    "A": "atrial premature beat",
    "V": "premature ventricular contraction",
    "Q": "unclassifiable beat",
}

#: Rhythm codes carried in the ``aux_note`` of a ``+`` annotation, and the nine
#: columns of the release's own ``tables.shtml``. **``AFL`` is deliberately absent**
#: — atrial flutter is not annotated in this database, unlike ``afdb``.
RHYTHM_NAMES = {
    "N": "sinus rhythm or any other unlisted rhythm",
    "AFIB": "atrial fibrillation",
    "SVTA": "supraventricular tachyarrhythmia",
    "VT": "ventricular tachycardia",
    "B": "ventricular bigeminy",
    "T": "ventricular trigeminy",
    "IVR": "idioventricular rhythm",
    "AB": "atrial bigeminy",
    "SBR": "sinus bradycardia",
}

#: Codes counted as AF for :func:`af_burden <load_labels>`. One code, not two:
#: this release annotates no flutter. Keeping it a tuple keeps the arithmetic
#: identical to ``afdb``'s so the two burdens are comparable in shape, and the
#: difference in membership is stated rather than hidden.
AF_CODES = ("AFIB",)

#: Free-text ``aux_note`` values on ``"`` comment annotations. ``MISSB`` and
#: ``PSE`` are the documented ones; ``M`` and ``MB`` are rare abbreviations of
#: ``MISSB`` appearing in five records (21, 49, 51, 105, 201) and are counted with
#: it rather than dropped.
MISSED_BEAT_NOTES = ("MISSB", "MB", "M")
PAUSE_NOTE = "PSE"

#: The WFDB annotation-file terminator: a ``"`` comment whose aux is this marker.
#: Exactly one per file in all 84 records, and its sample position is not a claim
#: about the signal — in record 30 it sits 2.98 hours past the end of the data.
TERMINATOR_AUX = "\x01 Aux"

#: ``T`` in a ``.qrs`` file marks a spontaneous AF termination, inserted by hand
#: by Swiryn and Moody to seed the AF Termination Challenge Database. In an
#: ``.atr`` file ``(T`` is a *rhythm* code meaning ventricular trigeminy. Same
#: letter, different annotator, unrelated meanings — hence two constants.
AF_TERMINATION_SYMBOL = "T"

#: AF-burden cut points for ``af_class``. Below the first, AF is an incidental
#: finding in an otherwise non-AF record; at or above the second, the record is in
#: AF essentially throughout. Same cuts as ``afdb``, so the two are comparable.
MINIMAL_AF_BURDEN = 0.05
SUSTAINED_AF_BURDEN = 0.95

#: Class names for ``af_class``, which is also the fold label here. See
#: :func:`attach_stratify_class` for why 84 records can afford three classes
#: where ``afdb``'s 25 could only afford two.
AF_MINIMAL = "minimal"
AF_PAROXYSMAL = "paroxysmal"
AF_SUSTAINED = "sustained"


def read_header(hea_path: Path) -> dict[str, object]:
    """Read the signal specification out of one header.

    Deliberately not ``wfdb.rdheader``: this is a scan over 84 files that must
    not stop because one of them is unreadable, and reading three lines of text
    cannot fail the way a record reader can.
    """
    lines = [
        line
        for line in hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    out: dict[str, object] = {
        "n_leads": 0,
        "n_samples": 0,
        "sampling_rate": 128,
        "start_time": "",
        "start_date": "",
        "lead_names": "",
        "adc_gains": "",
    }
    if not lines:
        logger.warning("Empty header: %s", hea_path.name)
        return out

    fields = lines[0].split()
    if len(fields) >= 4:
        out["n_leads"] = int(fields[1])
        out["sampling_rate"] = int(float(fields[2]))
        out["n_samples"] = int(fields[3])
    if len(fields) >= 5:
        out["start_time"] = fields[4]
    # Unlike afdb, every header here carries a date as well as a time of day.
    if len(fields) >= 6:
        out["start_date"] = fields[5]

    # Last field of each signal line is the description. Uniformly the bare
    # string "ECG" for both channels here — read rather than assumed, because it
    # is the only statement the files make about what the channels are.
    out["lead_names"] = "|".join(line.split()[-1] for line in lines[1:])
    # Field 3 of a signal line is "<gain>/<units>". These are measured per record
    # and per channel and range over 42 distinct values, which is why nothing
    # downstream rescales: wfdb applies them and the samples arrive in mV.
    out["adc_gains"] = "|".join(line.split()[2].split("/")[0] for line in lines[1:])
    return out


def summarise_annotations(record_path: Path, span_samples: int) -> dict[str, object]:
    """Summarise one record's reference ``.atr`` annotations.

    ``span_samples`` closes the final rhythm episode, which has no annotation
    after it. Using the record length here is PhysioNet's own convention in
    ``tables.shtml``, so the seconds this returns reproduce that table.

    Returns per-symbol beat counts, seconds and episode counts per rhythm code,
    the dominant rhythm by duration, AF burden, and how much of the record the
    beat annotations actually cover.
    """
    import wfdb

    out: dict[str, object] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
    out.update({f"rhythm_secs_{code}": 0.0 for code in RHYTHM_NAMES})
    out.update({f"n_episodes_{code}": 0 for code in RHYTHM_NAMES})
    out.update(
        {
            "n_beats": 0,
            "n_rhythm_changes": 0,
            "n_comment_annotations": 0,
            "n_missed_beats": 0,
            "n_pauses": 0,
            "rhythm_annotated_secs": 0.0,
            "rhythms": "",
            "dominant_rhythm": "",
            "dominant_rhythm_fraction": np.nan,
            "af_burden": np.nan,
            "longest_af_episode_secs": np.nan,
            "last_beat_sample": 0,
            "annotated_secs": np.nan,
            "unannotated_tail_secs": np.nan,
            "mean_heart_rate_bpm": np.nan,
        }
    )

    try:
        annotation = wfdb.rdann(str(record_path), REFERENCE_ANNOTATOR)
    except Exception as e:  # one unreadable .atr must not kill the whole scan
        logger.warning("Could not read %s.%s: %s", record_path.name, REFERENCE_ANNOTATOR, e)
        return out

    fs = float(getattr(annotation, "fs", 128) or 128)
    symbols = list(annotation.symbol)
    samples = np.asarray(annotation.sample, dtype=np.int64)
    notes = [str(aux).strip("\x00") for aux in annotation.aux_note]

    episodes: list[tuple[str, int]] = []
    beat_samples: list[int] = []

    for symbol, sample, raw_note in zip(symbols, samples, notes):
        note = raw_note.strip()
        if symbol in BEAT_SYMBOLS:
            out[f"beat_{symbol}"] = int(out[f"beat_{symbol}"]) + 1
            beat_samples.append(int(sample))
        elif symbol == "+":
            out["n_rhythm_changes"] = int(out["n_rhythm_changes"]) + 1
            if not note.startswith("("):
                logger.warning("%s: rhythm change with no code (%r)", record_path.name, raw_note)
                continue
            episodes.append((note[1:], int(sample)))
        elif symbol == '"':
            # Counted with the terminator included but out-of-range annotations
            # excluded, which is exactly what the release's own summary table
            # counts in its last column: record 30's terminator sits past the end
            # of the data and its published count is 1, not 2.
            if int(sample) < span_samples:
                out["n_comment_annotations"] = int(out["n_comment_annotations"]) + 1
            if raw_note == TERMINATOR_AUX:
                continue  # the file terminator; its sample is not a position
            if note in MISSED_BEAT_NOTES:
                out["n_missed_beats"] = int(out["n_missed_beats"]) + 1
            elif note == PAUSE_NOTE:
                out["n_pauses"] = int(out["n_pauses"]) + 1
            else:
                logger.warning("%s: unrecognised comment note %r", record_path.name, raw_note)
        else:
            logger.warning("%s: unexpected annotation symbol %r", record_path.name, symbol)

    out["n_beats"] = sum(int(out[f"beat_{symbol}"]) for symbol in BEAT_SYMBOLS)

    longest_af = 0.0
    for index, (code, start) in enumerate(episodes):
        end = episodes[index + 1][1] if index + 1 < len(episodes) else span_samples
        if code not in RHYTHM_NAMES:
            logger.warning("%s: unknown rhythm code %r", record_path.name, code)
            continue
        seconds = max(0.0, (end - start) / fs)
        out[f"rhythm_secs_{code}"] = float(out[f"rhythm_secs_{code}"]) + seconds
        out[f"n_episodes_{code}"] = int(out[f"n_episodes_{code}"]) + 1
        if code in AF_CODES:
            longest_af = max(longest_af, seconds)

    seconds_by_code = {
        code: float(out[f"rhythm_secs_{code}"])
        for code in RHYTHM_NAMES
        if float(out[f"rhythm_secs_{code}"]) > 0
    }
    total = sum(seconds_by_code.values())
    # The denominator of af_burden, made explicit. It is NOT record_seconds: the
    # first rhythm annotation is a few seconds into the record (47.5 s in record
    # 20), and nothing classifies what came before it. Over the release that
    # lead-in accounts for 25.7 of the 1,960.6 recorded hours.
    out["rhythm_annotated_secs"] = float(total)
    if total > 0:
        ordered = sorted(seconds_by_code, key=lambda code: -seconds_by_code[code])
        out["rhythms"] = "|".join(ordered)
        out["dominant_rhythm"] = ordered[0]
        out["dominant_rhythm_fraction"] = seconds_by_code[ordered[0]] / total
        out["af_burden"] = sum(seconds_by_code.get(code, 0.0) for code in AF_CODES) / total
        out["longest_af_episode_secs"] = longest_af

    if beat_samples:
        positions = np.asarray(beat_samples, dtype=np.float64)
        out["last_beat_sample"] = int(positions[-1])
        out["annotated_secs"] = float(positions[-1] / fs)
        # The stretch of signal past the last annotated beat. Milliseconds in most
        # records and hours in a few — see the module docstring.
        out["unannotated_tail_secs"] = max(0.0, (span_samples - positions[-1]) / fs)
        rr = np.diff(positions) / fs
        rr = rr[(rr > 0.2) & (rr < 2.5)]  # drop detector glitches, not real beats
        if len(rr):
            out["mean_heart_rate_bpm"] = float(60.0 / rr.mean())

    return out


def summarise_detector(record_path: Path) -> dict[str, object]:
    """Summarise the unaudited ``.qrs`` detections, kept apart from the reference.

    ``n_detections`` is not a beat count in the sense ``n_beats`` is: ``sqrs``
    emits ``N`` for everything it finds, so this counts detections of any type.
    The one thing here that exists nowhere else is ``n_af_terminations`` — the
    hand-placed ``T`` markers that the AF Termination Challenge Database's
    one-minute excerpts were cut around.
    """
    import wfdb

    out: dict[str, object] = {
        "n_detections": 0,
        "n_detector_artifacts": 0,
        "n_af_terminations": 0,
    }

    try:
        detections = wfdb.rdann(str(record_path), DETECTOR_ANNOTATOR)
    except Exception as e:
        logger.warning("Could not read %s.%s: %s", record_path.name, DETECTOR_ANNOTATOR, e)
        return out

    for symbol in detections.symbol:
        if symbol == "N":
            out["n_detections"] = int(out["n_detections"]) + 1
        elif symbol == "|":
            out["n_detector_artifacts"] = int(out["n_detector_artifacts"]) + 1
        elif symbol == AF_TERMINATION_SYMBOL:
            out["n_af_terminations"] = int(out["n_af_terminations"]) + 1
        else:
            logger.warning("%s.qrs: unexpected symbol %r", record_path.name, symbol)
    return out


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so
    the answer does not depend on what happens to be on disk — which matters here,
    because a partial download of this 3.4 GB release looks exactly like a smaller
    database.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. Long Term AF Database labels are "
            "the .atr annotations themselves, so point data_path at the dataset "
            "root — the flat directory holding 00.hea, RECORDS and ANNOTATORS. "
            "Get it from https://physionet.org/content/ltafdb/1.0.0/"
        )

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue

        row: dict[str, object] = {"record_name": name}
        header = read_header(hea)
        row.update(header)

        n_samples = int(header["n_samples"])
        row["record_seconds"] = n_samples / float(header["sampling_rate"])
        row["record_hours"] = row["record_seconds"] / 3600.0

        record_stem = hea.with_suffix("")
        row.update(summarise_annotations(record_stem, n_samples))
        row.update(summarise_detector(record_stem))
        # Flat tree: wfdb takes the bare stem, no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort on the record name as a STRING, which is what it is: "00" before "100"
    # before "20". Sorting numerically would silently assert the ids are numbers,
    # which is the same mistake zero_padded_identifiers exists to prevent.
    df = df.sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d LTAFDB records; %.1f h of signal, %d reference beats, "
        "%d rhythm episodes, %d AF terminations",
        len(df),
        df["record_hours"].sum(),
        int(df["n_beats"].sum()),
        int(sum(df[f"n_episodes_{code}"].sum() for code in RHYTHM_NAMES)),
        int(df["n_af_terminations"].sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``af_class``, and use it as ``stratify_class`` unchanged.

    This is the **only** derivation of the stratification label — ``LTAFDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    ``af_class`` bins ``af_burden`` into ``minimal`` (AF under 5% of the record),
    ``paroxysmal`` (5-95%) and ``sustained`` (95% or more), the same cuts
    ``afdb`` uses. Here the fold label **is** that column, where in ``afdb`` it
    had to be coarsened to a binary one: ``StratifiedKFold`` needs at least
    ``n_folds`` members per class, and 84 records leave every one of the three
    classes comfortably above the floor of 10 where 25 records did not. So the
    label a reader wants and the label the folds use are the same label, which is
    the arrangement to prefer whenever the counts allow it.

    Neither this nor ``dominant_rhythm`` is a beat- or sample-level label. For
    anything sample-level, use the ``.atr`` episodes directly.
    """
    out = df.copy()
    burden = out["af_burden"].fillna(0.0)

    out["af_class"] = np.select(
        [burden < MINIMAL_AF_BURDEN, burden >= SUSTAINED_AF_BURDEN],
        [AF_MINIMAL, AF_SUSTAINED],
        default=AF_PAROXYSMAL,
    )
    out["stratify_class"] = out["af_class"]

    logger.info("AF burden classes: %s", out["af_class"].value_counts().to_dict())
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Long Term AF Database labels indexed by record name.

    Columns:

    - ``dominant_rhythm``, ``rhythms``, ``dominant_rhythm_fraction`` and
      ``rhythm_secs_<CODE>`` — seconds spent in each of the nine annotated rhythms
      (see :data:`RHYTHM_NAMES`), from the manually verified ``.atr`` episodes.
      ``n_episodes_<CODE>`` counts episodes rather than duration; the two say
      different things, and this database makes the gap vivid — record 74 has 1,044
      AFIB episodes covering 23.5 h while record 12 has one covering 24.1 h.
    - ``af_burden`` — fraction of the record in **AFIB**. Not AFIB-or-flutter as in
      ``afdb``: there is no ``AFL`` code in this release. ``longest_af_episode_secs``
      complements it, and ``n_episodes_AFIB`` is the paroxysm count.
    - ``af_class`` — ``minimal`` / ``paroxysmal`` / ``sustained``, binned from
      ``af_burden``, and used unchanged as ``stratify_class``.
    - ``n_af_terminations`` — hand-placed ``T`` markers in the ``.qrs`` file
      recording spontaneous ends of AF episodes lasting a minute or more. These
      exist only in records 00-75; the 100- and 200-series carry none. They are
      what the AF Termination Challenge Database's excerpts were cut around.
    - ``beat_N`` / ``beat_A`` / ``beat_V`` / ``beat_Q`` and ``n_beats`` — typed
      reference beats from ``.atr``. ``n_detections`` and ``n_detector_artifacts``
      are the separate, **unaudited** ``.qrs`` detector's output, where every beat
      is called ``N`` regardless of type; do not add the two together.
    - ``n_missed_beats``, ``n_pauses``, ``n_comment_annotations`` — the ``"``
      comment markers. ``n_comment_annotations`` counts all of them including the
      file terminator, which is what the release's own summary table counts.
    - ``mean_heart_rate_bpm`` — from the ``.atr`` RR intervals, glitches outside
      0.2-2.5 s dropped.
    - ``record_seconds`` / ``record_hours`` / ``n_samples`` — record length, which
      varies from 6.1 h (record 30) to 26.2 h (record 11).
    - ``annotated_secs`` and ``unannotated_tail_secs`` — where the beat annotations
      stop, and how much signal follows. Usually milliseconds; **hours** in 17
      records, up to 8.05 h in record 117. Check it before windowing near the end
      of a record.
    - ``lead_names``, ``adc_gains``, ``n_leads``, ``sampling_rate``, ``start_time``,
      ``start_date`` — from the header. ``lead_names`` is ``ECG|ECG`` for every
      record: the release names both channels identically and states no electrode
      placement anywhere.

    There are **no demographics**: no age, no sex, no medication and no subject
    identifier beyond the record name. That is why the config sets
    ``patient_id_column: null`` — one record per subject is all that can be
    asserted, and PhysioNet's "84 subjects" is the source of even that.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
