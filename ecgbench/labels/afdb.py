"""
MIT-BIH Atrial Fibrillation labels: the reference rhythm episodes, per record.

Nothing machine-readable ships with this dataset — no metadata CSV, and unlike
``mitdb`` not even a header comment. Every ``.hea`` here is three lines of pure
signal specification::

    04015 2 250 9205760  9:00:00
    04015.dat 212 0 12 0 -55 -27172 0 ECG1
    04015.dat 212 0 12 0 -42 -28460 0 ECG2

So there are no demographics, no medications and no clinical description to
parse. **The labels are the annotations**, and they are what makes this database
worth having: manually reviewed rhythm episodes over 254.7 hours of two-lead
Holter, in four codes (``N``, ``AFIB``, ``AFL``, ``J``), plus 1,221,559 machine
beat annotations.

Three annotators ship, per the release's own ``ANNOTATORS`` file:

- ``.atr`` — the **manually reviewed rhythm** annotations, one per record. Each is
  a ``+`` marking where an episode *starts*; it runs until the next one. 623
  episodes across the 25 records.
- ``.qrs`` — **unaudited** beat annotations from an automatic detector, every
  symbol ``N``. Present for all 25 records. These are not beat *classifications*:
  the detector emits ``N`` for everything it finds, so counting ``beat_N`` here
  is counting detections, not normal beats.
- ``.qrsc`` — **manually corrected** beat annotations, for **two** records only
  (05091 and 07859). ``notes.txt`` names 05091; 07859's was added in 2014, long
  after that file was written, which is why it is not listed there.

Everything below was verified against the files.

**Two records ship no signals at all.** ``00735`` and ``03665`` have annotations
but no ``.dat``, and their headers declare ``0`` signals and ``0`` samples, which
the release's ``notes.txt`` states outright ("Signals unavailable"). They are
still in ``RECORDS`` and still carry rhythm labels, so this loader keeps them and
lets validation exclude them — the ``original`` version has all 25 records and
``clean`` has the 23 that can be read. ``wfdb.rdrecord`` fails on them with
``ValueError: sampto must be greater than sampfrom``, which is what ends up in
``quality_issues``; it means "this header declares an empty record", not a
corrupt file.

**Record length is not uniform.** 22 records hold 9,205,760 samples (36,823.04 s,
10 h 13.7 min) and ``06453`` holds 8,325,000 (33,300 s), which ``notes.txt``
explains — "Recording ends after about 9 hours, 15 minutes". The two signal-less
records declare 0.

**The signals run past the annotations.** Beat annotation stops at sample
~9,000,000, i.e. the nominal 10 h, while the ``.dat`` files hold 9,205,760
samples. So the last ~13.7 minutes of every full-length record carries signal
that nobody annotated — 823 s in 21 of the 23, and 1,692 s in 04048, whose
detector output stops earlier still. ``unannotated_tail_secs`` reports it per
record, measured against ``.qrs``; 07859's 2014 ``.qrsc`` is the one annotator
file that does cover a whole record.

**The amplitude is uncalibrated by the header's own declaration.** Every signal
line declares a gain of ``0``, WFDB's code for "uncalibrated", so ``wfdb`` falls
back to its default 200 adu/mV and reports the samples as millivolts. PhysioNet's
description says 12-bit over a ±10 mV range, which would make the true gain
204.8 adu/mV — so the values ECGBench returns are nominally 2.4% larger than that
figure implies. ECGBench keeps wfdb's 200, because that is what the files
declare and what every other AFDB pipeline applies; see the config for the full
argument. It affects absolute calibration only, never waveform shape.
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

#: Rhythm annotator — the manually reviewed one, and the only rhythm source.
RHYTHM_ANNOTATOR = "atr"

#: Beat annotators, in the order this loader prefers to report them. ``qrs`` is
#: unaudited and present everywhere; ``qrsc`` is corrected and present twice.
BEAT_ANNOTATOR = "qrs"
CORRECTED_BEAT_ANNOTATOR = "qrsc"

#: The four rhythm codes in the release's ``.atr`` files. Verified exhaustive:
#: all 623 ``+`` annotations across the 25 records carry one of these and nothing
#: else.
RHYTHM_NAMES = {
    "N": "sinus rhythm or any other non-AF rhythm",
    "AFIB": "atrial fibrillation",
    "AFL": "atrial flutter",
    "J": "AV junctional rhythm",
}

#: Codes counted as atrial fibrillation/flutter for ``af_burden``. ``J``
#: (junctional) is an escape rhythm, not AF, so it is excluded — it would
#: otherwise inflate 03665's burden from 16% to 68%.
AF_CODES = ("AFIB", "AFL")

#: Samples in a full-length record: 22 of the 23 records with signals declare
#: exactly this (36,823.04 s at 250 Hz). Used to close the final rhythm episode
#: of the two records whose headers declare 0 samples, which have annotations
#: covering the usual 10 hours but no signal file to measure against.
NOMINAL_SAMPLES = 9205760

#: AF-burden cut points for ``af_class``, the clinically shaped 3-level label.
#: Below the first, AF is an incidental finding in an otherwise non-AF record;
#: above the second, the record is in AF essentially throughout.
MINIMAL_AF_BURDEN = 0.05
SUSTAINED_AF_BURDEN = 0.95

#: AF-burden cut for ``stratify_class``, the 2-level fold label. See
#: :func:`attach_stratify_class` for why the fold label is binary and why the
#: threshold is 20% rather than either of the cuts above.
STRATIFY_AF_BURDEN = 0.20
AF_HIGH = "af_high"
AF_LOW = "af_low"


def read_header(hea_path: Path) -> dict[str, object]:
    """Read the signal specification out of one header.

    Deliberately not ``wfdb.rdheader``: it is called for the two records whose
    headers declare zero signals, and reading three lines of text cannot fail the
    way a record reader can.
    """
    lines = [
        line for line in hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    out: dict[str, object] = {
        "n_leads": 0,
        "n_samples": 0,
        "sampling_rate": 250,
        "start_time": "",
        "lead_names": "",
    }
    if not lines:
        logger.warning("Empty header: %s", hea_path.name)
        return out

    fields = lines[0].split()
    if len(fields) >= 4:
        out["n_leads"] = int(fields[1])
        out["sampling_rate"] = int(float(fields[2]))
        out["n_samples"] = int(fields[3])
    # Field 5 is the start time of day, absent from six headers: 08378, 08405 and
    # 08455, which notes.txt records as "No start time"; 04936, which it does
    # not; and the two signal-less records, whose headers are four fields long.
    # Empty rather than invented.
    if len(fields) >= 5:
        out["start_time"] = fields[4]

    # Last field of each signal line is the description, i.e. the lead name.
    # Uniformly ECG1/ECG2 here, but read rather than assumed: it is the only
    # statement the files make about what the two channels are.
    out["lead_names"] = "|".join(line.split()[-1] for line in lines[1:])
    return out


def summarise_rhythms(record_path: Path, span_samples: int) -> dict[str, object]:
    """Summarise one record's rhythm episodes into seconds per code.

    ``span_samples`` closes the final episode, which has no annotation after it.

    Returns seconds and episode counts per code, the dominant code by duration,
    the longest single AF episode, and ``af_burden`` — the fraction of annotated
    time in AFIB or AFL, which is the quantity most AF work on this database
    actually uses.
    """
    import wfdb

    out: dict[str, object] = {f"rhythm_secs_{code}": 0.0 for code in RHYTHM_NAMES}
    out.update({f"n_episodes_{code}": 0 for code in RHYTHM_NAMES})
    out.update(
        {
            "n_rhythm_annotations": 0,
            "rhythms": "",
            "dominant_rhythm": "",
            "dominant_rhythm_fraction": np.nan,
            "af_burden": np.nan,
            "longest_af_episode_secs": np.nan,
        }
    )

    try:
        annotation = wfdb.rdann(str(record_path), RHYTHM_ANNOTATOR)
    except Exception as e:  # one unreadable .atr must not kill the whole scan
        logger.warning("Could not read %s.%s: %s", record_path.name, RHYTHM_ANNOTATOR, e)
        return out

    fs = float(getattr(annotation, "fs", 250) or 250)
    out["n_rhythm_annotations"] = len(annotation.symbol)

    # A rhythm annotation is a '+' whose aux_note is "(CODE": it opens an episode
    # that runs to the next '+', or to the end of the record for the last one.
    episodes: list[tuple[str, int]] = []
    for symbol, sample, aux in zip(annotation.symbol, annotation.sample, annotation.aux_note):
        note = str(aux).strip().strip("\x00")
        if symbol != "+" or not note.startswith("("):
            logger.warning(
                "%s: unexpected rhythm annotation %r/%r, skipped",
                record_path.name, symbol, note,
            )
            continue
        episodes.append((note[1:], int(sample)))

    longest_af = 0.0
    for i, (code, start) in enumerate(episodes):
        end = episodes[i + 1][1] if i + 1 < len(episodes) else span_samples
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
    if total > 0:
        ordered = sorted(seconds_by_code, key=lambda c: -seconds_by_code[c])
        out["rhythms"] = "|".join(ordered)
        out["dominant_rhythm"] = ordered[0]
        out["dominant_rhythm_fraction"] = seconds_by_code[ordered[0]] / total
        out["af_burden"] = sum(seconds_by_code.get(c, 0.0) for c in AF_CODES) / total
        out["longest_af_episode_secs"] = longest_af

    return out


def summarise_beats(record_path: Path, span_samples: int) -> dict[str, object]:
    """Count beat annotations, and measure how much of the record they cover.

    ``n_beats`` comes from the **unaudited** ``.qrs``, which is all 25 records
    have. ``n_beats_corrected`` is the manually corrected ``.qrsc`` where it
    exists (05091 and 07859) and NaN otherwise — never silently substituted, so a
    user choosing between them is doing so knowingly.
    """
    import wfdb

    out: dict[str, object] = {
        "n_beats": 0,
        "n_beats_corrected": np.nan,
        "has_corrected_beats": False,
        "last_beat_sample": 0,
        "unannotated_tail_secs": np.nan,
        "mean_heart_rate_bpm": np.nan,
    }

    fs = 250.0
    try:
        beats = wfdb.rdann(str(record_path), BEAT_ANNOTATOR)
    except Exception as e:
        logger.warning("Could not read %s.%s: %s", record_path.name, BEAT_ANNOTATOR, e)
        beats = None

    if beats is not None and len(beats.sample):
        fs = float(getattr(beats, "fs", 250) or 250)
        samples = np.asarray(beats.sample, dtype=np.float64)
        out["n_beats"] = int(len(samples))
        out["last_beat_sample"] = int(samples[-1])
        # The stretch of record past the last annotated beat. ~823 s in every
        # full-length record, because annotation stopped at the nominal 10 h
        # while the signal runs 10 h 13.7 min.
        out["unannotated_tail_secs"] = max(0.0, (span_samples - samples[-1]) / fs)
        rr = np.diff(samples) / fs
        rr = rr[(rr > 0.2) & (rr < 2.5)]  # drop detector glitches, not real beats
        if len(rr):
            out["mean_heart_rate_bpm"] = float(60.0 / rr.mean())

    corrected_file = record_path.with_suffix(f".{CORRECTED_BEAT_ANNOTATOR}")
    if corrected_file.exists():
        try:
            corrected = wfdb.rdann(str(record_path), CORRECTED_BEAT_ANNOTATOR)
            out["n_beats_corrected"] = float(len(corrected.sample))
            out["has_corrected_beats"] = True
        except Exception as e:
            logger.warning(
                "Could not read %s.%s: %s", record_path.name, CORRECTED_BEAT_ANNOTATOR, e
            )

    return out


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file, so the ``old/``
    subdirectory — which holds the pre-2001 revisions of 25 ``.atr`` files under
    the same names — cannot leak in.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. MIT-BIH AFDB labels are the "
            ".atr rhythm annotations themselves, so point data_path at the "
            "dataset root — the flat directory holding 04015.hea, RECORDS and "
            "ANNOTATORS. Get it from https://physionet.org/content/afdb/1.0.0/"
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
        # 00735 and 03665 declare 0 samples because their signals were never
        # released. Their annotations still cover the usual 10 hours, so the
        # rhythm accounting uses the length every other record declares — and
        # has_signals is what tells a reader that is an assumption, not a
        # measurement.
        row["has_signals"] = n_samples > 0 and int(header["n_leads"]) > 0
        span = n_samples if n_samples > 0 else NOMINAL_SAMPLES
        row["rhythm_span_samples"] = span
        row["record_seconds"] = span / float(header["sampling_rate"])

        record_stem = hea.with_suffix("")
        row.update(summarise_rhythms(record_stem, span))
        row.update(summarise_beats(record_stem, span))
        # Flat tree: wfdb takes the bare stem, no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d AFDB records (%d with signals); %.1f h annotated, "
        "%d rhythm episodes, %d beat annotations",
        len(df),
        int(df["has_signals"].sum()),
        df["record_seconds"].sum() / 3600.0,
        int(sum(df[f"n_episodes_{c}"].sum() for c in RHYTHM_NAMES)),
        int(df["n_beats"].sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``af_class`` and ``stratify_class``, and use the latter to stratify.

    This is the **only** derivation of the stratification label — ``AFDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    ``af_class`` is the label a reader wants: ``minimal`` (AF under 5% of the
    record, 8 records), ``paroxysmal`` (5-95%, 14) and ``sustained`` (95% or more,
    3). It describes the database accurately — PhysioNet calls this cohort
    "mostly paroxysmal" and that is what the burden distribution shows.

    ``stratify_class`` is **coarser on purpose**, and it is binary because 25
    records leave no choice. ``StratifiedKFold`` needs at least as many members in
    every class as there are folds, so with ``n_folds=10`` no class may hold fewer
    than 10 records: ``af_class`` would fail on ``sustained`` (3), and
    ``dominant_rhythm`` on ``J`` (1 record, 03665). A single cut at 20% AF burden
    gives 14 / 11 — the only split of this data with margin over that floor, and a
    fixed threshold rather than a median that would move if the release ever
    changed. Records either side of it differ in what a fold means: below, AF is
    an incidental finding; above, it is most of what the record shows.

    Neither column is a beat- or sample-level label. Train on ``dominant_rhythm``,
    ``af_burden``, the ``rhythm_secs_*`` columns, or — for anything sample-level —
    the ``.atr`` episodes directly.
    """
    out = df.copy()
    burden = out["af_burden"].fillna(0.0)

    out["af_class"] = np.select(
        [burden < MINIMAL_AF_BURDEN, burden >= SUSTAINED_AF_BURDEN],
        ["minimal", "sustained"],
        default="paroxysmal",
    )
    out["stratify_class"] = np.where(burden >= STRATIFY_AF_BURDEN, AF_HIGH, AF_LOW)

    logger.info(
        "AF burden classes: %s; fold classes: %s",
        out["af_class"].value_counts().to_dict(),
        out["stratify_class"].value_counts().to_dict(),
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return MIT-BIH AFDB labels indexed by record name.

    Columns:

    - ``dominant_rhythm``, ``rhythms``, ``dominant_rhythm_fraction`` and
      ``rhythm_secs_<CODE>`` — seconds spent in each of the four annotated rhythms
      (see :data:`RHYTHM_NAMES`), from the manually reviewed ``.atr`` episodes.
      ``n_episodes_<CODE>`` counts the episodes rather than their duration; the two
      say different things, and 04043 shows why — 82 AFIB episodes for 21.5% of the
      record, against 07162's single episode for 100% of it.
    - ``af_burden`` — fraction of annotated time in AFIB or AFL. This is the
      headline label: it ranges from 0.002 (05091) to 1.0 (07162, 07859) and every
      one of the 25 records has some AF. ``longest_af_episode_secs`` complements it.
    - ``af_class`` — ``minimal`` / ``paroxysmal`` / ``sustained``, binned from
      ``af_burden``. ``stratify_class`` is the coarser binary fold label; do not
      train on it.
    - ``n_beats`` — beat annotations from the **unaudited** ``.qrs`` detector. Every
      symbol in it is ``N``, so this is a detection count, not a normal-beat count.
      ``n_beats_corrected`` / ``has_corrected_beats`` carry the manually corrected
      ``.qrsc``, which exists for 05091 and 07859 only.
    - ``mean_heart_rate_bpm`` — from the ``.qrs`` RR intervals, glitches outside
      0.2-2.5 s dropped.
    - ``has_signals`` — **False for 00735 and 03665**, whose ECG was never
      released. Their labels are real; their waveforms do not exist, and the
      ``clean`` version of the split excludes them.
    - ``n_samples``, ``record_seconds``, ``rhythm_span_samples`` — record length.
      ``n_samples`` is what the header declares (0 for the two above);
      ``rhythm_span_samples`` is the span used for rhythm accounting.
    - ``unannotated_tail_secs`` — record past the last ``.qrs`` beat: 823 s in 21
      of the 23 records with signals, because annotation stopped at the nominal
      10 h while the signal runs to 10 h 13.7 min. 1,692 s for 04048 and 100 s for
      the short record 06453.
    - ``lead_names``, ``n_leads``, ``sampling_rate``, ``start_time`` — from the
      header. ``start_time`` is empty for the six records that declare none.

    There are **no demographics**: this release's headers carry no age, sex,
    medication or clinical description, and no subject identifier beyond the
    record name. That is why the config sets ``patient_id_column: null`` — one
    record per subject is all that can be asserted.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
