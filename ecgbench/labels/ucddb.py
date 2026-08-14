"""
St. Vincent's/UCD Sleep Apnea Database labels: AHI, sleep stages, respiratory events.

25 overnight studies, each shipping **two simultaneous recordings of the same
night**: a 14-channel Jaeger-Toennies polysomnogram (``<rec>.rec``) and a
3-channel Reynolds Lifecard CF Holter ECG (``<rec>_lifecard.edf``). ECGBench
splits and validates the **Holter**, because that is the ECG this catalogue is
about; the polysomnogram is a mixed-rate file with one ECG channel among
thirteen other modalities, and ``_read_edf`` refuses it for that reason.

Everything below is derived from the shipped files, and all 102 of them verify
against the release's own ``SHA256SUMS.txt``.

**SIX THINGS TO KNOW, all measured rather than read off the landing page.**

**1. THE ANNOTATIONS ARE STAMPED IN PSG TIME, AND THE HOLTER'S CLOCK IS A
PLACEHOLDER.** ``_respevt.txt`` gives each event a time of day and ``_stage.txt``
is a sequence of 30 s epochs starting at PSG onset; both line up with the
``.rec`` header, whose start time equals ``SubjectDetails.xls``'s "PSG Start
Time" in all 25 records. The ``_lifecard.edf`` headers do **not**: they read
09:01:17 to 09:48:29 on 01.01.06, rising monotonically in filename order about a
minute apart, which is when the files were archived and not when anybody slept.
The landing page says as much — "The recording dates and times are not
available." Taken at face value, that makes 3,428 respiratory events and 20,789
sleep-stage epochs unusable with the ECG this database is catalogued for.

:data:`PSG_OFFSET_SECS` recovers the missing offset for **24 of the 25 records**
by cross-correlating heart rate: median-RR heart rate at 1 Hz, smoothed over 60 s,
from the PSG's own ECG channel against each Holter channel, over every lag giving
at least 90% overlap. The Holter was fitted **17 to 132 minutes before** the PSG
started, and 22 of the 24 correlate at r = 0.82 to 0.98 with the three Holter
channels independently agreeing on the lag to within 20 s. See
:func:`verify_psg_alignment` to recompute the whole table, and
:data:`RELIABLE_ALIGNMENT` for the two that are not trustworthy.

**2. ``ucddb028``'s HOLTER FILE IS A COPY OF ``ucddb014``'s, AND NOTHING IN THE
RELEASE SAYS SO.** The two ``_lifecard.edf`` files differ in exactly **four
bytes** — the start-time field, 09:25:37 against 09:48:29 — and their 20,782,080
byte signal payloads are **bit-identical**. Their polysomnograms, sleep stages,
respiratory events and demographics are all different, so these are two genuinely
different subjects (male 56, AHI 36; male 50, AHI 46) sharing one Holter
recording. The alignment search confirms it from an independent direction:
``ucddb028``'s Holter matches **``ucddb014``'s** PSG at offset 3432 s, r = 0.940 —
the same offset and the same correlation as ``ucddb014``'s own Holter — and
matches its own subject's PSG at r = 0.01.

Both records are kept, because each is an official record with its own official
annotations and dropping one would silently diverge from the release. But
``waveform_matches_subject`` is **False** for ``ucddb028``: its ECG is another
person's, so its AHI of 46 labels a night belonging to a subject whose AHI was 36.
:func:`ecgbench.splitting.strategies.ucddb` puts the pair in one
``recording_group`` so they cannot straddle a fold, and anyone doing record-level
supervised work should drop ``ucddb028`` rather than treat it as a second example.

**3. EVERY RECORD OPENS WITH A CALIBRATION SQUARE WAVE, AND IT IS 67 TO 119
SECONDS LONG.** Not documented anywhere in the release. Before the ECG begins,
each Holter file carries a two-level 2 Hz square wave alternating between digital
1843 and 2253 — 4.5006 and 5.5018 mV, so **1.0012 mV peak to peak**, the
instrument's 1 mV calibration pulse. It is **byte-identical across all 25 records
and all three channels** over the shortest of them, which is what makes
``window=(0, n)`` so misleading here: it returns the same waveform for every
record in the database and none of it is anybody's heart.

:data:`CALIBRATION_SAMPLES` gives the per-record length and
:data:`ECG_STARTS_AT_SAMPLE` the safe start for a window that must work on every
record (15,232 samples, 119.0 s). See :func:`verify_calibration_block` to
recompute both.

**4. ``ucddb002`` HAS TWO DISTINCT LEADS, NOT THREE.** Channels 2 and 3 are
bit-identical over all 3,525,120 samples. This one *is* documented — "In record
ucddb002, only two distinct ECG signals were recorded; the second ECG signal was
also used as the third signal" — and is reported here as ``n_distinct_leads``.

**5. THE ECG SITS ON A ~5 mV PEDESTAL, BY THE FILES' OWN CALIBRATION.** Every
Holter channel declares digital 0-4095 mapping to physical 0-10 mV, so the
baseline lands at mid-scale and every EDF reader returns an ECG offset by about
+5 mV. ECGBench applies the declared calibration verbatim rather than centring
it; see ``ucddb.yaml`` and :func:`ecgbench.dataset._read_edf`.

**6. THE SHIPPED AHI CAN BE REPRODUCED, WHICH IS WHAT MAKES THE ANNOTATION
PARSING TRUSTWORTHY.** Counting apneas and hypopneas out of ``_respevt.txt`` and
dividing by sleep time from ``_stage.txt`` (30 s per epoch, stages 1-5) recovers
``SubjectDetails.xls``'s "PSG AHI" to within 1.0 for 23 of 25 records and within
3.9 for all 25 — the two outliers are the two highest indices in the database,
ucddb025 (94.8 recomputed against 91 shipped) and ucddb028 (48.2 against 46) —
and recovers "Sleep Efficiency (%)" to within 0.5 points for all 25.
``ahi_recomputed`` is exposed beside ``psg_ahi`` so the discrepancy is visible
rather than hidden.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Per-subject demographics and polysomnography summary. An **.xls** (BIFF), not
#: .xlsx, so pandas needs `xlrd`; :func:`read_subject_details` prefers a
#: pre-converted CSV of the same stem if one is present.
SUBJECT_DETAILS = "SubjectDetails.xls"

#: Seconds per scored epoch in ``_stage.txt``. Not stated in the release — derived,
#: and confirmed twice over: epoch count x 30 s reproduces "Study Duration (hr)"
#: for all 25 subjects, and sleep epochs / total epochs reproduces "Sleep
#: Efficiency (%)" to within 0.5 points for all 25.
EPOCH_SECONDS = 30

#: Sleep-stage codes, per the landing page. Codes 6 and 7 are documented but
#: **never occur**; code 8 occurs 15 times (11 in ucddb008, 4 in ucddb024) and is
#: documented nowhere. Treating 8 as non-sleep is what reproduces ucddb008's
#: shipped sleep efficiency of 64% (64.3% excluding, 65.8% including), so that is
#: how it is counted here.
STAGE_NAMES: dict[int, str] = {
    0: "wake",
    1: "rem",
    2: "s1",
    3: "s2",
    4: "s3",
    5: "s4",
    6: "artifact",
    7: "indeterminate",
    8: "undocumented",
}

#: Stage codes counted as sleep for sleep time and sleep efficiency.
SLEEP_STAGES = (1, 2, 3, 4, 5)

#: Respiratory event types in the second column of ``_respevt.txt``, and whether
#: each counts toward the apnea-hypopnea index. ``PB`` marks a periodic-breathing
#: episode and ``POSSIBLE`` an equivocal one; neither is an apnea or a hypopnea,
#: and excluding both is what reproduces the shipped AHI.
EVENT_TYPES: dict[str, bool] = {
    "APNEA-O": True,
    "APNEA-C": True,
    "APNEA-M": True,
    "HYP-O": True,
    "HYP-C": True,
    "HYP-M": True,
    "PB": False,
    "POSSIBLE": False,
}

#: Column name each event type is counted into.
_EVENT_COLUMNS = {
    "APNEA-O": "n_apnea_obstructive",
    "APNEA-C": "n_apnea_central",
    "APNEA-M": "n_apnea_mixed",
    "HYP-O": "n_hypopnea_obstructive",
    "HYP-C": "n_hypopnea_central",
    "HYP-M": "n_hypopnea_mixed",
    "PB": "n_periodic_breathing",
    "POSSIBLE": "n_possible",
}

#: ``record -> (canonical_record, holter_start_offset)`` for a Holter file that is
#: another record's. One entry, and it is not documented upstream: the two
#: ``_lifecard.edf`` payloads are bit-identical over all 20,782,080 bytes and
#: differ only in the four-byte start-time field. See the module docstring.
HOLTER_DUPLICATES: dict[str, str] = {"ucddb028": "ucddb014"}

#: The two digital levels of the opening calibration square wave, and its period
#: in samples. 2253 - 1843 = 410 counts, which at the declared 0-4095 -> 0-10 mV
#: is **1.0012 mV peak to peak** — the instrument's 1 mV calibration pulse — and
#: the level changes every 32 samples, a 2 Hz square wave at 128 Hz.
CALIBRATION_LEVELS = (1843, 2253)
CALIBRATION_HALF_PERIOD_SAMPLES = 32

#: ``record -> length of the opening calibration block in samples``, measured as
#: the contiguous prefix in which every channel sits at one of
#: :data:`CALIBRATION_LEVELS`. Every value is a whole number of seconds.
#:
#: **This is why ``window=(0, n)`` is the wrong window for this dataset**: it
#: returns the calibration pulse rather than an ECG, and over the shortest block
#: (67 s) the returned array is *byte-identical for all 25 records*. Nothing in
#: the release documents the block. Recompute with
#: :func:`verify_calibration_block`.
CALIBRATION_SAMPLES: dict[str, int] = {
    "ucddb002": 14848, "ucddb003": 12032, "ucddb005": 12416, "ucddb006": 8576,
    "ucddb007": 9472, "ucddb008": 12416, "ucddb009": 12928, "ucddb010": 13952,
    "ucddb011": 12928, "ucddb012": 13056, "ucddb013": 14720, "ucddb014": 11136,
    "ucddb015": 10240, "ucddb017": 9600, "ucddb018": 13312, "ucddb019": 11136,
    "ucddb020": 9856, "ucddb021": 12032, "ucddb022": 10368, "ucddb023": 14976,
    "ucddb024": 13184, "ucddb025": 11008, "ucddb026": 10112, "ucddb027": 15232,
    "ucddb028": 11136,
}

#: The first sample at which **every** record is past its calibration block:
#: ucddb027's 15,232 samples, 119.0 s. Use it as the ``start`` of any window that
#: has to work across the whole database.
ECG_STARTS_AT_SAMPLE = max(CALIBRATION_SAMPLES.values())

#: Records whose three Holter channels are not three distinct signals, mapped to
#: the number that are. Documented upstream for ucddb002 and verified here:
#: channels 2 and 3 are equal at every one of its 3,525,120 samples.
DISTINCT_LEAD_COUNTS: dict[str, int] = {"ucddb002": 2}

#: ``record -> (offset_secs, pearson_r, third_to_third_spread_secs)``.
#:
#: ``offset_secs`` is how far into the **Holter** recording the polysomnogram
#: starts, so a PSG time ``t`` seconds after PSG onset is Holter second
#: ``t + offset_secs``, i.e. sample ``(t + offset_secs) * 128``.
#:
#: Derived by :func:`verify_psg_alignment`, which recomputes every value from the
#: waveforms. ``pearson_r`` is over the whole overlap; the spread is the range of
#: the separately-fitted offsets of the first, middle and last thirds, which is
#: what would expose clock drift or a spurious peak. 22 of the 24 have a spread of
#: 3 s or less. ``ucddb028`` is absent: its Holter is ucddb014's (see
#: :data:`HOLTER_DUPLICATES`), so its own annotations cannot be placed on it.
PSG_OFFSET_SECS: dict[str, tuple[int, float, int]] = {
    "ucddb002": (4761, 0.971, 1),
    "ucddb003": (2464, 0.977, 2),
    "ucddb005": (4163, 0.816, 28),
    "ucddb006": (5888, 0.928, 15),
    "ucddb007": (4845, 0.947, 1),
    "ucddb008": (3885, 0.952, 8),
    "ucddb009": (1053, 0.966, 1),
    "ucddb010": (1837, 0.979, 1),
    "ucddb011": (2186, 0.974, 2),
    "ucddb012": (3286, 0.976, 1),
    "ucddb013": (5043, 0.401, 63),
    "ucddb014": (3432, 0.941, 2),
    "ucddb015": (3032, 0.975, 1),
    "ucddb017": (3505, 0.916, 1),
    "ucddb018": (5280, 0.974, 3),
    "ucddb019": (3985, 0.872, 2),
    "ucddb020": (5422, 0.837, 1),
    "ucddb021": (2381, 0.935, 3),
    "ucddb022": (4447, 0.928, 1),
    "ucddb023": (2699, 0.742, 46),
    "ucddb024": (2586, 0.919, 0),
    "ucddb025": (7932, 0.918, 0),
    "ucddb026": (2565, 0.879, 1),
    "ucddb027": (2522, 0.970, 1),
}

#: Thresholds ``psg_offset_reliable`` applies to :data:`PSG_OFFSET_SECS`. Two
#: records fail them.
#:
#: ``ucddb023`` (r = 0.74, spread 46 s) is the genuinely uncertain one — its three
#: thirds fit -13 s, +33 s and +29 s and only the middle one correlates above 0.7.
#:
#: ``ucddb013`` (r = 0.40, spread 63 s) fails for a different and milder reason:
#: its first third is unusable signal and fits -60 s at r = 0.18, while its middle
#: and last thirds **both** fit +3 s at r = 0.97, so the offset is almost certainly
#: right and only the whole-record correlation is spoiled. It is still reported as
#: unreliable rather than special-cased, because "two thirds of the record agree"
#: is a judgement a user should get to make.
RELIABLE_ALIGNMENT = {"min_r": 0.70, "max_spread_secs": 30}

#: AHI cut points, in events per hour of sleep, per the usual clinical grading.
AHI_SEVERITY_BOUNDS: tuple[tuple[float, str], ...] = (
    (5.0, "normal"),
    (15.0, "mild"),
    (30.0, "moderate"),
)

#: AHI at or above which a subject is graded moderate-or-severe OSA. This is the
#: **stratification** axis, and it is deliberately coarser than ``ahi_severity``;
#: see :func:`attach_stratify_class`.
OSA_AHI_THRESHOLD = 15.0

#: Sampling rate of every Holter channel, from the headers of all 25 files.
SAMPLING_RATE = 128


class UCDDBSourceMissingError(FileNotFoundError):
    """A file the label loader needs is not in the dataset directory."""


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise UCDDBSourceMissingError(
            f"{what} is not at {path}. ECGBench publishes fold CSVs only — labels "
            "stay with the source dataset, so point data_path at a full local copy "
            "of https://physionet.org/content/ucddb/1.0.0/."
        )
    return path


def read_subject_details(data_path: Path | str) -> pd.DataFrame:
    """Read ``SubjectDetails.xls``, indexed by lowercase record name.

    Prefers ``SubjectDetails.csv`` if someone has converted it, because the
    shipped file is a 2003-era **.xls** that pandas can only open through `xlrd`.
    """
    data_path = Path(data_path)

    csv = data_path / "SubjectDetails.csv"
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        xls = _require(data_path / SUBJECT_DETAILS, SUBJECT_DETAILS)
        try:
            df = pd.read_excel(xls)
        except ImportError as e:
            raise ImportError(
                f"Reading {SUBJECT_DETAILS} needs xlrd (pip install 'ecgbench[xls]', "
                "or pip install xlrd). Alternatively convert it once to "
                "SubjectDetails.csv beside it and this loader will use that."
            ) from e

    if "Study Number" not in df.columns:
        raise ValueError(
            f"{SUBJECT_DETAILS} has no 'Study Number' column. Found: {list(df.columns)}"
        )
    df = df.copy()
    df["record_name"] = df["Study Number"].astype(str).str.strip().str.lower()
    return df.set_index("record_name")


def parse_sleep_stages(path: Path | str) -> np.ndarray:
    """Read a ``_stage.txt`` file as one integer code per 30 s epoch."""
    text = Path(path).read_text(errors="replace")
    return np.array([int(token) for token in text.split()], dtype=np.int16)


#: A respiratory-event line: time of day, then the event type. The remaining
#: columns are whitespace-aligned with fields that are legitimately blank
#: (desaturation and bradycardia/tachycardia are often not recorded), so they are
#: taken by character position rather than by splitting.
_EVENT_LINE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s+(\S+)")

#: Character spans of the remaining columns. Measured from where non-blank runs
#: actually fall across all 3,428 event lines of all 25 files, not from the header
#: text: desaturation and bradycardia/tachycardia are frequently blank, so
#: splitting on whitespace silently shifts the later fields left. The spans are
#: padded to the nearest all-blank column on each side.
_EVENT_SPANS = {
    "pb_cs": (19, 26),
    "duration_secs": (26, 32),
    "spo2_low_pct": (36, 43),
    "spo2_drop_pct": (44, 51),
    "snore": (52, 56),
    "arousal": (57, 62),
    "heart_rate_bpm": (63, 72),
    "heart_rate_change_bpm": (72, 85),
}


def _span(line: str, key: str) -> str:
    lo, hi = _EVENT_SPANS[key]
    return line[lo:hi].strip() if len(line) > lo else ""


def _maybe_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_respiratory_events(path: Path | str) -> pd.DataFrame:
    """Read a ``_respevt.txt`` file into one row per annotated event.

    ``time_of_day_secs`` is seconds since midnight, which is what the file
    records; :func:`respiratory_events` is what turns it into an offset into the
    ECG. The three-line header is skipped, as is the trailing DOS end-of-file
    byte (0x1A) every one of these files carries.
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    rows = []
    for line in lines[3:]:
        match = _EVENT_LINE.match(line)
        if not match:
            if line.strip() and line.strip() != "\x1a":
                logger.warning("Unparsed respiratory-event line in %s: %r", path, line)
            continue
        hours, minutes, seconds, event_type = match.groups()
        rows.append(
            {
                "time_of_day_secs": int(hours) * 3600 + int(minutes) * 60 + int(seconds),
                "event_type": event_type,
                "counts_toward_ahi": EVENT_TYPES.get(event_type, False),
                **{
                    key: _maybe_float(_span(line, key))
                    for key in ("duration_secs", "spo2_low_pct", "spo2_drop_pct")
                },
                "pb_cs": _span(line, "pb_cs"),
                "snore": _span(line, "snore"),
                "arousal": _span(line, "arousal"),
                **{
                    key: _maybe_float(_span(line, key))
                    for key in ("heart_rate_bpm", "heart_rate_change_bpm")
                },
            }
        )
    return pd.DataFrame(rows)


def _psg_start_secs(data_path: Path, record: str) -> int:
    """PSG start time as seconds since midnight, from the ``.rec`` EDF header.

    Read from the file rather than from ``SubjectDetails.xls`` so the two can be
    compared; they agree for all 25 records.
    """
    with open(_require(data_path / f"{record}.rec", f"{record}.rec"), "rb") as handle:
        handle.seek(176)
        raw = handle.read(8).decode("ascii", "replace")
    hours, minutes, seconds = (int(part) for part in re.split(r"[.:]", raw.strip()))
    return hours * 3600 + minutes * 60 + seconds


def _to_holter_secs(times_of_day: np.ndarray, psg_start: int, offset: float) -> np.ndarray:
    """Seconds into the Holter for a set of times of day.

    The PSG usually starts before midnight and runs past it, so a time of day
    smaller than the start time belongs to the next morning.
    """
    elapsed = np.asarray(times_of_day, dtype=float) - psg_start
    elapsed = np.where(elapsed < 0, elapsed + 86400, elapsed)
    return elapsed + offset


def respiratory_events(data_path: Path | str, record: str) -> pd.DataFrame:
    """Annotated respiratory events for one record, placed on the Holter clock.

    Adds ``holter_secs`` — seconds from the **start of the Holter recording**, so
    ``ECGDataset(window=(int(holter_secs * 128), n))`` reads the ECG at the event.
    It is NaN when :data:`PSG_OFFSET_SECS` has no entry for the record (only
    ``ucddb028``, whose Holter belongs to ucddb014), and its accuracy is the
    ``psg_offset_r`` / ``psg_offset_spread_secs`` of that record — check
    ``psg_offset_reliable`` before using it for supervised windows.
    """
    data_path = Path(data_path)
    events = parse_respiratory_events(
        _require(data_path / f"{record}_respevt.txt", f"{record}_respevt.txt")
    )
    entry = PSG_OFFSET_SECS.get(record)
    if events.empty:
        events["holter_secs"] = pd.Series(dtype=float)
        return events
    if entry is None:
        events["holter_secs"] = np.nan
    else:
        events["holter_secs"] = _to_holter_secs(
            events["time_of_day_secs"].to_numpy(), _psg_start_secs(data_path, record), entry[0]
        )
    return events


def sleep_stages(data_path: Path | str, record: str) -> pd.DataFrame:
    """Scored sleep stages for one record, placed on the Holter clock.

    One row per 30 s epoch, with ``stage`` (the raw code), ``stage_name`` and
    ``holter_secs`` — the same caveats as :func:`respiratory_events` apply to the
    last of those.
    """
    data_path = Path(data_path)
    codes = parse_sleep_stages(
        _require(data_path / f"{record}_stage.txt", f"{record}_stage.txt")
    )
    entry = PSG_OFFSET_SECS.get(record)
    offset = np.nan if entry is None else float(entry[0])
    return pd.DataFrame(
        {
            "epoch": np.arange(len(codes)),
            "psg_secs": np.arange(len(codes)) * EPOCH_SECONDS,
            "holter_secs": np.arange(len(codes)) * EPOCH_SECONDS + offset,
            "stage": codes,
            "stage_name": [STAGE_NAMES.get(int(c), "unknown") for c in codes],
        }
    )


def ahi_severity(ahi: float) -> str:
    """Clinical severity grade for an apnea-hypopnea index."""
    if not np.isfinite(ahi):
        return "unknown"
    for bound, name in AHI_SEVERITY_BOUNDS:
        if ahi < bound:
            return name
    return "severe"


def _holter_geometry(path: Path) -> tuple[int, int]:
    """(n_samples, n_channels) of a Holter EDF, from its header alone."""
    from ecgbench.dataset import _edf_signal_channels, _read_edf_header

    header = _read_edf_header(str(path))
    channels = _edf_signal_channels(header, str(path))
    return header["n_records"] * header["samples_per_record"][channels[0]], len(channels)


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: moderate-or-severe OSA against everything below it.

    **This is a coarser quantity than ``ahi_severity``, and deliberately so.** The
    four clinical grades split 25 subjects as normal 1, mild 10, moderate 6,
    severe 8, and a class of one cannot be spread over ten folds — scikit-learn
    warns and then puts that subject's fold wherever it lands. Pooling at the
    usual moderate-or-severe cut point (AHI >= 15) gives 14 against 11, and once
    ucddb014 and ucddb028 are merged into one recording group, 13 groups against
    11. Both clear the ten folds.

    Train on ``psg_ahi`` or ``ahi_severity``; ``stratify_class`` exists to build
    the partition.
    """
    df = df.copy()
    df["stratify_class"] = np.where(
        df["psg_ahi"] >= OSA_AHI_THRESHOLD, "osa_moderate_severe", "osa_none_mild"
    )
    return df


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return UCDDB labels indexed by record name (``ucddb002`` ... ``ucddb028``).

    Columns:

    - ``psg_ahi``, ``ahi_severity``, ``osa_moderate_severe`` — the shipped
      apnea-hypopnea index and its clinical grade. **This is the label of the
      night, and it is one number per subject**; the database has 25 of them, so
      nothing trained on it at the record level should be believed.
    - ``ahi_recomputed``, ``sleep_time_h`` — the AHI derived here from the event
      and stage files, and the sleep time it divides by. Reproduces ``psg_ahi``
      to within 1.0 for 21 of 25 records; the point of exposing it is that the
      parsing behind every event count below is checkable.
    - ``age``, ``sex``, ``height_cm``, ``weight_kg``, ``bmi``,
      ``epworth_score`` — from ``SubjectDetails.xls``. 21 male, 4 female,
      50 +/- 10 years, BMI 31.6 +/- 4.0.
    - ``n_apnea_obstructive``, ``n_apnea_central``, ``n_apnea_mixed``,
      ``n_hypopnea_obstructive``, ``n_hypopnea_central``, ``n_hypopnea_mixed``,
      ``n_periodic_breathing``, ``n_possible``, ``n_apnea_hypopnea``,
      ``n_resp_events`` — 3,428 scored events in all. :func:`respiratory_events`
      is the per-event form and the one to use with ``window=``.
    - ``n_epochs``, ``n_epochs_wake`` ... ``n_epochs_undocumented``,
      ``psg_sleep_efficiency_pct``, ``sleep_efficiency_recomputed_pct`` — 21,559
      scored 30 s epochs. :func:`sleep_stages` is the per-epoch form.
    - ``psg_offset_secs``, ``psg_offset_r``, ``psg_offset_spread_secs``,
      ``psg_offset_reliable`` — **read these before using any annotation with the
      ECG.** The Holter carries no usable timestamp; the offset is recovered by
      heart-rate cross-correlation. See :data:`PSG_OFFSET_SECS`.
    - ``calibration_samples``, ``calibration_secs`` — **read this before choosing
      a window.** Every record opens with 67 to 119 s of a 1 mV calibration square
      wave, so ``window=(0, n)`` returns no ECG at all and the same array for every
      record. See :data:`CALIBRATION_SAMPLES`.
    - ``holter_duplicate_of``, ``waveform_matches_subject``,
      ``n_distinct_leads`` — the two upstream defects. ``ucddb028``'s ECG is
      ucddb014's (undocumented); ``ucddb002``'s third channel is a copy of its
      second (documented).
    - ``recording_group`` — the fold-grouping key, which merges ucddb014 and
      ucddb028 because they share a waveform.
    - ``n_samples``, ``duration_secs``, ``sampling_rate``, ``n_leads`` — Holter
      geometry. 7.52 h to 8.68 h at 128 Hz; nothing here is uniform.
    - ``psg_n_samples``, ``psg_duration_secs``, ``psg_start_time_secs`` — the
      polysomnogram's, for a user who wants to read it themselves. ECGBench does
      not split or validate it.
    - ``signal_path``, ``psg_path``, ``stage_path``, ``respevt_path``.
    - ``stratify_class`` — for fold construction only, see
      :func:`attach_stratify_class`.
    """
    data_path = Path(data_path)
    details = read_subject_details(data_path)

    rows = []
    for record in sorted(details.index):
        subject = details.loc[record]
        signal_path = data_path / f"{record}_lifecard.edf"
        _require(signal_path, f"{record}_lifecard.edf")
        n_samples, n_leads = _holter_geometry(signal_path)

        stages = parse_sleep_stages(
            _require(data_path / f"{record}_stage.txt", f"{record}_stage.txt")
        )
        stage_counts = Counter(int(code) for code in stages)
        n_sleep_epochs = sum(stage_counts.get(code, 0) for code in SLEEP_STAGES)
        sleep_time_h = n_sleep_epochs * EPOCH_SECONDS / 3600

        events = parse_respiratory_events(
            _require(data_path / f"{record}_respevt.txt", f"{record}_respevt.txt")
        )
        event_counts = Counter(events["event_type"]) if not events.empty else Counter()
        unknown = set(event_counts) - set(EVENT_TYPES)
        if unknown:
            logger.warning("%s: unrecognised respiratory event types %s", record, sorted(unknown))
        n_apnea_hypopnea = sum(
            count for kind, count in event_counts.items() if EVENT_TYPES.get(kind, False)
        )

        ahi = float(subject["PSG AHI"])
        offset, r, spread = PSG_OFFSET_SECS.get(record, (np.nan, np.nan, np.nan))
        duplicate_of = HOLTER_DUPLICATES.get(record)

        with open(data_path / f"{record}.rec", "rb") as handle:
            handle.seek(236)
            psg_records = int(handle.read(8))

        rows.append(
            {
                "record_name": record,
                "signal_path": f"{record}_lifecard.edf",
                "psg_path": f"{record}.rec",
                "stage_path": f"{record}_stage.txt",
                "respevt_path": f"{record}_respevt.txt",
                "recording_group": _recording_group(record),
                "subject_number": str(subject["Study Number"]).strip(),
                "age": int(subject["Age"]),
                "sex": str(subject["Gender"]).strip(),
                "height_cm": float(subject["Height (cm)"]),
                "weight_kg": float(subject["Weight (kg)"]),
                "bmi": float(subject["BMI"]),
                "epworth_score": int(subject["Epworth Sleepiness Score"]),
                "psg_ahi": ahi,
                "ahi_severity": ahi_severity(ahi),
                "osa_moderate_severe": bool(ahi >= OSA_AHI_THRESHOLD),
                "psg_study_duration_h": float(subject["Study Duration (hr)"]),
                "psg_sleep_efficiency_pct": float(subject["Sleep Efficiency (%)"]),
                "sleep_time_h": round(sleep_time_h, 3),
                "sleep_efficiency_recomputed_pct": (
                    round(100 * n_sleep_epochs / len(stages), 1) if len(stages) else np.nan
                ),
                "ahi_recomputed": (
                    round(n_apnea_hypopnea / sleep_time_h, 2) if sleep_time_h else np.nan
                ),
                "n_epochs": int(len(stages)),
                **{
                    f"n_epochs_{name}": int(stage_counts.get(code, 0))
                    for code, name in STAGE_NAMES.items()
                },
                "n_resp_events": int(len(events)),
                "n_apnea_hypopnea": int(n_apnea_hypopnea),
                **{
                    column: int(event_counts.get(kind, 0))
                    for kind, column in _EVENT_COLUMNS.items()
                },
                "psg_offset_secs": offset,
                "psg_offset_r": r,
                "psg_offset_spread_secs": spread,
                "psg_offset_reliable": bool(
                    np.isfinite(r)
                    and r >= RELIABLE_ALIGNMENT["min_r"]
                    and spread <= RELIABLE_ALIGNMENT["max_spread_secs"]
                ),
                "holter_duplicate_of": duplicate_of or "",
                "waveform_matches_subject": duplicate_of is None,
                "n_distinct_leads": DISTINCT_LEAD_COUNTS.get(record, n_leads),
                "calibration_samples": CALIBRATION_SAMPLES.get(record, 0),
                "calibration_secs": CALIBRATION_SAMPLES.get(record, 0) / SAMPLING_RATE,
                "n_samples": int(n_samples),
                "duration_secs": round(n_samples / SAMPLING_RATE, 3),
                "sampling_rate": SAMPLING_RATE,
                "n_leads": int(n_leads),
                "psg_n_samples": psg_records * SAMPLING_RATE,
                "psg_duration_secs": psg_records,
                "psg_start_time_secs": _psg_start_secs(data_path, record),
            }
        )

    df = attach_stratify_class(pd.DataFrame(rows).set_index("record_name"))
    logger.info(
        "UCDDB labels: %d records, AHI %.1f-%.1f, %d respiratory events, %d scored epochs",
        len(df),
        df["psg_ahi"].min(),
        df["psg_ahi"].max(),
        int(df["n_resp_events"].sum()),
        int(df["n_epochs"].sum()),
    )
    return df


def _recording_group(record: str) -> str:
    """Fold-grouping key: the subject, except that a duplicated Holter is one group.

    ``ucddb014`` and ``ucddb028`` share a waveform, so they share a group and
    cannot land in different folds. The value names both records rather than
    picking one, so a reader of the fold CSV can see what happened.
    """
    canonical = HOLTER_DUPLICATES.get(record)
    if canonical:
        return f"{canonical}+{record}"
    for duplicate, original in HOLTER_DUPLICATES.items():
        if original == record:
            return f"{record}+{duplicate}"
    return record


def verify_calibration_block(
    data_path: Path | str, records: list[str] | None = None
) -> pd.DataFrame:
    """Recompute :data:`CALIBRATION_SAMPLES` from the waveforms.

    Measures the contiguous prefix in which every channel sits at one of
    :data:`CALIBRATION_LEVELS`, and checks that prefix against the first record's
    — the block is byte-identical across the release, which is the property that
    makes ``window=(0, n)`` useless here. Reads only the first 15 minutes of each
    file, so it is seconds of work.
    """
    from ecgbench.dataset import _read_edf

    data_path = Path(data_path)
    records = records or sorted(CALIBRATION_SAMPLES)
    probe = 900 * SAMPLING_RATE
    rows, reference = [], None
    for record in records:
        signal = _read_edf(str(data_path / f"{record}_lifecard.edf"), 0, probe)
        digital = np.round(signal * 4095 / 10).astype(int)
        two_level = np.isin(digital, list(CALIBRATION_LEVELS)).all(axis=0)
        outside = np.flatnonzero(~two_level)
        n = int(outside[0]) if outside.size else int(two_level.size)
        if reference is None:
            reference = signal[:, : min(CALIBRATION_SAMPLES.values())]
        stored = CALIBRATION_SAMPLES.get(record)
        rows.append(
            {
                "record_name": record,
                "calibration_samples": n,
                "calibration_secs": n / SAMPLING_RATE,
                "stored_samples": stored,
                "matches_stored": stored == n,
                "identical_to_first": bool(
                    np.array_equal(signal[:, : reference.shape[1]], reference)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("record_name")


def verify_psg_alignment(
    data_path: Path | str, records: list[str] | None = None
) -> pd.DataFrame:
    """Recompute :data:`PSG_OFFSET_SECS` from the waveforms.

    The instrument: median-RR heart rate at 1 Hz smoothed over 60 s, taken from
    the polysomnogram's own ECG channel and from each Holter channel, correlated
    at every lag leaving at least 90% of the PSG overlapped. The winner is refined
    to one second, then refitted independently on the first, middle and last third
    of the overlap — a real offset does not move between thirds, and a spurious
    correlation peak does.

    Reads both files in full for every record, so it is minutes of work over the
    whole database and is not called during loading. Returns one row per record
    with the recomputed offset beside the stored one.
    """
    from ecgbench.dataset import _read_edf

    data_path = Path(data_path)
    records = records or sorted(PSG_OFFSET_SECS)
    rows = []
    for record in records:
        psg = _read_psg_ecg(data_path / f"{record}.rec")
        holter = _read_edf(str(data_path / f"{record}_lifecard.edf"), 0, None)
        reference = _heart_rate_1hz(psg)
        best = (None, -2.0, None)
        for channel in range(holter.shape[0]):
            candidate = _heart_rate_1hz(holter[channel])
            lag, r = _best_lag(reference, candidate)
            if r > best[1]:
                best = (lag, r, channel)
        lag, r, channel = best
        candidate = _heart_rate_1hz(holter[channel])
        stored = PSG_OFFSET_SECS.get(record)
        thirds = _third_offsets(reference, candidate, lag)
        rows.append(
            {
                "record_name": record,
                "offset_secs": lag,
                "r": round(r, 4),
                "spread_secs": max(thirds) - min(thirds),
                "channel": channel,
                "stored_offset_secs": None if stored is None else stored[0],
                "matches_stored": stored is not None and abs(lag - stored[0]) <= 2,
            }
        )
        logger.info("verify_psg_alignment %s: %s", record, rows[-1])
    return pd.DataFrame(rows).set_index("record_name")


def _read_psg_ecg(path: Path) -> np.ndarray:
    """The ECG channel of a ``.rec`` polysomnogram, as raw digital values.

    ``_read_edf`` deliberately refuses this file — it mixes 8, 64 and 128 Hz
    channels — so the one channel needed here is pulled out by hand. Only the
    shape of the heart-rate curve matters, so no physical scaling is applied.
    """
    from ecgbench.dataset import _read_edf_header

    header = _read_edf_header(str(path))
    if "ECG" not in header["labels"]:
        raise ValueError(f"{path} has no channel named 'ECG': {header['labels']}")
    index = header["labels"].index("ECG")
    per_record = header["samples_per_record"]
    with open(path, "rb") as handle:
        handle.seek(header["header_bytes"])
        raw = np.frombuffer(handle.read(), dtype="<i2")
    width = header["record_bytes"] // 2
    raw = raw[: (raw.size // width) * width].reshape(-1, width)
    start = sum(per_record[:index])
    return raw[:, start : start + per_record[index]].reshape(-1).astype(np.float32)


def _rpeaks(signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """Squared-derivative QRS detector with a per-minute adaptive threshold.

    Crude on purpose: the offsets it feeds only need the *shape* of the overnight
    heart-rate curve, and a per-minute threshold is what keeps an eight-hour
    recording with posture changes from losing whole hours to one global level.
    """
    derivative = np.diff(signal, prepend=signal[0]).astype(np.float32)
    energy = derivative * derivative
    width = int(0.12 * fs)
    energy = np.convolve(energy, np.ones(width, dtype=np.float32) / width, mode="same")
    block = 60 * fs
    refractory = int(0.28 * fs)
    peaks: list[int] = []
    for start in range(0, len(energy), block):
        segment = energy[start : start + block]
        if segment.size < fs:
            continue
        threshold = np.percentile(segment, 98) * 0.35
        if threshold <= 0:
            continue
        above = segment > threshold
        onsets = np.flatnonzero(above[1:] & ~above[:-1]) + 1
        last = -refractory
        for onset in onsets:
            if onset - last >= refractory:
                peaks.append(start + onset)
                last = onset
    return np.asarray(peaks, dtype=np.int64)


#: RR intervals outside this range are detector errors, not heartbeats.
_RR_RANGE_SECS = (0.3, 2.0)


def _heart_rate_1hz(signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """Median-RR heart rate at 1 Hz, smoothed over 60 s, NaN across gaps."""
    peaks = _rpeaks(signal, fs)
    n_secs = len(signal) // fs
    out = np.full(n_secs, np.nan)
    if peaks.size < 5:
        return out
    rr = np.diff(peaks) / fs
    seconds = (peaks[:-1] / fs).astype(int)
    keep = (rr > _RR_RANGE_SECS[0]) & (rr < _RR_RANGE_SECS[1]) & (seconds < n_secs)
    if keep.sum() < 10:
        return out
    out[seconds[keep]] = 60.0 / rr[keep]
    finite = np.isfinite(out)
    filled = np.interp(np.arange(n_secs), np.flatnonzero(finite), out[finite])
    smoothed = np.convolve(filled, np.ones(60) / 60, mode="same")
    # A second with no detected beat within +/-5 s is a gap, not an interpolation.
    smoothed[~(np.convolve(finite.astype(float), np.ones(11), mode="same") > 0)] = np.nan
    return smoothed


def _corr_at(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    start_a, start_b = max(0, -lag), max(0, lag)
    n = min(len(a) - start_a, len(b) - start_b)
    if n < 600:
        return float("nan")
    x, y = a[start_a : start_a + n], b[start_b : start_b + n]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 600:
        return float("nan")
    xs, ys = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    denominator = np.sqrt((xs * xs).sum() * (ys * ys).sum())
    return float((xs * ys).sum() / denominator) if denominator else float("nan")


#: The Holter may have been fitted after the PSG started, so the search is not
#: restricted to lags that put the PSG wholly inside it.
_LAG_SLACK_SECS = 1800

#: A lag leaving less of the PSG than this overlapped is not evidence.
_MIN_OVERLAP_FRACTION = 0.90


def _best_lag(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    """Coarse-to-fine search for the lag maximising corr(a, shifted b)."""
    limit = (len(b) - len(a)) + _LAG_SLACK_SECS
    coarse = (0, -2.0)
    for lag in range(-_LAG_SLACK_SECS, limit + 1, 10):
        if min(len(a) + min(lag, 0), len(b) - max(lag, 0)) < _MIN_OVERLAP_FRACTION * len(a):
            continue
        r = _corr_at(a, b, lag)
        if np.isfinite(r) and r > coarse[1]:
            coarse = (lag, r)
    best = coarse
    for lag in range(coarse[0] - 60, coarse[0] + 61):
        r = _corr_at(a, b, lag)
        if np.isfinite(r) and r > best[1]:
            best = (lag, r)
    return best


def _third_offsets(a: np.ndarray, b: np.ndarray, lag: int, span: int = 60) -> list[int]:
    """Refit the offset separately on each third of the overlap."""
    start_a, start_b = max(0, -lag), max(0, lag)
    n = min(len(a) - start_a, len(b) - start_b)
    third = n // 3
    out = []
    for index in range(3):
        piece_a = a[start_a + index * third : start_a + (index + 1) * third]
        best = (0, -2.0)
        for delta in range(-span, span + 1):
            lo = start_b + index * third + delta
            if lo < 0 or lo + len(piece_a) > len(b):
                continue
            r = _corr_at(piece_a, b[lo : lo + len(piece_a)], 0)
            if np.isfinite(r) and r > best[1]:
                best = (delta, r)
        out.append(best[0])
    return out
