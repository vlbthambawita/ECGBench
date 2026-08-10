"""
MIT-BIH ST Change labels: record geometry, beat counts, exercise heart-rate profile.

**THE ST CHANGE DATABASE CONTAINS NO ST CHANGE ANNOTATIONS.** That is the first
thing to know about it and PhysioNet says so plainly on the landing page: "the
annotation files contain only beat labels; they do not include ST change
annotations, as in the European ST-T Database". This module confirms it from the
files — 76,175 of the 76,181 annotations in the 28 ``.atr`` files are beat labels
and the other six are signal-quality markers, there is not a single ``+`` rhythm
marker or ``s`` ST episode anywhere, and every ``aux_note`` in the release is
empty. If you want annotated ST episodes, use
``edb`` (European ST-T, 802 episodes) or ``ltstdb``. What this database offers is
28 recordings *selected* for exhibiting ST change, with reference beat
annotations and nothing else.

Nothing else ships either. The headers carry no comment line at all — not the
``# <age> <sex>`` of ``nsrdb``, not the ``#vfon:`` of ``sddb``, not even a start
time — so there is no age, no sex, no diagnosis, no subject identifier and no
recording date anywhere in the 142 files. Everything below is derived from the
signal headers and the ``.atr`` annotations, except the two group columns, which
are transcribed from one sentence of the landing page.

Five things worth knowing, all verified against the files.

**1. Ten of the 28 records hold ONE channel, not two.** Records 313, 314, 315,
316, 317, 319, 320, 321, 322 and 323 declare a single signal; the other 18 declare
two. Both are described in every header only as ``ECG``, with no electrode
placement stated, so the config names them positionally as ``ECG1``/``ECG2`` and
declares ``alternate_lead_names: {1: ["ECG1"]}``. The consequence for users is
concrete: ``ecg_collate_fn`` stacks signals with torch's ``default_collate``, so a
batch mixing 1- and 2-channel records raises ``RuntimeError``. Batch this dataset
with ``leads=["ECG1"]``, or filter on :func:`n_channels` first. ``n_channels`` is
exposed for exactly that.

**2. The exercise/long-term grouping is the release's, not the files'.** The
landing page says the database "includes 28 ECG recordings of varying lengths,
most of which were recorded during exercise stress tests and which exhibit
transient ST depression", and that "the last five records (323 through 327) are
excerpts of long-term ECG recordings and exhibit ST elevation". So the release
asserts *per record* only for the last five; the other 23 are assigned to
``exercise_stress``/``depression`` **by exclusion**, and the word is "most", not
"all". ``record_group`` and ``st_change_type`` record that assignment, and
``group_source`` is ``"landing_page"`` for every row so nobody mistakes them for
something measured.

**3. The heart-rate profile corroborates the grouping for three of the five, and
contradicts it for one.** ``hr_rise_bpm`` is peak minus opening 60-second mean
heart rate over the reference beats, and an exercise stress test has a
characteristic shape: ramp to peak, then recovery. Records 324, 325 and 326 rise
by 0.0, 3.1 and 7.9 bpm and never exceed 78.9 bpm — flat, as an ambulatory excerpt
should be. But **record 323 ramps 84 → 172 bpm and is still at 117 bpm in its
final minute**, which is not what the other four long-term excerpts look like; it
is also the only single-channel record among the five. Within the exercise group the rise runs
15.1–114.8 bpm, and the three weakest (303 at 15.1, 300 at 21.2, 304 at 29.5) are
consistent with the release's "most". Nothing here overrides the release's own
grouping — ``st_change_type`` still follows the landing page — but ``peak_hr_bpm``
and ``hr_rise_bpm`` are the per-record evidence, and they do not all agree with it.

**4. Beat annotation is essentially complete, unlike every other MIT-BIH-family
long-term database in this catalogue.** Annotation starts 0.2–1.0 s into each
record and ends 0.1–0.9 s before its end, so 99.77–99.98% of every record, and
99.9% of the 13.49 recorded hours, carries reference beats. There is no multi-hour
unannotated tail to window around — contrast ``nsrdb``, which leaves 12.1% of its
signal unannotated in silence.
These are short records that were annotated end to end.

**5. Signal-quality annotation exists in exactly one record.** Record 319 carries
the release's only six ``~`` annotations; all other 27 records have none. Their
subtypes are the ordinary WFDB channel bitmask (0 clean, 1 first signal noisy) plus
``-1`` for unreadable, and the last one opens at 567.5 s and runs to the end of the
record, so 86.2% of 319 is marked not-clean: 1,097.6 s noisy and 124.5 s
unreadable against 196.2 s clean. The absence of ``~`` elsewhere is **not** an
assertion that the other 27 records are clean — it means nobody marked them. Do
not read ``clean_secs`` as a quality measurement across records.

The 28 ``.xws`` files are WAVE workspace files for PhysioNet's viewer, three lines
naming the record and annotator, and the 28 ``.hea-`` files are the superseded
pre-2008 headers, whose signal descriptions read "record 300, signal 0" instead of
"ECG". Neither holds data. All 142 shipped files verify against the release's own
``SHA256SUMS.txt``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ecgbench.labels.svdb import AAMI_CLASSES, AAMI_ORDER

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Annotator extension. The shipped ANNOTATORS file names exactly one:
#: "atr    reference beat annotations".
ANNOTATOR = "atr"

#: The five records the landing page names as excerpts of long-term ECG recordings
#: exhibiting ST elevation. Every other record falls into the exercise-stress group
#: **by exclusion** — see the module docstring, point 2.
LONG_TERM_EXCERPTS = ("323", "324", "325", "326", "327")

#: Where ``record_group`` and ``st_change_type`` come from. Constant for every row,
#: and present so a consumer cannot mistake a transcribed grouping for a measurement.
GROUP_SOURCE = "landing_page"

#: Group name -> the ST deviation the landing page attributes to it.
ST_CHANGE_BY_GROUP = {
    "exercise_stress": "depression",
    "long_term_excerpt": "elevation",
}

#: PhysioBank beat symbols. Only N, V and S occur in this release; the rest are
#: listed so a re-release using them is counted rather than warned about.
BEAT_SYMBOLS = (
    "N", "L", "R", "V", "/", "A", "f", "F", "j",
    "a", "E", "J", "Q", "e", "S", "n", "B", "r",
)

BEAT_NAMES = {
    "N": "normal beat",
    "V": "premature ventricular contraction",
    "S": "supraventricular premature beat",
}

#: Non-beat annotation symbols occurring here, mapped to the column counting them.
#: Never add these to ``n_beats``. Only ``~`` occurs, and only in record 319.
NON_BEAT_COLUMNS = {
    "|": "n_isolated_artifacts",
    "~": "n_quality_changes",
    "+": "n_rhythm_changes",
}

#: ``subtype`` of a ``~`` annotation: a bitmask over the channels, plus WFDB's -1
#: for "signals unreadable". Both 1 and -1 occur here, in record 319 alone.
QUALITY_SUBTYPES = {
    0: "clean",
    1: "noisy_ECG1",
    2: "noisy_ECG2",
    3: "noisy_both",
    -1: "unreadable",
}

#: RR intervals outside this range are dropped before any heart-rate summary —
#: double detections below, artefact-spanning gaps above. The upper bound is 2.0 s
#: rather than the 2.5 s some references use because nothing here is bradycardic.
RR_RANGE_SECS = (0.3, 2.0)

#: Heart-rate profile: window width and step, in seconds. A window is used only if
#: it holds at least :data:`HR_WINDOW_MIN_BEATS` accepted RR intervals, which keeps
#: a couple of stray detections from becoming a "peak".
HR_WINDOW_SECS = 60.0
HR_STEP_SECS = 30.0
HR_WINDOW_MIN_BEATS = 20


def record_group(record_name: str) -> str:
    """Return ``"long_term_excerpt"`` or ``"exercise_stress"`` for one record.

    Transcribed from the landing page, not derived from the signal. See the module
    docstring, point 2: only the five long-term excerpts are named per record, and
    the rest are the release's "most ... were recorded during exercise stress
    tests" read by exclusion.
    """
    return "long_term_excerpt" if record_name in LONG_TERM_EXCERPTS else "exercise_stress"


def _quality_seconds(events: list[tuple[int, int]], sig_len: int, fs: float) -> dict[str, float]:
    """Turn ``~`` transitions into seconds spent in each quality state.

    Each event opens an interval running to the next one, or to the end of the
    record. The span before the first event is counted as clean.

    A record with no ``~`` at all comes back wholly clean, which is what the WFDB
    convention implies and **not** a statement that anyone inspected it: 27 of the
    28 records here carry no quality annotation whatsoever.
    """
    secs = {name: 0.0 for name in QUALITY_SUBTYPES.values()}
    if not events:
        secs["clean"] = sig_len / fs
        return secs

    secs["clean"] += events[0][0] / fs
    for i, (start, subtype) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else sig_len
        name = QUALITY_SUBTYPES.get(subtype)
        if name is None:
            logger.warning("Unknown signal-quality subtype %r, not counted", subtype)
            continue
        secs[name] += (end - start) / fs
    return secs


def _heart_rate_profile(beat_samples: list[int], sig_len: int, fs: float) -> dict[str, float]:
    """Opening, peak and closing heart rate over a sliding window.

    This is the only per-record quantity here that speaks to what the recording
    *was* — an exercise stress test ramps and recovers, an ambulatory excerpt does
    not. It is computed from the reference beats, so it inherits their accuracy and
    nothing else.

    ``hr_rise_bpm`` is peak minus opening, not peak minus minimum: the question is
    how far the subject was driven from where they started, and a single low window
    mid-recovery would flatter every record equally.
    """
    out = {
        "baseline_hr_bpm": np.nan,
        "peak_hr_bpm": np.nan,
        "final_hr_bpm": np.nan,
        "hr_rise_bpm": np.nan,
        "n_hr_windows": 0,
    }
    if len(beat_samples) < 3:
        return out

    times = np.asarray(beat_samples, dtype=np.float64) / fs
    rr = np.diff(times)
    mid = times[:-1]
    low, high = RR_RANGE_SECS
    keep = (rr >= low) & (rr <= high)
    rr, mid = rr[keep], mid[keep]
    if rr.size < HR_WINDOW_MIN_BEATS:
        return out

    hr = 60.0 / rr
    means = []
    for start in np.arange(0.0, sig_len / fs, HR_STEP_SECS):
        window = (mid >= start) & (mid < start + HR_WINDOW_SECS)
        if int(window.sum()) >= HR_WINDOW_MIN_BEATS:
            means.append(float(hr[window].mean()))
    if not means:
        return out

    out["baseline_hr_bpm"] = means[0]
    out["peak_hr_bpm"] = max(means)
    out["final_hr_bpm"] = means[-1]
    out["hr_rise_bpm"] = max(means) - means[0]
    out["n_hr_windows"] = len(means)
    return out


def summarise_annotations(record_path: Path, sig_len: int, fs: float) -> dict[str, object]:
    """Summarise one record's reference annotations.

    Returns per-symbol beat counts and their AAMI EC57 reduction, non-beat marker
    counts, seconds in each annotated signal-quality state, the annotated span
    against the record length, whole-record HRV, and the heart-rate profile.
    """
    import wfdb

    counts: dict[str, object] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
    counts.update({f"aami_{cls}": 0 for cls in AAMI_ORDER})
    counts.update({column: 0 for column in NON_BEAT_COLUMNS.values()})
    counts.update({f"{name}_secs": 0.0 for name in QUALITY_SUBTYPES.values()})
    counts.update(
        {
            "n_beats": 0,
            "n_annotations": 0,
            "n_ectopic_beats": 0,
            "ectopic_per_100k_beats": np.nan,
            "annotated_secs": np.nan,
            "unannotated_head_secs": np.nan,
            "unannotated_tail_secs": np.nan,
            "annotated_fraction": np.nan,
            "noisy_secs": 0.0,
            "noisy_fraction": np.nan,
            "mean_hr_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "n_rr_rejected": 0,
        }
    )
    counts.update(_heart_rate_profile([], sig_len, fs))

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .atr must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    beat_set = set(BEAT_SYMBOLS)
    unexpected: set[str] = set()
    quality_events: list[tuple[int, int]] = []
    beat_samples: list[int] = []

    counts["n_annotations"] = len(annotation.symbol)
    for symbol, sample, subtype in zip(annotation.symbol, annotation.sample, annotation.subtype):
        if symbol in beat_set:
            counts[f"beat_{symbol}"] = int(counts[f"beat_{symbol}"]) + 1
            counts["n_beats"] = int(counts["n_beats"]) + 1
            aami = AAMI_CLASSES.get(symbol)
            if aami is not None:
                counts[f"aami_{aami}"] = int(counts[f"aami_{aami}"]) + 1
            beat_samples.append(int(sample))
        elif symbol in NON_BEAT_COLUMNS:
            column = NON_BEAT_COLUMNS[symbol]
            counts[column] = int(counts[column]) + 1
            if symbol == "~":
                quality_events.append((int(sample), int(subtype)))
        else:
            # Worth seeing rather than silently dropping: it would mean this
            # release uses symbols this module does not know about.
            unexpected.add(symbol)

    if unexpected:
        logger.warning(
            "%s: annotation symbols outside BEAT_SYMBOLS and NON_BEAT_COLUMNS, not counted: %s",
            record_path.name,
            sorted(unexpected),
        )

    counts["n_ectopic_beats"] = int(counts["n_beats"]) - int(counts["beat_N"])
    if int(counts["n_beats"]) > 0:
        counts["ectopic_per_100k_beats"] = (
            1e5 * int(counts["n_ectopic_beats"]) / int(counts["n_beats"])
        )

    for name, secs in _quality_seconds(quality_events, sig_len, fs).items():
        counts[f"{name}_secs"] = secs
    counts["noisy_secs"] = (
        float(counts["noisy_ECG1_secs"])
        + float(counts["noisy_ECG2_secs"])
        + float(counts["noisy_both_secs"])
        + float(counts["unreadable_secs"])
    )
    if sig_len > 0:
        counts["noisy_fraction"] = float(counts["noisy_secs"]) / (sig_len / fs)

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

    counts.update(_heart_rate_profile(beat_samples, sig_len, fs))
    return counts


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so
    neither a stray file nor one of the 28 superseded ``.hea-`` headers can enter
    the partition.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. MIT-BIH ST Change labels are derived "
            "from the record headers and .atr annotation files, so point data_path at "
            "the dataset root — the flat directory holding 300.hea, RECORDS and "
            "ANNOTATORS. Get it from https://physionet.org/content/stdb/1.0.0/"
        )

    import wfdb

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue

        header = wfdb.rdheader(str(hea.with_suffix("")))
        sig_len = int(header.sig_len)
        fs = float(header.fs)

        row: dict[str, object] = {"record_name": name}
        row["n_channels"] = int(header.n_sig)
        row["n_samples"] = sig_len
        row["duration_secs"] = sig_len / fs
        row["sampling_rate"] = fs
        # Every header describes every channel as the bare word "ECG", so this is
        # "ECG" or "ECG|ECG" and carries no placement. It is kept because a
        # re-release that finally names the electrodes would show up here.
        row["signal_descriptions"] = "|".join(header.sig_name or [])
        # Per-record and per-channel: 31 distinct values from 161 to 500 adu/mV.
        # The 12-bit rail moves with it, which is what the config's
        # amplitude_range_mv is computed from.
        gains = list(header.adc_gain or [])
        row["adc_gain_ECG1"] = float(gains[0]) if len(gains) > 0 else np.nan
        row["adc_gain_ECG2"] = float(gains[1]) if len(gains) > 1 else np.nan

        row["record_group"] = record_group(name)
        row["st_change_type"] = ST_CHANGE_BY_GROUP[row["record_group"]]
        row["group_source"] = GROUP_SOURCE

        row.update(summarise_annotations(hea.with_suffix(""), sig_len, fs))
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d records: %.2f h of signal, %d reference beats (%d ectopic); "
        "%d records hold one channel, %d hold two",
        len(df),
        df["duration_secs"].sum() / 3600,
        int(df["n_beats"].sum()),
        int(df["n_ectopic_beats"].sum()),
        int((df["n_channels"] == 1).sum()),
        int((df["n_channels"] == 2).sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: the ST-change group crossed with the channel count.

    This is the **only** derivation of the stratification label — ``STDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **Why these two axes.** The release documents exactly one grouping of its
    records (23 exercise-stress/ST-depression, 5 long-term-excerpt/ST-elevation),
    and it is the obvious thing to balance: left to chance, all five elevation
    records could land in two folds. But channel count is crossed in because it is
    the property that decides whether a record can be *used* alongside another —
    ``ecg_collate_fn`` cannot stack a 1-channel record with a 2-channel one, so a
    fold's usable size under ``leads=["ECG1","ECG2"]`` depends entirely on how many
    single-channel records it drew. Balancing one axis and not the other would
    leave that to luck.

    The four resulting classes are ``depression_2ch`` (14), ``depression_1ch`` (9),
    ``elevation_2ch`` (4) and ``elevation_1ch`` (1, record 323).
    ``StratifiedKFold`` raises only when *every* class is smaller than ``n_folds``,
    so the 14-record class carries the split and the singleton is tolerated —
    sklearn warns, and with 28 records over 10 folds each fold holds two or three
    records regardless.

    **What was rejected.** Beat-derived bands, the axis ``svdb`` and ``chfdb`` use,
    are unusable here: 75,038 of the 76,175 beats are ``N``, nineteen records have
    fewer than five ectopic beats in total, and nine have none at all, so any burden
    banding would be noise. ``peak_hr_bpm`` banding is measured rather than
    asserted, but it is a *consequence* of the exercise protocol this already
    stratifies on, and cutting it into bands would double-count that axis while
    adding a threshold nobody can justify.

    It is not a clinical grouping and must not be trained on as one — in
    particular, ``st_change_type`` is transcribed from the landing page and is
    **not** an ST measurement. For real per-record quantities use ``peak_hr_bpm``,
    ``hr_rise_bpm``, ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms`` or the ``beat_*``
    counts.
    """
    out = df.copy()
    out["stratify_class"] = (
        out["st_change_type"].astype(str) + "_" + out["n_channels"].astype(int).astype(str) + "ch"
    )
    logger.info(
        "Stratification classes (ST change x channel count): %s",
        out["stratify_class"].value_counts().to_dict(),
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return MIT-BIH ST Change labels indexed by record name.

    **There are no ST change annotations in this database** — see the module
    docstring. The columns are:

    - ``record_group``, ``st_change_type``, ``group_source`` — the landing page's
      own grouping, transcribed: ``exercise_stress``/``depression`` for 23 records
      and ``long_term_excerpt``/``elevation`` for 323–327. ``group_source`` is
      ``"landing_page"`` for every row because none of this is measured, and the
      release asserts it per record only for the five. Check it against
      ``hr_rise_bpm`` before relying on it.
    - ``n_channels`` — **1 for ten records, 2 for eighteen.** Filter or select
      ``leads=["ECG1"]`` before batching; a mixed batch raises in collation.
    - ``adc_gain_ECG1``, ``adc_gain_ECG2`` — per-record, per-channel gain, 161 to
      500 adu/mV over 31 distinct values. ``wfdb`` applies each record's own, so
      the returned signal is millivolts either way; the gain matters only because
      it sets that record's 12-bit rail.
    - ``baseline_hr_bpm``, ``peak_hr_bpm``, ``final_hr_bpm``, ``hr_rise_bpm``,
      ``n_hr_windows`` — the heart-rate profile over 60 s windows stepped 30 s.
      The one per-record measurement that speaks to the recording protocol.
    - ``beat_N``, ``beat_V``, ``beat_S`` … — reference beat counts per symbol (see
      :data:`BEAT_NAMES`), their AAMI EC57 reduction ``aami_N``/``aami_S``/
      ``aami_V``/``aami_F``/``aami_Q``, and ``n_beats``, ``n_ectopic_beats``,
      ``ectopic_per_100k_beats``. 1,137 of 76,175 beats are not normal, and two thirds
      of that is three records: 305 holds 265 of the release's 322 ``V``, and 324 and
      326 together hold 699 of its 815 ``S``.
    - ``annotated_secs``, ``unannotated_head_secs``, ``unannotated_tail_secs``,
      ``annotated_fraction`` — annotation coverage. Effectively complete here
      (99.77–99.98% per record), unlike ``nsrdb`` or ``sddb``, but exposed for the
      same reason.
    - ``clean_secs``, ``noisy_ECG1_secs``, ``noisy_ECG2_secs``, ``noisy_both_secs``,
      ``unreadable_secs``, ``noisy_secs``, ``noisy_fraction`` — from the ``~``
      annotations, which exist **only in record 319**. Everywhere else these say
      "wholly clean" because nobody marked anything, not because anybody looked.
    - ``n_isolated_artifacts``, ``n_quality_changes``, ``n_rhythm_changes`` —
      non-beat markers, excluded from ``n_beats``. All zero except six quality
      changes in record 319; there is not one ``+`` in the release.
    - ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, ``n_rr_rejected`` — whole-record
      HRV over RR intervals in :data:`RR_RANGE_SECS`. For a stress test these
      summarise a deliberately non-stationary recording; they describe the record,
      they are not an HRV result.
    - ``n_samples``, ``duration_secs``, ``sampling_rate``, ``signal_descriptions``
      — record geometry. All 28 lengths differ, 784.3 s to 4,032.9 s.
    - ``stratify_class`` — for fold construction only. See
      :func:`attach_stratify_class`.

    There is no patient identifier column and cannot be one: the release carries no
    subject information of any kind, so whether any two of these 28 recordings came
    from the same person is unknowable from the files.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
