"""
MIT-BIH Normal Sinus Rhythm labels: demographics, beat counts, signal quality, HRV.

Nothing machine-readable ships with this dataset. Each record's ``.hea`` carries a
single comment line::

    16265 2 128 11730944  8:04:00
    16265.dat 212 0 12 0 -33 15756 0 ECG1
    16265.dat 212 0 12 0 -65 -21174 0 ECG2
    # 32 M

That ``# <age> <sex>`` line is the whole of the shipped metadata — no medications,
no clinical description, no tape or recorder number, unlike ``mitdb``, whose
header comments this format is a stripped-down version of. Everything else here
is derived from the companion ``.atr``, which the shipped ``ANNOTATORS`` file
describes as "reference beat and signal quality annotations".

**THIS DATABASE HAS NO DIAGNOSTIC LABEL, AND THAT IS THE POINT.** All 18 subjects
were "found to have had no significant arrhythmias"; that is what PhysioNet
asserts about the cohort and it is what ``cohort_label`` records, identically, for
every record. It is not derived from the annotations — there are no rhythm
annotations in this release at all. Use this database as a normal-sinus reference
or a negative class, not as a classification task.

Four things worth knowing, all verified against the files.

**1. The beat annotations stop long before the signal does.** This is the largest
trap in the release. Beat annotation covers 79.5%–95.7% of each record — 52.9 of
the 437.5 recorded hours, **12.1%**, carry no beat annotation at all. The tails
run from 3,826 s (16539) to 17,822 s (19090), i.e. one to five hours per record,
and they are *silent*: nothing in the header or the annotation file says the
recording continues. Five records (19088, 19090, 19093, 19140, 19830) also open
with an unannotated head of 23–34 s. ``annotated_secs``,
``unannotated_tail_secs`` and ``unannotated_head_secs`` report it per record, so
a window meant for supervised work can be kept inside the annotated span.

**2. Ectopy exists, but only just.** 1,729,629 reference beats across the 18
records, of which **127 are not normal** — 91 supraventricular premature (``S``),
26 ventricular premature (``V``), 8 fusion (``F``) and 2 nodal premature (``J``).
That is 7.3 ectopic beats per 100,000. Three records (16795, 18184, 19140) have
none whatsoever, and the worst two (16265 and 16773) have 27 each. There is no
usable ectopy class here; ``n_ectopic_beats`` is exposed because "how clean is
clean" is a real question, not because it can be trained on.

**3. Signal quality is annotated, per channel, and is exposed as time.** The
``~`` annotations mark quality transitions, and their ``subtype`` is a bitmask
over the two channels — 0 clean, 1 ECG1 noisy, 2 ECG2 noisy, 3 both. Each opens
an interval running to the next transition. Across the release **98.61% of the
recorded time is annotated clean**; 16272 is the noisiest at 9.60% and 16786 the
cleanest at 0.23%. The state before a record's first ``~`` is taken as clean,
which is safe here rather than assumed: in all 18 records the first ``~`` is a
transition *into* noise (subtype 1, 2 or 3), never a return to clean.

**4. HRV summaries come from the reference beats, with a physiologic filter.**
``mean_hr_bpm``, ``sdnn_ms`` and ``rmssd_ms`` are computed over RR intervals in
[0.3 s, 2.0 s] — the filter is what keeps the unannotated gaps in point 1, which
appear as multi-hour "RR intervals", out of the statistics. ``n_rr_rejected``
reports how many intervals that dropped. These are whole-record summaries over
~24 h of mixed activity and sleep, not the segmented, artefact-corrected analysis
an HRV study would run; take them as a description of the record, not a result.

The ``.xws`` files shipped alongside each record are WAVE workspace files — two
lines naming the record and annotator for PhysioNet's viewer. They hold no data.
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

#: Annotator extension, per the shipped ANNOTATORS file. It is the only one.
ANNOTATOR = "atr"

#: The one thing this release asserts about every subject, and the value of
#: ``cohort_label`` for all 18 records. From PhysioNet's own description:
#: "Subjects included in this database were found to have had no significant
#: arrhythmias."
COHORT_LABEL = "normal_sinus_rhythm"

#: PhysioBank beat symbols. Only N, V, S, F and J occur in this release; the rest
#: are listed so a re-release using them is counted rather than warned about.
BEAT_SYMBOLS = ("N", "L", "R", "V", "/", "A", "f", "F", "j", "a", "E", "J", "Q", "e", "S")

BEAT_NAMES = {
    "N": "normal beat",
    "V": "premature ventricular contraction",
    "S": "supraventricular premature beat",
    "F": "fusion of ventricular and normal beat",
    "J": "nodal (junctional) premature beat",
}

#: Non-beat annotation symbols occurring here, mapped to the column counting them.
#: Never add these to ``n_beats``.
NON_BEAT_COLUMNS = {
    "|": "n_isolated_artifacts",
    "~": "n_quality_changes",
}

#: ``subtype`` of a ``~`` annotation: a bitmask over the two channels. -1 is
#: WFDB's "signals unreadable"; it does not occur in this release.
QUALITY_SUBTYPES = {
    0: "clean",
    1: "noisy_ECG1",
    2: "noisy_ECG2",
    3: "noisy_both",
    -1: "unreadable",
}

#: RR intervals outside this range are dropped before any HRV summary. The upper
#: bound is what keeps the multi-hour unannotated tails from being counted as
#: single enormous intervals; the lower bound drops double detections.
RR_RANGE_SECS = (0.3, 2.0)

#: Header comment: ``# <age> <sex>``. That is the entire shipped metadata.
_DEMOGRAPHICS_RE = re.compile(r"^#\s*(?P<age>-?\d+)\s+(?P<sex>[MF])\s*$")


def parse_header_comments(hea_path: Path) -> dict[str, object]:
    """Parse the one comment line into age and sex.

    A header without a parseable comment comes back with NaN/empty rather than
    raising, so one malformed file cannot fail the whole scan — genuinely broken
    records are what ``corrupt_header`` is for.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    comments = [line for line in lines if line.startswith("#")]

    out: dict[str, object] = {"age": np.nan, "sex": ""}
    if not comments:
        logger.warning("%s carries no comment line, so no demographics", hea_path.name)
        return out

    match = _DEMOGRAPHICS_RE.match(comments[0])
    if match:
        age = int(match.group("age"))
        out["age"] = float(age) if age >= 0 else np.nan
        out["sex"] = match.group("sex")
    else:
        logger.warning("Unparsed demographics comment in %s: %r", hea_path.name, comments[0])
    return out


def _quality_seconds(events: list[tuple[int, int]], sig_len: int, fs: float) -> dict[str, float]:
    """Turn ``~`` transitions into seconds spent in each quality state.

    Each event opens an interval running to the next one, or to the end of the
    record. The span before the first event is counted as clean: in all 18
    records the first ``~`` is a transition *into* noise, so there is nothing
    before it that was ever marked otherwise.
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


def summarise_annotations(record_path: Path, sig_len: int) -> dict[str, object]:
    """Summarise one record's reference annotations.

    Returns per-symbol beat counts, artefact and quality-change counts, seconds in
    each annotated signal-quality state, the annotated span against the record
    length, and whole-record HRV summaries.
    """
    import wfdb

    counts: dict[str, object] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
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

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .atr must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    fs = float(getattr(annotation, "fs", 128) or 128)
    beat_set = set(BEAT_SYMBOLS)
    unexpected: set[str] = set()
    quality_events: list[tuple[int, int]] = []
    beat_samples: list[int] = []

    counts["n_annotations"] = len(annotation.symbol)
    for symbol, sample, subtype in zip(annotation.symbol, annotation.sample, annotation.subtype):
        if symbol in beat_set:
            counts[f"beat_{symbol}"] = int(counts[f"beat_{symbol}"]) + 1
            counts["n_beats"] = int(counts["n_beats"]) + 1
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

    return counts


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so
    a stray file in the directory cannot enter the partition.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. MIT-BIH Normal Sinus Rhythm labels "
            "live in the record headers and .atr annotation files, so point data_path "
            "at the dataset root — the flat directory holding 16265.hea, RECORDS and "
            "ANNOTATORS. Get it from https://physionet.org/content/nsrdb/1.0.0/"
        )

    import wfdb

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue

        row: dict[str, object] = {"record_name": name}
        row.update(parse_header_comments(hea))

        header = wfdb.rdheader(str(hea.with_suffix("")))
        sig_len = int(header.sig_len)
        row["n_samples"] = sig_len
        row["duration_secs"] = sig_len / float(header.fs)
        row["lead_names"] = "|".join(header.sig_name or [])
        # Time of day the Holter tape started. No date ships, here or anywhere
        # else in the release.
        row["start_time"] = str(header.base_time) if header.base_time else ""

        row.update(summarise_annotations(hea.with_suffix(""), sig_len))
        # The whole database is one class, and it is asserted by the release
        # rather than derived from anything in the files.
        row["cohort_label"] = COHORT_LABEL
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d records: %.1f h of signal, %d reference beats (%d ectopic), "
        "%.2f%% of recorded time annotated clean",
        len(df),
        df["duration_secs"].sum() / 3600,
        int(df["n_beats"].sum()),
        int(df["n_ectopic_beats"].sum()),
        100 * df["clean_secs"].sum() / df["duration_secs"].sum(),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``, the subject's sex, and explain why it is that.

    This is the **only** derivation of the stratification label — ``NSRDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **There is no clinical label to stratify on.** Every subject here is a normal
    sinus rhythm control, and the release ships no rhythm annotations, so
    ``cohort_label`` is one value for all 18 records and carries no information a
    fold could be balanced on. What is left is the cohort itself, which PhysioNet
    describes in exactly these terms: "5 men, aged 26 to 45, and 13 women, aged 20
    to 50". Sex is the one documented axis on which these 18 recordings differ.

    **18 records over 10 folds admits nothing finer.** ``StratifiedKFold`` raises
    when *every* class holds fewer members than there are folds, so a usable split
    needs at least one class of 10 or more. Sex gives 13/5 and clears it; a median
    cut on age gives 10/8 and clears it by nothing; anything ectopy-based fails
    outright (three records have no ectopic beats at all, and the rest differ by
    single-digit counts out of ~100,000). Folds hold one or two records each, so
    the five men land in five different folds and half the folds have none — that
    is the arithmetic of 18 records, not a defect in the stratification.

    It is not a clinical grouping and must not be trained on as one. For real
    per-record quantities use ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, the
    ``beat_*`` counts, or the signal-quality seconds.
    """
    out = df.copy()
    # Missing sex would be its own class and would break the fold balance
    # silently; there is none in this release, and "U" makes it visible if a
    # re-release introduces one.
    out["stratify_class"] = out["sex"].replace("", "U")
    logger.info("Stratification classes (sex): %s", out["stratify_class"].value_counts().to_dict())
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return MIT-BIH Normal Sinus Rhythm labels indexed by record name.

    Columns:

    - ``cohort_label`` — ``"normal_sinus_rhythm"`` for **all 18 records**. The
      release asserts it of the cohort; nothing in the files derives it, because
      there are no rhythm annotations here. This database is a reference or a
      negative class, not a classification task.
    - ``age``, ``sex`` — the entire shipped metadata, from the one header comment
      line. 5 men aged 26–45 and 13 women aged 20–50, which reproduces PhysioNet's
      description exactly.
    - ``beat_N`` … ``beat_S`` — reference beat counts per symbol (see
      :data:`BEAT_NAMES`), with ``n_beats``, ``n_ectopic_beats`` and
      ``ectopic_per_100k_beats``. 127 of 1,729,629 beats are not normal.
    - ``annotated_secs``, ``unannotated_head_secs``, ``unannotated_tail_secs``,
      ``annotated_fraction`` — **read these before choosing a window.** Beat
      annotation covers 79.5%–95.7% of each record; the rest is signal with no
      reference behind it.
    - ``clean_secs``, ``noisy_ECG1_secs``, ``noisy_ECG2_secs``,
      ``noisy_both_secs``, ``unreadable_secs``, ``noisy_secs``,
      ``noisy_fraction`` — time in each annotated signal-quality state, from the
      ``~`` annotations, per channel.
    - ``n_isolated_artifacts``, ``n_quality_changes`` — annotation markers that
      are **not** beats and are excluded from ``n_beats``. The artefact count
      varies by three orders of magnitude (52 in 16273, 30,782 in 16773).
    - ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, ``n_rr_rejected`` — whole-record
      HRV summaries over RR intervals in :data:`RR_RANGE_SECS`. Descriptive, not a
      segmented HRV analysis.
    - ``n_samples``, ``duration_secs``, ``start_time``, ``lead_names`` — record
      geometry. Duration is **not** uniform (23.13 h to 25.96 h).
    - ``stratify_class`` — the subject's sex, **for fold construction only**. See
      :func:`attach_stratify_class`.

    There is no patient identifier column: PhysioNet describes 18 recordings from
    18 subjects, and the headers carry nothing that would group them further.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
