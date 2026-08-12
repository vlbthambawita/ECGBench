"""
BUT QDB labels: sample-by-sample ECG signal quality, three experts and a consensus.

**The ground truth of this database is a label per sample, not a label per record.**
Every other dataset in ECGBench answers "what is wrong with this patient"; this one
answers "can this stretch of signal be analysed at all", and it answers it at 1 kHz
resolution over 99.4 hours of free-living wearable ECG. The per-record columns
returned by :func:`load_labels` are *summaries* of that, provided so folds can be
balanced and records filtered. The real labels come from
:func:`load_quality_intervals` and :func:`quality_vector`.

The three quality classes are the release's own (see :data:`QUALITY_CLASSES`):
class 1 all waveforms measurable, class 2 QRS detectable but nothing finer, class 3
QRS not reliably detectable and the signal unusable. Class 0 is not a quality
verdict — it means *this segment was never annotated*.

**Six things worth knowing, all verified against the shipped files.**

**1. Only 20.8% of the recorded time carries an annotation, and it is concentrated
in three records.** Three recordings (100001, 105001, 111001) are annotated end to
end: 24.19 h, 38.65 h and 25.18 h, which is 88.6% of all annotated time in the
release. The other 15 get two 20-minute segments each, plus five extra segments the
authors picked for being noisy — four of 20 minutes (113001, 124001 x2, 114001) and
one of 2 minutes (114001). So 113001 has 60 annotated minutes, 114001 62, 124001 80
and the remaining twelve exactly 40.

**And the two standard segments sit at a fixed offset in every one of those 15
records**, which the release does not mention: samples 28,800,000-30,000,000 and
57,600,000-58,800,000, i.e. 8 h 00 m to 8 h 20 m and 16 h 00 m to 16 h 20 m into the
recording. So ``window=(0, n)`` — the obvious first thing to try — lands on
unannotated signal for 15 of the 18 records however small ``n`` is. The five extra
segments are elsewhere (113001 at 36,119,999; 114001 at 11,214,750 and 11,674,750;
124001 at 33,699,999 and 65,099,999), and 114001's 2-minute segment is 120,001
samples rather than 120,000, which is why its annotated total is 3720.001 s rather
than 3720. ``annotated_secs`` and ``annotated_fraction`` report coverage per record,
and :func:`annotated_blocks` returns the bounds a window has to sit inside.

**2. The fourth column triple is the consensus, and it behaves as a majority vote.**
The release says the CSV holds "3 columns x 3 annotators + consensus" and does not say
how the consensus was formed. Measured over all 357,799,001 annotated samples: a
majority of the three experts exists at 99.863% of them, and the consensus equals that
majority at all but **3,103** of those samples (99.99913%). Where all three disagree
it almost always still adopts one of their three opinions — it differs from every
expert at only **378** samples in the whole release, all inside 111001, 124001 and
125001. The residuals cluster at interval boundaries, which is what one would expect
from a consensus drawn segment by segment rather than sample by sample.
``consensus_matches_majority`` and ``expert_majority_fraction`` record it per record,
so a re-release formed by a different rule shows up in the columns rather than being
assumed away.

**3. The experts disagree a great deal, and expert 1 is the outlier.** All three
agree on 248,410,246 of the 357,799,001 annotated samples — **69.43%**. Pairwise
agreement, averaged over the 18 records, is 0.82 (experts 1-2), 0.81 (1-3) and 0.89
(2-3); expert 1 is systematically stricter,
calling class 2 where the others call class 1 — on 121001 expert 1 says 51.8% class
1 and expert 2 says 90.4%. Anything reported as accuracy against "the" label is
reporting agreement with a consensus that three humans reached this loosely; the
per-expert fractions are exposed so that ceiling is visible.

**4. The annotated 20-minute segments are not a random sample of the recording.**
Five of them were "subjectively selected" to raise the proportion of poor signal,
so the class mix over annotated time is deliberately worse than over the recordings
as a whole. Class 3 is 15.2% of annotated time across the release, and 87% of that
comes from a single record (105001, 13.2 h of class 3). Do not read
``consensus_class3_fraction`` as a property of 24-hour wearable ECG.

**5. Every record saturates the 16-bit converter, and no check can see it.** All 18
records attain both ADC rails, so ``amplitude_range_mv`` has to be set to the rail
and ``amplitude_outlier`` cannot fail anything (see the config). ``clipped_fraction``
is the column that measures it: 114001 is at a rail for 0.015% of its samples and
100001 for two samples out of 87 million. This is a much milder pathology than
``tollet``'s, but it is the reason the check is a no-op here.

**6. WFDB's invalid-sample marker does not occur.** Format 16 reserves -32768, which
``wfdb.rdrecord`` turns into NaN — and ``nan_values`` fails a record on a single NaN.
There is not one anywhere in this release, so the check passes on all 18 records
rather than passing by luck. ``n_invalid_samples`` reports it per record so a
re-release that introduces the marker is caught rather than silently dropping
records.

The gain and baseline differ per record (gain 0.99998 to 1.996 ADC units per µV,
baseline -18289 to +11462), so the physical span differs per record too: 100001
covers -10.30 to +22.53 mV and 104001 -32.77 to +32.77 mV. ``min_mv``/``max_mv``
report each record's attained extremes, which are its rails.

The 3-axis accelerometer ships as a separate WFDB record per recording, at 100 Hz in
milli-g. ECGBench gives it no records of its own — it is not an ECG and its rate is
not one of ``sampling_rates`` — but ``acc_path`` points at it, because the release's
own usage notes propose motion as an input to quality assessment.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: The release's own quality classes, verbatim in intent. 0 is not a verdict.
QUALITY_CLASSES = {
    0: "not annotated",
    1: "all significant waveforms (P, QRS, T) clearly visible, onsets and offsets "
       "reliably detectable",
    2: "noise increased and significant points unclear, but QRS complexes clearly "
       "visible and reliably detectable",
    3: "QRS complexes cannot be detected reliably; unsuitable for any analysis",
}

#: Column triples of ``<record>_ANN.csv``, in file order. The release describes the
#: layout as "3 columns x 3 annotators + consensus"; the last triple is the
#: consensus, and point 2 of the module docstring is the check of that claim.
ANNOTATORS: tuple[str, ...] = ("expert_1", "expert_2", "expert_3", "consensus")

#: 0-based index of each annotator's first column in the 12-column CSV.
_FIRST_COLUMN = {name: 3 * i for i, name in enumerate(ANNOTATORS)}

#: The label everything downstream should use unless it is studying the annotators.
DEFAULT_ANNOTATOR = "consensus"

#: Format-16 rails. ``-32768`` is WFDB's invalid-sample marker and becomes NaN, so
#: the negative rail a real sample can reach is ``-32767`` (point 6).
_ADC_RAILS = (-32767, 32767)
_ADC_INVALID = -32768

#: Fraction of *annotated* time in class 3 above which a record is put in the
#: ``class3_high`` stratification class. See :func:`attach_stratify_class` — the
#: value is immaterial anywhere in [0.004, 0.028] because the release has a gap
#: there.
CLASS3_HIGH_THRESHOLD = 0.01

#: Read the .dat in chunks rather than whole: 18 records are 3.4 GB of int16, and
#: the scan only needs counts.
_SCAN_CHUNK = 20_000_000


def _records_file(data_path: Path) -> list[str]:
    """Return the ECG record stems listed in the shipped ``RECORDS`` file.

    ``RECORDS`` lists 36 entries — one ``<id>/<id>_ACC`` and one ``<id>/<id>_ECG``
    per recording. Only the ECG ones become ECGBench records; taking the list from
    the release rather than a glob is what keeps the accelerometer records, and any
    stray file, out of the partition.
    """
    from ecgbench.labels import LabelSourceMissingError

    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. BUT QDB labels live in the per-record "
            "<id>_ANN.csv files and subject-info.csv, so point data_path at the version "
            "directory — the one holding 100001/, RECORDS and subject-info.csv. Get it "
            "from https://physionet.org/content/butqdb/1.0.0/"
        )
    stems = [line.strip() for line in records_file.read_text().split() if line.strip()]
    ecg = [s for s in stems if s.endswith("_ECG")]
    if not ecg:
        raise LabelSourceMissingError(
            f"{records_file} lists {len(stems)} entries and none of them ends in '_ECG'. "
            "This does not look like a BUT QDB 1.0.0 tree."
        )
    return ecg


def parse_annotation_csv(csv_path: Path, n_samples: int) -> pd.DataFrame:
    """Parse one ``<record>_ANN.csv`` into a tidy interval table.

    The shipped file has 12 headerless columns and one row per interval per
    annotator, padded with blanks because the four annotators cut the record into
    different numbers of intervals. Each triple is ``(first sample, last sample,
    quality class)`` and is **1-based and inclusive**, which is what
    ``ann_reader.m`` expects. This function returns 0-based half-open bounds
    instead, so ``signal[start:end]`` is the annotated stretch.

    Returns a frame with columns ``annotator``, ``start``, ``end``, ``n_samples``,
    ``duration_secs``, ``quality_class``, sorted by annotator then start.

    Raises:
        ValueError: if an annotator's intervals do not tile ``[0, n_samples)``
            exactly. They do in all 18 shipped records, and a gap or an overlap
            would silently corrupt every fraction computed from them.
    """
    raw = pd.read_csv(csv_path, header=None)
    if raw.shape[1] != 3 * len(ANNOTATORS):
        raise ValueError(
            f"{csv_path.name} has {raw.shape[1]} columns; BUT QDB annotation files have "
            f"{3 * len(ANNOTATORS)} (3 per annotator for {len(ANNOTATORS)} annotators)."
        )

    frames = []
    for name in ANNOTATORS:
        col = _FIRST_COLUMN[name]
        sub = raw.iloc[:, [col, col + 1, col + 2]].dropna()
        start = sub.iloc[:, 0].astype(np.int64).to_numpy() - 1  # 1-based -> 0-based
        end = sub.iloc[:, 1].astype(np.int64).to_numpy()  # inclusive -> exclusive
        klass = sub.iloc[:, 2].astype(np.int8).to_numpy()

        if start.size == 0:
            raise ValueError(f"{csv_path.name}: annotator {name} has no intervals.")
        if start[0] != 0 or end[-1] != n_samples:
            raise ValueError(
                f"{csv_path.name}: annotator {name} covers samples "
                f"[{start[0]}, {end[-1]}) but the record holds {n_samples}."
            )
        if not np.array_equal(start[1:], end[:-1]):
            raise ValueError(
                f"{csv_path.name}: annotator {name}'s intervals have a gap or an "
                "overlap, so no fraction computed from them would be meaningful."
            )
        unknown = set(np.unique(klass).tolist()) - set(QUALITY_CLASSES)
        if unknown:
            raise ValueError(
                f"{csv_path.name}: annotator {name} uses quality classes {sorted(unknown)}, "
                f"which are not in {sorted(QUALITY_CLASSES)}."
            )

        frames.append(
            pd.DataFrame(
                {
                    "annotator": name,
                    "start": start,
                    "end": end,
                    "n_samples": end - start,
                    "quality_class": klass,
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    out["duration_secs"] = out["n_samples"] / 1000.0
    return out


def load_quality_intervals(
    data_path: Path | str,
    record_id: str,
    annotator: str | None = DEFAULT_ANNOTATOR,
) -> pd.DataFrame:
    """Return one record's quality intervals, 0-based and half-open.

    This is the dataset's actual ground truth. ``annotator=None`` returns all four
    (three experts and the consensus) so their disagreement can be studied;
    otherwise pass one of :data:`ANNOTATORS`.

    Rows with ``quality_class == 0`` are **unannotated stretches**, not a fourth
    quality level. Drop them, or use :func:`annotated_blocks`.
    """
    data_path = Path(data_path)
    record_id = str(record_id)
    csv_path = data_path / record_id / f"{record_id}_ANN.csv"
    if not csv_path.exists():
        from ecgbench.labels import LabelSourceMissingError

        raise LabelSourceMissingError(
            f"No quality annotations at {csv_path}. BUT QDB ships one <id>_ANN.csv per "
            "recording alongside its signals; point data_path at the version directory "
            "of a full local copy (https://physionet.org/content/butqdb/1.0.0/)."
        )

    n_samples = _record_length(data_path, record_id)
    intervals = parse_annotation_csv(csv_path, n_samples)
    if annotator is None:
        return intervals
    if annotator not in ANNOTATORS:
        raise ValueError(f"annotator must be one of {ANNOTATORS} or None, got {annotator!r}")
    return intervals[intervals["annotator"] == annotator].reset_index(drop=True)


def annotated_blocks(
    data_path: Path | str,
    record_id: str,
    annotator: str = DEFAULT_ANNOTATOR,
) -> pd.DataFrame:
    """Return the contiguous stretches of one record that were annotated at all.

    Coverage is identical across the four annotators in every shipped record — they
    graded the same segments — so the block list does not depend on ``annotator``;
    it is a parameter only so a re-release with per-annotator coverage is not
    silently averaged.

    Use this to place a ``window=``: a window outside these bounds returns signal
    with no label behind it. Three records return a single block covering the whole
    recording; the rest return two to four blocks of 2 or 20 minutes.
    """
    intervals = load_quality_intervals(data_path, record_id, annotator)
    annotated = intervals[intervals["quality_class"] > 0]
    if annotated.empty:
        return annotated.assign(block=pd.Series(dtype="int64"))

    # Merge intervals that touch: consecutive class-1/class-2 stretches inside one
    # graded 20-minute segment are one block of annotated time, not two.
    blocks: list[dict[str, int]] = []
    for row in annotated.itertuples(index=False):
        if blocks and row.start == blocks[-1]["end"]:
            blocks[-1]["end"] = row.end
        else:
            blocks.append({"start": int(row.start), "end": int(row.end)})

    out = pd.DataFrame(blocks)
    out.insert(0, "block", np.arange(1, len(out) + 1))
    out["n_samples"] = out["end"] - out["start"]
    out["duration_secs"] = out["n_samples"] / 1000.0
    return out


def quality_vector(
    data_path: Path | str,
    record_id: str,
    annotator: str = DEFAULT_ANNOTATOR,
    start: int = 0,
    length: int | None = None,
) -> np.ndarray:
    """Expand the intervals into one quality class per sample.

    Returns an ``int8`` array of ``length`` samples (to the end of the record when
    ``length`` is None), aligned to ``signal[start:start + length]`` — so the
    arguments are the same ``(start, length)`` pair as ``ECGDataset(window=...)``
    and the two line up sample for sample.

    Values are 0-3 as in :data:`QUALITY_CLASSES`; **0 means unannotated** and is the
    majority of every record but three, so mask it before computing anything.
    """
    data_path = Path(data_path)
    n_samples = _record_length(data_path, str(record_id))
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    stop = n_samples if length is None else start + int(length)
    if stop > n_samples:
        raise ValueError(
            f"window ({start}, {length}) runs to sample {stop}, past the end of record "
            f"{record_id}, which holds {n_samples} samples."
        )

    out = np.zeros(stop - start, dtype=np.int8)
    for row in load_quality_intervals(data_path, record_id, annotator).itertuples(index=False):
        lo, hi = max(int(row.start), start), min(int(row.end), stop)
        if hi > lo:
            out[lo - start : hi - start] = row.quality_class
    return out


def _record_length(data_path: Path, record_id: str) -> int:
    """Sample count from the ECG header's first line, without reading the signal."""
    hea = data_path / record_id / f"{record_id}_ECG.hea"
    if not hea.exists():
        from ecgbench.labels import LabelSourceMissingError

        raise LabelSourceMissingError(f"No ECG header at {hea}.")
    return int(hea.read_text().splitlines()[0].split()[3])


def summarise_quality(intervals: pd.DataFrame, n_samples: int) -> dict[str, object]:
    """Reduce one record's intervals to the per-record columns of :func:`load_labels`.

    Every fraction is over **annotated** samples, not over the record, because 15 of
    the 18 records are annotated for 40 minutes of a 24-hour recording and a
    fraction over the record would be a fraction of unannotated time.
    """
    vectors: dict[str, np.ndarray] = {}
    for name in ANNOTATORS:
        v = np.zeros(n_samples, dtype=np.int8)
        for row in intervals[intervals["annotator"] == name].itertuples(index=False):
            v[row.start : row.end] = row.quality_class
        vectors[name] = v

    mask = vectors[DEFAULT_ANNOTATOR] > 0
    n_ann = int(mask.sum())
    out: dict[str, object] = {
        "annotated_samples": n_ann,
        "annotated_secs": n_ann / 1000.0,
        "annotated_fraction": n_ann / n_samples if n_samples else np.nan,
        "fully_annotated": bool(n_ann == n_samples),
        # Coverage is identical across annotators in every shipped record. Recorded
        # rather than assumed: a re-release where one expert graded less would make
        # every cross-annotator figure below an average over different samples.
        "same_coverage_all_annotators": bool(
            all(np.array_equal(mask, vectors[n] > 0) for n in ANNOTATORS)
        ),
    }

    for name in ANNOTATORS:
        graded = vectors[name][mask]
        prefix = "consensus" if name == DEFAULT_ANNOTATOR else name
        for k in (1, 2, 3):
            count = int((graded == k).sum())
            out[f"{prefix}_class{k}_fraction"] = count / n_ann if n_ann else np.nan
            if name == DEFAULT_ANNOTATOR:
                out[f"consensus_class{k}_secs"] = count / 1000.0

    experts = [vectors[n][mask] for n in ANNOTATORS if n != DEFAULT_ANNOTATOR]
    consensus = vectors[DEFAULT_ANNOTATOR][mask]
    if n_ann:
        pairs = [float((a == b).mean()) for a, b in combinations(experts, 2)]
        out["mean_expert_agreement"] = float(np.mean(pairs))
        out["expert_unanimous_fraction"] = float(
            ((experts[0] == experts[1]) & (experts[1] == experts[2])).mean()
        )
        # Point 2 of the module docstring: does the fourth triple behave like a
        # majority vote of the other three? Computed per record rather than
        # asserted, so a re-release that changes the rule shows up in the column.
        majority = np.zeros_like(consensus)
        for k in (1, 2, 3):
            majority[sum((e == k).astype(np.int8) for e in experts) >= 2] = k
        has_majority = majority > 0
        out["expert_majority_fraction"] = float(has_majority.mean())
        out["consensus_matches_majority"] = (
            float((consensus[has_majority] == majority[has_majority]).mean())
            if has_majority.any()
            else np.nan
        )
        out["dominant_consensus_class"] = int(
            max((1, 2, 3), key=lambda k: float((consensus == k).sum()))
        )
    else:  # pragma: no cover - no shipped record is unannotated
        out.update(
            mean_expert_agreement=np.nan,
            expert_unanimous_fraction=np.nan,
            expert_majority_fraction=np.nan,
            consensus_matches_majority=np.nan,
            dominant_consensus_class=0,
        )
    return out


def scan_signal(data_path: Path, record_id: str) -> dict[str, object]:
    """Measure one record's geometry, physical span and converter saturation.

    Reads the ``.dat`` as memory-mapped ``int16`` in chunks and applies the header's
    own ``(sample - baseline) / gain`` conversion, rather than going through
    ``wfdb.rdrecord``: the whole release is 1.72 billion samples, and a float64
    ``p_signal`` for the longest record alone is 1.1 GB.

    ``clipped_samples`` counts samples at either ADC rail (point 5) and
    ``n_invalid_samples`` counts WFDB's invalid marker, which becomes NaN and would
    fail ``nan_values`` (point 6).
    """
    hea = (data_path / record_id / f"{record_id}_ECG.hea").read_text().splitlines()
    n_samples = int(hea[0].split()[3])
    fs = float(hea[0].split()[2])
    # "1.996(-12200)/uV" -> gain 1.996 ADC units per microvolt, baseline -12200.
    gain_field = hea[1].split()[2]
    gain = float(gain_field.split("(")[0])
    baseline = int(gain_field.split("(")[1].split(")")[0])
    units = gain_field.split("/")[-1]

    dat = np.memmap(data_path / record_id / f"{record_id}_ECG.dat", dtype="<i2", mode="r")
    if dat.size != n_samples:
        raise ValueError(
            f"{record_id}_ECG.dat holds {dat.size} int16 samples but its header declares "
            f"{n_samples}."
        )

    adc_min, adc_max = _ADC_RAILS[1], _ADC_RAILS[0]
    n_clipped = n_invalid = 0
    low_rail, high_rail = _ADC_RAILS
    for offset in range(0, n_samples, _SCAN_CHUNK):
        chunk = np.asarray(dat[offset : offset + _SCAN_CHUNK])
        invalid = chunk == _ADC_INVALID
        chunk_invalid = int(invalid.sum())
        n_invalid += chunk_invalid
        # The marker is not a sample, so it must not enter the extremes; masking is
        # skipped when this chunk has none, which is every chunk in the 1.0.0 release.
        valid = chunk[~invalid] if chunk_invalid else chunk
        if valid.size:
            adc_min = min(adc_min, int(valid.min()))
            adc_max = max(adc_max, int(valid.max()))
            n_clipped += int(((valid == high_rail) | (valid == low_rail)).sum())
    del dat

    to_mv = 0.001 / gain
    return {
        "n_samples": n_samples,
        "duration_secs": n_samples / fs,
        "sampling_rate": int(fs),
        "adc_gain": gain,
        "adc_baseline": baseline,
        "signal_units": units,
        "min_mv": (adc_min - baseline) * to_mv,
        "max_mv": (adc_max - baseline) * to_mv,
        "clipped_samples": n_clipped,
        "clipped_fraction": n_clipped / n_samples if n_samples else np.nan,
        "n_invalid_samples": n_invalid,
    }


def load_subject_info(data_path: Path) -> pd.DataFrame:
    """Read ``subject-info.csv``, which is semicolon-separated and keyed by RECORD.

    Its ``ID`` column holds the six-digit *recording* name, not the three-digit
    subject, so the two extra recordings of subject 100 and the three of subject 103
    appear as separate rows carrying identical demographics. That redundancy is what
    confirms the subject grouping recovered from the record name (see
    :func:`scan_records`).
    """
    from ecgbench.labels import LabelSourceMissingError

    path = data_path / "subject-info.csv"
    if not path.exists():
        raise LabelSourceMissingError(
            f"No subject-info.csv under {data_path}. It holds the only demographics in "
            "BUT QDB (gender, age, height, weight, smoking status); get a full local "
            "copy from https://physionet.org/content/butqdb/1.0.0/"
        )
    df = pd.read_csv(path, sep=";", dtype={"ID": str})
    return df.set_index("ID")


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header, annotation file and demographic row into one frame.

    One row per ECG record, 18 of them. The accelerometer records listed alongside
    them in ``RECORDS`` are not ECG and get no row; ``acc_path`` points at each.

    The subject identifier is **recovered from the record name**, which the release
    documents: "six-digit numbers where the first three numbers are unique subject
    identifiers and the next three indicate the measurement number of this subject".
    That is checked rather than trusted — the demographics of the records sharing a
    prefix must be identical, and they are, which is how 18 records reduce to 15
    subjects (100 twice, 103 three times).
    """
    data_path = Path(data_path)
    stems = _records_file(data_path)
    info = load_subject_info(data_path)

    rows = []
    for stem in stems:
        record_id = Path(stem).name.removesuffix("_ECG")
        row: dict[str, object] = {
            "record_id": record_id,
            "subject_id": record_id[:3],
            "session_index": int(record_id[3:]),
            # wfdb takes a stem with no extension; the tree is one directory per
            # recording holding both the ECG and the ACC record.
            "signal_path": f"{record_id}/{record_id}_ECG",
            "acc_path": f"{record_id}/{record_id}_ACC",
        }

        if record_id in info.index:
            d = info.loc[record_id]
            height = float(d["Height"])
            weight = float(d["Weight"])
            row.update(
                sex=str(d["Gender"]),
                age=int(d["Age"]),
                height_cm=height,
                weight_kg=weight,
                bmi=weight / (height / 100.0) ** 2 if height else np.nan,
                smoker=bool(int(d["Smoker"])),
            )
        else:
            logger.warning("subject-info.csv has no row for record %s", record_id)
            row.update(sex="U", age=-1, height_cm=np.nan, weight_kg=np.nan,
                       bmi=np.nan, smoker=False)

        row.update(scan_signal(data_path, record_id))
        intervals = parse_annotation_csv(
            data_path / record_id / f"{record_id}_ANN.csv", int(row["n_samples"])
        )
        row.update(summarise_quality(intervals, int(row["n_samples"])))
        blocks = annotated_blocks(data_path, record_id)
        row["n_annotated_blocks"] = int(len(blocks))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)

    # The subject grouping is the whole basis of the fold partition, so verify it
    # instead of assuming the naming convention held.
    demographic = ["sex", "age", "height_cm", "weight_kg"]
    inconsistent = [
        subject
        for subject, group in df.groupby("subject_id")
        if any(group[column].nunique() > 1 for column in demographic)
    ]
    if inconsistent:
        logger.warning(
            "Records sharing a subject prefix carry different demographics for %s. The "
            "first three digits of the record name may not be the subject identifier in "
            "this release, which would make the fold grouping wrong.",
            inconsistent,
        )

    total_secs = float(df["duration_secs"].sum())
    logger.info(
        "Parsed %d ECG records from %d subjects: %.1f h of signal, %.1f h annotated "
        "(%.1f%%), of which class 1 %.1f%% / class 2 %.1f%% / class 3 %.1f%% by the "
        "expert consensus. %d records annotated end to end.",
        len(df),
        df["subject_id"].nunique(),
        total_secs / 3600,
        df["annotated_secs"].sum() / 3600,
        100 * df["annotated_secs"].sum() / total_secs,
        *(
            100 * (df[f"consensus_class{k}_fraction"] * df["annotated_samples"]).sum()
            / df["annotated_samples"].sum()
            for k in (1, 2, 3)
        ),
        int(df["fully_annotated"].sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: whether the record carries an appreciable class-3 burden.

    This is the **only** derivation of the stratification label — ``BUTQDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **What the folds have to balance here is unusable signal, because that is the
    task.** A fold with no class-3 record cannot evaluate a quality classifier's
    ability to reject signal, which is the reason this database exists. Six of the 18
    records have more than 1% of their annotated time in class 3 —
    ``105001`` 34.0%, ``122001`` 50.0%, ``114001`` 22.6%, ``124001`` 14.3%,
    ``111001`` 4.4%, ``113001`` 2.8% — and the other twelve have between 0.00% and
    0.38%. **The threshold is arbitrary anywhere in [0.4%, 2.8%]**: the release has
    a sevenfold gap there, so 1% is not a tuned value.

    Measured over the shipped files at ``random_state=42``, that puts a class-3
    record in **6 of the 10 folds — the arithmetic maximum**, since there are only
    six such records. It also happens to put the three fully-annotated records
    (88.6% of all annotated time) in three different folds, so no cross with
    annotation depth is needed.

    **The alternatives were measured, and each is worse:**

    ============================  ==========================  =========================
    Stratified on                 classes (records/subjects)  folds holding a class-3 rec
    ============================  ==========================  =========================
    ``class3_high``/``low``       12/6 records, 9/6 subjects   **6 of 10**
    sex                          10 F / 8 M, 9/6 subjects      5 of 10
    age < 45                      13/5 records, 10/5 subjects   6 of 10
    dominant consensus class      15/2/1 records               5 of 10
    annotated fully or not        3/15 records                 5 of 10
    sex x class-3 burden          all four classes < 10        raises
    ============================  ==========================  =========================

    Sex is what ``nsrdb`` and ``chfdb`` use and it is *not* better here: 15 subjects
    over 10 folds put one or two records in each fold, so the female fraction per
    fold is 0, 0.5 or 1 whatever the stratification — stratifying on sex buys
    nothing that the fold arithmetic does not already destroy. The age cut matches
    class-3 coverage only by luck and would need an arbitrary boundary (a cut at the
    subject median of 37 gives 5 of 10). Every cross of two axes raises, because
    ``StratifiedGroupKFold`` needs at least one class with ``n_folds`` records and no
    cross of 18 records has one.

    Not a clinical grouping, and not a training target. For per-record quantities use
    ``consensus_class1_fraction`` … ``consensus_class3_fraction``, and for the real
    label use :func:`quality_vector`.
    """
    out = df.copy()
    high = out["consensus_class3_fraction"] > CLASS3_HIGH_THRESHOLD
    out["stratify_class"] = np.where(high, "class3_high", "class3_low")
    logger.info(
        "Stratification classes (class-3 burden > %.0f%% of annotated time): %s; "
        "%d subjects high, %d low",
        100 * CLASS3_HIGH_THRESHOLD,
        out["stratify_class"].value_counts().to_dict(),
        out.loc[high, "subject_id"].nunique(),
        out.loc[~high, "subject_id"].nunique(),
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return BUT QDB per-record labels indexed by record id.

    **These are summaries. The label of this dataset is per sample** — use
    :func:`quality_vector` (aligned to ``ECGDataset(window=...)``) or
    :func:`load_quality_intervals`, and :func:`annotated_blocks` to find out where a
    window may legitimately go.

    Columns:

    - ``subject_id``, ``session_index`` — recovered from the six-digit record name,
      which the release documents as ``<3-digit subject><3-digit session>``, and
      checked against the demographics. 15 subjects for 18 records: subject 100 was
      recorded twice and 103 three times.
    - ``sex``, ``age``, ``height_cm``, ``weight_kg``, ``bmi``, ``smoker`` — the whole
      of ``subject-info.csv``. 9 women and 6 men; subject ages 21-83, mean 40.6,
      median 37.
    - ``annotated_samples``, ``annotated_secs``, ``annotated_fraction``,
      ``fully_annotated``, ``n_annotated_blocks`` — **read these first.** Three
      records are annotated end to end and the other 15 for 40-80 minutes of a
      24-hour recording, so 20.8% of the release carries a label at all.
    - ``consensus_class1_fraction`` … ``consensus_class3_fraction`` and the matching
      ``_secs`` — the expert consensus over **annotated** samples. Class 3 is 15.2%
      of annotated time, and 87% of that is one record (105001).
    - ``expert_1_class1_fraction`` … ``expert_3_class3_fraction`` — the same for each
      expert individually. Expert 1 is systematically stricter than the other two.
    - ``mean_expert_agreement``, ``expert_unanimous_fraction``,
      ``expert_majority_fraction``, ``consensus_matches_majority`` — how much the
      three experts actually agreed (69.4% unanimous release-wide) and the evidence
      that the fourth column triple is a majority vote of the other three.
    - ``dominant_consensus_class`` — the modal class over annotated samples, and the
      config's ``label_column``. **A reduction, not the ground truth**: it is 1 for
      15 of the 18 records, and for 15 of them it describes 40 minutes of a 24-hour
      recording. Do not train on it.
    - ``clipped_samples``, ``clipped_fraction``, ``min_mv``, ``max_mv``,
      ``n_invalid_samples``, ``adc_gain``, ``adc_baseline`` — converter behaviour.
      Every record attains both ADC rails, which is why ``amplitude_outlier`` cannot
      fire on this dataset; ``clipped_fraction`` is what measures it instead.
    - ``n_samples``, ``duration_secs``, ``sampling_rate`` — geometry. Duration is
      **not** uniform: 24.01 h to 38.65 h.
    - ``acc_path`` — the companion 3-axis accelerometer record (100 Hz, milli-g).
      ECGBench gives it no records of its own.
    - ``stratify_class`` — class-3 burden band, **for fold construction only**. See
      :func:`attach_stratify_class`.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_id")
    df.index.name = config.record_id_column
    return df
