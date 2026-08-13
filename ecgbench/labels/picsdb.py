"""
Preterm Infant Cardio-Respiratory Signals labels: bradycardia onsets, R peaks, HRV.

Nothing tabular ships with this database. There is no metadata file of any kind —
just 10 ECG records, 10 respiration records, ``RECORDS``, ``ANNOTATORS`` and
``SHA256SUMS.txt``. Every column below is derived from a header or an annotation
file, and the whole of it verifies against the release's own checksums.

**THE PRODUCT OF THIS DATABASE IS AN EVENT TIME, NOT A DIAGNOSIS.** Every one of
the ten infants is a preterm infant in the same NICU, so ``cohort_label`` is one
value for all ten records and there is nothing to classify at the record level.
What is worth having is the 622 manually validated **bradycardia onsets** in the
``.atr`` files and the 3,797,503 manually verified **R peaks** in the ``.qrsc``
files, both of which are *time series* — use :func:`bradycardia_onsets` and
:func:`rpeaks`, which align to the same ``window=`` ``ECGDataset`` takes.

Six things to know, all verified against the files.

**1. The bradycardia onset marks the R peak that opens the first long RR
interval, one sample late.** The release defines a bradycardia as heart rate
below 100 bpm (RR > 0.6 s) for at least two beats (> 1.2 s), with successive
events inside a 3-minute window aggregated. Measured against the ``.qrsc`` peaks:
**526 of 622 onsets sit within two samples** of the beat opening the first RR
interval longer than 0.6 s, and 493 of those are exactly one sample *after* it —
which is why only 32 of 622 coincide with a ``.qrsc`` sample and a naive
``np.isin(onsets, rpeaks)`` finds almost nothing. The other 96 follow it by up to
10 s, and every one of the 622 is within 10 s of one. Observed spacing between
consecutive onsets in a record is 369 s at the shortest, comfortably above the
3-minute aggregation window.

**2. Two of the ten records are 250 Hz, so a window in samples is not a window in
time.** infant1 and infant5 are the "compound" recordings and sample at 250 Hz;
the other eight are 500 Hz. ``window=(0, 15000)`` is 30 s of infant2 and 60 s of
infant5. The rate is a per-record property, not a choice — ``sampling_rate`` is
exposed here per record, and ``picsdb.yaml`` keys its single path column on the
nominal 500 Hz.

**3. Every record clips at the 16-bit converter rail, and two of them do it for
tens of minutes.** All ten touch both ±32767, but for eight it is a handful of
samples. infant5 sits at the negative rail for **422,773 samples (1,691 s)** and
infant1 for 160,567 (642 s) — 0.96% and 0.39% of those recordings, at −40.96 mV,
which is nothing an infant heart did. ``rail_secs`` and ``rail_fraction`` report
it. The good news is that WFDB's invalid-sample marker (−32768) appears nowhere,
so no record produces NaN.

**4. There are long stretches of perfectly constant signal, and whole-record
variance cannot see them.** Every record carries between 239 s and 3,147 s of
signal in constant runs of a second or more, the longest single run being 1,456 s
(24 minutes) in infant5. ECGBench's ``flat_line`` check tests variance over the
*whole* record, which these 20-70 hour recordings pass easily, so all ten are
valid and ``clean`` equals ``original`` — read ``flat_secs``, ``flat_fraction``
and ``longest_flat_secs`` before choosing a window, not the validation report.

**5. R-peak annotation does not cover the whole recording.** Coverage runs 94.0%
to 99.9%. infant10 has a **7,667 s (2.13 h) unannotated tail** and 49 internal
gaps longer than 10 s; infant5 opens with 1,631 s of unannotated signal and
infant2 with 409 s. ``annotated_fraction``, ``annotated_head_secs``,
``annotated_tail_secs`` and ``annotation_gap_secs`` report it per record. The HRV
summaries filter RR intervals to :data:`RR_RANGE_SECS`, which is what keeps those
multi-minute "intervals" out of them.

**6. The respiration signal is a companion record, not a lead of the ECG one.**
``resp_path`` points at it; it is 50 Hz for nine infants and 500 Hz for infant1,
and its ``.resp`` peaks are **algorithmic and were never manually vetted**, unlike
the R peaks and the bradycardia onsets. Its length also does not always match the
ECG: infant5's respiration record is 1.0 h *longer* than its ECG, and infant10's
is the same length while its ECG annotation stops 2.1 h early. ECGBench splits and
validates the ECG records only.

No per-record demographics ship. The release states the cohort — 10 infants of
post-conceptional age 29 3/7 to 34 2/7 weeks and study weight 843 to 2,100 g —
without saying which infant is which, so no age or weight is attached to any row.
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

#: Annotator extensions, per the shipped ANNOTATORS file. ``atr`` and ``qrsc``
#: belong to the ECG record, ``resp`` to the respiration record beside it.
BRADYCARDIA_ANNOTATOR = "atr"
RPEAK_ANNOTATOR = "qrsc"
RESPIRATION_ANNOTATOR = "resp"

#: The one thing this release asserts about every subject, and the value of
#: ``cohort_label`` for all 10 records. It is not derived from anything in the
#: files — there is no diagnosis, no outcome and no control group here.
COHORT_LABEL = "preterm_infant"

#: 16-bit converter rails. −32768 is WFDB's invalid-sample marker, which
#: ``wfdb.rdrecord`` turns into NaN; it appears in no record of this release, and
#: ``n_invalid_samples`` exists so a re-release that introduces one is visible
#: rather than failing ``nan_values`` with no explanation.
_ADC_RAILS = (32767, -32767)
_ADC_INVALID = -32768

#: Chunk size for the memory-mapped signal scan. The release is 1.58 billion
#: samples and infant9 alone is 126.6 million, so a float64 ``p_signal`` for it
#: would be 1.0 GB — the scan reads int16 instead and never materialises one.
_SCAN_CHUNK = 8_000_000

#: RR intervals outside this range are dropped before any HRV summary. The upper
#: bound is what keeps the unannotated gaps of point 5 — which appear as single
#: intervals of up to 973 s — out of the statistics; the lower one drops double
#: detections. Both are wider than an adult filter would be: these infants run at
#: 130-167 bpm, so a normal RR here is 0.36-0.46 s and a bradycardia is > 0.6 s.
RR_RANGE_SECS = (0.2, 3.0)

#: An RR interval longer than this is read as a gap in the annotation rather than
#: a very slow heart: the longest bradycardia the release describes is a few
#: seconds, and the intervals this catches run to 973 s.
ANNOTATION_GAP_SECS = 10.0

#: The bradycardia definition the release states, kept here because
#: :func:`verify_bradycardia_onsets` measures the shipped onsets against it.
BRADYCARDIA_RR_SECS = 0.6


def _records_file(data_path: Path) -> list[str]:
    """ECG record stems, from the shipped ``RECORDS`` file rather than a glob.

    ``RECORDS`` lists 20 names — an ``_ecg`` and a ``_resp`` record per infant.
    Only the ECG half gets a row: the respiration records are not ECG, carry no
    beat annotation and must not enter the partition. Each one is reachable from
    its infant's row through ``resp_path``.
    """
    from ecgbench.labels import LabelSourceMissingError

    path = data_path / "RECORDS"
    if not path.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. Preterm Infant Cardio-Respiratory "
            "labels live entirely in the record headers and the .atr/.qrsc "
            "annotation files, so point data_path at the dataset root — the flat "
            "directory holding infant1_ecg.hea, RECORDS and ANNOTATORS. Get it "
            "from https://physionet.org/content/picsdb/1.0.0/"
        )

    names = [line.strip() for line in path.read_text().split() if line.strip()]
    ecg = [name for name in names if name.endswith("_ecg")]
    if not ecg:
        raise LabelSourceMissingError(
            f"RECORDS under {data_path} names no *_ecg records: {names[:5]}"
        )
    # infant1 .. infant10, not infant1, infant10, infant2 — the fold assignment is
    # a pure function of this order, so it must not depend on string sorting.
    return sorted(ecg, key=lambda name: int(name.removeprefix("infant").removesuffix("_ecg")))


def _flat_and_rail_stats(dat: np.memmap, fs: float) -> dict[str, object]:
    """Constant-run and converter-rail statistics, in one chunked pass.

    Runs are tracked across chunk boundaries — the longest constant run in the
    release is 728,100 samples, which no sensible chunk size contains — by
    carrying the final run of each chunk forward and merging it if the next chunk
    opens on the same value.
    """
    high_rail, low_rail = _ADC_RAILS
    n_rail = n_invalid = 0
    adc_min, adc_max = low_rail, high_rail
    longest = 0
    longest_value = 0
    flat_samples = 0  # samples inside a constant run of >= 1 s
    min_run = int(round(fs))

    carry_value: int | None = None
    carry_len = 0

    def account(values: np.ndarray, lengths: np.ndarray) -> None:
        nonlocal longest, longest_value, flat_samples
        if lengths.size == 0:
            return
        keep = lengths >= min_run
        flat_samples += int(lengths[keep].sum())
        top = int(np.argmax(lengths))
        if int(lengths[top]) > longest:
            longest = int(lengths[top])
            longest_value = int(values[top])

    for offset in range(0, dat.size, _SCAN_CHUNK):
        chunk = np.asarray(dat[offset : offset + _SCAN_CHUNK], dtype=np.int32)
        if chunk.size == 0:
            continue

        invalid = chunk == _ADC_INVALID
        n_chunk_invalid = int(invalid.sum())
        n_invalid += n_chunk_invalid
        # The marker is not a sample, so it must not enter the extremes.
        valid = chunk[~invalid] if n_chunk_invalid else chunk
        if valid.size:
            adc_min = min(adc_min, int(valid.min()))
            adc_max = max(adc_max, int(valid.max()))
            n_rail += int(((valid == high_rail) | (valid == low_rail)).sum())

        starts = np.concatenate(([0], np.flatnonzero(np.diff(chunk)) + 1))
        ends = np.concatenate((starts[1:], [chunk.size]))
        values = chunk[starts]
        lengths = ends - starts

        if carry_value is not None and int(values[0]) == carry_value:
            lengths = lengths.copy()
            lengths[0] += carry_len
        elif carry_value is not None:
            account(np.array([carry_value]), np.array([carry_len]))

        # The last run may continue into the next chunk, so it is carried rather
        # than counted here.
        account(values[:-1], lengths[:-1])
        carry_value, carry_len = int(values[-1]), int(lengths[-1])

    if carry_value is not None:
        account(np.array([carry_value]), np.array([carry_len]))

    return {
        "adc_min": adc_min,
        "adc_max": adc_max,
        "n_rail_samples": n_rail,
        "n_invalid_samples": n_invalid,
        "longest_flat_samples": longest,
        "longest_flat_value_adu": longest_value,
        "flat_samples": flat_samples,
    }


def scan_signal(data_path: Path, record_name: str) -> dict[str, object]:
    """Measure one ECG record's geometry, physical span, clipping and flat runs.

    Reads the ``.dat`` as memory-mapped ``int16`` and applies the header's own
    ``(sample - baseline) / gain`` conversion rather than going through
    ``wfdb.rdrecord``, which would build a 1.0 GB float64 array for infant9.

    ``rail_low_mv`` / ``rail_high_mv`` are where the converter saturates *for this
    record*: gain and baseline differ per record, so the same ±32767 is a
    different pair of millivolt values in each. Their union across the ten is what
    ``amplitude_range_mv`` in ``picsdb.yaml`` has to cover.
    """
    import wfdb

    header = wfdb.rdheader(str(data_path / record_name))
    fs = float(header.fs)
    n_samples = int(header.sig_len)
    gain = float(header.adc_gain[0])
    baseline = float(header.baseline[0])

    dat = np.memmap(data_path / f"{record_name}.dat", dtype="<i2", mode="r")
    if dat.size != n_samples:
        raise ValueError(
            f"{record_name}.dat holds {dat.size} int16 samples but its header "
            f"declares {n_samples}."
        )
    stats = _flat_and_rail_stats(dat, fs)
    del dat

    high_rail, low_rail = _ADC_RAILS
    n_rail = int(stats["n_rail_samples"])
    flat_samples = int(stats["flat_samples"])
    longest = int(stats["longest_flat_samples"])
    return {
        "n_samples": n_samples,
        "duration_secs": n_samples / fs,
        "sampling_rate": int(fs),
        "lead_name": (header.sig_name or [""])[0],
        "adc_gain": gain,
        "adc_baseline": baseline,
        "min_mv": (int(stats["adc_min"]) - baseline) / gain,
        "max_mv": (int(stats["adc_max"]) - baseline) / gain,
        "rail_low_mv": (low_rail - baseline) / gain,
        "rail_high_mv": (high_rail - baseline) / gain,
        "n_rail_samples": n_rail,
        "rail_secs": n_rail / fs,
        "rail_fraction": n_rail / n_samples if n_samples else np.nan,
        "n_invalid_samples": int(stats["n_invalid_samples"]),
        "flat_secs": flat_samples / fs,
        "flat_fraction": flat_samples / n_samples if n_samples else np.nan,
        "longest_flat_secs": longest / fs,
        "longest_flat_value_adu": int(stats["longest_flat_value_adu"]),
    }


def summarise_annotations(data_path: Path, record_name: str, n_samples: int, fs: float) -> dict:
    """Summarise one record's bradycardia onsets, R peaks and HRV.

    The onset times are pipe-joined into ``bradycardia_onsets_secs`` rather than
    left in a list, because the metadata CSV and the fold pipeline are tabular.
    97 values is the most any record carries; :func:`bradycardia_onsets` is the
    array form, and it is the one to use with ``window=``.
    """
    import wfdb

    out: dict[str, object] = {
        "n_bradycardias": 0,
        "bradycardias_per_hour": np.nan,
        "bradycardia_onsets_secs": "",
        "first_bradycardia_secs": np.nan,
        "last_bradycardia_secs": np.nan,
        "min_interevent_secs": np.nan,
        "n_rpeaks": 0,
        "annotated_head_secs": np.nan,
        "annotated_tail_secs": np.nan,
        "annotation_gap_secs": 0.0,
        "n_annotation_gaps": 0,
        "annotated_fraction": np.nan,
        "mean_hr_bpm": np.nan,
        "sdnn_ms": np.nan,
        "rmssd_ms": np.nan,
        "n_rr_rejected": 0,
    }

    try:
        brady = wfdb.rdann(str(data_path / record_name), BRADYCARDIA_ANNOTATOR)
    except Exception as e:  # a missing .atr must not kill the whole scan
        logger.warning("Could not read %s.%s: %s", record_name, BRADYCARDIA_ANNOTATOR, e)
        brady = None

    if brady is not None and len(brady.sample):
        onsets = np.asarray(brady.sample, dtype=np.int64)
        symbols = set(brady.symbol)
        # Every onset in the 1.0.0 release is "[". The symbol is WFDB's
        # start-of-ventricular-flutter marker being reused as a generic episode
        # start; it does not mean flutter, and nothing here should read it as a
        # beat. Warn rather than silently reinterpreting if that ever changes.
        if symbols - {"["}:
            logger.warning(
                "%s.atr uses annotation symbols beyond '[': %s — all are still "
                "counted as bradycardia onsets",
                record_name,
                sorted(symbols),
            )
        secs = onsets / fs
        gaps = np.diff(secs)
        out.update(
            n_bradycardias=len(onsets),
            bradycardias_per_hour=len(onsets) / (n_samples / fs / 3600.0),
            bradycardia_onsets_secs="|".join(f"{s:.3f}" for s in secs),
            first_bradycardia_secs=float(secs[0]),
            last_bradycardia_secs=float(secs[-1]),
            min_interevent_secs=float(gaps.min()) if gaps.size else np.nan,
        )

    try:
        peaks = wfdb.rdann(str(data_path / record_name), RPEAK_ANNOTATOR)
    except Exception as e:  # pragma: no cover - every shipped record has one
        logger.warning("Could not read %s.%s: %s", record_name, RPEAK_ANNOTATOR, e)
        return out

    q = np.asarray(peaks.sample, dtype=np.int64)
    if q.size < 3:
        return out

    rr = np.diff(q) / fs
    gap_mask = rr > ANNOTATION_GAP_SECS
    head = float(q[0] / fs)
    tail = float((n_samples - q[-1]) / fs)
    gap_secs = float(rr[gap_mask].sum())
    duration = n_samples / fs
    out.update(
        n_rpeaks=int(q.size),
        annotated_head_secs=head,
        annotated_tail_secs=tail,
        annotation_gap_secs=gap_secs,
        n_annotation_gaps=int(gap_mask.sum()),
        annotated_fraction=(duration - head - tail - gap_secs) / duration,
    )

    low, high = RR_RANGE_SECS
    keep = (rr >= low) & (rr <= high)
    out["n_rr_rejected"] = int((~keep).sum())
    rr = rr[keep]
    if rr.size > 1:
        out.update(
            mean_hr_bpm=float(60.0 / rr.mean()),
            sdnn_ms=float(1000.0 * rr.std(ddof=1)),
            rmssd_ms=float(1000.0 * np.sqrt(np.mean(np.diff(rr) ** 2))),
        )
    return out


def summarise_respiration(data_path: Path, record_name: str) -> dict[str, object]:
    """Geometry and peak count of the companion respiration record.

    Reported, never validated or split: it is not an ECG. Its peaks are the one
    annotation layer in this release the authors describe as **not** manually
    vetted.
    """
    import wfdb

    resp_name = record_name.removesuffix("_ecg") + "_resp"
    out: dict[str, object] = {
        "resp_path": resp_name,
        "resp_sampling_rate": -1,
        "resp_n_samples": 0,
        "resp_duration_secs": np.nan,
        "n_resp_peaks": 0,
    }
    try:
        header = wfdb.rdheader(str(data_path / resp_name))
    except Exception as e:
        logger.warning("No respiration record for %s (%s)", record_name, e)
        out["resp_path"] = ""
        return out

    out.update(
        resp_sampling_rate=int(header.fs),
        resp_n_samples=int(header.sig_len),
        resp_duration_secs=header.sig_len / float(header.fs),
    )
    try:
        peaks = wfdb.rdann(str(data_path / resp_name), RESPIRATION_ANNOTATOR)
        out["n_resp_peaks"] = len(peaks.sample)
    except Exception as e:  # pragma: no cover - every shipped record has one
        logger.warning("Could not read %s.%s: %s", resp_name, RESPIRATION_ANNOTATOR, e)
    return out


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header, signal and annotation file into one frame, one row per ECG record.

    Reads all 1.58 billion samples once, to count converter clipping and constant
    runs that no header states and no ECGBench check can see. That takes a minute
    or two; ``PICSDBSplitter`` caches the result as ``ecgbench_metadata.csv`` so it
    happens once per dataset copy.
    """
    data_path = Path(data_path)
    rows = []
    for name in _records_file(data_path):
        row: dict[str, object] = {
            "record_name": name,
            # One record per infant, so this is 1:1 — but it is set rather than
            # left null so the split is grouped and stays correct if a future
            # release adds a second recording for an infant.
            "subject_id": name.removesuffix("_ecg"),
            # Flat tree: wfdb takes the stem, no extension and no subdirectory.
            "signal_path": name,
        }
        row.update(scan_signal(data_path, name))
        row.update(
            summarise_annotations(
                data_path, name, int(row["n_samples"]), float(row["sampling_rate"])
            )
        )
        row.update(summarise_respiration(data_path, name))
        # Asserted by the release of the whole cohort; nothing in the files
        # derives it, and every record carries the same value.
        row["cohort_label"] = COHORT_LABEL
        rows.append(row)

    df = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(
        "Parsed %d records from %d infants: %.1f h of ECG, %d bradycardia onsets, "
        "%d verified R peaks; %.1f h clipped at a converter rail and %.1f h in "
        "constant runs of a second or more",
        len(df),
        df["subject_id"].nunique(),
        df["duration_secs"].sum() / 3600,
        int(df["n_bradycardias"].sum()),
        int(df["n_rpeaks"].sum()),
        df["rail_secs"].sum() / 3600,
        df["flat_secs"].sum() / 3600,
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``, and explain why it is a constant.

    This is the **only** derivation of the stratification label — ``PICSDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **Ten records from ten infants over ten folds leaves nothing to balance.**
    Each fold is exactly one infant whatever the label says, and every non-constant
    axis fails arithmetically before it can even be tried: ``StratifiedGroupKFold``
    raises ``n_splits=10 cannot be greater than the number of members in each
    class`` unless every class holds ten records, which over ten records means one
    class. Measured, so that this is a fact rather than an argument — a median cut
    on bradycardia rate (0.85-1.83/h) gives 5/5 and raises; sampling rate gives 8/2
    and raises; lead name gives 7/2/1 and raises. A constant reduces
    ``StratifiedGroupKFold`` to a plain partition of the ten infants, which is the
    leave-one-infant-out structure this database wants anyway.

    Do not read the fold layout as balanced on anything. For real per-record
    quantities use ``bradycardias_per_hour``, ``mean_hr_bpm``, ``sdnn_ms``,
    ``rmssd_ms`` or the coverage columns.
    """
    out = df.copy()
    out["stratify_class"] = COHORT_LABEL
    logger.info(
        "Stratification class is constant (%s) over %d records; folds are a plain "
        "partition of %d infants",
        COHORT_LABEL,
        len(out),
        out["subject_id"].nunique() if "subject_id" in out else 0,
    )
    return out


def _annotation_samples(
    data_path: Path | str,
    record_name: str,
    annotator: str,
    start: int = 0,
    length: int | None = None,
) -> np.ndarray:
    """Annotation sample indices, re-based on ``start`` and clipped to ``length``."""
    import wfdb

    data_path = Path(data_path)
    ann = wfdb.rdann(str(data_path / record_name), annotator)
    samples = np.asarray(ann.sample, dtype=np.int64)
    if start:
        samples = samples[samples >= start]
    if length is not None:
        samples = samples[samples < start + length]
    return samples - start


def bradycardia_onsets(
    data_path: Path | str,
    record_name: str,
    start: int = 0,
    length: int | None = None,
) -> np.ndarray:
    """Bradycardia onset sample indices for one record, aligned to ``window=``.

    ``start`` and ``length`` are in samples and mean exactly what they mean in
    ``ECGDataset(window=(start, length))``, so::

        ds = ECGDataset("picsdb", split="train", window=(0, 15_000), data_path=...)
        onsets = bradycardia_onsets(data_path, ds[0]["record_id"], 0, 15_000)

    indexes the returned tensor directly. **Remember that samples are not seconds
    here**: eight records run at 500 Hz and infant1 and infant5 at 250 Hz.

    The onsets are manually validated, and each marks the R peak that opens the
    first RR interval longer than 0.6 s — one sample *after* it, in 493 of 622
    cases. See :func:`verify_bradycardia_onsets`.
    """
    return _annotation_samples(
        data_path, record_name, BRADYCARDIA_ANNOTATOR, start, length
    )


def rpeaks(
    data_path: Path | str,
    record_name: str,
    start: int = 0,
    length: int | None = None,
) -> np.ndarray:
    """R-peak sample indices for one record, aligned to ``window=``.

    3,797,503 peaks across the ten records, from a modified Pan-Tompkins detector
    with visual inspection for artefact removal — the release describes these as
    verified, unlike the respiration peaks.

    They do **not** cover the whole recording: infant10 stops 2.13 h before the end
    of its signal and infant5 starts 1,631 s in. An empty return therefore means
    "not annotated here", not "no beats"; check ``annotated_head_secs``,
    ``annotated_tail_secs`` and ``annotation_gap_secs`` in :func:`load_labels`.
    """
    return _annotation_samples(data_path, record_name, RPEAK_ANNOTATOR, start, length)


def respiration_peaks(
    data_path: Path | str,
    record_name: str,
    start: int = 0,
    length: int | None = None,
) -> np.ndarray:
    """Respiration peak indices for a record's companion ``*_resp`` record.

    ``start`` and ``length`` are in samples of the **respiration** record, which is
    50 Hz for nine infants and 500 Hz for infant1 — they are not the ECG's samples,
    and converting between the two needs both rates. These peaks were extracted
    algorithmically and, unlike the R peaks and bradycardia onsets, were never
    manually vetted.
    """
    resp_name = record_name.removesuffix("_ecg") + "_resp"
    return _annotation_samples(
        data_path, resp_name, RESPIRATION_ANNOTATOR, start, length
    )


def verify_bradycardia_onsets(data_path: Path | str, tolerance_secs: float = 10.0) -> pd.DataFrame:
    """Measure the shipped onsets against the release's own bradycardia definition.

    For every ``.atr`` onset, finds the nearest R peak that opens an RR interval
    longer than :data:`BRADYCARDIA_RR_SECS` and reports ``onset - peak`` in
    samples, so **+1 means the onset sits one sample after that peak**. Over the
    1.0.0 release: 526 of 622 onsets land within two samples of one, 493 of those
    exactly one sample after it, 32 exactly on it, and all 622 within 10 s — which
    is the evidence for the reading in :func:`bradycardia_onsets`. Reads only
    annotation files, so it is quick, but it is not run at load time.
    """
    data_path = Path(data_path)
    import wfdb

    rows = []
    for name in _records_file(data_path):
        fs = float(wfdb.rdheader(str(data_path / name)).fs)
        q = np.asarray(wfdb.rdann(str(data_path / name), RPEAK_ANNOTATOR).sample, dtype=np.int64)
        a = np.asarray(
            wfdb.rdann(str(data_path / name), BRADYCARDIA_ANNOTATOR).sample, dtype=np.int64
        )
        long_openers = q[:-1][(np.diff(q) / fs) > BRADYCARDIA_RR_SECS]
        offsets = []
        for onset in a:
            if long_openers.size == 0:  # pragma: no cover - no such record ships
                offsets.append(np.nan)
                continue
            delta = onset - long_openers
            offsets.append(int(delta[np.argmin(np.abs(delta))]))
        offsets = np.asarray(offsets, dtype=float)
        rows.append(
            {
                "record_name": name,
                "sampling_rate": int(fs),
                "n_bradycardias": len(a),
                "n_within_2_samples": int(np.sum(np.abs(offsets) <= 2)),
                "n_one_sample_after_rpeak": int(np.sum(offsets == 1)),
                "n_on_an_rpeak": int(np.sum(offsets == 0)),
                "n_within_tolerance": int(np.sum(np.abs(offsets) <= tolerance_secs * fs)),
                "median_offset_samples": float(np.nanmedian(offsets)),
                "max_offset_secs": float(np.nanmax(np.abs(offsets)) / fs),
            }
        )
    return pd.DataFrame(rows)


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Preterm Infant Cardio-Respiratory labels indexed by record name.

    Columns:

    - ``cohort_label`` — ``"preterm_infant"`` for **all 10 records**, asserted by
      the release of the cohort. There is no diagnosis, no outcome and no control
      group in this database; it is an event-detection dataset, not a
      classification task.
    - ``n_bradycardias``, ``bradycardias_per_hour``, ``bradycardia_onsets_secs``
      (pipe-joined), ``first_bradycardia_secs``, ``last_bradycardia_secs``,
      ``min_interevent_secs`` — the 622 manually validated onsets, 28 to 97 per
      record. :func:`bradycardia_onsets` is the array form and the one to use with
      ``window=``.
    - ``n_rpeaks``, ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, ``n_rr_rejected`` —
      3,797,503 verified R peaks and whole-record HRV summaries over RR intervals
      in :data:`RR_RANGE_SECS`. Descriptive of a 20-70 hour recording, not a
      segmented HRV analysis.
    - ``annotated_head_secs``, ``annotated_tail_secs``, ``annotation_gap_secs``,
      ``n_annotation_gaps``, ``annotated_fraction`` — **read these before choosing
      a window for supervised work.** Coverage is 94.0%-99.9%, and infant10's last
      2.13 h carry no R peaks at all.
    - ``rail_secs``, ``rail_fraction``, ``n_rail_samples``, ``min_mv``, ``max_mv``,
      ``rail_low_mv``, ``rail_high_mv``, ``n_invalid_samples`` — converter clipping.
      infant5 spends 1,691 s and infant1 642 s pinned at −40.96 mV.
    - ``flat_secs``, ``flat_fraction``, ``longest_flat_secs``,
      ``longest_flat_value_adu`` — time in constant runs of a second or more, which
      whole-record variance cannot detect. The longest single run is 1,456 s.
    - ``sampling_rate`` — **250 for infant1 and infant5, 500 for the other eight.**
      A window in samples is a different length of time in each.
    - ``lead_name`` — ``"II"`` in seven records, ``"ECG"`` in infant1 and infant5,
      ``"I"`` in infant10. One channel either way.
    - ``resp_path``, ``resp_sampling_rate``, ``resp_n_samples``,
      ``resp_duration_secs``, ``n_resp_peaks`` — the companion respiration record.
      Not an ECG, so it is never split or validated; its peaks were not manually
      vetted.
    - ``n_samples``, ``duration_secs``, ``adc_gain``, ``adc_baseline`` — record
      geometry. Duration runs 20.34 h to 70.32 h; nothing here is uniform.
    - ``stratify_class`` — a constant, **for fold construction only**. See
      :func:`attach_stratify_class`.

    No demographics ship per record. The release states the cohort's
    post-conceptional age (29 3/7 to 34 2/7 weeks) and study weight (843-2,100 g)
    without saying which infant is which, so nothing here attaches either.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
