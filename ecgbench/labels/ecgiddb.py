"""
ECG-ID labels: subject identity, demographics, session structure, ten annotated beats.

**The label of this database is the person.** It was collected to test whether a
20-second Lead I recording identifies who produced it, and the thesis reports 96%
correct identification over its 90 individuals. There is no diagnosis anywhere in
the release, no clinical assessment, and no rhythm annotation — ``subject_id`` is
the ground truth, and it is the same column as the config's
``patient_id_column``. That is deliberate, and it has a consequence stated in
:func:`load_labels` and on the dataset page: ECGBench's folds group by subject, so
they cannot be used for the identification task.

Nothing machine-readable ships except ``RECORDS``. Everything below comes from
three comment lines per ``.hea``::

    # Age: 25
    # Sex: male
    # ECG date: 07.12.2004

and from the 20 annotations in each ``.atr``. All 310 headers carry all three
lines, in that order, and all 310 parse.

**Seven things worth knowing, all verified against the files.**

**1. Record names collide across subjects.** Every ``Person_NN/`` directory
numbers its records ``rec_1``, ``rec_2``, …, contiguously from 1, so ``rec_1``
names 90 different recordings. ``record_id`` is ``"Person_01_rec_1"``;
``signal_path`` keeps the release's own ``"Person_01/rec_1"``. Anything keyed on
the bare record name silently merges 90 recordings into one.

**2. The .atr annotations stop about 40% of the way in.** Exactly 20 annotations
per record in all 310 files — 10 ``N`` R-peaks and 10 ``t`` T-peaks, strictly
alternating ``NtNtNt…`` — so they describe the first ten beats and nothing after
them. The last annotation lands at sample 2,542 to 5,869 of 10,000 (mean 4,002,
median 3,970), i.e. 5.1 s to 11.7 s into a 20.000 s record; only 18 records have
any annotation past the halfway mark. ``annotated_fraction`` records this per
record. A window past ``last_annotation_sample`` contains beats nobody marked.

**3. They are also unaudited.** The shipped ``ANNOTATORS`` file says so in one
line: "unaudited R- and T-wave peaks annotations from an automated detector". So
``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms`` and ``mean_rt_interval_ms`` are
estimates over ten machine-detected beats, and this is not a beat-detection
reference dataset. Use ``qtdb`` or ``ludb`` for that.

**4. There are only ten acquisition days in the entire release, and one of them
holds 43% of it.** 134 of the 310 records were recorded on 12 May 2005, and the
ten dates run 7 December 2004 to 24 May 2005. ``ecg_date`` is therefore a session
identifier as much as a date, which is why ``session_index`` and
``days_since_first_session`` are derived from it rather than from anything the
release states.

**5. Only 20 of the 90 subjects were recorded on more than one day.** 70 have all
their records from a single session, so the longitudinal question the database was
built to answer rests on the other 20. ``is_multi_session`` and
``session_span_days`` flag them; Person_02 is the extreme, 22 records over six
sessions spanning 156 days.

**6. Age is constant within a subject, even across 156 days.** Person_02 is 23 in
all six of its sessions. So age is a subject attribute here, not a per-record one,
and no subject's age changes mid-database — which is what makes
:func:`attach_stratify_class` safe to compute per record under subject grouping.

**7. The per-subject record count contradicts the release's own documentation.**
The README and the PhysioNet abstract both say "from 2 … to 20"; ``RECORDS`` and
the files on disk say 1 to 22, because Person_74 has one record and Person_02 has
22. Every other figure in the README reproduces exactly: 310 records, 90 subjects,
44 men and 46 women, ages 13 to 75.

Signals are not read here. The raw-versus-filtered noise level is the one
interesting per-record quantity that needs them, and
:func:`scan_noise_levels` computes it on request — see its docstring for why it is
not a label.
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

#: Newline-delimited list of ``Person_NN/rec_N`` record paths in the dataset root.
RECORDS_FILE = "RECORDS"

#: What ANNOTATORS says about the .atr files, verbatim. Exposed as a column so the
#: caveat travels with the numbers derived from them.
ANNOTATION_SOURCE = "unaudited automated detector"

#: The two annotation symbols in the release, and nothing else occurs: 3,100 of
#: each over the 310 files. ``N`` is an R-wave peak and ``t`` a T-wave peak — note
#: that ``N`` here means "the detector found a beat", not "normal beat", because
#: no beat in this database was ever classified.
R_PEAK_SYMBOL = "N"
T_PEAK_SYMBOL = "t"

#: RR intervals outside this range are dropped before any heart-rate summary. With
#: only ten beats per record this almost never fires — no interval in the release
#: falls outside it — but a re-release with a missed beat would produce a doubled
#: interval that would otherwise halve the reported rate.
RR_RANGE_SECS = (0.3, 2.0)

#: Age at or below this goes in the ``le30`` stratification cell, above it in
#: ``gt30``. See :func:`attach_stratify_class` for why 30 and not anything finer.
AGE_CUT_YEARS = 30

#: ``# Age: 25`` / ``# Sex: male`` / ``# ECG date: 07.12.2004``. All three are
#: present, in this order, in all 310 headers.
_COMMENT_RE = re.compile(r"^#\s*(?P<key>Age|Sex|ECG date)\s*:\s*(?P<value>.*?)\s*$")

#: The release writes dates day-first with dots.
_DATE_FORMAT = "%d.%m.%Y"


def parse_header_comments(hea_path: Path) -> dict[str, object]:
    """Parse the three comment lines into age, sex and recording date.

    A header whose comments do not parse comes back with NaN/empty and a warning
    rather than raising, so one malformed file cannot fail the whole scan —
    genuinely unreadable records are what the ``corrupt_header`` check is for.

    ``ecg_date`` is normalised to ISO ``YYYY-MM-DD``; ``ecg_date_raw`` keeps the
    release's own day-first ``DD.MM.YYYY`` so nothing is lost if a future release
    writes an ambiguous date.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()

    out: dict[str, object] = {
        "age": np.nan,
        "sex": "",
        "ecg_date": "",
        "ecg_date_raw": "",
    }
    found: dict[str, str] = {}
    for line in lines:
        if not line.startswith("#"):
            continue
        match = _COMMENT_RE.match(line)
        if match:
            found[match.group("key")] = match.group("value")
        else:
            logger.warning("Unparsed comment line in %s: %r", hea_path.name, line)

    if "Age" in found:
        try:
            out["age"] = float(int(found["Age"]))
        except ValueError:
            logger.warning("Non-integer age in %s: %r", hea_path.name, found["Age"])
    if "Sex" in found:
        out["sex"] = found["Sex"].strip().lower()
    if "ECG date" in found:
        raw = found["ECG date"].strip()
        out["ecg_date_raw"] = raw
        parsed = pd.to_datetime(raw, format=_DATE_FORMAT, errors="coerce")
        if pd.isna(parsed):
            logger.warning("Unparsed ECG date in %s: %r", hea_path.name, raw)
        else:
            out["ecg_date"] = parsed.strftime("%Y-%m-%d")

    missing = {"Age", "Sex", "ECG date"} - set(found)
    if missing:
        logger.warning("%s is missing header comments: %s", hea_path.name, sorted(missing))
    return out


def summarise_annotations(record_path: Path, sig_len: int, fs: float) -> dict[str, object]:
    """Summarise one record's ten annotated beats.

    Returns the peak counts, where the annotated span sits inside the record, and
    heart-rate summaries over the R-peaks. Everything here comes from an unaudited
    automatic detector covering only the first ten beats — see points 2 and 3 of
    the module docstring.
    """
    import wfdb

    out: dict[str, object] = {
        "n_annotations": 0,
        "n_r_peaks": 0,
        "n_t_peaks": 0,
        "annotation_source": ANNOTATION_SOURCE,
        "first_annotation_sample": np.nan,
        "last_annotation_sample": np.nan,
        "annotated_secs": np.nan,
        "annotated_fraction": np.nan,
        "unannotated_tail_secs": np.nan,
        "mean_hr_bpm": np.nan,
        "sdnn_ms": np.nan,
        "rmssd_ms": np.nan,
        "n_rr_used": 0,
        "n_rr_rejected": 0,
        "mean_rt_interval_ms": np.nan,
    }

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .atr must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return out

    symbols = np.asarray(annotation.symbol)
    samples = np.asarray(annotation.sample, dtype=np.int64)
    if len(symbols) == 0:
        return out

    unexpected = sorted(set(symbols) - {R_PEAK_SYMBOL, T_PEAK_SYMBOL})
    if unexpected:
        # Worth seeing rather than silently dropping: the release uses exactly two
        # symbols, so anything else means a re-release changed the annotation set.
        logger.warning("%s: unexpected annotation symbols %s", record_path.name, unexpected)

    r_peaks = np.sort(samples[symbols == R_PEAK_SYMBOL])
    t_peaks = np.sort(samples[symbols == T_PEAK_SYMBOL])

    out["n_annotations"] = int(len(symbols))
    out["n_r_peaks"] = int(len(r_peaks))
    out["n_t_peaks"] = int(len(t_peaks))
    out["first_annotation_sample"] = float(samples.min())
    out["last_annotation_sample"] = float(samples.max())
    # Measured from sample 0, not from the first annotation: what a user needs to
    # know is how much of the record the annotations reach, and every record's
    # first beat is annotated.
    out["annotated_secs"] = float(samples.max()) / fs
    if sig_len > 0:
        out["annotated_fraction"] = float(samples.max()) / float(sig_len)
        out["unannotated_tail_secs"] = float(sig_len - samples.max()) / fs

    if len(r_peaks) > 1:
        rr = np.diff(r_peaks) / fs
        low, high = RR_RANGE_SECS
        keep = (rr >= low) & (rr <= high)
        out["n_rr_rejected"] = int((~keep).sum())
        rr = rr[keep]
        out["n_rr_used"] = int(len(rr))
        if len(rr):
            out["mean_hr_bpm"] = float(60.0 / rr.mean())
            out["sdnn_ms"] = float(np.std(rr) * 1000.0)
        if len(rr) > 1:
            out["rmssd_ms"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0)

    if len(r_peaks) and len(t_peaks):
        # Each R paired with the first T after it. Not a QT interval — the
        # annotation marks the T *peak*, not its offset — so this is R-peak to
        # T-peak and is named accordingly.
        intervals = [
            (t_peaks[t_peaks > peak][0] - peak) for peak in r_peaks if (t_peaks > peak).any()
        ]
        if intervals:
            out["mean_rt_interval_ms"] = float(np.mean(intervals) / fs * 1000.0)
    return out


def _record_paths(data_path: Path) -> list[str]:
    """Read the shipped ``RECORDS`` file, or explain what is missing."""
    from ecgbench.labels import LabelSourceMissingError

    records_file = data_path / RECORDS_FILE
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. ECG-ID labels are derived from the "
            "record headers and .atr annotation files, so point data_path at the "
            "dataset root — the directory holding RECORDS, ANNOTATORS and the 90 "
            "Person_NN/ subdirectories. Get it from "
            "https://physionet.org/content/ecgiddb/1.0.0/"
        )
    return [line.strip() for line in records_file.read_text().split() if line.strip()]


def attach_session_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-subject session columns, derived from ``ecg_date``.

    The release states nothing about sessions — there is no session column and no
    session count anywhere. What it does give is a recording date per record, and
    there are only ten distinct dates in the whole database (point 4), so a
    subject's distinct dates are its sessions.

    Adds ``n_records_for_subject``, ``n_sessions_for_subject``, ``session_index``
    (1-based, in date order), ``days_since_first_session``, ``session_span_days``
    and ``is_multi_session``. 20 of the 90 subjects are multi-session; for the
    other 70 the span is 0 and the index is 1 for every record.
    """
    out = df.copy()
    dates = pd.to_datetime(out["ecg_date"], format="%Y-%m-%d", errors="coerce")
    out["n_records_for_subject"] = out.groupby("subject_id")["record_id"].transform("size")
    out["n_sessions_for_subject"] = (
        out.assign(_d=dates).groupby("subject_id")["_d"].transform("nunique")
    )
    first = out.assign(_d=dates).groupby("subject_id")["_d"].transform("min")
    last = out.assign(_d=dates).groupby("subject_id")["_d"].transform("max")
    out["days_since_first_session"] = (dates - first).dt.days
    out["session_span_days"] = (last - first).dt.days
    # Rank over the subject's distinct dates, so two records on the same day share
    # a session index.
    out["session_index"] = (
        out.assign(_d=dates).groupby("subject_id")["_d"].rank(method="dense").astype("Int64")
    )
    out["is_multi_session"] = out["n_sessions_for_subject"] > 1
    logger.info(
        "%d of %d subjects were recorded on more than one day; longest span %s days",
        int(out.groupby("subject_id")["is_multi_session"].first().sum()),
        int(out["subject_id"].nunique()),
        out["session_span_days"].max(),
    )
    return out


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: sex crossed with a single age cut at 30 years.

    This is the **only** derivation of the stratification label — ``ECGIDDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **Why a demographic cross at all.** ``subject_id`` is this database's ground
    truth and cannot balance a grouped split: every group is exactly one class, so
    there is no fold in which a subject both appears and is held out. Demographics
    are what remains, and both axes need balancing. Neither works alone:

    - Stratifying on **sex** alone leaves four of the ten folds with no subject
      over 45 in them, because age is skewed — 45 of the 90 subjects are 21-30 and
      only 11 are over 45.
    - Stratifying on **age bands** alone leaves the sex balance to chance, and here
      chance is not benign: group sizes run 1 to 22 records, so one subject can
      swing a fold, and one fold comes out 9 female / 19 male.

    **Why the cut is at 30 and nothing finer.** A class must contain at least
    ``n_folds`` subjects to appear in every fold. With the cut at 30 the four cells
    hold 36, 27, 17 and 10 subjects (``female_le30`` 124 records, ``male_le30``
    99, ``male_gt30`` 55, ``female_gt30`` 32) — the smallest is exactly 10, which
    is the floor for 10 folds. Crossing sex with four age bands instead produces
    5-subject cells, which cannot be spread over ten folds and leaves empty cells
    in the fold table.

    **What was rejected.** Heart rate from the ``.atr`` annotations: it is not a
    subject attribute here — the thesis deliberately did not restrict physical or
    emotional state, and within-record RR SD runs to 393 ms — so it cannot balance
    a split whose groups are subjects, and it comes from an unaudited detector over
    ten beats. Session count (``is_multi_session``, 20 subjects against 70) does
    balance and was considered for exactly the reason ``stdb`` crosses in its
    channel count: whether a fold contains multi-session subjects decides whether
    it can support a within-subject evaluation. It loses to the demographic cross
    because stratifying on it leaves sex at 6 female / 21 male in one fold and age
    with five empty cells, a worse joint outcome. ``is_multi_session`` is exposed
    as a label so a user who needs it can filter or re-split.

    This is a fold-construction device, not a clinical grouping, and must not be
    trained on as one.
    """
    out = df.copy()
    age_cell = np.where(out["age"] <= AGE_CUT_YEARS, f"le{AGE_CUT_YEARS}", f"gt{AGE_CUT_YEARS}")
    # An unparsed age would silently land in gt30 above, so say so instead.
    unknown = out["age"].isna()
    if unknown.any():
        logger.warning(
            "%d records have no parsed age; their stratify_class is age_unknown",
            int(unknown.sum()),
        )
        age_cell = np.where(unknown, "age_unknown", age_cell)
    out["stratify_class"] = out["sex"].astype(str) + "_" + age_cell
    counts = out["stratify_class"].value_counts()
    logger.info("Stratification classes (sex x age cut at %d): %s", AGE_CUT_YEARS, counts.to_dict())
    subjects = out.groupby("stratify_class")["subject_id"].nunique()
    smallest = int(subjects.min()) if len(subjects) else 0
    if smallest < 10:
        # StratifiedGroupKFold cannot put a class into every fold with fewer
        # subjects than folds, and its own message names neither the config nor the
        # column. Say it here.
        logger.warning(
            "Smallest stratification class holds %d subjects, fewer than the 10 folds "
            "ECGBench generates; some folds will contain none of it. Widen the cells in "
            "ecgbench.labels.ecgiddb.attach_stratify_class.",
            smallest,
        )
    return out


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so
    a stray file cannot enter the partition and the 90 subject directories do not
    have to be walked.
    """
    import wfdb

    data_path = Path(data_path)
    paths = _record_paths(data_path)

    rows: list[dict[str, object]] = []
    for rel in paths:
        subject_id, _, record_name = rel.partition("/")
        if not record_name:
            logger.warning("RECORDS entry %r has no subject directory, skipped", rel)
            continue

        hea = data_path / f"{rel}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", rel, hea)
            continue

        header = wfdb.rdheader(str(data_path / rel))
        sig_len = int(header.sig_len)
        fs = float(header.fs)

        row: dict[str, object] = {
            # "Person_01_rec_1". The bare record name is not unique — see point 1.
            "record_id": f"{subject_id}_{record_name}",
            "subject_id": subject_id,
            "record_name": record_name,
        }
        # "Person_01" -> 1 and "rec_1" -> 1, for anyone who wants to sort or index
        # numerically. Kept alongside the strings, never instead of them.
        row["subject_number"] = _trailing_int(subject_id)
        row["record_number"] = _trailing_int(record_name)

        row["n_channels"] = int(header.n_sig)
        row["n_samples"] = sig_len
        row["duration_secs"] = sig_len / fs
        row["sampling_rate"] = fs
        # "ECG I|ECG I filtered" in all 310 records: one physical lead stored twice,
        # raw first. Kept because a re-release that changes it would show up here.
        row["signal_descriptions"] = "|".join(header.sig_name or [])
        gains = list(header.adc_gain or [])
        row["adc_gain"] = float(gains[0]) if gains else np.nan

        row.update(parse_header_comments(hea))
        row.update(summarise_annotations(data_path / rel, sig_len, fs))
        # The release's own path, which is what wfdb is handed.
        row["signal_path"] = rel
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["subject_number", "record_number"]).reset_index(drop=True)
    df = attach_session_structure(df)
    logger.info(
        "Parsed %d records from %d subjects (%d records each at most): %.1f min of signal, "
        "%d annotated beats over %d records",
        len(df),
        df["subject_id"].nunique(),
        int(df["n_records_for_subject"].max()),
        df["duration_secs"].sum() / 60,
        int(df["n_r_peaks"].sum()),
        len(df),
    )
    return df


def _trailing_int(value: str) -> int | float:
    """``"Person_01"`` -> 1, ``"rec_12"`` -> 12; NaN if there is no trailing number."""
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else np.nan


def scan_noise_levels(data_path: Path | str) -> pd.DataFrame:
    """Per-record noise level, as the difference between the two stored channels.

    **Not part of :func:`load_labels`, on purpose.** Every other column in this
    module comes from a header or an annotation file, so ``load_labels`` reads no
    signal and ``ecgbench splits`` does not read the waveforms twice. This one
    needs all 310 records decoded, which is the only reason it is separate — at
    12.5 MB it takes seconds, so call it directly when you want it.

    ``raw - filtered`` is what the thesis's preprocessing removed: baseline drift,
    50 Hz power-line interference and high-frequency noise. The subtraction is
    meaningful because the two channels are sample-aligned — peak cross-correlation
    between them is at lag 0 in 273 of the 310 records and at ±1 sample in the rest
    — so the residual is noise rather than a phase difference.

    Measured on v1.0.0 (all files verified against the release's own
    ``SHA256SUMS.txt``): ``removed_rms_mv`` has a median of 0.187 mV and a 90th
    percentile of 0.562 mV, and then a long tail — 2.214 mV at the 99th percentile
    and 41.948 mV for Person_76/rec_2, whose raw channel drifts to -154.155 mV.
    Correlation between the two channels runs 0.103 to 0.978, median 0.691; a low
    value means drift dominates the raw channel, not that the filter failed.

    Returns a frame indexed by ``record_id`` with ``removed_rms_mv``,
    ``removed_ptp_mv``, ``raw_filtered_corr`` and the per-channel extremes.
    """
    import wfdb

    data_path = Path(data_path)
    rows = []
    for rel in _record_paths(data_path):
        subject_id, _, record_name = rel.partition("/")
        record = wfdb.rdrecord(str(data_path / rel))
        signal = np.asarray(record.p_signal, dtype=np.float64)
        if signal.shape[1] < 2:
            logger.warning("%s holds %d channel(s), so no residual", rel, signal.shape[1])
            continue
        raw, filtered = signal[:, 0], signal[:, 1]
        removed = raw - filtered
        rows.append(
            {
                "record_id": f"{subject_id}_{record_name}",
                "removed_rms_mv": float(np.sqrt(np.mean(removed**2))),
                "removed_ptp_mv": float(np.ptp(removed)),
                "raw_filtered_corr": float(np.corrcoef(raw, filtered)[0, 1]),
                "raw_min_mv": float(raw.min()),
                "raw_max_mv": float(raw.max()),
                "filtered_min_mv": float(filtered.min()),
                "filtered_max_mv": float(filtered.max()),
            }
        )
    return pd.DataFrame(rows).set_index("record_id")


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return ECG-ID labels indexed by ``record_id``.

    **The label is the subject.** ``subject_id`` is this database's ground truth —
    the thesis's task is 90-class identification — and it is also the config's
    ``patient_id_column``. Two consequences follow, and both matter:

    - **ECGBench's folds cannot be used for the identification task.** Folds group
      by subject, so a subject's records are all in one fold and no fold's model
      has ever seen the person it is asked to recognise. That is the right default
      for every other use of these recordings and the wrong one for this one. For
      identification, split *within* subject — for instance by
      ``session_index``, holding out a subject's later sessions — using the 20
      multi-session subjects flagged by ``is_multi_session``.
    - **There is no diagnosis to predict.** No clinical assessment was collected,
      so the cohort can be assumed neither healthy nor unhealthy.

    Columns:

    - ``subject_id``, ``subject_number``, ``record_name``, ``record_number`` —
      identity. ``record_name`` is **not unique**: ``rec_1`` names 90 different
      recordings, which is why ``record_id`` is ``"Person_01_rec_1"``.
    - ``age``, ``sex``, ``ecg_date``, ``ecg_date_raw`` — the three header comment
      lines, which are the entire shipped metadata. Ages 13 to 75, median 23; 46
      female and 44 male subjects (156 and 154 records). Age is constant within a
      subject even across 156 days, so it is a subject attribute.
    - ``n_records_for_subject``, ``n_sessions_for_subject``, ``session_index``,
      ``days_since_first_session``, ``session_span_days``, ``is_multi_session`` —
      the session structure, derived from ``ecg_date`` by
      :func:`attach_session_structure`. **1 to 22 records per subject**, against
      the README's "2 to 20"; 20 of the 90 subjects span more than one day.
    - ``n_annotations``, ``n_r_peaks``, ``n_t_peaks``, ``annotation_source`` — 20,
      10 and 10 in every record, from an unaudited automatic detector.
    - ``first_annotation_sample``, ``last_annotation_sample``, ``annotated_secs``,
      ``annotated_fraction``, ``unannotated_tail_secs`` — **where the annotations
      stop**: at 25.4% to 58.7% of the record, mean 40.0%. Beats after that are
      unmarked in every record.
    - ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, ``n_rr_used``, ``n_rr_rejected``,
      ``mean_rt_interval_ms`` — over the nine RR intervals between the ten detected
      R-peaks. Heart rate runs 50.0 to 132.7 bpm, mean 78.5. These describe ten
      beats of a 20-second recording from a detector nobody audited; they are not
      an HRV result, and ``mean_rt_interval_ms`` is R-peak to T-*peak*, not a QT
      interval.
    - ``n_samples``, ``duration_secs``, ``sampling_rate``, ``n_channels``,
      ``signal_descriptions``, ``adc_gain`` — record geometry, identical in all 310
      records: 10,000 samples, 20.000 s, 500 Hz, 2 channels named
      ``ECG I|ECG I filtered``, gain 200 adu/mV.
    - ``signal_path`` — the release's own ``Person_01/rec_1``.
    - ``stratify_class`` — for fold construction only. See
      :func:`attach_stratify_class`.

    The raw-versus-filtered noise level is not here because it needs the waveforms;
    :func:`scan_noise_levels` computes it on request.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_id")
    df.index.name = config.record_id_column
    return df
