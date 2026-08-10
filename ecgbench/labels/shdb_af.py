"""
SHDB-AF labels: the clinical table, and the beat-level rhythm annotations.

Unusually for a Holter database, this one has **both** layers. ``AdditionalData.csv``
carries 45 columns of clinical and demographic data per recording — diagnosis,
comorbidities, medications, echo measurements, ablation history — and the ``.atr``
files carry rhythm marks at beat resolution for 98 of the 128 recordings. So the
clinical table is the label source for anything record-level and the annotations
are the label source for anything time-resolved, and this module returns both,
joined on the recording id.

Everything below was verified against the shipped files, and all 488 of them
verify against the release's own ``SHA256SUMS.txt``.

**1. THE ANNOTATION FILES DO NOT LOOK LIKE ANY OTHER WFDB DATABASE IN THIS
CATALOGUE, AND CODE WRITTEN FOR ``ltafdb`` OR ``afdb`` READS THEM AS EMPTY.**
Everywhere else, a rhythm episode is a ``+`` annotation whose ``aux_note`` names
the rhythm, and beats carry typed symbols (``N``, ``A``, ``V``, ...). Here **every
single annotation is a ``"`` comment** — 10,349,733 of them across the 98 files,
not one ``+`` and not one typed beat — and the rhythm code rides in the
``aux_note`` of the ``"`` that sits on the **first beat of the interval**::

    sample   symbol   aux_note
    1595     "        (AFIB      <- interval starts here
    1675     "                   <- same interval, no note
    1756     "
    ...

A reader filtering on ``symbol == "+"`` therefore finds zero episodes, and one
filtering on ``symbol in ("N", "A", "V", "Q")`` finds zero beats, in a database
that has 10.3 million of the latter. :func:`summarise_annotations` forward-fills
the marks instead, which is the only way to get a per-beat rhythm out of this
layout.

**2. THE ``.atr`` AND ``.qrs`` FILES HOLD THE SAME BEAT POSITIONS, EXACTLY.** All
98 pairs are identical, sample for sample, so ``.atr`` is ``.qrs`` plus the rhythm
marks rather than an independently detected beat set. Both are machine output:
the positions come from the ``epltd`` implementation of Pan-Tompkins, and only the
rhythm *labels* were placed by a human. That matters for how much the beat
positions can be trusted — nothing in this release audited them — and it means
``n_beats`` and ``n_detections`` are the same number and neither is a verified
beat count. Contrast ``ltafdb``, where ``.atr`` is manually verified and ``.qrs``
is a separate unaudited detector.

**3. ONLY SUPRAVENTRICULAR ARRHYTHMIA WAS ANNOTATED. ``(N`` MEANS "NOT LABELLED",
NOT "SINUS RHYTHM".** The release states it plainly — "Normal sinus rhythm and
other rhythms were not annotated" — so ``(N`` is a residual class holding sinus
rhythm, ventricular ectopy, pauses, noise and everything else the protocol did
not cover. **Do not train a sinus-rhythm detector on ``(N``**, and do not read
``rhythm_secs_N`` as sinus time. This is the single most important difference from
``ltafdb``, whose nine codes include ventricular bigeminy, trigeminy, VT, SVT,
idioventricular rhythm and sinus bradycardia as positive classes.

**4. THERE IS A SIXTH RHYTHM CODE IN THE FILES THAT THE RELEASE DOES NOT
DOCUMENT: ``(AB``.** The landing page and the shipped ``README.md`` both list five
categories (``AFIB``, ``AFL``, ``AT``, ``PAT``/``NOD``, ``N``). ``(AB`` — atrial
bigeminy in the standard WFDB rhythm vocabulary — appears in **3 intervals across
2 records** (047 once, 051 twice) covering 5,021 beats and 3,674.2 s, and it appears
in no row of the release's published beat table. It is carried in
:data:`RHYTHM_NAMES` and counted like any other code, because dropping an
annotation for want of documentation would silently reassign those beats.

**And the published beat table does not reproduce, for a reason worth knowing
before you chase it.** Recomputed over the 98 annotated records, three of its five
rows match exactly — ``AFL`` 195,659 beats in 45 intervals, ``AT`` 48,800 in 57,
``PAT``+``NOD`` 4,416 in 9 — while ``N`` comes out 170,276 beats short of the
published 7,812,308 and ``AFIB`` 59,154 short of 2,512,959, with 794 ``AFIB``
intervals against a published 809. The shortfall is 224,409 beats, which is **2.12
times the mean 105,610 beats per record**: two records' worth. v1.0.1 withdrew
exactly two annotated records, 016 and 030, as duplicates, and the table was
evidently not regenerated afterwards — it describes v1.0.0's 100 annotated
recordings. The same explanation covers the interval-duration summary, which gives
a 2.5 s minimum and a 47.5 s median (IQR 17.0-270.25) where the shipped files give
0.630 s and 58.91 s (IQR 16.52-500.76), with 50 intervals under the stated 2.5 s
floor. So a recomputation that differs from the landing page in ``N``, ``AFIB`` and
the duration quartiles — and only those — is right rather than broken.

**5. RECORDS 005 AND 020 ARE THE SAME RECORDING, ANNOTATED TWICE.** ``005.dat``
and ``020.dat`` have the **same SHA-256 in the release's own ``SHA256SUMS.txt``**,
and so do ``005.qrs`` and ``020.qrs``; only the ``.atr`` differs. The clinical
table presents them as two separate Holters three years apart — ``Age_at_Holter``
47 and 50, ``Date_Holter`` 2021-05-21 and 2024-02-21 — so nothing except the
checksum reveals it. v1.0.1's release notes say duplicates 016 and 030 were
removed; this pair was missed. **The release holds 127 distinct recordings, not
128.** Two consequences:

- ECGBench's fold assignment is *not* leaky because of it, and that is luck rather
  than design: both rows carry ``Subject_ID`` 4899921, and folds are grouped on
  ``Subject_ID``, so the two copies always land in the same fold. ``duplicate_of``
  below names the partner anyway, so anyone computing a per-record metric can drop
  one.
- The two ``.atr`` files are an accidental **annotation-repeatability sample**: the
  same 17 marks and the same 9 ``N`` / 8 ``AFIB`` episode structure, with three
  beats moved across boundaries (94,795/7,207 beats against 94,798/7,204) and an
  AF burden of 0.03640 against 0.03637. That is the only estimate of annotator
  reproducibility this database offers.

**6. THE PUBLISHED DEMOGRAPHIC SUMMARY MIXES RECORDING-LEVEL AND SUBJECT-LEVEL
COUNTS, AND ITS AGE IS NOT REPRODUCIBLE.** The landing page's table gives "Female:
47 (38.5%), Male: 75 (61.4%)", which sums to the 122 *subjects*, but 47 is the
number of *recordings* from female patients (per-subject it is 44). Age is given as
68.0 +/- 11.3 where the file yields 65.8 +/- 12.1 per recording and 65.7 +/- 12.0
per subject; nothing in the release accounts for the 2.2-year difference. Stroke is
given as 11.7% where 19/128 is 14.8%. Recompute from the columns rather than
quoting the table — the per-recording figures are what this module returns.

**7. THE SHIPPED ``README.md`` IS THE v1.0.0 ONE AND ITS COLUMN NAMES ARE WRONG.**
It documents ``<Study ID>``, ``<UID>``, ``<Height (m)>`` and "127 unique patients";
the file that actually ships has ``Subject_ID``, ``Data_ID``, ``Height`` and 122
subjects. It also says the ``.dat`` files carry ``base_year`` and ``base_time``
fields, which v1.0.1 moved out of the headers into ``Holter_start_time`` — every
shipped header is exactly three lines with no timestamp at all. Follow the landing
page, not the README.

**8. THE RECORD LIST IS ``RECORDS.txt``, NOT ``RECORDS``.** Every other WFDB
release in this catalogue ships an extensionless ``RECORDS``. Code that globs for
it finds nothing here and silently falls back to whatever ``*.hea`` files are on
disk — which for a 7.7 GB release is how a partial download passes for a smaller
database. :func:`scan_records` reads ``RECORDS.txt`` and raises if it is absent.

**9. THE CLINICAL TABLE'S BOOLEAN COLUMNS SHIP IN TWO DIFFERENT TYPES.**
``Annotated``, ``Previously_Documented_AFL``, ``Previous_AF_Ablation``,
``PPM_on_Holter``, ``PPM_after_Holter``, ``HTN``, ``Age_75_or_Older``, ``DM``,
``Stroke`` and ``Vascular_Diseases`` are the strings ``True``/``False``, while
``CHF``, ``Moderate_or_Severe_MR``/``TR``/``AS``/``AR``, ``Ablation1_PVI`` and
``Ablation1_CTI`` are the floats ``1.0``/``0.0``/blank for the same kind of yes/no
fact. :data:`NUMERIC_FLAG_COLUMNS` are converted to pandas' nullable ``boolean``
so a user can write ``df["CHF"] & df["HTN"]`` without checking which encoding each
column happened to get. Nothing else is coerced.

**10. DATES ARE SHIFTED, SO DATE ARITHMETIC ACROSS COLUMNS IS NOT SAFE.** The
release shifted every subject-linked date "by at least one year" for
de-identification, which is why ``Date_Holter`` reaches 2024-02-21 for a cohort
collected to May 2023. The shift is per subject rather than global, so a difference
between two dates *within* one row may survive it and a comparison *between* rows
does not. ``AF_Duration_Months`` is the release's own pre-shift interval and is the
column to use for time since diagnosis.
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

#: The rhythm annotator: beat positions from ``epltd``, rhythm marks by hand.
#: Present for 98 of the 128 records.
RHYTHM_ANNOTATOR = "atr"

#: The R-peak detector, present for all 128 records. Holds the *same* positions as
#: ``.atr`` where both exist — see point 2 of the module docstring — so for the 30
#: unannotated records this is the only annotation layer there is.
DETECTOR_ANNOTATOR = "qrs"

#: The one annotation symbol in this database. Every annotation in every ``.atr``
#: is a ``"`` comment, and every annotation in every ``.qrs`` is an ``N``.
COMMENT_SYMBOL = '"'
DETECTION_SYMBOL = "N"

#: Rhythm codes as they appear in ``aux_note``, without the leading ``(``.
#:
#: ``AB`` is in the files and in no documentation — see point 4 above. ``N`` is a
#: residual "not labelled" class, **not** sinus rhythm — see point 3.
RHYTHM_NAMES = {
    "N": "not annotated (sinus rhythm and any other rhythm outside the protocol)",
    "AFIB": "atrial fibrillation",
    "AFL": "atrial flutter",
    "AT": "atrial tachycardia",
    "PAT": "other supraventricular tachycardia, e.g. Wolff-Parkinson-White",
    "NOD": "intranodal (AV-nodal) tachycardia",
    "AB": "atrial bigeminy (present in the files, absent from the release's docs)",
}

#: Codes counted as AF for ``af_burden``. AFIB alone, so the figure means the same
#: thing as ``afdb``'s and ``ltafdb``'s. Flutter *is* annotated here (it is not in
#: ``ltafdb``), so it gets its own ``afl_burden`` rather than being folded in —
#: they are different arrhythmias with different treatment implications, and a
#: combined figure cannot be taken apart again.
AF_CODES = ("AFIB",)

#: Every code the annotation protocol treated as a positive finding: the four
#: categories the release names, plus the undocumented ``AB``. The complement of
#: ``N``, and the denominator-free view of "how much of this record is abnormal".
SVT_CODES = ("AFIB", "AFL", "AT", "PAT", "NOD", "AB")

#: The residual class. Named so nothing has to spell the string twice.
UNLABELLED_CODE = "N"

#: Clinical table, and the record list. Note the ``.txt`` — see point 8.
CLINICAL_CSV = "AdditionalData.csv"
RECORDS_FILE = "RECORDS.txt"

#: Yes/no columns the source ships as ``1.0``/``0.0``/blank floats rather than
#: ``True``/``False`` strings. Converted to nullable ``boolean``; see point 9.
NUMERIC_FLAG_COLUMNS = (
    "CHF",
    "Moderate_or_Severe_MR",
    "Moderate_or_Severe_TR",
    "Moderate_or_Severe_AS",
    "Moderate_or_Severe_AR",
    "Ablation1_PVI",
    "Ablation1_CTI",
)

#: The one duplicated pair, keyed both ways. Established from the release's own
#: ``SHA256SUMS.txt`` rather than inferred — see point 5.
DUPLICATE_RECORDINGS = {"005": "020", "020": "005"}

#: AF-burden cut points for ``af_class``, the same as ``afdb``'s and ``ltafdb``'s
#: so the three are comparable. Below the first, AF is incidental; at or above the
#: second, the record is in AF throughout.
MINIMAL_AF_BURDEN = 0.05
SUSTAINED_AF_BURDEN = 0.95

AF_MINIMAL = "minimal"
AF_PAROXYSMAL = "paroxysmal"
AF_SUSTAINED = "sustained"

#: What ``af_class`` says for a record with no ``.atr``. A distinct string rather
#: than NaN, because "we did not measure the AF burden of this recording" is not
#: the same claim as "its AF burden is zero" and the 30 unannotated records
#: include 11 with a PAF diagnosis.
AF_UNANNOTATED = "unannotated"

#: The clinical diagnosis, per recording, from ``AF_Type``. Verbatim source values.
AF_TYPE_VALUES = ("PAF", "PerAF", "non-AF")


def read_header(hea_path: Path) -> dict[str, object]:
    """Read the signal specification out of one three-line header.

    Deliberately not ``wfdb.rdheader``: this is a scan over 128 files that must not
    stop because one of them is unreadable.

    Every header here is exactly three lines and carries **no** timestamp — v1.0.1
    moved the start time into the clinical table's ``Holter_start_time`` column —
    and no comment lines, so there is nothing clinical to parse out of it. The two
    signal descriptions are always ``ECG1`` and ``ECG2``, which the release does
    document as modified CC5 and NASA respectively.

    ``adc_gains`` and ``adc_baselines`` are worth having per record: both vary,
    over 236 distinct gains across the 256 channels, because each channel was
    scaled to fill the 16-bit range. See the config for why that makes the
    millivolt rail record-dependent.
    """
    lines = [
        line
        for line in hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    out: dict[str, object] = {
        "n_leads": 0,
        "n_samples": 0,
        "sampling_rate": 200,
        "lead_names": "",
        "adc_gains": "",
        "adc_baselines": "",
    }
    if not lines:
        logger.warning("Empty header: %s", hea_path.name)
        return out

    fields = lines[0].split()
    if len(fields) >= 4:
        out["n_leads"] = int(fields[1])
        out["sampling_rate"] = int(float(fields[2]))
        out["n_samples"] = int(fields[3])

    out["lead_names"] = "|".join(line.split()[-1] for line in lines[1:])
    gains, baselines = [], []
    for line in lines[1:]:
        # Field 3 of a signal line is "<gain>(<baseline>)/<units>". The baseline is
        # not optional here: it is nonzero on all 256 channels, so a reader that
        # drops it is off by up to 19,701 adu.
        spec = line.split()[2].split("/")[0]
        if "(" in spec:
            gain, baseline = spec.split("(")
            baselines.append(baseline.rstrip(")"))
        else:
            gain, _ = spec, baselines.append("0")
        gains.append(gain)
    out["adc_gains"] = "|".join(gains)
    out["adc_baselines"] = "|".join(baselines)
    return out


def summarise_annotations(record_path: Path, span_samples: int) -> dict[str, object]:
    """Summarise one record's ``.atr`` rhythm marks.

    The layout this has to cope with is described in point 1 of the module
    docstring: every annotation is a ``"``, and only the first beat of each
    interval carries a code in ``aux_note``. So the rhythm of a beat is the most
    recent mark at or before it, and this function forward-fills.

    ``span_samples`` closes the final interval, which has no mark after it. Using
    the record length is PhysioNet's own convention (it is what ``ltafdb``'s
    published summary tables do) and it matters more here than there: the last
    interval is often the longest, and 18 of the 98 records place a mark on their
    very last annotated beat, which without this would be a zero-second interval.

    Returns beat counts, seconds and interval counts per rhythm code, the dominant
    rhythm, AF burden measured two ways, and where the annotations stop.
    """
    import wfdb

    out: dict[str, object] = {f"beats_{code}": 0 for code in RHYTHM_NAMES}
    out.update({f"rhythm_secs_{code}": 0.0 for code in RHYTHM_NAMES})
    out.update({f"n_episodes_{code}": 0 for code in RHYTHM_NAMES})
    out.update(
        {
            "n_beats": 0,
            "n_rhythm_marks": 0,
            "rhythm_annotated_secs": 0.0,
            "rhythms": "",
            "dominant_rhythm": "",
            "dominant_rhythm_fraction": np.nan,
            "af_burden": np.nan,
            "af_beat_fraction": np.nan,
            "afl_burden": np.nan,
            "svt_burden": np.nan,
            "longest_af_episode_secs": np.nan,
            "first_mark_sample": -1,
            "last_beat_sample": -1,
            "annotated_secs": np.nan,
            "unannotated_tail_secs": np.nan,
            "mean_heart_rate_bpm": np.nan,
        }
    )

    try:
        annotation = wfdb.rdann(str(record_path), RHYTHM_ANNOTATOR)
    except Exception as e:  # one unreadable .atr must not kill the whole scan
        logger.warning("Could not read %s.%s: %s", record_path.name, RHYTHM_ANNOTATOR, e)
        return out

    fs = float(getattr(annotation, "fs", 200) or 200)
    samples = np.asarray(annotation.sample, dtype=np.int64)
    if samples.size == 0:
        logger.warning("%s.atr holds no annotations", record_path.name)
        return out

    unexpected = {s for s in annotation.symbol if s != COMMENT_SYMBOL}
    if unexpected:
        # Not an error — it would mean a future release switched to the ordinary
        # WFDB layout, which is worth knowing about rather than silently absorbing.
        logger.warning(
            "%s.atr carries symbols other than %r: %s. This module forward-fills "
            "aux_note over every annotation; a release using '+' rhythm changes "
            "and typed beats needs summarise_annotations revisited.",
            record_path.name,
            COMMENT_SYMBOL,
            sorted(unexpected),
        )

    notes = np.array(
        [str(aux).strip("\x00").strip() for aux in annotation.aux_note], dtype=object
    )
    mark_positions = np.flatnonzero(notes != "")
    out["n_beats"] = int(samples.size)
    out["n_rhythm_marks"] = int(mark_positions.size)
    out["last_beat_sample"] = int(samples[-1])
    out["annotated_secs"] = float(samples[-1] / fs)
    out["unannotated_tail_secs"] = max(0.0, (span_samples - samples[-1]) / fs)

    if mark_positions.size == 0:
        logger.warning("%s.atr has annotations but no rhythm marks", record_path.name)
        return out

    out["first_mark_sample"] = int(samples[mark_positions[0]])
    codes = [str(note).lstrip("(") for note in notes[mark_positions]]
    # A mark's interval ends where the next mark's beat starts; the last one ends
    # at the end of the record.
    ends = np.append(mark_positions[1:], samples.size)

    longest_af = 0.0
    for code, start_index, end_index in zip(codes, mark_positions, ends):
        if code not in RHYTHM_NAMES:
            logger.warning("%s: unknown rhythm code %r", record_path.name, code)
            continue
        end_sample = int(samples[end_index]) if end_index < samples.size else int(span_samples)
        seconds = max(0.0, (end_sample - int(samples[start_index])) / fs)
        out[f"rhythm_secs_{code}"] = float(out[f"rhythm_secs_{code}"]) + seconds
        out[f"n_episodes_{code}"] = int(out[f"n_episodes_{code}"]) + 1
        out[f"beats_{code}"] = int(out[f"beats_{code}"]) + int(end_index - start_index)
        if code in AF_CODES:
            longest_af = max(longest_af, seconds)

    seconds_by_code = {
        code: float(out[f"rhythm_secs_{code}"])
        for code in RHYTHM_NAMES
        if float(out[f"rhythm_secs_{code}"]) > 0
    }
    total = sum(seconds_by_code.values())
    # Stated explicitly rather than left to be inferred: it is the record length
    # minus the lead-in before the first detected beat, roughly 8 s per record,
    # NOT the record length.
    out["rhythm_annotated_secs"] = float(total)
    if total > 0:
        ordered = sorted(seconds_by_code, key=lambda code: -seconds_by_code[code])
        out["rhythms"] = "|".join(ordered)
        out["dominant_rhythm"] = ordered[0]
        out["dominant_rhythm_fraction"] = seconds_by_code[ordered[0]] / total
        out["af_burden"] = sum(seconds_by_code.get(c, 0.0) for c in AF_CODES) / total
        out["afl_burden"] = seconds_by_code.get("AFL", 0.0) / total
        out["svt_burden"] = sum(seconds_by_code.get(c, 0.0) for c in SVT_CODES) / total
        out["longest_af_episode_secs"] = longest_af
    if out["n_beats"]:
        # The beat-based counterpart of af_burden, and the quantity the release's
        # own published table counts. It is not the same number: AF episodes run
        # faster than the rest of the record, so the beat fraction exceeds the time
        # fraction in every AF record here.
        out["af_beat_fraction"] = sum(
            int(out[f"beats_{c}"]) for c in AF_CODES
        ) / float(out["n_beats"])

    intervals = np.diff(samples) / fs
    intervals = intervals[(intervals > 0.2) & (intervals < 2.5)]  # drop detector glitches
    if intervals.size:
        out["mean_heart_rate_bpm"] = float(60.0 / intervals.mean())
    return out


def summarise_detector(record_path: Path, span_samples: int) -> dict[str, object]:
    """Summarise the ``.qrs`` detections — the only annotation layer 30 records have.

    These are the same positions the ``.atr`` file holds where one exists, so this
    exists mainly for the unannotated 30, and for the mean heart rate and coverage
    of a record whose rhythm was never labelled.
    """
    import wfdb

    out: dict[str, object] = {
        "n_detections": 0,
        "detector_last_sample": -1,
        "detector_unannotated_tail_secs": np.nan,
        "detector_mean_heart_rate_bpm": np.nan,
    }

    try:
        detections = wfdb.rdann(str(record_path), DETECTOR_ANNOTATOR)
    except Exception as e:
        logger.warning("Could not read %s.%s: %s", record_path.name, DETECTOR_ANNOTATOR, e)
        return out

    samples = np.asarray(detections.sample, dtype=np.int64)
    unexpected = {s for s in detections.symbol if s != DETECTION_SYMBOL}
    if unexpected:
        logger.warning(
            "%s.qrs carries symbols other than %r: %s",
            record_path.name,
            DETECTION_SYMBOL,
            sorted(unexpected),
        )
    out["n_detections"] = int(samples.size)
    if samples.size:
        fs = float(getattr(detections, "fs", 200) or 200)
        out["detector_last_sample"] = int(samples[-1])
        out["detector_unannotated_tail_secs"] = max(0.0, (span_samples - samples[-1]) / fs)
        intervals = np.diff(samples) / fs
        intervals = intervals[(intervals > 0.2) & (intervals < 2.5)]
        if intervals.size:
            out["detector_mean_heart_rate_bpm"] = float(60.0 / intervals.mean())
    return out


def load_clinical(data_path: Path | str) -> pd.DataFrame:
    """Read ``AdditionalData.csv``, one row per recording, indexed by ``Data_ID``.

    ``Data_ID`` and ``Subject_ID`` are read as strings. ``Data_ID`` has to be:
    the ids are zero-padded three-digit numbers (``001``, not ``1``) and pandas'
    default inference destroys the padding, after which the id no longer names a
    record and ``data_path / "1"`` is not a file. ``Subject_ID`` happens to have no
    leading zeros today, and is read as a string anyway so a future release cannot
    change that quietly.

    The only value conversion is :data:`NUMERIC_FLAG_COLUMNS` to nullable
    ``boolean`` — see point 9 of the module docstring. Free-text columns are
    returned verbatim, inconsistencies included: ``Echo_LV_Asynergy`` mixes
    ``anteroseptal`` and ``Anteroseptal``, and ``Antiarrhythmic_Drug_nonBB``
    contains at least one misspelt drug name.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    csv_path = data_path / CLINICAL_CSV
    if not csv_path.exists():
        raise LabelSourceMissingError(
            f"SHDB-AF clinical labels come from {CLINICAL_CSV}, which is not in "
            f"{data_path}. ECGBench publishes fold CSVs only — labels stay with the "
            "source dataset, so point data_path at a full local copy from "
            "https://physionet.org/content/shdb-af/1.0.1/"
        )

    df = pd.read_csv(csv_path, dtype={"Data_ID": "str", "Subject_ID": "str"})
    missing = [c for c in ("Data_ID", "Subject_ID", "Annotated") if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing {missing}. v1.0.0 of this release named these "
            "columns 'Study ID' and 'UID' and had no Subject_ID at all — if that is "
            "what you have, get v1.0.1: the older file cannot express the six "
            "subjects who contributed two recordings each, so patient-grouped folds "
            "are impossible from it."
        )

    for column in NUMERIC_FLAG_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("boolean")

    return df.set_index("Data_ID")


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS.txt`` rather than a glob, so
    the answer does not depend on what happens to be on disk. That is not a
    theoretical concern for a 7.7 GB release: a partial download is
    indistinguishable from a smaller database, and the copy this config was written
    against was missing ten records and every metadata file when it was first
    examined.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / RECORDS_FILE
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No {RECORDS_FILE} under {data_path}. Note the '.txt' — this release "
            "does not ship the extensionless RECORDS file every other WFDB database "
            "in this catalogue does. Point data_path at the dataset root, the flat "
            f"directory holding 001.hea, {CLINICAL_CSV} and {RECORDS_FILE}, from "
            "https://physionet.org/content/shdb-af/1.0.1/"
        )

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("%s names %s but %s is missing", RECORDS_FILE, name, hea.name)
            continue

        row: dict[str, object] = {"Data_ID": name}
        header = read_header(hea)
        row.update(header)

        n_samples = int(header["n_samples"])
        rate = float(header["sampling_rate"]) or 200.0
        row["record_seconds"] = n_samples / rate
        row["record_hours"] = row["record_seconds"] / 3600.0

        stem = hea.with_suffix("")
        row["has_rhythm_annotation"] = (data_path / f"{name}.{RHYTHM_ANNOTATOR}").exists()
        row.update(summarise_detector(stem, n_samples))
        if row["has_rhythm_annotation"]:
            row.update(summarise_annotations(stem, n_samples))
        else:
            # Placeholders with the same keys, so the frame has one schema whether
            # or not a record was annotated and a groupby does not silently drop
            # the unannotated 30.
            row.update({f"beats_{c}": pd.NA for c in RHYTHM_NAMES})
            row.update({f"rhythm_secs_{c}": np.nan for c in RHYTHM_NAMES})
            row.update({f"n_episodes_{c}": pd.NA for c in RHYTHM_NAMES})
            row.update(
                {
                    "n_beats": row.get("n_detections", 0),
                    "n_rhythm_marks": 0,
                    "rhythm_annotated_secs": np.nan,
                    "rhythms": "",
                    "dominant_rhythm": "",
                    "dominant_rhythm_fraction": np.nan,
                    "af_burden": np.nan,
                    "af_beat_fraction": np.nan,
                    "afl_burden": np.nan,
                    "svt_burden": np.nan,
                    "longest_af_episode_secs": np.nan,
                    "first_mark_sample": -1,
                    "last_beat_sample": row.get("detector_last_sample", -1),
                    "annotated_secs": np.nan,
                    "unannotated_tail_secs": row.get("detector_unannotated_tail_secs", np.nan),
                    "mean_heart_rate_bpm": row.get("detector_mean_heart_rate_bpm", np.nan),
                }
            )

        row["duplicate_of"] = DUPLICATE_RECORDINGS.get(name, "")
        # Flat tree: wfdb takes the bare zero-padded stem, no extension.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort on the id as a STRING, which is what it is. Sorting numerically would
    # assert the ids are numbers, the same mistake zero_padded_identifiers exists
    # to prevent.
    df = df.sort_values("Data_ID").reset_index(drop=True)
    logger.info(
        "Parsed %d SHDB-AF records: %.1f h of signal, %d beat detections, "
        "%d with rhythm annotations",
        len(df),
        df["record_hours"].sum(),
        int(df["n_detections"].sum()),
        int(df["has_rhythm_annotation"].sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``af_class`` and the fold label ``stratify_class``.

    Two different quantities, and the distinction is the point:

    ``af_class`` bins the measured ``af_burden`` into ``minimal`` (AF under 5% of
    the annotated time), ``paroxysmal`` (5-95%) and ``sustained`` (95% or more) —
    the same cuts ``afdb`` and ``ltafdb`` use, so the three are comparable. It is
    **not** usable as a fold label here, because 30 records have no ``.atr`` and
    therefore no burden to bin; they get :data:`AF_UNANNOTATED`.

    ``stratify_class`` is built from ``AF_Type`` — the release's own clinical
    diagnosis, present for all 128 recordings — crossed with whether the recording
    was annotated, because those are the two axes a user will slice on and an
    unlucky fold that held no annotated persistent-AF record would be useless for
    either. ``StratifiedGroupKFold`` needs enough members per class to spread over
    ten folds, so the cross is taken only where it can afford to be:

    ===================  =====
    ``PAF+annotated``       69
    ``PAF+unannotated``     11
    ``PerAF``               15
    ``non-AF+annotated``    20
    ``non-AF+unannotated``  13
    ===================  =====

    ``PerAF`` stays whole because its own cross is 9 annotated and 6 unannotated,
    and a class of 6 cannot be spread over ten folds. This is the **only**
    derivation of the fold label — ``SHDBAFSplitter`` reads the column rather than
    recomputing it, so the exposed label and the fold label cannot drift.

    Neither column is a sample-level label. For anything time-resolved use the
    ``.atr`` marks directly; ``af_burden`` is a whole-record summary and
    ``AF_Type`` is a clinical diagnosis rather than a measurement. The two agree
    better than one might expect and not perfectly: **none** of the 20 annotated
    ``non-AF`` recordings carries a single second of annotated AFIB, and 68 of the 69
    annotated ``PAF`` ones do — the exception being record 107, which is also the
    shortest recording in the release at 9.00 h. Flutter cuts across the diagnosis:
    10 ``PAF`` and 2 ``PerAF`` recordings carry annotated ``AFL``. Note too that
    ``PerAF`` is a diagnosis and not a burden: its 9 annotated recordings have a
    median AF burden of 0.023, lower than ``PAF``'s 0.116, because the diagnosis
    describes the subject's history rather than this particular day.
    """
    out = df.copy()

    burden = out["af_burden"]
    out["af_class"] = np.select(
        [burden.isna(), burden < MINIMAL_AF_BURDEN, burden >= SUSTAINED_AF_BURDEN],
        [AF_UNANNOTATED, AF_MINIMAL, AF_SUSTAINED],
        default=AF_PAROXYSMAL,
    )

    annotated = out["has_rhythm_annotation"].astype(bool)
    af_type = out["AF_Type"].astype(str)
    suffix = np.where(annotated, "+annotated", "+unannotated")
    out["stratify_class"] = np.where(af_type == "PerAF", af_type, af_type + suffix)

    logger.info("af_class: %s", out["af_class"].value_counts().to_dict())
    logger.info("stratify_class: %s", out["stratify_class"].value_counts().to_dict())
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return SHDB-AF labels indexed by ``Data_ID``.

    Two layers joined on the recording id.

    **Clinical, from ``AdditionalData.csv``** — all 45 source columns verbatim,
    except the seven in :data:`NUMERIC_FLAG_COLUMNS` which become nullable
    booleans. The ones most likely to be wanted:

    - ``AF_Type`` — the post-Holter diagnosis, and the closest thing this release
      has to a record-level ground truth: ``PAF`` (80), ``PerAF`` (15), ``non-AF``
      (33). Counts are per *recording*; see point 6 of the module docstring before
      quoting any demographic figure.
    - ``Annotated`` — whether a ``.atr`` ships. Cross-checked against the files by
      :func:`load_labels` and reported as ``has_rhythm_annotation``.
    - ``Subject_ID`` — 122 distinct values over 128 recordings. Six subjects
      contributed two recordings each, which is why folds are grouped on it.
    - ``Age_at_Holter``, ``Sex``, ``Height``, ``Weight``, ``BMI``,
      ``Holter_start_time``, ``Holter_recording_length``, ``Indication_Holter``.
    - History and comorbidity: ``Previously_Documented_AFL``,
      ``Previous_AF_Ablation``, ``AF_Duration_Months``, ``CHF``, ``HTN``, ``DM``,
      ``Stroke``, ``Vascular_Diseases``, ``Age_75_or_Older``.
    - Treatment: ``Antiarrhythmic_Drug_nonBB``, ``Antiarrhythmic_Drug_BB``,
      ``Anticoagulation``, the ``Ablation1_*`` and ``PPM_*`` columns.
    - Echo: ``Echo_LAD``, ``Echo_LVEF``, ``Echo_LV_Asynergy``, and the four
      ``Moderate_or_Severe_*`` valve flags.

    **Annotation-derived, from ``.atr`` and ``.qrs``**:

    - ``beats_<CODE>``, ``rhythm_secs_<CODE>``, ``n_episodes_<CODE>`` for each of
      the seven codes in :data:`RHYTHM_NAMES`. Remember ``N`` is "not annotated",
      not sinus rhythm, and ``AB`` is undocumented upstream.
    - ``af_burden`` — AFIB seconds over ``rhythm_annotated_secs``. ``afl_burden``
      and ``svt_burden`` are the flutter-only and any-arrhythmia counterparts, and
      ``af_beat_fraction`` is the beat-count version, which is the larger number
      because AF beats are faster. ``longest_af_episode_secs`` and
      ``n_episodes_AFIB`` describe the paroxysms.
    - ``af_class`` — ``minimal``/``paroxysmal``/``sustained``, or ``unannotated``
      for the 30 records with no ``.atr``. Not the fold label; see
      :func:`attach_stratify_class`.
    - ``dominant_rhythm``, ``dominant_rhythm_fraction``, ``rhythms``.
    - ``n_beats`` / ``n_detections`` — the same number, and neither is audited:
      both files hold ``epltd`` output. ``mean_heart_rate_bpm`` comes from the RR
      intervals with glitches outside 0.2-2.5 s dropped.
    - ``annotated_secs`` / ``unannotated_tail_secs`` — where the beats stop and how
      much signal follows.
    - ``record_seconds`` / ``record_hours`` / ``n_samples`` / ``sampling_rate``,
      and ``lead_names``, ``adc_gains``, ``adc_baselines`` from the header.
    - ``duplicate_of`` — ``"020"`` on record 005 and ``"005"`` on record 020, empty
      everywhere else. See point 5: those two are the same recording.
    """
    clinical = load_clinical(data_path)
    scanned = scan_records(data_path).set_index("Data_ID")

    only_in_files = sorted(set(scanned.index) - set(clinical.index))
    only_in_csv = sorted(set(clinical.index) - set(scanned.index))
    if only_in_files or only_in_csv:
        logger.warning(
            "%s and %s disagree: %d ids only in the files (%s), %d only in the CSV (%s)",
            RECORDS_FILE,
            CLINICAL_CSV,
            len(only_in_files),
            only_in_files[:10],
            len(only_in_csv),
            only_in_csv[:10],
        )

    df = scanned.join(clinical, how="left")

    # The clinical table's own Annotated column against the presence of a .atr
    # file. They agree for all 128 records today; a disagreement means either a
    # partial download or a re-release, and both change every annotation figure.
    if "Annotated" in df.columns:
        claimed = df["Annotated"].astype("boolean").fillna(False).astype(bool)
        found = df["has_rhythm_annotation"].astype(bool)
        if not claimed.equals(found):
            mismatched = df.index[claimed != found].tolist()
            logger.warning(
                "AdditionalData.csv's 'Annotated' disagrees with the presence of a "
                ".atr file for %d records: %s. If these files came from a partial "
                "download the annotation figures below are wrong; verify against "
                "the release's SHA256SUMS.txt.",
                len(mismatched),
                mismatched[:10],
            )

    df = attach_stratify_class(df)
    df.index.name = config.record_id_column
    return df
