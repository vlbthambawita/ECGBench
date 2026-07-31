"""
MIMIC-IV-ECG labels: machine report text and cardiologist-machine measurements.

``machine_measurements.csv`` carries, per study, up to 18 free-text report lines
(``report_0``..``report_17``) plus nine numeric measurements, the ECG cart's
bandwidth and filter settings. It joins ``record_list.csv`` 1:1 on ``study_id``:
the two files hold exactly the same 800,035 studies.

Two things about this source make a plain declarative block insufficient.

**The report is spread across 18 columns.** A variable number are populated —
median 3, up to 17 — and the data dictionary warns they "can be separated by
blank rows", so a populated line may follow an empty one. This module joins the
non-empty lines, in order, into a single ``report_text``.

**Missing measurements are encoded as integer sentinels, not as NaN.** Every one
of the nine numeric columns is 100% populated, which makes them look complete;
they are not. The markers are:

===========  ======================================  =============================
Sentinel     Meaning                                 Records affected
===========  ======================================  =============================
``29999``    wave timing not measurable              ``p_onset`` 123,434,
                                                     ``p_end`` 230,323
``32767``    axis not measurable (int16 max)         ``p_axis`` 7,199,
``-32768``                                           ``t_axis`` 1,440,
                                                     ``qrs_axis`` 1
``65535``    RR interval not measurable (uint16 max) 5
===========  ======================================  =============================

That the sentinel means "not measurable" rather than "not recorded" is verifiable
from the data: ``p_end == 29999`` in **100.0%** of atrial-fibrillation records and
``p_onset == 29999`` in 90.7% of them, against **0%** ``p_onset`` sentinels among
sinus-rhythm records — atrial fibrillation has no organised P wave to measure.
Averaging ``p_axis`` without handling this yields nonsense.

This loader replaces sentinels with ``NaN``. That is lossless, because the source
columns have no genuine NaNs, so ``NaN`` here can only mean "was a sentinel".
On the values that remain, ``qrs_onset < qrs_end < t_end`` holds in 99.98% of
records and the derived QRS duration has a median of 94 ms — physiologic, so the
surviving numbers can be trusted.

**``report_0`` is the first report line, not "the rhythm".** It usually is a
rhythm statement, but it is sometimes a data-quality warning
(``--- warning: data quality may affect interpretation ---``, 7,079 records), an
input note (``age not entered, assumed to be 50 years old…``, 4,980) or a finding
(``*** consider acute st elevation mi ***``, 8,516). It is exposed as
``primary_report`` for that reason, and the stratification label derived from it
is named ``stratify_class``. Train on ``report_text``.

Labels are **not** on the HuggingFace Hub and never will be: MIMIC-IV is
credentialed, so redistributing its report text is not permitted. ``labels=True``
needs a local copy of the full release.
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

#: The measurements file, as the official release names it. If a local copy has
#: been filtered or renamed, this is the name that must hold the real content —
#: verify against the release's SHA256SUMS.txt.
SOURCE_CSV = "machine_measurements.csv"

#: Column holding the study identifier in both source files.
JOIN_COLUMN = "study_id"

#: Free-text report columns, in report order.
REPORT_COLUMNS = [f"report_{i}" for i in range(18)]

#: Numeric measurement columns, all in milliseconds except the three axes (degrees).
TIMING_COLUMNS = ["rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end"]
AXIS_COLUMNS = ["p_axis", "qrs_axis", "t_axis"]
MEASUREMENT_COLUMNS = TIMING_COLUMNS + AXIS_COLUMNS

#: "Not measurable" markers. Integer rails, not NaN — see the module docstring.
SENTINELS = (29999, 32767, -32768, 65535, -32767)

#: Passthrough columns from the measurements file.
_PASSTHROUGH = ["cart_id", "bandwidth", "filtering"]

#: Stratification classes with fewer than this many records are pooled into
#: OTHER. 47 normalised report_0 values clear 1,000 records and together cover
#: 92.6% of the dataset, so this keeps the label interpretable without a long
#: tail of near-empty classes.
MIN_CLASS_SIZE = 1000

OTHER = "OTHER"
UNKNOWN = "UNKNOWN"


def normalise_report_line(value: object) -> str:
    """Lower-case, collapse whitespace and drop trailing punctuation.

    The same statement appears with varied spacing and trailing periods, which
    splits one class into several: 1,571 raw ``report_0`` values collapse to 950.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text.rstrip(". ").strip().lower()


def _join_report(row: pd.Series) -> str:
    """Join the populated report lines, in order, with ' | '.

    Empty lines are skipped rather than terminating the scan: the data dictionary
    states populated lines can follow blank ones.
    """
    parts = [str(row[c]).strip() for c in REPORT_COLUMNS if pd.notna(row[c])]
    return " | ".join(p for p in parts if p)


def clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the integer 'not measurable' sentinels with NaN, in place.

    Lossless: the source columns contain no genuine NaN, so a NaN afterwards can
    only mean the value was a sentinel.
    """
    out = df.copy()
    for column in MEASUREMENT_COLUMNS:
        if column not in out.columns:
            continue
        numeric = pd.to_numeric(out[column], errors="coerce")
        n_sentinel = int(numeric.isin(SENTINELS).sum())
        numeric = numeric.mask(numeric.isin(SENTINELS))
        # Anything left outside a physiologic range is also not a measurement.
        if column in AXIS_COLUMNS:
            implausible = (numeric < -180) | (numeric > 180)
        else:
            implausible = (numeric < 0) | (numeric > 3000)
        n_implausible = int(implausible.sum())
        out[column] = numeric.mask(implausible)
        if n_sentinel or n_implausible:
            logger.debug(
                "%s: %d sentinel + %d implausible value(s) set to NaN",
                column,
                n_sentinel,
                n_implausible,
            )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return MIMIC-IV-ECG labels indexed by ``study_id``.

    Columns:

    - ``report_text`` — the machine report, populated lines joined with ``' | '``.
      **This is the ground truth**; it is free text, not a code, and only one
      record of 800,035 has none.
    - ``primary_report`` — normalised ``report_0``, i.e. the *first* report line.
      Usually the rhythm, but sometimes a data-quality warning or an input note,
      so do not read it as a rhythm label.
    - ``report_0``..``report_17`` — the raw lines, verbatim, so nothing is lost.
    - ``rr_interval``, ``p_onset``, ``p_end``, ``qrs_onset``, ``qrs_end``,
      ``t_end`` (ms) and ``p_axis``, ``qrs_axis``, ``t_axis`` (degrees), with the
      integer sentinels replaced by NaN — see the module docstring.
    - ``qrs_duration`` — ``qrs_end - qrs_onset`` (ms), NaN where either side is.
    - ``cart_id``, ``bandwidth``, ``filtering``, ``ecg_time``.
    - ``stratify_class`` — pooled ``primary_report``, **for fold construction
      only**.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    csv_path = data_path / SOURCE_CSV
    if not csv_path.exists():
        raise LabelSourceMissingError(
            f"MIMIC-IV-ECG labels come from {SOURCE_CSV}, which is not in {data_path}. "
            "ECGBench publishes fold CSVs only, and MIMIC-IV is credentialed, so its "
            "report text is never redistributed — point data_path at a full local copy "
            "of https://physionet.org/content/mimic-iv-ecg/1.0/ ."
        )

    df = pd.read_csv(csv_path, low_memory=False)
    if JOIN_COLUMN not in df.columns:
        raise ValueError(f"{csv_path} has no '{JOIN_COLUMN}' column. Found: {list(df.columns)}")

    present_reports = [c for c in REPORT_COLUMNS if c in df.columns]
    df["report_text"] = df.apply(_join_report, axis=1) if present_reports else ""
    df["primary_report"] = df.get("report_0", pd.Series(dtype=object)).map(normalise_report_line)

    df = clean_measurements(df)
    if {"qrs_onset", "qrs_end"} <= set(df.columns):
        df["qrs_duration"] = df["qrs_end"] - df["qrs_onset"]

    df = attach_stratify_class(df)

    keep = (
        ["report_text", "primary_report", "stratify_class"]
        + present_reports
        + [c for c in MEASUREMENT_COLUMNS if c in df.columns]
        + [c for c in ("qrs_duration", "ecg_time") if c in df.columns]
        + [c for c in _PASSTHROUGH if c in df.columns]
    )
    df = df.set_index(JOIN_COLUMN)[keep]
    df.index.name = config.record_id_column

    logger.info(
        "Loaded MIMIC-IV-ECG labels for %d studies; %d distinct primary_report values, "
        "%d records with no report text",
        len(df),
        df["primary_report"].nunique(),
        int((df["report_text"] == "").sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: pooled ``primary_report``, for folds only.

    This is the **only** derivation of the stratification label —
    ``MimicIVECGSplitter`` reads the column rather than recomputing it, so the
    exposed label and the fold label cannot drift.
    """
    out = df.copy()
    labels = out["primary_report"].fillna("").replace("", UNKNOWN)

    counts = labels.value_counts()
    rare = set(counts[counts < MIN_CLASS_SIZE].index)
    if rare:
        logger.info(
            "Pooling %d primary_report value(s) with <%d records into '%s' "
            "(%d records affected)",
            len(rare),
            MIN_CLASS_SIZE,
            OTHER,
            int(labels.isin(rare).sum()),
        )
        labels = labels.where(~labels.isin(rare), OTHER)

    out["stratify_class"] = labels
    return out
