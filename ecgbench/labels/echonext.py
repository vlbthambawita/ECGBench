"""
EchoNext labels — structural heart disease read off a matched echocardiogram.

Each of the 100,000 ECGs is paired with an echo, and the labels are that echo's
findings: eleven per-condition binary flags, the composite
``shd_moderate_or_greater_flag`` the dataset is built around, the ordinal and
continuous measurements those flags were thresholded from, and demographics.

**The one thing to know before training on any flag: a missing measurement is
recorded as a negative, not as missing.** Every flag is 0/1 with no nulls, but the
underlying measurement is absent for a large minority of records, and in every
such case the flag reads 0 — verified across all seven measurable conditions,
0 records have a positive flag with a missing value. The worst two:

===================================  ==========  ================================
Flag                                 unmeasured  what a 0 therefore means
===================================  ==========  ================================
``tr_max_gte_32_flag``               54,996      "<3.2 m/s **or never measured**"
``pasp_gte_45_flag``                 43,424      "<45 mmHg **or never measured**"
``pericardial_effusion_...``         11,823      "none/small **or never assessed**"
``lvef_lte_45_flag``                  8,944      ">45% **or never measured**"
===================================  ==========  ================================

So a model trained on ``tr_max_gte_32_flag`` learns "measured and high" against
"low or never imaged", and the second class is mostly an artefact of who got a
full study. :func:`load_labels` therefore emits a ``<flag>_measured`` boolean
beside every such flag; mask on it before computing a prevalence or a loss.
``shd_moderate_or_greater_flag`` is a composite of the other ten and has no single
measurement behind it, so it gets no ``_measured`` companion.

The ordinal columns carry a **``presumed none``** level distinct from ``none``
(6,561 records for aortic stenosis) — an inference by the report parser rather
than a measured absence. It is preserved rather than merged into ``none``.

**A separate defect, in the release's own README.** The tabular feature arrays
``EchoNext_<split>_tabular_features.npy`` are documented as
``sex, ventricular_rate, atrial_rate, pr_interval, qrs_duration, qt_corrected,
age_at_ecg``. They are not. ``age_at_ecg`` is column **1**, not column 6, and
everything between shifts down one. Recovered by rank-correlating each array
column against the metadata: the corrected order gives Spearman 1.000 on all
100,000 rows of all four splits, the documented order gives 0.05-0.32. Anyone
following the README trains with age labelled as ventricular rate.
:data:`TABULAR_FEATURE_COLUMNS` and :func:`load_tabular_features` use the true
order.

Labels come from ``echonext_metadata_100k.csv``, the file the release ships, so
they do not depend on ``ecgbench splits`` having been run first. That table covers
all 100,000 records including the 17,457 ``no_split`` ones that ECGBench's
partition excludes; reindex against a split to drop them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: The release's own metadata table. Its README calls this
#: ``EchoNext_metadata_100k.csv``; the shipped and checksummed file is lowercase.
SOURCE_CSV = "echonext_metadata_100k.csv"

#: Written by ``EchoNextSplitter``; used only as a fallback.
GENERATED_CSV = "ecgbench_metadata.csv"

JOIN_COLUMN = "ecg_key"

#: The composite target the dataset is named for: moderate or greater structural
#: heart disease. A disjunction of the ten per-condition flags below.
COMPOSITE_FLAG = "shd_moderate_or_greater_flag"

#: Per-condition flag -> the measurement column it was thresholded from. Used to
#: emit ``<flag>_measured``, because a missing measurement reads as a 0 flag.
#: ``lvwt_gte_13_flag`` is thresholded on the thicker of two walls, so it counts
#: as measured only when both are present.
FLAG_SOURCES: dict[str, tuple[str, ...]] = {
    "lvef_lte_45_flag": ("lvef_value",),
    "lvwt_gte_13_flag": ("ivs_measurement", "lvpw_measurement"),
    "aortic_stenosis_moderate_or_greater_flag": ("aortic_stenosis_value",),
    "aortic_regurgitation_moderate_or_greater_flag": ("aortic_regurgitation_value",),
    "mitral_regurgitation_moderate_or_greater_flag": ("mitral_regurgitation_value",),
    "tricuspid_regurgitation_moderate_or_greater_flag": ("tricuspid_regurgitation_value",),
    "pulmonary_regurgitation_moderate_or_greater_flag": ("pulmonary_regurgitation_value",),
    "rv_systolic_dysfunction_moderate_or_greater_flag": ("rv_systolic_function_value",),
    "pericardial_effusion_moderate_large_flag": ("pericardial_effusion_value",),
    "pasp_gte_45_flag": ("pasp_value",),
    "tr_max_gte_32_flag": ("tr_max_velocity_value",),
}

#: Every binary label, composite last.
FLAG_COLUMNS: tuple[str, ...] = (*FLAG_SOURCES, COMPOSITE_FLAG)

#: Ordinal severity scales, least to most severe. ``presumed none`` is the report
#: parser inferring an absence rather than measuring one, and is kept distinct.
ORDINAL_LEVELS: dict[str, tuple[str, ...]] = {
    "aortic_stenosis_value": ("presumed none", "none", "mild", "moderate", "severe"),
    "aortic_regurgitation_value": ("presumed none", "none", "mild", "moderate", "severe"),
    "mitral_regurgitation_value": ("presumed none", "none", "mild", "moderate", "severe"),
    "tricuspid_regurgitation_value": (
        "presumed none",
        "none",
        "mild",
        "moderate",
        "severe",
    ),
    "pulmonary_regurgitation_value": (
        "presumed none",
        "none",
        "mild",
        "moderate",
        "severe",
    ),
    "rv_systolic_function_value": (
        "normal",
        "mildly_reduced",
        "moderately_reduced",
        "severely_reduced",
    ),
    "pericardial_effusion_value": ("none", "trace", "small", "moderate", "large"),
}

#: Continuous echo measurements the flags were thresholded from.
CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "lvef_value",
    "ivs_measurement",
    "lvpw_measurement",
    "pasp_value",
    "tr_max_velocity_value",
)

#: ECG-derived measurements and demographics carried in the same table.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "patient_key",
    "age_at_ecg",
    "sex",
    "acquisition_year",
    "location_setting",
    "race_ethnicity",
    "most_recent_ecg",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
    "split",
)

#: TRUE column order of ``EchoNext_<split>_tabular_features.npy``. The release's
#: README lists ``age_at_ecg`` last; it is actually second. See the module
#: docstring — recovered by rank correlation, Spearman 1.000 on every split.
TABULAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "sex",
    "age_at_ecg",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
)

#: What the README claims the order is, kept so the discrepancy is greppable.
README_TABULAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "sex",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
    "age_at_ecg",
)

TABULAR_TEMPLATE = "EchoNext_{split}_tabular_features.npy"
SPLITS = ("train", "val", "test", "no_split")


def _source_path(data_path: Path) -> Path:
    from ecgbench.labels import LabelSourceMissingError

    for name in (SOURCE_CSV, GENERATED_CSV):
        candidate = Path(data_path) / name
        if candidate.exists():
            return candidate
    raise LabelSourceMissingError(
        f"EchoNext labels come from {SOURCE_CSV}, which is not in {data_path}. "
        "ECGBench publishes no fold CSVs for this dataset and never its labels, so "
        "point data_path at your own credentialed copy from "
        "https://physionet.org/content/echonext/1.1.0/ ."
    )


def load_labels(data_path: Path | str, config: DatasetConfig | None = None) -> pd.DataFrame:
    """Echo-derived labels for every EchoNext record, indexed by ``ecg_key``.

    Returns the eleven per-condition flags and the composite, a ``<flag>_measured``
    boolean beside each per-condition flag, the ordinal severities as ordered
    categoricals, the continuous echo measurements, and demographic/ECG context.

    Covers all 100,000 records, including the 17,457 ``no_split`` ones outside
    ECGBench's partition — reindex against a split's record ids to drop them.
    """
    path = _source_path(Path(data_path))
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])

    if JOIN_COLUMN not in df.columns:
        raise ValueError(f"'{JOIN_COLUMN}' not in {path}; found {list(df.columns)[:10]}")

    out = pd.DataFrame(index=pd.Index(df[JOIN_COLUMN], name=JOIN_COLUMN))

    for flag in FLAG_COLUMNS:
        if flag in df.columns:
            out[flag] = df[flag].astype("Int64").values

    # A 0 flag can mean "measured and below threshold" or "never measured". These
    # companions are the only way to tell the two apart.
    for flag, sources in FLAG_SOURCES.items():
        present = [s for s in sources if s in df.columns]
        if not present or flag not in df.columns:
            continue
        measured = df[present].notna().all(axis=1)
        out[f"{flag}_measured"] = measured.values

    for column, levels in ORDINAL_LEVELS.items():
        if column in df.columns:
            out[column] = pd.Categorical(df[column].values, categories=list(levels), ordered=True)

    for column in (*CONTINUOUS_COLUMNS, *CONTEXT_COLUMNS):
        if column in df.columns and column not in out.columns:
            out[column] = df[column].values

    n_flags = sum(c in out.columns for c in FLAG_COLUMNS)
    logger.info(
        "EchoNext labels: %d records x %d columns (%d flags, %d _measured masks)",
        len(out),
        out.shape[1],
        n_flags,
        sum(c.endswith("_measured") for c in out.columns),
    )
    return out


def load_tabular_features(data_path: Path | str, split: str = "train") -> pd.DataFrame:
    """The publisher's preprocessed tabular features, with the columns named right.

    Shape ``(N, 7)``, standardised (except ``sex``, which is binary), missing values
    median-imputed — ``atrial_rate`` and ``pr_interval`` were set to 0 instead. The
    rows align with ``echonext_metadata_100k.csv`` filtered to ``split``, in order.

    Columns are named from :data:`TABULAR_FEATURE_COLUMNS`, **not** the order the
    release's README gives: ``age_at_ecg`` is column 1 there, not column 6.
    """
    import numpy as np

    from ecgbench.labels import LabelSourceMissingError

    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    path = Path(data_path) / TABULAR_TEMPLATE.format(split=split)
    if not path.exists():
        raise LabelSourceMissingError(
            f"EchoNext tabular features for split {split!r} not found at {path}."
        )

    array = np.load(path)
    if array.shape[1] != len(TABULAR_FEATURE_COLUMNS):
        raise ValueError(
            f"{path.name} has {array.shape[1]} columns; expected "
            f"{len(TABULAR_FEATURE_COLUMNS)}. The column order this module corrects "
            "may have changed — re-verify against the metadata before trusting it."
        )
    return pd.DataFrame(array, columns=list(TABULAR_FEATURE_COLUMNS))
