"""
IKEM labels: demographics and two cart-measured rates. **No diagnoses.**

The first thing to know is what is *not* here. ``exams.csv`` carries ``age``,
``is_male``, ``weight``, ``height``, ``ventricular_rate``, ``atrial_rate`` and
``acquisition_date``, and that is the whole of it. The paper this release
accompanies trains on IKEM diagnostic labels which were not published, so this
dataset supports age and sex estimation, rate regression and self-supervised
pretraining — not diagnosis classification. There is no label vocabulary to
expose and no multi-label question to answer.

The second thing is that **every numeric column uses -1 as a missing-value
sentinel**, and the missingness is severe rather than incidental:

======================  ===========  =======================================
Column                  ``-1`` rows  share
======================  ===========  =======================================
``weight``                   87,916  89.59%
``height``                   87,674  89.34%
``age``                       8,816  8.98%
``atrial_rate``                 410  0.42%
``is_male``                     376  0.38%
``ventricular_rate``              6  0.006%
======================  ===========  =======================================

Read literally the cohort has a mean weight of about -76 kg. This is the trap
the checklist calls out for MIMIC-IV-ECG's integer rails, in a more aggressive
form: here the sentinel is in *every* numeric column including sex, so a
``notna()`` count reports complete data for all 98,130 records. Every -1 is
converted to NaN here, ``is_male`` becomes a nullable boolean so its 376
unknowns survive, and ``has_weight``/``has_height``/``has_age`` make the
commonest cases explicit. This is lossless: the source has no genuine blanks and
no genuine -1 measurement.

Values that survive the sentinel filter still deserve suspicion. 546 records
give ``age`` 0 and 21 give 100 or more; one gives ``ventricular_rate`` 0. Those
are left as they are — they are not sentinels and guessing which are real is not
this module's job — but ``stratify_class`` treats a zero rate as unmeasured.

Other quirks, all recomputed from the shipped ``exams.csv`` (md5
``fd6e9e3a6cfca32f74744d3e839bf6fa``, matching Zenodo):

- **Patients repeat heavily.** 30,290 patients over 98,130 records; 19,078 have
  more than one and the largest has 96, so 86,918 records (88.6%) belong to a
  multi-record patient. Folds are grouped on ``patient_id``, a 40-character
  SHA-1 surrogate.
- **``acquisition_date`` is ``MM-DD-YYYY``**, which sorts wrongly as a string —
  taking string min/max suggests a 2018-2021 range when the real range is
  2004-03-18 to 2022-07-26. It is parsed here, and ``acquisition_year`` added.
  57,179 records are from 2017 and 33,355 from 2018; the whole 2004-2016 tail is
  190 records.
- **48 records are shorter than they look.** The ``tracings`` arrays are
  rectangular at 4,096 samples, but the parts carry a ``real_lengths`` dataset
  and 48 records have only 2,500 real samples (5.0 s), zero-padded to 4,096.
  ``real_length_samples`` and ``real_duration_seconds`` are read from those
  datasets — three 1-D reads, about 200 kB — because a user windowing a record
  needs the true length and cannot get it from the array shape.
- **No duplicate recordings.** Each part carries a ``hashes`` dataset holding
  the SHA-1 of every record's raw int16 bytes, and all 98,130 are distinct. That
  is a stronger guarantee than CODE-15% offers, where 47 byte-identical
  duplicates sit in part 0 alone.

``signal_path`` is deliberately not produced here — resolving an ``exam_id`` to
its row is a statement about file layout, and
:mod:`ecgbench.splitting.strategies.ikem` makes it.
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

__all__ = [
    "EXAMS_CSV",
    "HASHES_DATASET",
    "MISSING_SENTINEL",
    "N_RECORDS",
    "N_SAMPLES",
    "PART_GLOB",
    "REAL_LENGTHS_DATASET",
    "SAMPLING_RATE",
    "SENTINEL_COLUMNS",
    "STRATIFY_BRADY",
    "STRATIFY_NORMAL",
    "STRATIFY_TACHY",
    "add_stratify_class",
    "load_labels",
    "read_real_lengths",
]

#: The release's single metadata table, one row per record.
EXAMS_CSV = "exams.csv"

#: Glob for the waveform parts. Three in v1.0.0, but matched rather than listed
#: so a re-release that splits them differently still resolves.
PART_GLOB = "exams_part_*.hdf5"

#: 1-D datasets inside each part. `exam_id` is the join key (see the splitter);
#: `real_lengths` is the true sample count before zero padding; `hashes` is the
#: SHA-1 of each record's raw int16 bytes.
EXAM_ID_DATASET = "exam_id"
REAL_LENGTHS_DATASET = "real_lengths"
HASHES_DATASET = "hashes"

#: Records in the published release. Checked, not assumed.
N_RECORDS = 98130

#: Stored samples per record — uniform, because the arrays are rectangular. The
#: *real* length is per-record; see :func:`read_real_lengths`.
N_SAMPLES = 4096

#: Constant across the release, and verified against the cart's own
#: ventricular_rate rather than taken from the description, which says 10 s.
SAMPLING_RATE = 500

#: The value every numeric column uses for "not recorded".
MISSING_SENTINEL = -1

#: Columns in which :data:`MISSING_SENTINEL` means missing.
SENTINEL_COLUMNS = (
    "age",
    "is_male",
    "weight",
    "height",
    "ventricular_rate",
    "atrial_rate",
)

#: Stratification classes: the cart's ventricular rate, banded.
STRATIFY_BRADY = "BRADY"
STRATIFY_NORMAL = "NORMAL"
STRATIFY_TACHY = "TACHY"

#: Band edges in bpm, exclusive of the sentinel. Conventional clinical cut-offs.
BRADY_BELOW = 60
TACHY_ABOVE = 100


def read_real_lengths(data_path: Path | str) -> pd.Series:
    """Return exam_id -> true sample count, read from the parts' 1-D datasets.

    Three small reads (about 200 kB in total), not a waveform scan: the
    ``tracings`` arrays are never opened. Returns an empty Series if no part is
    present, so ``load_labels`` still works against a metadata-only copy.
    """
    import h5py

    root = Path(data_path)
    out: dict[int, int] = {}
    for part in sorted(root.glob(PART_GLOB)):
        with h5py.File(part, "r") as handle:
            if EXAM_ID_DATASET not in handle or REAL_LENGTHS_DATASET not in handle:
                logger.warning(
                    "%s has no '%s'/'%s' dataset; real lengths unavailable for it",
                    part.name,
                    EXAM_ID_DATASET,
                    REAL_LENGTHS_DATASET,
                )
                continue
            ids = handle[EXAM_ID_DATASET][:]
            lengths = handle[REAL_LENGTHS_DATASET][:]
        if len(ids) != len(lengths):
            raise ValueError(
                f"{part} has {len(ids)} exam_ids but {len(lengths)} real_lengths; "
                "they are meant to be parallel 1-D arrays."
            )
        out.update({int(i): int(n) for i, n in zip(ids, lengths)})
    return pd.Series(out, dtype="int64", name="real_length_samples")


def add_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: the ventricular rate banded into three classes.

    This is the **only** derivation of the column in ECGBench; ``IKEMSplitter``
    reaches it through :func:`load_labels` rather than repeating the banding.

    Three classes and not four: only 7 records have no usable rate (6 sentinels
    and one literal 0), and a 7-member class cannot be spread across 10 folds at
    all. Those 7 join ``NORMAL``, which is arbitrary but affects 0.007% of the
    release. A rate band is a **measurement, not a diagnosis** — it says nothing
    about rhythm, and a rate of 75 is as compatible with atrial fibrillation as
    with sinus rhythm. It exists to keep the folds balanced on the one clinical
    axis this release actually measures.
    """
    out = df.copy()
    rate = out["ventricular_rate"].to_numpy(dtype=float)
    strat = np.full(len(out), STRATIFY_NORMAL, dtype=object)
    # NaN comparisons are False, so sentinel/zero rates keep the NORMAL default.
    strat[rate < BRADY_BELOW] = STRATIFY_BRADY
    strat[rate > TACHY_ABOVE] = STRATIFY_TACHY
    strat[~np.isfinite(rate) | (rate <= 0)] = STRATIFY_NORMAL
    out["stratify_class"] = strat
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return IKEM metadata indexed by ``exam_id``. No diagnoses — see the module docstring.

    Columns:
        ``patient_id``, ``age``, ``has_age``, ``is_male`` (nullable boolean),
        ``sex``, ``weight``, ``has_weight``, ``height``, ``has_height``,
        ``ventricular_rate``, ``atrial_rate``, ``acquisition_date`` (datetime),
        ``acquisition_year``, ``stratify_class``, ``real_length_samples``,
        ``real_duration_seconds``, ``n_samples``, ``duration_seconds``,
        ``sampling_rate``.

    Every ``-1`` in the source becomes NaN, so ``weight`` and ``height`` are
    absent for about 89.5% of records and ``age`` for 9.0%. Never train on
    ``stratify_class``; it is a banded rate measurement kept for fold balance.

    Raises:
        LabelSourceMissingError: ``exams.csv`` is not under ``data_path``.
    """
    from ecgbench.labels import LabelSourceMissingError

    root = Path(data_path)
    path = root / EXAMS_CSV
    if not path.exists():
        raise LabelSourceMissingError(
            f"IKEM metadata comes from {EXAMS_CSV}, which is not in {root}. ECGBench "
            "publishes no IKEM fold CSVs at all and never labels, so point data_path at "
            f"a full local copy (see {config.url})."
        )

    raw = pd.read_csv(path)
    expected = {"exam_id", "acquisition_date", "patient_id", *SENTINEL_COLUMNS}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    if raw["exam_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate exam_id values; the join would multiply rows.")

    # .to_numpy() throughout: passing index= alongside Series values *reindexes*
    # them against the new labels rather than relabelling, yielding all-NaN.
    index = pd.Index(raw["exam_id"].astype(int).to_numpy(), name="exam_id")
    df = pd.DataFrame({"patient_id": raw["patient_id"].to_numpy()}, index=index)

    # Every -1 becomes NaN. Lossless: the source carries no genuine blanks, so
    # nothing here can be mistaken for a real measurement of -1.
    for col in SENTINEL_COLUMNS:
        values = raw[col].to_numpy(dtype=float)
        df[col] = np.where(values == MISSING_SENTINEL, np.nan, values)

    # A nullable boolean, so the 376 records of unknown sex stay unknown rather
    # than becoming female.
    df["is_male"] = pd.array(
        [None if not np.isfinite(v) else bool(v) for v in df["is_male"].to_numpy()],
        dtype="boolean",
    )
    df["sex"] = pd.array(
        [None if v is pd.NA or v is None else ("M" if v else "F") for v in df["is_male"]],
        dtype="string",
    )
    for col in ("age", "weight", "height"):
        df[f"has_{col}"] = df[col].notna().to_numpy()

    # MM-DD-YYYY. Parsed explicitly: as a string this column sorts wrongly, and
    # dayfirst inference would silently swap month and day for the first 12 days
    # of each month.
    dates = pd.to_datetime(raw["acquisition_date"], format="%m-%d-%Y", errors="coerce")
    df["acquisition_date"] = dates.to_numpy()
    df["acquisition_year"] = dates.dt.year.to_numpy()
    if int(dates.isna().sum()):
        logger.warning(
            "%d acquisition_date values did not parse as MM-DD-YYYY", int(dates.isna().sum())
        )

    # True pre-padding length, from the parts' own 1-D dataset.
    real = read_real_lengths(root)
    if len(real):
        df["real_length_samples"] = real.reindex(df.index).to_numpy()
        df["real_duration_seconds"] = df["real_length_samples"] / SAMPLING_RATE
        short = int((df["real_length_samples"] < N_SAMPLES).sum())
        if short:
            logger.info(
                "%d record(s) are zero-padded: fewer than %d real samples", short, N_SAMPLES
            )
    else:
        logger.warning(
            "No %s found under %s, so real_length_samples is unavailable. Every record "
            "is stored as %d samples but 48 of them are zero-padded from 2,500.",
            PART_GLOB,
            root,
            N_SAMPLES,
        )

    df["n_samples"] = N_SAMPLES
    df["duration_seconds"] = N_SAMPLES / SAMPLING_RATE
    df["sampling_rate"] = SAMPLING_RATE

    df = add_stratify_class(df)
    logger.info(
        "Loaded IKEM metadata: %d records, %d patients; missing age %d, weight %d, "
        "height %d, sex %d (no diagnoses ship)",
        len(df),
        df["patient_id"].nunique(),
        int(df["age"].isna().sum()),
        int(df["weight"].isna().sum()),
        int(df["height"].isna().sum()),
        int(df["is_male"].isna().sum()),
    )
    return df
