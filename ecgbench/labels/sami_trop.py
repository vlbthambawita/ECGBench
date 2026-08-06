"""
SaMi-Trop labels: demographics, a single normality flag, and mortality follow-up.

**This module owns the positional join, and that is its main reason to exist.**
``exams.hdf5`` holds one ``(1631, 4096, 12)`` ``tracings`` array and no
identifier dataset, so the only possible link between a waveform and a row of
``exams.csv`` is row position. The sibling CODE-15% release proves that cannot
be assumed — its ``exams.csv`` is *not* in file order, which is why
``CODE15Splitter`` has to read 18 ``exam_id`` datasets to recover the mapping.

So the alignment was tested. QRS amplitude is reliably larger in men; splitting
the precordial peak-to-peak amplitude by ``exams.csv``'s own ``is_male`` in row
order gives a Welch *t* of 4.98, while the largest |t| over 2,000 random
permutations of the same rows is 3.44. Row order it is. :data:`N_RECORDS` is
enforced on every read, because a positional join against a file of the wrong
length cannot produce a partial match that a warning could flag — it produces
1,631 confidently wrong rows.

Quirks worth knowing, all recomputed from the shipped ``exams.csv`` (md5
``6c9007a0427f7c3d9e1b6fb091231a67``, matching Zenodo):

- **There is no diagnostic label vocabulary.** Unlike CODE-15% and CODE-test,
  which share this file format, SaMi-Trop ships no abnormality flags at all. The
  only ECG label is the binary ``normal_ecg``: 286 of 1,631 records (17.5%) are
  flagged normal, and the other 1,345 are only known to be *not* normal. Nothing
  says what is wrong with them.
- **"Normal ECG" does not mean healthy.** Every patient in this cohort has
  chronic Chagas cardiomyopathy. A ``normal_ecg`` record is a normal tracing in
  a diseased patient, so these 286 records are not usable as healthy controls —
  a mistake that is easy to make when pooling this release with others.
- **Mortality follow-up is complete, unlike CODE-15%'s.** All 1,631 records have
  a ``death`` flag and a ``timey``, so there is no missingness to reason about
  and no nullable-boolean dance: 104 patients died, and ``followup_years`` is
  time to death for them and time to censoring for the other 1,527. Median 2.07
  years, range 0.07-3.39.
- **One recording per patient.** The release is each patient's *first* exam, so
  ``exam_id`` is patient-unique, the config sets no ``patient_id_column``, and
  there is no grouping constraint on the folds. This is the unusual case in the
  catalogue, not the normal one.
- **``nn_predicted_age`` is a model output, not an observation.** It is the age
  estimated from the tracing by the network of Lima et al., and it is the
  quantity that paper studies. Exposed because it ships; training against it is
  training against another model. It runs 22.6-95.9 against true ages of 26-98.

``signal_path`` is deliberately not produced here — it is a statement about
file layout, and :mod:`ecgbench.splitting.strategies.sami_trop` makes it.
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
    "N_RECORDS",
    "N_SAMPLES",
    "SAMPLING_RATE",
    "STRATIFY_ABNORMAL_ALIVE",
    "STRATIFY_DEATH",
    "STRATIFY_NORMAL",
    "TRACINGS_DATASET",
    "TRACINGS_HDF5",
    "load_labels",
]

#: The release's single metadata table, one row per record, in waveform order.
EXAMS_CSV = "exams.csv"

#: The waveform array, and the dataset inside it. Named here because the
#: positional join is validated against its length.
TRACINGS_HDF5 = "exams.hdf5"
TRACINGS_DATASET = "tracings"

#: Every file in this release has exactly this many rows, and the positional
#: join is only sound because they all do.
N_RECORDS = 1631

#: Samples per record, uniform — the array is one rectangular block.
N_SAMPLES = 4096

#: Constant across the release.
SAMPLING_RATE = 400

#: Stratification classes. Three, not the four of the death x normal_ecg cross:
#: only 3 records are both dead and normal, and a class of 3 cannot be spread
#: over 10 folds at all, so those 3 go to DEATH and the outcome stays balanced.
STRATIFY_DEATH = "DEATH"
STRATIFY_NORMAL = "NORMAL"
STRATIFY_ABNORMAL_ALIVE = "ABNORMAL_ALIVE"


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return SaMi-Trop labels and metadata indexed by ``exam_id``.

    Columns:
        ``row`` (the record's index in the ``tracings`` array), ``age``,
        ``is_male``, ``sex``, ``nn_predicted_age``, ``normal_ecg``, ``death``,
        ``followup_years``, ``stratify_class``, ``n_samples``,
        ``duration_seconds``, ``sampling_rate``.

    Single-label and not diagnostic: ``normal_ecg`` is the only ECG label, and a
    record that is not normal carries no statement of what is wrong with it. For
    survival work use ``death`` with ``followup_years``. Never train on
    ``stratify_class`` — it exists to make the folds well defined.

    Raises:
        LabelSourceMissingError: ``exams.csv`` is not under ``data_path``.
        ValueError: it does not have exactly :data:`N_RECORDS` rows, which would
            make the positional join to the waveforms unsound.
    """
    from ecgbench.labels import LabelSourceMissingError

    root = Path(data_path)
    path = root / EXAMS_CSV
    if not path.exists():
        raise LabelSourceMissingError(
            f"SaMi-Trop labels come from {EXAMS_CSV}, which is not in {root}. ECGBench "
            "publishes fold CSVs only — labels stay with the source dataset, so point "
            f"data_path at a full local copy (see {config.url})."
        )

    raw = pd.read_csv(path)
    expected = {"exam_id", "age", "is_male", "normal_ecg", "death", "timey", "nn_predicted_age"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")

    if len(raw) != N_RECORDS:
        raise ValueError(
            f"{path} has {len(raw)} rows, expected {N_RECORDS}. This release ships no "
            "record identifier inside exams.hdf5 — the CSV is aligned to the tracings "
            "array by row position — so a table of a different length cannot be joined "
            "at all, only mis-joined."
        )
    if raw["exam_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate exam_id values; the join would multiply rows.")

    # .to_numpy() throughout: passing index= alongside Series values *reindexes*
    # them against the new labels rather than relabelling, yielding all-NaN.
    df = pd.DataFrame(
        {
            # The row this record occupies in the tracings array. Explicit
            # because it IS the join, and the splitter builds paths from it.
            "row": np.arange(len(raw), dtype=int),
            "age": raw["age"].to_numpy(),
            "is_male": raw["is_male"].astype(bool).to_numpy(),
            "nn_predicted_age": raw["nn_predicted_age"].to_numpy(),
            "normal_ecg": raw["normal_ecg"].astype(bool).to_numpy(),
            "death": raw["death"].astype(bool).to_numpy(),
            "followup_years": raw["timey"].to_numpy(),
        },
        index=pd.Index(raw["exam_id"].astype(int).to_numpy(), name="exam_id"),
    )
    df["sex"] = np.where(df["is_male"].to_numpy(), "M", "F")

    # Uniform across the release, so no file has to be opened to learn them.
    df["n_samples"] = N_SAMPLES
    df["duration_seconds"] = N_SAMPLES / SAMPLING_RATE
    df["sampling_rate"] = SAMPLING_RATE

    # Single-label reduction for stratification ONLY. Death first so the rare
    # outcome (104 records) is what the folds balance; the 3 records that are
    # both dead and normal land in DEATH rather than forming a 3-member class.
    strat = np.where(df["normal_ecg"].to_numpy(), STRATIFY_NORMAL, STRATIFY_ABNORMAL_ALIVE).astype(
        object
    )
    strat[df["death"].to_numpy()] = STRATIFY_DEATH
    df["stratify_class"] = strat

    logger.info(
        "Loaded SaMi-Trop labels: %d records, %d flagged normal, %d deaths, "
        "median follow-up %.2f years",
        len(df),
        int(df["normal_ecg"].sum()),
        int(df["death"].sum()),
        float(df["followup_years"].median()),
    )
    return df
