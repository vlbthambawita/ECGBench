"""
CODE-15% labels: six binary abnormalities, demographics and mortality follow-up.

Everything comes from one file, ``exams.csv``, so this module is close to what
the declarative path would give. It exists for three derivations that path
cannot express, all of which the splitter needs to reach through here rather
than recompute:

- ``abnormality_codes``, the six flags reduced to a ``,``-separated list, which
  is the multi-label target users should train on;
- ``stratify_class``, the single-label reduction the folds are stratified on;
- the mortality columns' missingness, which is real and has to stay visible.

Quirks worth knowing, all verified against the shipped file (md5
``0107516d3f63864498fb77d15799cc95``, matching Zenodo):

- **"No abnormality" is not "normal", and the gap is enormous.** 308,004 of the
  345,779 records carry none of the six flags, but only 134,657 records are
  flagged ``normal_ecg``. The other 173,347 have some finding the six-class
  vocabulary does not name — a left-axis deviation, an old infarct, anything
  outside 1dAVb/RBBB/LBBB/SB/ST/AF. A model trained on the six flags alone
  treats all 173,347 as negative examples of everything, which is wrong for
  173,347 records rather than merely uninformative. ``normal_ecg`` is exposed so
  that a user can restrict the negatives to genuine normals, and
  ``stratify_class`` separates ``NORMAL`` from ``OTHER`` for the same reason.
  The two are exactly disjoint from the flags: no record is both ``normal_ecg``
  and abnormal.
- **Multi-label, mildly.** 37,775 records (10.9%) carry at least one
  abnormality and 3,671 carry more than one. RBBB is the commonest at 9,672 and
  SB the rarest at 5,605.
- **Mortality follow-up is missing for a third of the cohort.** ``death`` and
  ``timey`` are blank for 112,132 records — no follow-up, not "survived" — and
  they ship as an object-dtype column mixing ``True``/``False`` with NaN.
  ``death`` is exposed as a nullable boolean and ``has_followup`` makes the
  distinction explicit, because reading NaN as False would turn 112,132
  unknowns into 112,132 survivors. Of the 233,647 records with follow-up, 8,341
  died.
- **``nn_predicted_age`` is a model output, not an observation.** It is the
  age estimated by the network of Lima et al. from the tracing itself. Exposed
  because it ships, but it is a prediction and training against it is training
  against another model.
- **Patients repeat, heavily.** 233,770 patients over 345,779 records, with
  66,929 contributing more than one and one contributing 38. Folds are grouped
  on ``patient_id``.
- **Patient grouping does not catch duplicate recordings.** Part 0 alone holds
  47 non-degenerate records whose waveforms are byte-identical to another record
  in the same part, in groups that sometimes span different ``patient_id``s. It
  is a small effect (0.24% of that part) but it is leakage that grouping cannot
  see, and it comes from the source.

``signal_path`` is deliberately **not** produced here: resolving an ``exam_id``
to its row in an HDF5 part means reading all 18 ``exam_id`` datasets, which is
the splitter's job (``splitting/strategies/code15.py``) and not something a
label load should pay for.
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
    "ABNORMALITIES",
    "EXAMS_CSV",
    "NO_ABNORMALITY_NORMAL",
    "NO_ABNORMALITY_OTHER",
    "SAMPLING_RATE",
    "add_derived_columns",
    "load_labels",
]

#: The release's single metadata table, one row per record.
EXAMS_CSV = "exams.csv"

#: The six binary abnormality columns, in the order exams.csv declares them.
#: This is the whole label vocabulary — there is no seventh class.
ABNORMALITIES = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")

#: Stratification classes for records carrying none of the six. Kept apart
#: because they are not the same thing: see the module docstring.
NO_ABNORMALITY_NORMAL = "NORMAL"
NO_ABNORMALITY_OTHER = "OTHER"

#: Constant across the release.
SAMPLING_RATE = 400

#: Separator for the list-valued label column. A comma is safe here — the six
#: codes contain none.
LIST_SEPARATOR = ","


def _rarest_first(df: pd.DataFrame) -> list[str]:
    """The six abnormalities ordered rarest to commonest, ties on name.

    Computed from the frame rather than hardcoded so the reduction stays a pure
    function of the data, and sorted on ``(count, name)`` so it does not depend
    on column or row order.
    """
    return sorted(ABNORMALITIES, key=lambda c: (int(df[c].sum()), c))


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``abnormality_codes``, ``n_abnormalities`` and ``stratify_class``.

    This is the **only** derivation of these columns in ECGBench.
    ``CODE15Splitter`` reaches them through ``load_labels`` rather than
    repeating the reduction, so the stratification label and the exposed labels
    cannot drift apart.
    """
    out = df.copy()
    codes = np.array(ABNORMALITIES)
    flags = out[list(ABNORMALITIES)].astype(bool).to_numpy()

    # Only the 37,775 records that carry something do the string work; the other
    # 89% keep the empty default. Row-wise joining over all 345,779 is slow
    # enough to notice.
    joined = np.full(len(flags), "", dtype=object)
    for i in np.flatnonzero(flags.any(axis=1)):
        joined[i] = LIST_SEPARATOR.join(codes[flags[i]])
    out["abnormality_codes"] = joined
    out["n_abnormalities"] = flags.sum(axis=1).astype(int)

    # Single-label reduction for stratification ONLY — see the module docstring.
    # Rarest-wins, so the small classes survive a ten-way split; the alternative
    # (first-listed wins) would starve SB and 1dAVb in favour of RBBB.
    # Assigned commonest-first so the rarest class a record carries lands last
    # and wins.
    order = _rarest_first(out)
    strat = np.where(out["normal_ecg"].to_numpy(), NO_ABNORMALITY_NORMAL, NO_ABNORMALITY_OTHER)
    strat = strat.astype(object)
    for code in reversed(order):
        strat[flags[:, ABNORMALITIES.index(code)]] = code
    out["stratify_class"] = strat
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return CODE-15% labels and metadata indexed by ``exam_id``.

    Columns:
        ``patient_id``, ``age``, ``is_male``, ``sex``, ``nn_predicted_age``,
        the six flags ``1dAVb``/``RBBB``/``LBBB``/``SB``/``ST``/``AF``,
        ``normal_ecg``, ``abnormality_codes``, ``n_abnormalities``,
        ``stratify_class``, ``death``, ``has_followup``,
        ``followup_years``, ``trace_file``, ``n_samples``,
        ``duration_seconds``, ``sampling_rate``.

    Multi-label: ``abnormality_codes`` is a ``,``-separated list and is empty
    for the 308,004 records carrying none of the six. Train on that together
    with ``normal_ecg`` — an empty list is *not* a normal ECG. Never train on
    ``stratify_class``; it exists to make the folds well defined.
    """
    from ecgbench.labels import LabelSourceMissingError

    root = Path(data_path)
    path = root / EXAMS_CSV
    if not path.exists():
        raise LabelSourceMissingError(
            f"CODE-15% labels come from {EXAMS_CSV}, which is not in {root}. ECGBench "
            "publishes fold CSVs only — labels stay with the source dataset, so point "
            f"data_path at a full local copy (see {config.url})."
        )

    raw = pd.read_csv(path)
    expected = {
        "exam_id",
        "patient_id",
        "age",
        "is_male",
        "nn_predicted_age",
        "normal_ecg",
        "death",
        "timey",
        "trace_file",
        *ABNORMALITIES,
    }
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")

    # .to_numpy() throughout: passing index= alongside Series values *reindexes*
    # them against the new labels rather than relabelling, which silently yields
    # a frame of NaN.
    df = pd.DataFrame(
        {
            "patient_id": raw["patient_id"].to_numpy(),
            "age": raw["age"].to_numpy(),
            "is_male": raw["is_male"].astype(bool).to_numpy(),
            "nn_predicted_age": raw["nn_predicted_age"].to_numpy(),
            "normal_ecg": raw["normal_ecg"].astype(bool).to_numpy(),
            "trace_file": raw["trace_file"].to_numpy(),
        },
        index=pd.Index(raw["exam_id"].astype(int).to_numpy(), name="exam_id"),
    )
    df["sex"] = np.where(df["is_male"].to_numpy(), "M", "F")
    for code in ABNORMALITIES:
        df[code] = raw[code].astype(bool).to_numpy()

    # `death` ships as object dtype mixing True/False with NaN for the 112,132
    # records without follow-up. A nullable boolean keeps the third state; the
    # explicit flag keeps it from being read as "survived".
    df["death"] = pd.array(raw["death"].to_numpy(), dtype="boolean")
    df["has_followup"] = df["death"].notna().to_numpy()
    df["followup_years"] = raw["timey"].to_numpy()

    # Uniform across the release, so no file has to be opened to learn them.
    df["n_samples"] = 4096
    df["duration_seconds"] = 4096 / SAMPLING_RATE
    df["sampling_rate"] = SAMPLING_RATE

    df = add_derived_columns(df)
    logger.info(
        "Loaded CODE-15%% labels: %d records, %d patients, %d with an abnormality, "
        "%d flagged normal, %d with mortality follow-up",
        len(df),
        df["patient_id"].nunique(),
        int((df["n_abnormalities"] > 0).sum()),
        int(df["normal_ecg"].sum()),
        int(df["has_followup"].sum()),
    )
    return df
