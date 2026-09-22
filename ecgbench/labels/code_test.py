"""
CODE-test labels: six abnormalities as read by six annotator groups plus a DNN.

This release exists for annotator agreement, so the loader exposes **every**
annotation set side by side rather than only the gold standard. Each of the six
abnormalities appears seven times, once per annotator, as
``<annotator>_<code>`` — ``gold_standard_AF``, ``cardiologist1_AF``,
``dnn_AF`` and so on — with the gold standard also mirrored into the unprefixed
``abnormality_codes`` list that the config names as ``label_column``.

**Everything here is joined by row position, because nothing else exists.**
``ecg_tracings.hdf5`` holds one ``(827, 4096, 12)`` array and no identifier;
``attributes.csv`` and the seven annotation CSVs each hold 827 rows and no key.
The bundled README is explicit: "the i-th line corresponds to the i-th tracing".
So the index this loader returns is the row number, 0-826, and every source
file is checked for exactly 827 rows before it is read. That check is the whole
safety net — a positional join against a file of the wrong length would
mislabel every record after the first discrepancy without raising anything, so
it refuses rather than truncating or padding.

The annotator sets, per the README:

===================== ==========================================================
``gold_standard``     Two cardiologists read every record independently; where
                      they agreed that reading stands, and where they disagreed
                      a third senior specialist, shown both, decided. This is
                      the evaluation target.
``cardiologist1/2``   The two independent reads the gold standard was built
                      from. They are **not** independent of it.
``cardiology_residents``  Two 4th-year cardiology residents, each annotating
                      half the set.
``emergency_residents``   Two 3rd-year emergency residents, each half.
``medical_students``  Two 5th-year medical students, each half.
``dnn``               The paper's own neural network, thresholded to maximise
                      F1. A model output, not a human read.
===================== ==========================================================

Two things that will bite otherwise:

- **The three non-cardiologist sets are each two people, split by half.** A
  disagreement between ``medical_students`` and ``gold_standard`` is partly a
  difference between two individuals and partly a difference in training level,
  and the release does not say which record went to which annotator. Nothing
  here can separate the two.
- **``dnn.csv`` has an extra leading unnamed index column** that the other six
  annotation files do not. It is dropped; the remaining six columns line up.

Gold-standard prevalence, recomputed from the shipped files: ST 37, RBBB 34,
LBBB 30, 1dAVb 28, SB 16, AF 13. 681 of 827 records (82.3%) carry none of the
six and 12 carry more than one. As in CODE-15%, "none of the six" is not
"normal" — this release publishes no normal flag at all, so the negative class
here is strictly "none of these six findings".

``signal_path`` is produced by the splitter, not here: it is
``ecg_tracings.hdf5:tracings:<row>``, which is a statement about file layout
rather than about labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ecgbench.labels._fields import Field

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ABNORMALITIES",
    "ANNOTATORS",
    "ANNOTATIONS_DIR",
    "ATTRIBUTES_CSV",
    "GOLD_STANDARD",
    "N_RECORDS",
    "NO_ABNORMALITY",
    "SAMPLING_RATE",
    "TRACINGS_HDF5",
    "load_annotations",
    "load_labels",
]

#: The waveform array. Named here because the splitter builds signal paths
#: against it and the loader validates the row count against it.
TRACINGS_HDF5 = "ecg_tracings.hdf5"

#: The dataset inside that file.
TRACINGS_DATASET = "tracings"

#: Age and sex, 827 rows, positionally aligned.
ATTRIBUTES_CSV = "attributes.csv"

#: Directory holding the seven annotation CSVs.
ANNOTATIONS_DIR = "annotations"

#: The six abnormalities, in the order the annotation CSVs declare them. Note
#: this differs from CODE-15%'s exams.csv, which orders ST before AF.
ABNORMALITIES = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")

#: Annotation set -> CSV stem, in the order they are exposed.
ANNOTATORS = (
    "gold_standard",
    "cardiologist1",
    "cardiologist2",
    "cardiology_residents",
    "emergency_residents",
    "medical_students",
    "dnn",
)

#: The annotator whose reading is the evaluation target and the source of the
#: unprefixed label columns.
GOLD_STANDARD = "gold_standard"

#: Stratification class for a record the gold standard flags with nothing. Not
#: called NORMAL: this release publishes no normal flag, so all that is known is
#: the absence of these six findings.
NO_ABNORMALITY = "NONE"

#: Every source file in this release has exactly this many rows, and the
#: positional join is only sound because they all do.
N_RECORDS = 827

#: Constant across the release.
SAMPLING_RATE = 400

#: Samples per record. Uniform: 7 s and 10 s acquisitions are both zero-padded
#: symmetrically to this length.
N_SAMPLES = 4096

LIST_SEPARATOR = ","


def _require(path: Path, what: str, url: str) -> None:
    from ecgbench.labels import LabelSourceMissingError

    if not path.exists():
        raise LabelSourceMissingError(
            f"CODE-test {what} comes from {path.name}, which is not in {path.parent}. "
            "ECGBench publishes fold CSVs only — labels stay with the source dataset, "
            f"so point data_path at a full local copy (see {url}). Note the archive "
            "extracts to a 'data/' subdirectory; data_path must be that directory."
        )


def _read_827(path: Path, what: str) -> pd.DataFrame:
    """Read a keyless CSV and refuse it unless it has exactly 827 rows.

    Every join in this module is positional, so a file of the wrong length does
    not produce a partial match that a warning could flag — it produces 827
    confidently wrong labels. Hence an exception, not a log line.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    if len(df) != N_RECORDS:
        raise ValueError(
            f"CODE-test {what} ({path}) has {len(df)} rows, expected {N_RECORDS}. "
            "This release has no record identifiers — every table is aligned to the "
            "waveform array by row position — so a table of a different length "
            "cannot be joined at all, only mis-joined."
        )
    return df


def load_annotations(data_path: Path | str, url: str = "") -> dict[str, pd.DataFrame]:
    """Read the seven annotation CSVs, each indexed by row number 0-826.

    Returns:
        Annotator name -> DataFrame with the six boolean abnormality columns.
    """
    root = Path(data_path) / ANNOTATIONS_DIR
    out: dict[str, pd.DataFrame] = {}
    for name in ANNOTATORS:
        path = root / f"{name}.csv"
        _require(path, f"'{name}' annotations", url)
        # dnn.csv carries an extra unnamed index column; _read_827 drops it.
        df = _read_827(path, f"'{name}' annotations")
        missing = set(ABNORMALITIES) - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
        out[name] = (
            df[list(ABNORMALITIES)]
            .astype(bool)
            .set_axis(pd.RangeIndex(N_RECORDS, name="record_id"))
        )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return CODE-test labels and metadata indexed by ``record_id`` (0-826).

    Columns:
        ``age``, ``sex``, ``n_samples``, ``duration_seconds``,
        ``sampling_rate``; the gold standard as the six unprefixed flags
        ``1dAVb``/``RBBB``/``LBBB``/``SB``/``AF``/``ST`` plus
        ``abnormality_codes``, ``n_abnormalities`` and ``stratify_class``;
        and every annotator's reading as ``<annotator>_<code>`` together with
        ``<annotator>_abnormality_codes`` and ``<annotator>_n_abnormalities``.

    Multi-label. Evaluate against ``abnormality_codes`` (the gold standard) or
    against a named annotator's columns; never against ``stratify_class``,
    which exists only to make the folds well defined.
    """
    root = Path(data_path)
    url = config.url

    attributes_path = root / ATTRIBUTES_CSV
    _require(attributes_path, "demographics", url)
    attributes = _read_827(attributes_path, "demographics")

    annotations = load_annotations(root, url)

    df = pd.DataFrame(
        {
            "age": attributes["age"].to_numpy(),
            "sex": attributes["sex"].to_numpy(),
        },
        index=pd.RangeIndex(N_RECORDS, name="record_id"),
    )
    df["n_samples"] = N_SAMPLES
    df["duration_seconds"] = N_SAMPLES / SAMPLING_RATE
    df["sampling_rate"] = SAMPLING_RATE

    codes = np.array(ABNORMALITIES)
    for name in ANNOTATORS:
        flags = annotations[name].to_numpy()
        joined = np.full(N_RECORDS, "", dtype=object)
        for i in np.flatnonzero(flags.any(axis=1)):
            joined[i] = LIST_SEPARATOR.join(codes[flags[i]])
        for j, code in enumerate(ABNORMALITIES):
            df[f"{name}_{code}"] = flags[:, j]
        df[f"{name}_abnormality_codes"] = joined
        df[f"{name}_n_abnormalities"] = flags.sum(axis=1).astype(int)

    # The gold standard again, unprefixed: it is the evaluation target, and
    # `label_column` in the config names the unprefixed list.
    gold = annotations[GOLD_STANDARD].to_numpy()
    for j, code in enumerate(ABNORMALITIES):
        df[code] = gold[:, j]
    df["abnormality_codes"] = df[f"{GOLD_STANDARD}_abnormality_codes"]
    df["n_abnormalities"] = df[f"{GOLD_STANDARD}_n_abnormalities"]

    # Single-label reduction for stratification ONLY. Rarest-wins, assigned
    # commonest-first so the rarest class a record carries lands last; with AF
    # at 13 records out of 827 anything else would leave folds without one.
    order = sorted(ABNORMALITIES, key=lambda c: (int(gold[:, ABNORMALITIES.index(c)].sum()), c))
    strat = np.full(N_RECORDS, NO_ABNORMALITY, dtype=object)
    for code in reversed(order):
        strat[gold[:, ABNORMALITIES.index(code)]] = code
    df["stratify_class"] = strat

    logger.info(
        "Loaded CODE-test labels: %d records, %d annotator sets, %d with a "
        "gold-standard abnormality",
        len(df),
        len(ANNOTATORS),
        int((df["n_abnormalities"] > 0).sum()),
    )
    return df


# --------------------------------------------------------------------------- fields

#: The columns ``load_labels`` returns, as data — see ``ecgbench.labels._fields``.
#: Literal on purpose: the metadata build reads it without importing this module.
FIELDS = (
    Field(
        "age",
        "integer",
        "Age from attributes.csv, joined by row position (this release has no record "
        "identifiers)",
        unit="year",
    ),
    Field(
        "sex",
        "string",
        "Sex from attributes.csv",
        vocabulary=("M", "F"),
    ),
    Field(
        "n_samples",
        "integer",
        "4096 for every record (7 s and 10 s acquisitions zero-padded)",
        nullable=False,
    ),
    Field(
        "duration_seconds",
        "number",
        "4096 / 400 = 10.24 s for every record",
        unit="s",
        nullable=False,
    ),
    Field(
        "sampling_rate",
        "integer",
        "400 Hz for every record",
        unit="Hz",
        nullable=False,
    ),
    Field(
        "gold_standard_1dAVb",
        "boolean",
        "first-degree AV block as read by the gold standard (two cardiologists, "
        "disagreements settled by a third senior specialist) - the evaluation target",
        nullable=False,
    ),
    Field(
        "gold_standard_RBBB",
        "boolean",
        "right bundle branch block as read by the gold standard (two cardiologists, "
        "disagreements settled by a third senior specialist) - the evaluation target",
        nullable=False,
    ),
    Field(
        "gold_standard_LBBB",
        "boolean",
        "left bundle branch block as read by the gold standard (two cardiologists, "
        "disagreements settled by a third senior specialist) - the evaluation target",
        nullable=False,
    ),
    Field(
        "gold_standard_SB",
        "boolean",
        "sinus bradycardia as read by the gold standard (two cardiologists, disagreements "
        "settled by a third senior specialist) - the evaluation target",
        nullable=False,
    ),
    Field(
        "gold_standard_AF",
        "boolean",
        "atrial fibrillation as read by the gold standard (two cardiologists, disagreements "
        "settled by a third senior specialist) - the evaluation target",
        nullable=False,
    ),
    Field(
        "gold_standard_ST",
        "boolean",
        "sinus tachycardia as read by the gold standard (two cardiologists, disagreements "
        "settled by a third senior specialist) - the evaluation target",
        nullable=False,
    ),
    Field(
        "gold_standard_abnormality_codes",
        "string",
        "Comma-separated flags set by the gold standard (two cardiologists, disagreements "
        "settled by a third senior specialist) - the evaluation target; empty when none",
        nullable=False,
    ),
    Field(
        "gold_standard_n_abnormalities",
        "integer",
        "Number of flags set by the gold standard (two cardiologists, disagreements settled "
        "by a third senior specialist) - the evaluation target",
        nullable=False,
    ),
    Field(
        "cardiologist1_1dAVb",
        "boolean",
        "first-degree AV block as read by cardiologist 1, one of the two reads the gold "
        "standard was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist1_RBBB",
        "boolean",
        "right bundle branch block as read by cardiologist 1, one of the two reads the gold "
        "standard was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist1_LBBB",
        "boolean",
        "left bundle branch block as read by cardiologist 1, one of the two reads the gold "
        "standard was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist1_SB",
        "boolean",
        "sinus bradycardia as read by cardiologist 1, one of the two reads the gold "
        "standard was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist1_AF",
        "boolean",
        "atrial fibrillation as read by cardiologist 1, one of the two reads the gold "
        "standard was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist1_ST",
        "boolean",
        "sinus tachycardia as read by cardiologist 1, one of the two reads the gold "
        "standard was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist1_abnormality_codes",
        "string",
        "Comma-separated flags set by cardiologist 1, one of the two reads the gold "
        "standard was built from (not independent of it); empty when none",
        nullable=False,
    ),
    Field(
        "cardiologist1_n_abnormalities",
        "integer",
        "Number of flags set by cardiologist 1, one of the two reads the gold standard was "
        "built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist2_1dAVb",
        "boolean",
        "first-degree AV block as read by cardiologist 2, the other read the gold standard "
        "was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist2_RBBB",
        "boolean",
        "right bundle branch block as read by cardiologist 2, the other read the gold "
        "standard was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist2_LBBB",
        "boolean",
        "left bundle branch block as read by cardiologist 2, the other read the gold "
        "standard was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist2_SB",
        "boolean",
        "sinus bradycardia as read by cardiologist 2, the other read the gold standard was "
        "built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist2_AF",
        "boolean",
        "atrial fibrillation as read by cardiologist 2, the other read the gold standard "
        "was built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist2_ST",
        "boolean",
        "sinus tachycardia as read by cardiologist 2, the other read the gold standard was "
        "built from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiologist2_abnormality_codes",
        "string",
        "Comma-separated flags set by cardiologist 2, the other read the gold standard was "
        "built from (not independent of it); empty when none",
        nullable=False,
    ),
    Field(
        "cardiologist2_n_abnormalities",
        "integer",
        "Number of flags set by cardiologist 2, the other read the gold standard was built "
        "from (not independent of it)",
        nullable=False,
    ),
    Field(
        "cardiology_residents_1dAVb",
        "boolean",
        "first-degree AV block as read by two 4th-year cardiology residents, each "
        "annotating half the set",
        nullable=False,
    ),
    Field(
        "cardiology_residents_RBBB",
        "boolean",
        "right bundle branch block as read by two 4th-year cardiology residents, each "
        "annotating half the set",
        nullable=False,
    ),
    Field(
        "cardiology_residents_LBBB",
        "boolean",
        "left bundle branch block as read by two 4th-year cardiology residents, each "
        "annotating half the set",
        nullable=False,
    ),
    Field(
        "cardiology_residents_SB",
        "boolean",
        "sinus bradycardia as read by two 4th-year cardiology residents, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "cardiology_residents_AF",
        "boolean",
        "atrial fibrillation as read by two 4th-year cardiology residents, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "cardiology_residents_ST",
        "boolean",
        "sinus tachycardia as read by two 4th-year cardiology residents, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "cardiology_residents_abnormality_codes",
        "string",
        "Comma-separated flags set by two 4th-year cardiology residents, each annotating "
        "half the set; empty when none",
        nullable=False,
    ),
    Field(
        "cardiology_residents_n_abnormalities",
        "integer",
        "Number of flags set by two 4th-year cardiology residents, each annotating half the "
        "set",
        nullable=False,
    ),
    Field(
        "emergency_residents_1dAVb",
        "boolean",
        "first-degree AV block as read by two 3rd-year emergency residents, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "emergency_residents_RBBB",
        "boolean",
        "right bundle branch block as read by two 3rd-year emergency residents, each "
        "annotating half the set",
        nullable=False,
    ),
    Field(
        "emergency_residents_LBBB",
        "boolean",
        "left bundle branch block as read by two 3rd-year emergency residents, each "
        "annotating half the set",
        nullable=False,
    ),
    Field(
        "emergency_residents_SB",
        "boolean",
        "sinus bradycardia as read by two 3rd-year emergency residents, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "emergency_residents_AF",
        "boolean",
        "atrial fibrillation as read by two 3rd-year emergency residents, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "emergency_residents_ST",
        "boolean",
        "sinus tachycardia as read by two 3rd-year emergency residents, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "emergency_residents_abnormality_codes",
        "string",
        "Comma-separated flags set by two 3rd-year emergency residents, each annotating "
        "half the set; empty when none",
        nullable=False,
    ),
    Field(
        "emergency_residents_n_abnormalities",
        "integer",
        "Number of flags set by two 3rd-year emergency residents, each annotating half the "
        "set",
        nullable=False,
    ),
    Field(
        "medical_students_1dAVb",
        "boolean",
        "first-degree AV block as read by two 5th-year medical students, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "medical_students_RBBB",
        "boolean",
        "right bundle branch block as read by two 5th-year medical students, each "
        "annotating half the set",
        nullable=False,
    ),
    Field(
        "medical_students_LBBB",
        "boolean",
        "left bundle branch block as read by two 5th-year medical students, each annotating "
        "half the set",
        nullable=False,
    ),
    Field(
        "medical_students_SB",
        "boolean",
        "sinus bradycardia as read by two 5th-year medical students, each annotating half "
        "the set",
        nullable=False,
    ),
    Field(
        "medical_students_AF",
        "boolean",
        "atrial fibrillation as read by two 5th-year medical students, each annotating half "
        "the set",
        nullable=False,
    ),
    Field(
        "medical_students_ST",
        "boolean",
        "sinus tachycardia as read by two 5th-year medical students, each annotating half "
        "the set",
        nullable=False,
    ),
    Field(
        "medical_students_abnormality_codes",
        "string",
        "Comma-separated flags set by two 5th-year medical students, each annotating half "
        "the set; empty when none",
        nullable=False,
    ),
    Field(
        "medical_students_n_abnormalities",
        "integer",
        "Number of flags set by two 5th-year medical students, each annotating half the set",
        nullable=False,
    ),
    Field(
        "dnn_1dAVb",
        "boolean",
        "first-degree AV block as read by the paper's neural network thresholded to "
        "maximise F1 - a model output, not a human read",
        nullable=False,
    ),
    Field(
        "dnn_RBBB",
        "boolean",
        "right bundle branch block as read by the paper's neural network thresholded to "
        "maximise F1 - a model output, not a human read",
        nullable=False,
    ),
    Field(
        "dnn_LBBB",
        "boolean",
        "left bundle branch block as read by the paper's neural network thresholded to "
        "maximise F1 - a model output, not a human read",
        nullable=False,
    ),
    Field(
        "dnn_SB",
        "boolean",
        "sinus bradycardia as read by the paper's neural network thresholded to maximise F1 "
        "- a model output, not a human read",
        nullable=False,
    ),
    Field(
        "dnn_AF",
        "boolean",
        "atrial fibrillation as read by the paper's neural network thresholded to maximise "
        "F1 - a model output, not a human read",
        nullable=False,
    ),
    Field(
        "dnn_ST",
        "boolean",
        "sinus tachycardia as read by the paper's neural network thresholded to maximise F1 "
        "- a model output, not a human read",
        nullable=False,
    ),
    Field(
        "dnn_abnormality_codes",
        "string",
        "Comma-separated flags set by the paper's neural network thresholded to maximise F1 "
        "- a model output, not a human read; empty when none",
        nullable=False,
    ),
    Field(
        "dnn_n_abnormalities",
        "integer",
        "Number of flags set by the paper's neural network thresholded to maximise F1 - a "
        "model output, not a human read",
        nullable=False,
    ),
    Field(
        "1dAVb",
        "boolean",
        "Gold-standard flag: first-degree AV block (equal to gold_standard_1dAVb)",
        nullable=False,
    ),
    Field(
        "RBBB",
        "boolean",
        "Gold-standard flag: right bundle branch block (equal to gold_standard_RBBB)",
        nullable=False,
    ),
    Field(
        "LBBB",
        "boolean",
        "Gold-standard flag: left bundle branch block (equal to gold_standard_LBBB)",
        nullable=False,
    ),
    Field(
        "SB",
        "boolean",
        "Gold-standard flag: sinus bradycardia (equal to gold_standard_SB)",
        nullable=False,
    ),
    Field(
        "AF",
        "boolean",
        "Gold-standard flag: atrial fibrillation (equal to gold_standard_AF)",
        nullable=False,
    ),
    Field(
        "ST",
        "boolean",
        "Gold-standard flag: sinus tachycardia (equal to gold_standard_ST)",
        nullable=False,
    ),
    Field(
        "abnormality_codes",
        "string",
        "The gold standard's flags as a comma-separated list (equal to "
        "gold_standard_abnormality_codes); the evaluation target. Empty for 681 of 827 "
        "records - none of the six, which is not the same as normal",
        nullable=False,
    ),
    Field(
        "n_abnormalities",
        "integer",
        "Number of gold-standard flags set (12 records carry more than one)",
        nullable=False,
    ),
    Field(
        "stratify_class",
        "string",
        "Single-label reduction: the rarest gold-standard abnormality the record carries, "
        "else NONE. For fold construction only",
        vocabulary=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST", "NONE"),
        nullable=False,
    ),
)
