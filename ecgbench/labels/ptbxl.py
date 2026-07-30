"""
PTB-XL labels: SCP-ECG statements, and the diagnostic hierarchy above them.

``ptbxl_database.csv`` stores ``scp_codes`` as a dict-string of statement ->
likelihood. The statement taxonomy lives in ``scp_statements.csv``, which flags
each statement as diagnostic, form and/or rhythm, and gives diagnostic
statements a ``diagnostic_class`` (the five superclasses) and a
``diagnostic_subclass``.

This module is the **single** source of that derivation. ``PTBXLSplitter`` used
to carry its own hardcoded code -> superclass dict, which had drifted from the
shipped statement table: it omitted five diagnostic codes (ANEUR, EL, IPLMI,
IPMI, ISCAN) and treated seven non-diagnostic ones as diagnostic (APTS, ISCA,
ISCI, NT_, STD_, STE_, TAB_), putting 465 records in OTHER where the table gives
411. Deriving from the shipped file makes that class of drift impossible.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: The five diagnostic superclasses, in a fixed order so multi-hot target
#: columns mean the same thing across runs.
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

#: Statement table shipped alongside ptbxl_database.csv.
STATEMENTS_CSV = "scp_statements.csv"

#: Label for records whose statements are all non-diagnostic.
OTHER = "OTHER"

#: Columns copied straight from ptbxl_database.csv.
_PASSTHROUGH = [
    "patient_id", "age", "sex", "height", "weight", "report", "heart_axis",
    "recording_date", "device", "validated_by_human", "strat_fold",
]


def _parse_scp_codes(value: str) -> dict[str, float]:
    """Parse the dict-string in the scp_codes column."""
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_statement_table(data_path: Path | str) -> pd.DataFrame:
    """Load scp_statements.csv, indexed by SCP statement code."""
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    path = data_path / STATEMENTS_CSV
    if not path.exists():
        raise LabelSourceMissingError(
            f"PTB-XL labels need {STATEMENTS_CSV} next to ptbxl_database.csv, but it "
            f"is not in {data_path}. It ships with the dataset — see "
            "https://physionet.org/content/ptb-xl/1.0.3/"
        )
    return pd.read_csv(path, index_col=0)


def load_labels(data_path: Path, config: DatasetConfig) -> pd.DataFrame:
    """Return PTB-XL labels indexed by ``ecg_id``.

    Columns:
        scp_codes           dict of statement -> likelihood, as recorded
        diagnostic_codes    statements flagged diagnostic in scp_statements.csv
        form_codes          statements flagged form
        rhythm_codes        statements flagged rhythm
        superclasses        sorted diagnostic_class values (multi-label)
        subclasses          sorted diagnostic_subclass values (multi-label)
        primary_superclass  single highest-likelihood superclass, else "OTHER"
        plus patient_id, age, sex, height, weight, report, heart_axis,
        recording_date, device, validated_by_human, strat_fold

    ``sex`` and the other demographic columns are passed through exactly as
    PTB-XL encodes them (sex: 0 = male, 1 = female; ages over 89 stored as 300),
    so values match anything else built against the source CSV.

    Multi-label by design: 5,144 of 21,799 records in v1.0.3 carry more than one
    superclass, and 411 carry none.

    ``primary_superclass`` is a lossy single-label view kept for stratification —
    **do not train on it.** 2,358 records (10.8%) have two or more superclasses
    tied on likelihood, so which one wins is decided by an arbitrary fixed order,
    not by the data. Train on ``superclasses`` (or ``subclasses``) instead.
    """
    from ecgbench.labels import LabelSourceMissingError

    source = config.labels.source_csv if config.labels else config.metadata_csv
    db_path = data_path / (source or config.metadata_csv)
    if not db_path.exists():
        raise LabelSourceMissingError(
            f"PTB-XL labels come from {db_path.name}, which is not in {data_path}. "
            "ECGBench publishes fold CSVs only — point data_path at a full local "
            "copy of PTB-XL."
        )

    db = pd.read_csv(db_path)
    stmt = load_statement_table(data_path)

    diagnostic = stmt.index[stmt.get("diagnostic", 0) == 1]
    to_class = stmt.loc[diagnostic, "diagnostic_class"].to_dict()
    to_subclass = stmt.loc[diagnostic, "diagnostic_subclass"].to_dict()
    form = set(stmt.index[stmt.get("form", 0) == 1])
    rhythm = set(stmt.index[stmt.get("rhythm", 0) == 1])

    codes = db["scp_codes"].map(_parse_scp_codes)

    out = pd.DataFrame(index=pd.Index(db[config.record_id_column], name="ecg_id"))
    out["scp_codes"] = codes.to_numpy()
    out["diagnostic_codes"] = [
        sorted(c for c in d if c in to_class) for d in codes
    ]
    out["form_codes"] = [sorted(c for c in d if c in form) for d in codes]
    out["rhythm_codes"] = [sorted(c for c in d if c in rhythm) for d in codes]
    out["superclasses"] = [
        sorted({to_class[c] for c in d if c in to_class}) for d in codes
    ]
    out["subclasses"] = [
        sorted({to_subclass[c] for c in d if c in to_subclass and pd.notna(to_subclass[c])})
        for d in codes
    ]
    out["primary_superclass"] = [_primary_superclass(d, to_class) for d in codes]

    for col in _PASSTHROUGH:
        if col in db.columns:
            out[col] = db[col].to_numpy()

    logger.info(
        "Loaded PTB-XL labels: %d records, %d with no diagnostic superclass",
        len(out), int((out["superclasses"].map(len) == 0).sum()),
    )
    return out


def _primary_superclass(scp_codes: dict[str, float], to_class: dict[str, str]) -> str:
    """Reduce a record's statements to one superclass by summed likelihood.

    Ties break on ``SUPERCLASSES`` order so the result is deterministic — a plain
    ``max()`` over a dict would depend on the order codes happen to appear in.
    Ties are not rare (10.8% of records), and PTB-XL likelihood 0.0 means
    "present but not graded" rather than "absent", so this reduction is a
    convenience for stratification only, never a training target.
    """
    scores: dict[str, float] = {}
    for code, likelihood in scp_codes.items():
        superclass = to_class.get(code)
        if superclass:
            scores[superclass] = scores.get(superclass, 0.0) + float(likelihood)
    if not scores:
        return OTHER
    return min(scores, key=lambda s: (-scores[s], SUPERCLASSES.index(s)))


def multi_hot(label_lists, classes=None):
    """Encode lists of class names as a (n, len(classes)) float32 array.

    Returns a numpy array; wrap in ``torch.from_numpy`` for a tensor.
    """
    import numpy as np

    classes = list(classes) if classes is not None else SUPERCLASSES
    index = {name: i for i, name in enumerate(classes)}
    out = np.zeros((len(label_lists), len(classes)), dtype="float32")
    for row, labels in enumerate(label_lists):
        for label in labels:
            position = index.get(label)
            if position is not None:
                out[row, position] = 1.0
    return out
