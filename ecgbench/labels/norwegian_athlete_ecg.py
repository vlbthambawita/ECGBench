"""
Norwegian athlete labels: two free-text interpretations per record, from the header.

The release ships no metadata table. Every record's labels are two comment lines
appended to its ``.hea``:

    #SL12: Sinus bradycardia with marked sinus arrhythmia, Right axis deviation, Borderline ECG
    #C: Sinus arrhythmia,  Normal ECG

``#SL12:`` is the GE Marquette SL12 algorithm (version 23, v243); ``#C:`` is a
cardiologist trained in athlete ECG interpretation. Both are comma-separated
statement lists whose **last** statement is an overall verdict. This module reads
the headers directly rather than a generated CSV, so ``load_labels`` works on a
fresh copy of the dataset with no prior ``ecgbench splits`` run — the splitter
then builds its metadata CSV from this loader instead of re-parsing the headers.

**Comparing the two readings is the intended use of this dataset**, so nothing is
merged: every field is prefixed ``sl12_`` or ``cardiologist_``, and both raw
strings are kept verbatim.

Four parsing traps, all verified against all 28 records:

- **A comma does not reliably end a statement.** Three GE statements contain
  commas of their own — ``"ST elevation, consider early repolarization,
  pericarditis, or injury"`` alone splits into four fragments. Splitting naively
  turns 2 records into 7 spurious findings.
- **...and the obvious fix is wrong.** The tempting rule is "a fragment starting
  lowercase is a continuation", but the cardiologist writes genuine statements in
  lowercase: ath_005 has ``"Sinus bradycardia, normal sinus rhythm, First degree
  AV block"`` and ath_017 has ``"first degree AV block"``. Both are real separate
  findings. So this module protects an explicit list of known comma-carrying
  statements (:data:`MULTI_COMMA_STATEMENTS`) and splits everything else, rather
  than guessing from capitalisation. The dataset is frozen at v1.0.0, so the list
  is complete; an unrecognised verdict raises rather than being silently dropped.
- **Asterisk noise carries meaning.** 4 records open with
  ``"***Critical test result: STEMI"`` and repeat it near the end as
  ``"** ** ACUTE MI/STEMI** **"``. Both are extracted into their own columns
  rather than left in the findings list as punctuation.
- **One verdict is misspelt.** ath_010 ends ``"Abnormal EKG"``, not "ECG". The
  normalised ``*_verdict`` column folds it into ``"Abnormal ECG"``; the raw
  string keeps it.

There are **no per-record demographics**. Age, sex and sport are published only
as cohort aggregates on the landing page, so they cannot be joined to a record
and are not exposed here.

Label distribution is degenerate and worth knowing before you model it: the
cardiologist reads 26 of 28 records as "Normal ECG" and 2 as "Borderline ECG",
with no abnormal reading at all. ``cardiologist_primary_rhythm`` (16 / 7 / 5) is
the only reasonably balanced ground-truth field, which is why the splitter
stratifies on it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Newline-delimited list of record stems shipped in the dataset root.
RECORDS_FILE = "RECORDS"

#: Header comment prefix -> column prefix.
SOURCES = {"SL12": "sl12", "C": "cardiologist"}

#: Statements that contain commas themselves. Their commas are masked before the
#: statement list is split, then restored. Complete for v1.0.0 — these three are
#: the only ones in the release, and they occur in ath_024 and ath_027 only.
MULTI_COMMA_STATEMENTS = (
    "ST elevation, consider early repolarization, pericarditis, or injury",
    "Minimal voltage criteria for LVH, may be normal variant",
    "ST elevation, probably due to early repolarization",
)

#: Sentinel standing in for a protected comma. Cannot occur in the headers.
_COMMA_SENTINEL = "\x00"

#: Overall-verdict vocabulary, lowercased -> canonical. ath_010 spells it "EKG".
VERDICTS = {
    "normal ecg": "Normal ECG",
    "otherwise normal ecg": "Otherwise normal ECG",
    "borderline ecg": "Borderline ECG",
    "abnormal ecg": "Abnormal ECG",
    "abnormal ekg": "Abnormal ECG",
}

#: Verdicts counted as a normal reading. GE's "Otherwise normal ECG" means normal
#: apart from the rhythm finding it lists — grouping it with "Normal ECG" is an
#: interpretation, and it is the one the ``*_is_normal`` columns make. Compare
#: ``*_verdict`` directly if you would rather draw the line elsewhere.
NORMAL_VERDICTS = frozenset({"Normal ECG", "Otherwise normal ECG"})

#: The complement — a reading that flags something.
NON_NORMAL_VERDICTS = frozenset({"Borderline ECG", "Abnormal ECG"})

#: Rhythms the cardiologist opens a reading with. Every one of the 28 records
#: starts with one of these, which is what makes primary_rhythm well defined.
RHYTHM_VOCABULARY = {
    "normal sinus rhythm": "Normal sinus rhythm",
    "sinus arrhythmia": "Sinus arrhythmia",
    "sinus bradycardia": "Sinus bradycardia",
}

_CRITICAL_RE = re.compile(r"^Critical test result\s*:\s*(.+)$", re.IGNORECASE)
_ACUTE_RE = re.compile(r"^ACUTE\b.*", re.IGNORECASE)


def _normalise(statement: str) -> str:
    """Strip the asterisk decoration and collapse whitespace in one statement."""
    return re.sub(r"\s+", " ", statement.replace("*", " ")).strip(" ,")


def split_statements(text: str) -> list[str]:
    """Split one interpretation line into its statements.

    Commas inside the statements listed in :data:`MULTI_COMMA_STATEMENTS` are
    masked first, so those statements survive intact instead of shattering into
    fragments.
    """
    protected = text
    for index, statement in enumerate(MULTI_COMMA_STATEMENTS):
        protected = protected.replace(
            statement, statement.replace(",", f"{_COMMA_SENTINEL}{index}{_COMMA_SENTINEL}")
        )

    statements = []
    for part in protected.split(","):
        for index in range(len(MULTI_COMMA_STATEMENTS)):
            part = part.replace(f"{_COMMA_SENTINEL}{index}{_COMMA_SENTINEL}", ",")
        cleaned = _normalise(part)
        if cleaned:
            statements.append(cleaned)
    return statements


def _read_comments(hea_path: Path) -> dict[str, str]:
    """Return {source: raw interpretation string} for one header file."""
    found: dict[str, str] = {}
    with open(hea_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            key, separator, value = line[1:].partition(":")
            if separator and key.strip() in SOURCES:
                found[key.strip()] = value.strip()

    missing = [s for s in SOURCES if s not in found]
    if missing:
        raise ValueError(
            f"{hea_path.name} is missing the {missing} interpretation comment line(s). "
            f"Every record in this dataset carries both #SL12: and #C:."
        )
    return found


def _parse_interpretation(raw: str, record: str, source: str) -> dict[str, object]:
    """Parse one interpretation line into findings, verdict and the alert flags."""
    statements = split_statements(raw)
    if not statements:
        raise ValueError(f"{record}: empty #{source}: interpretation line")

    critical: str | None = None
    acute: str | None = None
    findings: list[str] = []
    for statement in statements:
        match = _CRITICAL_RE.match(statement)
        if match:
            critical = match.group(1).strip()
            continue
        if _ACUTE_RE.match(statement):
            acute = statement
            continue
        findings.append(statement)

    verdict_raw = findings.pop() if findings else ""
    verdict = VERDICTS.get(verdict_raw.lower())
    if verdict is None:
        # A silent None here would look like a record with no reading at all.
        raise ValueError(
            f"{record}: unrecognised #{source}: verdict {verdict_raw!r}. "
            f"Known verdicts: {sorted(set(VERDICTS.values()))}. Add it to "
            "ecgbench.labels.norwegian_athlete_ecg.VERDICTS if the release changed."
        )

    return {
        "raw": raw,
        "findings": findings,
        "verdict": verdict,
        "verdict_raw": verdict_raw,
        "critical_test_result": critical,
        "acute_alert": acute,
    }


def _primary_rhythm(findings: list[str], record: str) -> str:
    """Return the rhythm the cardiologist's reading opens with."""
    first = findings[0] if findings else ""
    rhythm = RHYTHM_VOCABULARY.get(first.lower())
    if rhythm is None:
        raise ValueError(
            f"{record}: first cardiologist finding {first!r} is not a known rhythm. "
            f"Known: {sorted(set(RHYTHM_VOCABULARY.values()))}. This is the "
            "stratification label, so a wrong value would silently skew the folds."
        )
    return rhythm


def _record_names(data_path: Path, config: DatasetConfig) -> list[str]:
    """Record stems, from the shipped RECORDS file if present, else by glob."""
    records_file = data_path / RECORDS_FILE
    if records_file.exists():
        names = [line.strip() for line in records_file.read_text().splitlines()]
        return [n for n in names if n]

    logger.warning("%s not found — falling back to globbing *.hea", records_file)
    names = sorted(p.stem for p in data_path.glob("*.hea"))
    if not names:
        raise FileNotFoundError(
            f"No .hea header files in {data_path}. Point data_path at the dataset "
            f"root, the directory holding ath_001.hea and RECORDS (see {config.url})."
        )
    return names


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return both interpretations of every record, indexed by record ID.

    Columns, per source (``sl12_`` = GE Marquette SL12 algorithm,
    ``cardiologist_`` = cardiologist trained in athlete ECG interpretation):

        *_raw                    the header line verbatim, including the "EKG"
                                 misspelling and stray asterisks
        *_findings               list of statements excluding the verdict and the
                                 two alert flags; empty for a bare-verdict reading
        *_verdict                one of Normal ECG / Otherwise normal ECG /
                                 Borderline ECG / Abnormal ECG
        *_verdict_raw            the verdict as written (ath_010: "Abnormal EKG")
        *_critical_test_result   e.g. "STEMI" — SL12 only, 4 records, else None
        *_acute_alert            e.g. "ACUTE MI/STEMI" — SL12 only, 4 records
        *_is_normal              verdict in NORMAL_VERDICTS, i.e. "Normal ECG" or
                                 GE's "Otherwise normal ECG" (13 SL12 / 26
                                 cardiologist). That grouping is a judgement —
                                 compare *_verdict yourself to draw it elsewhere.
        *_n_findings             len(*_findings)

    Plus three derived fields:

        cardiologist_primary_rhythm   the rhythm the cardiologist's reading opens
                                      with (16 Normal sinus rhythm / 7 Sinus
                                      arrhythmia / 5 Sinus bradycardia). This is
                                      the stratification label — it is a single
                                      reduction of a multi-statement reading, so
                                      use ``cardiologist_findings`` to train.
        verdicts_match                the two verdict strings are equal (4 of 28,
                                      all of them "Normal ECG")
        sl12_overcalls                SL12 returned borderline/abnormal where the
                                      cardiologist read normal (13 of 28). This is
                                      the quantity the dataset was published to
                                      show; SL12 is the system under test here,
                                      not the ground truth.

    Every field is multi-statement: a record carries 0-6 findings per source, so
    ``*_findings`` is a list and does not reduce to one class. Not one record is
    read as abnormal by the cardiologist, so there is no positive class for a
    normal/abnormal task on human ground truth.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    if not data_path.is_dir():
        raise LabelSourceMissingError(
            f"Norwegian athlete labels come from the per-record .hea headers, but "
            f"{data_path} is not a directory. ECGBench publishes fold CSVs only — "
            f"point data_path at a full local copy (see {config.url})."
        )

    names = _record_names(data_path, config)
    rows: list[dict[str, object]] = []
    for name in names:
        hea_path = data_path / f"{name}.hea"
        if not hea_path.exists():
            raise LabelSourceMissingError(
                f"{RECORDS_FILE} lists {name}, but {hea_path} is missing. The labels "
                f"for this dataset are in the headers, so point data_path at a "
                f"complete local copy (see {config.url})."
            )

        comments = _read_comments(hea_path)
        row: dict[str, object] = {config.record_id_column: name}
        for source, prefix in SOURCES.items():
            parsed = _parse_interpretation(comments[source], name, source)
            for field, value in parsed.items():
                row[f"{prefix}_{field}"] = value
            row[f"{prefix}_is_normal"] = parsed["verdict"] in NORMAL_VERDICTS
            row[f"{prefix}_n_findings"] = len(parsed["findings"])
        row["cardiologist_primary_rhythm"] = _primary_rhythm(
            row["cardiologist_findings"], name
        )
        rows.append(row)

    out = pd.DataFrame(rows).set_index(config.record_id_column)
    out["verdicts_match"] = out["sl12_verdict"] == out["cardiologist_verdict"]
    out["sl12_overcalls"] = (
        out["sl12_verdict"].isin(NON_NORMAL_VERDICTS) & out["cardiologist_is_normal"]
    )

    n_critical = int(out["sl12_critical_test_result"].notna().sum())
    overcalled = int(out["sl12_overcalls"].sum())
    logger.info(
        "Loaded Norwegian athlete labels: %d records; SL12 raised %d critical alert(s) "
        "and read %d record(s) as borderline/abnormal that the cardiologist called normal",
        len(out), n_critical, overcalled,
    )
    return out
