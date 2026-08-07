"""
MIT-BIH Arrhythmia labels: header comments plus a summary of the reference annotations.

Nothing machine-readable ships with this dataset. Each record's ``.hea`` carries
two or more comment lines, and the reference annotations live in a companion
``.atr``::

    # 68 M 1960 2851 x1
    # Digoxin, Hydrochlorthiazide, Inderal, KCl
    # The PVCs are uniform and late-cycle.  This record was taken from the same
    # analog tape as record 201.

Line 1 is ``<age> <sex> <tape> <recorder> x<speed>``, line 2 is the subject's
medications (``None`` when there were none), and any further lines are the
free-text clinical description of *this* record. All of it is exposed verbatim,
plus per-symbol annotation counts and time-in-rhythm derived from the ``.atr``.

The two numeric fields on line 1 are not documented in the header itself; both
were decoded from the files and cross-checked against the shipped directory
(``mitdbdir/intro.htm``), which is why this module can key patient grouping on
one of them:

- **Field 3 is the analog tape**, and therefore the subject. It is distinct for
  every record except 201 and 202, which share ``1960`` — exactly the pair the
  directory names ("Records 201 and 202 came from the same male subject"), giving
  **47 subjects over 48 records**. Recomputed demographics agree with the
  directory to the record: 25 men aged 32-89 and 22 women aged 23-89.
- **Field 4 is the Del Mar Avionics recorder.** Grouping records by it reproduces
  the directory's recorder table exactly — ``654`` is its recorder E (13
  records), ``1629`` is G (8), ``694`` is F (8), ``167`` is A (5), ``2851`` is H
  (4), ``3655`` is I (4), ``653`` is D (3), ``171`` is B (1), ``356`` is C (1) —
  and record 208, the one the directory says was never traced to a recorder,
  is the one record whose field reads ``N/A``. Exposed as ``recorder`` because
  recorder-specific artefact is a real confounder in this data.

The ``x1``/``x2`` suffix is the playback speed at digitisation. The 18 records
marked ``x2`` are exactly the set the directory lists as played back at twice
real time (112, 115-124, 205, 220, 223, 230-234).

Other quirks worth knowing, all verified against the files:

- **Age is ``-1`` for records 103 and 219** — unknown, not a parse failure. This
  module returns them as NaN, so the mean age is over 45 subjects, not 47.
- **Two annotation files are not part of the released annotator set and are
  ignored.** ``102-0.atr`` holds annotations byte-identical in content to
  ``102.atr`` (2,192 annotations, same symbol counts) but has no header of its
  own, and ``108.at_`` is a superseded copy of ``108.atr``. The shipped
  ``ANNOTATORS`` file lists ``atr`` and nothing else.
- **The ``x_mitdb/`` subdirectory is not part of this dataset.** Its 23 records
  are ``xform`` extracts of the first 600 s of records already here, with the
  baseline removed — correlation 1.0 against the parent over all 216,000
  samples. Including them would put the same recording in the partition twice,
  so this loader takes its record list from the shipped ``RECORDS`` file, which
  names the 48 and none of the extracts.
- **Beat positions are reliable, but the two channels are not synchronous.** The
  directory records skew of up to 40 ms between the two signals, fixed per
  recorder plus a variable component from tape wobble. Anything comparing the
  channels sample-for-sample has to allow for it.
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

#: Samples per record. Uniform across all 48 (650,000 at 360 Hz = 1805.6 s), and
#: used to close the final rhythm episode, which has no annotation after it.
RECORD_SAMPLES = 650000

#: PhysioBank beat symbols occurring in this release, descending by frequency.
#: These fifteen sum to 109,494 — the count the database has published since
#: 1980 — and the remaining 3,153 annotations are the non-beat markers below.
BEAT_SYMBOLS = ("N", "L", "R", "V", "/", "A", "f", "F", "j", "a", "E", "J", "Q", "e", "S")

BEAT_NAMES = {
    "N": "normal beat",
    "L": "left bundle branch block beat",
    "R": "right bundle branch block beat",
    "V": "premature ventricular contraction",
    "/": "paced beat",
    "A": "atrial premature beat",
    "f": "fusion of paced and normal beat",
    "F": "fusion of ventricular and normal beat",
    "j": "nodal (junctional) escape beat",
    "a": "aberrated atrial premature beat",
    "E": "ventricular escape beat",
    "J": "nodal (junctional) premature beat",
    "Q": "unclassifiable beat",
    "e": "atrial escape beat",
    "S": "supraventricular premature beat",
}

#: Non-beat annotation symbols, mapped to the column that counts them. These must
#: never be added to ``n_beats`` — doing so is how a "110,000 beats" figure gets
#: quoted for a database with 109,494 of them.
NON_BEAT_COLUMNS = {
    "+": "n_rhythm_changes",
    "~": "n_signal_quality_changes",
    "!": "n_ventricular_flutter_waves",
    '"': "n_comment_annotations",
    "x": "n_nonconducted_p_waves",
    "|": "n_isolated_artifacts",
    "[": "n_vfl_episode_starts",
    "]": "n_vfl_episode_ends",
}

#: Rhythm codes carried in the ``aux_note`` of a ``+`` annotation. Every one of
#: the 1,291 ``+`` annotations in the release carries one of these.
RHYTHM_NAMES = {
    "N": "normal sinus rhythm",
    "AFIB": "atrial fibrillation",
    "AFL": "atrial flutter",
    "B": "ventricular bigeminy",
    "T": "ventricular trigeminy",
    "VT": "ventricular tachycardia",
    "SVTA": "supraventricular tachyarrhythmia",
    "P": "paced rhythm",
    "NOD": "nodal (AV junctional) rhythm",
    "IVR": "idioventricular rhythm",
    "VFL": "ventricular flutter",
    "BII": "second degree heart block",
    "AB": "atrial bigeminy",
    "PREX": "pre-excitation (WPW)",
    "SBR": "sinus bradycardia",
}

#: Free-text ``aux_note`` values carried by ``"`` comment annotations, i.e. every
#: aux note that is not a rhythm. Counted per record because they mark exactly the
#: places a beat detector has nothing to detect.
COMMENT_NOTE_NAMES = {
    "MISSB": "missed beat",
    "PSE": "pause",
    "TS": "tape slippage",
}

#: The two documented halves of the database, and the stratification label.
RANDOM_SAMPLE = "random_sample"
SELECTED = "selected"

#: Record numbers below this are the randomly chosen half; at or above it, the
#: half selected for rare phenomena. Straight from the directory's introduction.
SELECTED_FROM = 200

#: Header comment 1: age, sex, analog tape, recorder, playback speed.
_DEMOGRAPHICS_RE = re.compile(
    r"^#\s*(?P<age>-?\d+)\s+(?P<sex>[MF])\s+(?P<tape>\S+)\s+(?P<recorder>\S+)"
    r"\s+x(?P<speed>\d+)\s*$"
)


def parse_header_comments(hea_path: Path) -> dict[str, object]:
    """Parse one MIT-BIH header into demographics, provenance and free text.

    Missing pieces come back empty rather than raising, so one malformed header
    cannot fail the whole scan — the validation engine flags genuinely broken
    records via ``corrupt_header``.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    comments = [line for line in lines if line.startswith("#")]

    out: dict[str, object] = {
        "age": np.nan,
        "sex": "",
        "patient_id": "",
        "recorder": "",
        "digitised_at_double_speed": False,
        "medications": "",
        "description": "",
    }

    if comments:
        match = _DEMOGRAPHICS_RE.match(comments[0])
        if match:
            age = int(match.group("age"))
            # -1 is the release's "unknown", on records 103 and 219. Keeping it as
            # a number would drag every mean age down by more than a year.
            out["age"] = float(age) if age >= 0 else np.nan
            out["sex"] = match.group("sex")
            out["patient_id"] = f"tape{match.group('tape')}"
            out["recorder"] = match.group("recorder")
            out["digitised_at_double_speed"] = match.group("speed") == "2"
        else:
            logger.warning("Unparsed demographics comment in %s: %r", hea_path.name, comments[0])

    if len(comments) > 1:
        medications = comments[1].lstrip("#").strip()
        # "None" is the release's way of saying no medications; an empty string
        # says the same thing without inviting a spurious drug name.
        out["medications"] = "" if medications.lower() == "none" else medications

    if len(comments) > 2:
        # Wrapped free text: the line breaks are typographic, not semantic, and
        # the source double-spaces after full stops.
        out["description"] = " ".join(line.lstrip("#").strip() for line in comments[2:]).strip()

    return out


def parse_lead_names(hea_path: Path) -> list[str]:
    """Lead names in the order this record stores them.

    Read per record rather than assumed, because they vary: 40 records store
    MLII/V1 but eight do not, and record 114 stores its two signals reversed
    (V5 then MLII). ``config.record_lead_layouts`` is what makes
    ``ECGDataset(leads=...)`` honour that; this column is what lets a user see it
    without opening a header.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    names = []
    for line in lines[1:]:
        if line.startswith("#") or not line.strip():
            continue
        names.append(line.split()[-1])
    return names


def summarise_annotations(record_path: Path) -> dict[str, object]:
    """Summarise one record's reference annotations.

    Returns per-symbol beat counts, the non-beat marker counts, seconds spent in
    each annotated rhythm, and the dominant rhythm by duration.
    """
    import wfdb

    counts: dict[str, object] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
    counts.update({column: 0 for column in NON_BEAT_COLUMNS.values()})
    counts.update({f"note_{note}": 0 for note in COMMENT_NOTE_NAMES})
    counts.update({f"rhythm_secs_{code}": 0.0 for code in RHYTHM_NAMES})
    counts["n_beats"] = 0
    counts["n_annotations"] = 0
    counts["rhythms"] = ""
    counts["dominant_rhythm"] = ""
    counts["dominant_rhythm_fraction"] = np.nan

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .atr must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    beat_set = set(BEAT_SYMBOLS)
    unexpected: set[str] = set()
    episodes: list[tuple[str, int]] = []

    counts["n_annotations"] = len(annotation.symbol)
    for symbol, sample, aux in zip(annotation.symbol, annotation.sample, annotation.aux_note):
        note = str(aux).strip().strip("\x00")
        if symbol in beat_set:
            counts[f"beat_{symbol}"] = int(counts[f"beat_{symbol}"]) + 1
            counts["n_beats"] = int(counts["n_beats"]) + 1
        elif symbol in NON_BEAT_COLUMNS:
            column = NON_BEAT_COLUMNS[symbol]
            counts[column] = int(counts[column]) + 1
            if symbol == "+" and note.startswith("("):
                episodes.append((note[1:], int(sample)))
            elif symbol == '"' and note in COMMENT_NOTE_NAMES:
                counts[f"note_{note}"] = int(counts[f"note_{note}"]) + 1
        else:
            # Worth seeing rather than silently dropping: it would mean this
            # release uses symbols this module does not know about.
            unexpected.add(symbol)

    if unexpected:
        logger.warning(
            "%s: annotation symbols outside BEAT_SYMBOLS and NON_BEAT_COLUMNS, " "not counted: %s",
            record_path.name,
            sorted(unexpected),
        )

    # A rhythm annotation marks where an episode STARTS; it runs until the next
    # one, and the last runs to the end of the record.
    fs = float(getattr(annotation, "fs", 360) or 360)
    for i, (code, start) in enumerate(episodes):
        end = episodes[i + 1][1] if i + 1 < len(episodes) else RECORD_SAMPLES
        if code not in RHYTHM_NAMES:
            logger.warning("%s: unknown rhythm code %r", record_path.name, code)
            continue
        counts[f"rhythm_secs_{code}"] = float(counts[f"rhythm_secs_{code}"]) + (end - start) / fs

    seconds = {
        code: float(counts[f"rhythm_secs_{code}"])
        for code in RHYTHM_NAMES
        if float(counts[f"rhythm_secs_{code}"]) > 0
    }
    if seconds:
        ordered = sorted(seconds, key=lambda c: -seconds[c])
        counts["rhythms"] = "|".join(ordered)
        counts["dominant_rhythm"] = ordered[0]
        counts["dominant_rhythm_fraction"] = seconds[ordered[0]] / sum(seconds.values())

    return counts


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob,
    which is what keeps the ``x_mitdb/`` extracts — 600 s copies of records
    already here — out of the partition.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. MIT-BIH labels live in the record "
            "headers and .atr annotation files, so point data_path at the dataset "
            "root — the flat directory holding 100.hea, RECORDS and ANNOTATORS. "
            "Get it from https://physionet.org/content/mitdb/1.0.0/"
        )

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue
        row: dict[str, object] = {"record_name": name}
        row.update(parse_header_comments(hea))
        leads = parse_lead_names(hea)
        row["lead_names"] = "|".join(leads)
        row.update(summarise_annotations(hea.with_suffix("")))
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    df["pvc_fraction"] = (df["beat_V"] / df["n_beats"]).where(df["n_beats"] > 0)
    logger.info(
        "Parsed %d MIT-BIH records from %d subjects; %d beat annotations, " "%d lead layouts",
        len(df),
        df["patient_id"].nunique(),
        int(df["n_beats"].sum()),
        df["lead_names"].nunique(),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``record_group``, the database's own two halves, and use it to stratify.

    This is the **only** derivation of the stratification label — ``MITDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    The database was assembled in two deliberate halves, and the directory says so:
    records 100-124 were "chosen at random" from the Beth Israel Holter collection
    to be representative, and records 200-234 were "selected to include a variety of
    rare but clinically important phenomena that would not be well-represented by a
    small random sample". Stratifying on that split — 23 against 25 — is what stops
    a fold from being all rare-arrhythmia records or none, which with only 48
    records is otherwise easy to get.

    It is not a clinical grouping. Train on ``dominant_rhythm``, ``rhythms``, the
    ``rhythm_secs_*`` columns and the beat counts.
    """
    out = df.copy()
    out["record_group"] = np.where(
        out["record_name"].astype(int) >= SELECTED_FROM, SELECTED, RANDOM_SAMPLE
    )
    out["stratify_class"] = out["record_group"]
    logger.info("Record groups: %s", out["record_group"].value_counts().to_dict())
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return MIT-BIH Arrhythmia labels indexed by record name.

    Columns:

    - ``dominant_rhythm``, ``rhythms``, ``dominant_rhythm_fraction`` and
      ``rhythm_secs_<CODE>`` — time spent in each annotated rhythm (see
      :data:`RHYTHM_NAMES`). Derived from the ``+`` annotations, each of which
      opens an episode running to the next one.
    - ``beat_N`` … ``beat_S`` — reference beat counts per symbol (see
      :data:`BEAT_NAMES`), with ``n_beats`` and ``pvc_fraction``. **Multi-class
      per record**: almost every record carries several beat types, so a single
      record-level beat label does not exist.
    - ``n_rhythm_changes``, ``n_signal_quality_changes``, ``n_isolated_artifacts``
      and the rest of :data:`NON_BEAT_COLUMNS` — annotation markers that are
      **not** beats and are excluded from ``n_beats``.
    - ``note_MISSB``, ``note_PSE``, ``note_TS`` — missed beats, pauses and tape
      slippage, from the ``"`` comment annotations. Record 231 alone holds 427 of
      the 428 missed-beat markers.
    - ``patient_id`` — ``tape<N>``, the analog tape the record came from. 47
      subjects over 48 records; 201 and 202 share tape 1960. Folds are grouped by
      this.
    - ``lead_names`` — the two leads *this* record stores, pipe-separated, because
      the layout is not constant: 40 records are MLII/V1, and record 114 stores
      them reversed.
    - ``age``, ``sex``, ``medications``, ``description`` — subject-level except
      ``description``, which describes this record.
    - ``recorder``, ``digitised_at_double_speed`` — acquisition provenance, both
      confounders worth being able to control for.
    - ``record_group`` / ``stratify_class`` — the database's random half against
      its selected half, **for fold construction**. Not a clinical label.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
