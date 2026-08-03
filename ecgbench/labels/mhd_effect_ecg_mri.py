"""
MHD-effect labels: acquisition conditions from the header, QRS counts from the .qrs.

This dataset has **no diagnosis to predict**. Every subject was healthy, and the
only annotations are manually marked QRS positions with no normal/ectopic
distinction. What varies — and what a user models against — is the *acquisition
condition*: magnetic field strength, B0 orientation, body position in the bore,
and which of the two ECG devices was used. All of it lives in each record's
``.hea`` comment block, in ``#--Key:Value`` form:

    #--Magnetic field strength:3T
    #--MR scanner:Siemens Magnetom Skyra
    #--Orientation of the static magnetic field (B0):Horizontal
    #--ECG recorder:Getemed CM 3000, 12-lead Holter ECG
    #--Sex:Male
    #--Positon in the scanner:Feet first (Ff)      <- the source's spelling

This module reads the headers and annotation files directly, so ``load_labels``
works on a fresh copy of the dataset with no prior ``ecgbench splits`` run; the
splitter then builds its metadata CSV from this loader rather than re-parsing.

Everything below was verified against all 53 records, and all 163 shipped files
match the release's own ``SHA256SUMS.txt`` — so these are upstream properties,
not download damage.

**The release contradicts itself, and this module does not paper over it.**

- **53 records, not 43.** The README, the PhysioNet page and the 2017 CinC paper
  all say 43 records / 23 subjects / 203 min. ``RECORDS`` lists 53, 53 exist, and
  they total 226.6 min. The release grew after publication and the prose did not.
- **``ECGMRI3T01Hf``'s filename says head-first; its header says "Feet first
  (Ff)".** Nothing in the release says which is right. Both are exposed —
  ``position`` from the filename, ``position_header`` from the header — and
  ``position_disagrees`` flags it. The filename is the more likely correct one
  (subject 3T01's record set is Ff/Hf/Out, a deliberate protocol, and the README
  documents the naming convention), but this module will not choose for you.
- **``ECGMRI1T01Out``'s field strength says "1T" and its B0 "Vertical" even though
  its position is "Outside the scanner".** The other 9 reference records say
  "Outside the scanner" for all three fields. So filtering references on the
  header's field-strength string silently misses one of the ten. Use
  :data:`REFERENCE_POSITION`/``is_reference``, or ``condition``, both of which are
  derived from the filename and position and get all 10.
- **No record is a breath-hold recording.** The README says breath-hold protocols
  "are noted in the header files"; all 53 headers say "Spontaneous respiration".
  ``respiration`` is exposed anyway, so a future release that adds them will show
  up rather than being assumed away.
- **A fourth scanner ships that the README never mentions**: ``ECGMRI3T02Ff`` and
  ``ECGMRI3T02Out`` were recorded on a Philips Achiva, while every other 3T record
  used a Siemens Magnetom Skyra. Both declare 3T, so this is extra detail rather
  than a contradiction — but a per-field-strength analysis that assumes one
  scanner per field strength is wrong.
- **The published demographics do not quite match either.** Recomputed over the 26
  subjects: age 18-30 (mean 24.6), weight 45-98 kg (mean 72.5), height 158-193 cm
  (mean 179.4). The README claims 27.1 +/- 3.2 years, 73.8 +/- 13.1 kg,
  181.7 +/- 10.5 cm — figures for the 23-subject version it describes.

**``subject_key`` is derived, and you must understand how before grouping on it.**

The release ships **no subject identifier**. The filename's subject number is
scoped *per scanner*: ``ECGMRI1T01`` is Male/27y/75kg/190cm while ``ECGMRI3T01``
is Female/29y/60kg/165cm. Grouping folds on that number would merge different
people and — far worse — split single people across folds, because three subjects
were recorded in more than one scanner:

    Male/27y/75kg/190cm    -> 1T01, 3T02, 7T05   (8 records)
    Female/29y/60kg/165cm  -> 3T01, 7T04         (6 records)

Since the point of the dataset is comparing one subject's ECG across field
strengths, letting 3T01 train and 7T04 test is textbook leakage. ``subject_key``
is therefore the ``sex/age/weight/height`` tuple, which collapses the 29 filename
subject-slots into **26 people** and reunites those two.

Two honest limitations:

- **It can over-merge.** Two genuinely different volunteers with identical sex,
  age, weight and height would become one group. That direction is safe — it
  costs a little fold balance and creates no leakage — but it is a real
  possibility in a cohort this homogeneous.
- **It can still under-merge, and 26 is not the release's 23.** Data was collected
  over 2011-2017, so one person's recorded weight or age can differ between
  sessions. ``Male/27y/75kg/190cm`` (1T01/3T02/7T05) and ``Male/30y/75kg/190cm``
  (3T09) are plausibly the same person three years apart, and
  ``Male/27y/65kg/180cm`` (3T03) against ``Male/28y/65kg/180cm`` (3T07) likewise.
  Neither the README's 23 nor any other count is reproducible from the files by a
  stated rule, so ECGBench reports the 26 it can defend and flags the rest here.
  If your work depends on exact subject identity, contact the author.
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

#: Newline-delimited list of record stems in the dataset root.
RECORDS_FILE = "RECORDS"

#: Annotator extension, per the shipped ANNOTATORS file ("qrs  QRS complexes").
ANNOTATOR = "qrs"

#: The only annotation symbol in the release: 14,950 'N' marks. They are QRS
#: positions, NOT beat classifications — the README is explicit that no
#: distinction was made between normal and ectopic beats, so do not read 'N' as
#: "normal beat".
QRS_SYMBOL = "N"

#: Header comment key -> exposed column. The source misspells "Position".
HEADER_FIELDS = {
    "Magnetic field strength": "field_strength_header",
    "MR scanner": "mr_scanner",
    "Orientation of the static magnetic field (B0)": "b0_orientation",
    "ECG recorder": "ecg_recorder",
    "ADC resolution": "adc_resolution",
    "ADC input voltage range": "adc_input_range",
    "ECG lead configuration": "lead_config",
    "Sex": "sex",
    "Age": "age_raw",
    "Weight": "weight_raw",
    "Height": "height_raw",
    "Positon in the scanner": "position_header",
    "Respiration": "respiration",
}

#: Filename position suffix -> canonical position. Ff1/Ff2/Ff3 are three separate
#: feet-first runs of subject 3T09, so the trailing digit is stripped.
POSITIONS = {
    "Ff": "Feet first",
    "Hf": "Head first",
    "Pro": "Prone",
    "Sup": "Supine",
    "Out": "Outside the scanner",
}

#: The position that marks a reference (outside-the-bore) recording. Derive
#: reference status from this, never from the header's field-strength string —
#: ECGMRI1T01Out declares "1T" there despite being a reference.
REFERENCE_POSITION = "Outside the scanner"

#: Value the condition column takes for a reference recording.
REFERENCE_CONDITION = "reference"

#: ECGMRI<field>T<subject><position>, with an optional run index.
_RECORD_RE = re.compile(
    r"^ECGMRI(?P<field>\d+)T(?P<subject>\d+)(?P<position>[A-Za-z]+)(?P<run>\d*)$"
)

_COMMENT_RE = re.compile(r"^#--([^:]+):\s*(.*)$")

#: Strip the unit off "27years" / "75kg" / "190cm".
_NUMBER_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def parse_record_name(record: str) -> dict[str, object]:
    """Split a record stem into field strength, subject slot, position and run.

    Raises:
        ValueError: the name does not follow the documented ECGMRI convention.
    """
    match = _RECORD_RE.match(record)
    if match is None:
        raise ValueError(
            f"Record name {record!r} does not match the documented convention "
            "ECGMRI<field>T<subject><position>, e.g. ECGMRI3T04Ff."
        )
    suffix = match.group("position")
    position = POSITIONS.get(suffix)
    if position is None:
        raise ValueError(
            f"Record {record!r} has unknown position suffix {suffix!r}. "
            f"Known: {sorted(POSITIONS)}."
        )
    run = match.group("run")
    scanner_field = int(match.group("field"))
    return {
        # The scanner the session belongs to, even for a reference recording.
        "scanner_field_T": scanner_field,
        # The field the subject was actually exposed to: 0 outside the bore.
        "field_strength_T": 0 if position == REFERENCE_POSITION else scanner_field,
        # Per-SCANNER subject number. Not a patient ID — see the module docstring.
        "subject_number": match.group("subject"),
        "scanner_subject_slot": f"{scanner_field}T{match.group('subject')}",
        "position": position,
        "run": int(run) if run else 1,
        "is_reference": position == REFERENCE_POSITION,
        "condition": (
            REFERENCE_CONDITION if position == REFERENCE_POSITION else f"{scanner_field}T"
        ),
    }


def parse_header_comments(hea_path: Path) -> dict[str, object]:
    """Read the ``#--Key:Value`` metadata block and channel layout of one header."""
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"{hea_path.name} is empty")

    fields = lines[0].split()
    n_signals = int(fields[1])
    out: dict[str, object] = {
        "n_signals": n_signals,
        "sampling_rate": int(fields[2]),
        "n_samples": int(fields[3]),
        # Channel names in file order, '|'-joined so the list survives a CSV round
        # trip. Two layouts ship and only channels 0-2 are shared, so resolve
        # anything past III against this rather than against a fixed index.
        "channel_names": "|".join(
            line.split()[-1] for line in lines[1 : 1 + n_signals]
        ),
    }
    out["duration_seconds"] = round(out["n_samples"] / out["sampling_rate"], 3)

    found = {}
    for line in lines:
        match = _COMMENT_RE.match(line)
        if match:
            found[match.group(1).strip()] = match.group(2).strip()

    for key, column in HEADER_FIELDS.items():
        out[column] = found.get(key, "")

    missing = [k for k in HEADER_FIELDS if k not in found]
    if missing:
        # Every one of the 53 headers carries all 13 keys, so an absence means a
        # changed release rather than an optional field.
        logger.warning("%s: header is missing %s", hea_path.name, missing)
    return out


def _numeric(value: object) -> float | None:
    """Pull the number out of '27years' / '75kg' / '190cm'."""
    match = _NUMBER_RE.match(str(value))
    return float(match.group(1)) if match else None


def count_qrs(record_path: Path) -> dict[str, int]:
    """Count the manual QRS annotations of one record.

    Returns ``n_qrs`` plus ``n_qrs_other`` for any symbol that is not
    :data:`QRS_SYMBOL`, which is 0 everywhere in v1.0.0.
    """
    import wfdb

    counts = {"n_qrs": 0, "n_qrs_other": 0}
    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .qrs must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    for symbol in annotation.symbol:
        if symbol == QRS_SYMBOL:
            counts["n_qrs"] += 1
        else:
            counts["n_qrs_other"] += 1
    if counts["n_qrs_other"]:
        logger.info(
            "%s: %d annotation(s) with a symbol other than %r",
            record_path.name, counts["n_qrs_other"], QRS_SYMBOL,
        )
    return counts


def _record_names(data_path: Path, config: DatasetConfig) -> list[str]:
    """Record stems, from the shipped RECORDS file if present, else by glob."""
    records_file = data_path / RECORDS_FILE
    if records_file.exists():
        names = [line.strip() for line in records_file.read_text().splitlines()]
        return [name for name in names if name]

    logger.warning("%s not found — falling back to globbing *.hea", records_file)
    names = sorted(path.stem for path in data_path.glob("*.hea"))
    if not names:
        raise FileNotFoundError(
            f"No .hea header files in {data_path}. Point data_path at the dataset "
            f"root — the directory holding ECGMRI3T01Ff.hea and RECORDS "
            f"(see {config.url})."
        )
    return names


def scan_records(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Parse every record's name, header block and QRS annotations into a frame."""
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    if not data_path.is_dir():
        raise LabelSourceMissingError(
            f"MHD-effect labels come from the per-record .hea and .qrs files, but "
            f"{data_path} is not a directory. ECGBench publishes fold CSVs only — "
            f"point data_path at a full local copy (see {config.url})."
        )

    rows = []
    for name in _record_names(data_path, config):
        hea_path = data_path / f"{name}.hea"
        if not hea_path.exists():
            raise LabelSourceMissingError(
                f"{RECORDS_FILE} lists {name}, but {hea_path} is missing. This "
                f"dataset's labels live in its headers, so point data_path at a "
                f"complete local copy (see {config.url})."
            )
        row: dict[str, object] = {config.record_id_column: name}
        row.update(parse_record_name(name))
        row.update(parse_header_comments(hea_path))
        row.update(count_qrs(data_path / name))
        rows.append(row)

    return pd.DataFrame(rows)


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return acquisition metadata and QRS counts per record, indexed by record ID.

    Acquisition condition (derived from the filename, which is self-consistent —
    unlike the header's field-strength string):

        condition           'reference' | '1T' | '3T' | '7T' — the stratification
                            label. 10 / 2 / 31 / 10 records.
        field_strength_T    0 for a reference recording, else 1, 3 or 7
        scanner_field_T     the session's scanner, 1/3/7 even for a reference
        is_reference        recorded outside the bore (10 records, 7 subjects)
        position            'Feet first' | 'Head first' | 'Prone' | 'Supine' |
                            'Outside the scanner'
        run                 1, except ECGMRI3T09Ff1/2/3 which are runs 1-3

    Straight from the header, verbatim:

        field_strength_header, mr_scanner, b0_orientation, ecg_recorder,
        adc_resolution, adc_input_range, lead_config, position_header, respiration

    Disagreements, surfaced rather than resolved:

        position_disagrees      filename and header positions differ. True for
                                ECGMRI3T01Hf only.
        reference_header_agrees False where position is outside the bore but the
                                header still names a field strength. False for
                                ECGMRI1T01Out only.

    Subject (see the module docstring before grouping on this):

        subject_key             sex/age/weight/height — the derived patient ID,
                                26 distinct values over 53 records
        scanner_subject_slot    e.g. '3T01'. Per-SCANNER, so NOT a patient ID.
        subject_number, sex, age, weight, height, age_raw, weight_raw, height_raw

    Signal shape and annotations:

        n_signals           12 (39 records) or 3 (14 records)
        channel_names       '|'-joined channel names in file order
        sampling_rate       1024 everywhere
        n_samples           25,000 to 740,001 — length is NOT uniform
        duration_seconds    24.4 to 722.7
        n_qrs               manually annotated QRS complexes, 27 to 887
        n_qrs_other         annotations with an unexpected symbol; 0 in v1.0.0
        mean_hr_bpm         n_qrs / duration_seconds * 60, a sanity figure only

    No diagnostic label exists: all 26 subjects were free of known cardiac
    disease and the QRS marks carry no beat classification.
    """
    data_path = Path(data_path)
    df = scan_records(data_path, config)

    for column, source in (("age", "age_raw"), ("weight", "weight_raw"), ("height", "height_raw")):
        df[column] = df[source].map(_numeric)

    # The release ships no subject ID and the filename's number is per-scanner, so
    # this tuple is the only thing that identifies a person across scanners.
    df["subject_key"] = (
        df["sex"].astype(str) + "/" + df["age_raw"].astype(str) + "/"
        + df["weight_raw"].astype(str) + "/" + df["height_raw"].astype(str)
    )

    df["position_disagrees"] = df["position"] != df["position_header"].map(
        lambda value: _canonical_header_position(value)
    )
    df["reference_header_agrees"] = ~(
        df["is_reference"] & (df["field_strength_header"] != REFERENCE_POSITION)
    )
    df["mean_hr_bpm"] = (
        df["n_qrs"] / df["duration_seconds"].replace(0, pd.NA) * 60
    ).round(1)

    out = df.set_index(config.record_id_column)

    n_subjects = out["subject_key"].nunique()
    n_slots = out["scanner_subject_slot"].nunique()
    logger.info(
        "Loaded MHD-effect labels: %d records, %d subjects by demographics "
        "(%d filename subject-slots), %d QRS annotations",
        len(out), n_subjects, n_slots, int(out["n_qrs"].sum()),
    )
    disagree = out.index[out["position_disagrees"]].tolist()
    if disagree:
        logger.info(
            "Filename and header positions disagree for %s — both kept; see "
            "position vs position_header", disagree,
        )
    inconsistent = out.index[~out["reference_header_agrees"]].tolist()
    if inconsistent:
        logger.info(
            "Reference record(s) %s still name a field strength in the header; "
            "use is_reference/condition, not field_strength_header", inconsistent,
        )
    return out


def _canonical_header_position(value: object) -> str:
    """Map a header position string onto the filename vocabulary.

    Headers write 'Feet first (Ff)', 'Head first (Hf)', 'Prone', 'Supine' or
    'Outside the scanner'; the parenthesised suffix is dropped so the two sources
    can be compared.
    """
    text = re.sub(r"\s*\([^)]*\)\s*$", "", str(value)).strip()
    return text or ""
