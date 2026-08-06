"""
STAFF III labels: the shipped annotation spreadsheet, the WFDB headers and the
``.event`` files, joined into one row per record.

STAFF III is a **protocol** dataset, not a diagnosis dataset. Each of the 104
patients contributed a short series of recordings around a single elective
percutaneous transluminal coronary angioplasty (PTCA), and the label that matters
is *where in that protocol* a recording sits:

    BR  baseline room        pre-procedure, on the ward
    BC  baseline cathlab     pre-procedure, in the catheterisation lab
    BI  balloon inflation    the balloon is up — controlled transient ischaemia
    PC  postinflation cathlab
    PR  postinflation room

``recording_type`` is that label, and it is the ground truth for the canonical
task on this dataset: detecting ischaemia from the surface ECG against the
patient's own baseline.

Three source files, all of which agree with each other (verified, see below):

- ``STAFF-III-Database-Annotations.xlsx`` — one row per patient, wide: which file
  number played which protocol role, the occluded artery per inflation, the
  inflation timings, contrast-injection times, age, sex and prior-MI location.
- the record headers — age and sex again, plus the authoritative record length.
- the ``.event`` annotation files — sample-accurate inflation, deflation and
  contrast-injection markers, one file per balloon-inflation record.

Quirks worth knowing, all verified against the files:

- **There are 9 leads, not 12.** The files store ``V1..V6, I, II, III`` in that
  order — precordials *first*, and the augmented leads aVR/aVL/aVF are not stored
  because they are exact linear combinations of I and II. Anything assuming
  ``signal[0]`` is lead I is wrong here. See ``lead_names`` in the config.
- **The spreadsheet's D2 field is unreliable.** ``D0;D1;D2`` is documented as
  time-to-inflation, inflation duration and time-from-deflation-to-end-of-file.
  D0 and D1 agree with the ``.event`` markers on all 152 inflations to within a
  second; D2 disagrees with the actual record length on 30 of 142 records, by up
  to 575 s. This module therefore takes every timing from the ``.event`` files
  and the record length from the header, and ignores D2.
- **``089d`` cannot be read.** Its header declares 468,554 samples but the
  ``.dat`` holds 300,000. Every one of the release's 1,189 files matches the
  shipped ``SHA256SUMS.txt``, so this is an upstream defect and not a damaged
  local copy. ``wfdb.rdrecord`` raises ``ValueError: Samples were not loaded
  correctly``; the validation engine records that as ``corrupt_header`` and drops
  the record from ``clean/``. The truncation is after the balloon deflation
  (inflation 0-278 s, file 0-300 s), so the ischaemic episode itself survives —
  read it with ``sampto=300000`` if you want it.
- **``089e`` has the opposite defect**, harmlessly: the header declares 300,000
  samples and the ``.dat`` holds ~366,667. wfdb reads the declared 300,000 and
  ignores the tail, so nothing downstream notices.
- **``016f`` has lead V6 recorded as all zeros** — caught by ``missing_leads``.
- **Patients 1, 4, 5, 6 and 89 have leads the depositors later identified as
  possibly reversed** (lead or sign). The spreadsheet says so in a free-text note
  and does not say *which* leads, so this module flags the whole patient with
  ``suspect_leads`` rather than guessing.
- **Four patient numbers are unused.** The spreadsheet has 108 rows but 28, 67,
  78 and 103 have no files; the cohort is 104 patients, which is why publications
  quoting 108 are wrong. The spreadsheet says so itself.
- **Age is missing for patients 14 and 15**, whose headers carry no ``# Age:``
  line at all (the spreadsheet writes ``?``). Sex is present for all 104.
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

#: Annotator extension, per the shipped ANNOTATORS file.
ANNOTATOR = "event"

#: The annotation spreadsheet, relative to the dataset root. An .ods of the same
#: content ships beside it; they were checked to agree and the .xlsx is read
#: because pandas needs no extra engine for it beyond openpyxl.
ANNOTATION_XLSX = "STAFF-III-Database-Annotations.xlsx"

#: Signal files live one level down, in a flat directory.
SIGNAL_SUBDIR = "data"

#: Protocol phase codes in recording order, with the spreadsheet's own wording.
RECORDING_TYPES = {
    "BR": "baseline room",
    "BC": "baseline cathlab",
    "BI": "balloon inflation",
    "PC": "postinflation cathlab",
    "PR": "postinflation room",
}

#: Coronary territory for each occluded-artery string the spreadsheet uses. The
#: twelve raw values are free text ("prox mid LAD"), so they are mapped onto the
#: three main vessels plus the left main stem rather than used directly.
_TERRITORY_PATTERNS = (
    ("left main", "LM"),
    ("LAD", "LAD"),
    ("RCA", "RCA"),
    ("circ", "LCx"),
)

#: Patients whose recordings the depositors flagged for possible lead or sign
#: reversal. The note does not say which leads, so the flag is per patient.
SUSPECT_LEAD_PATIENTS = frozenset({1, 4, 5, 6, 89})

#: Stratification classes with fewer than this many PATIENTS are pooled into
#: OTHER. Patients, not records, because folds are grouped by patient.
MIN_CLASS_PATIENTS = 10

OTHER = "OTHER"
UNKNOWN = "UNKNOWN"

#: Spreadsheet layout. Row 8 holds the group headings and row 9 the column
#: names; data starts at row 10. Column indices are positional because the sheet
#: has merged headings that pandas cannot turn into unique names.
_DATA_START_ROW = 10
_COL_PATIENT, _COL_AGE, _COL_SEX = 0, 1, 2
_COL_PRIOR_MI = 28

#: (file column, protocol code) for the single-valued phases.
_SIMPLE_PHASE_COLUMNS = (
    (3, "BR", 1),
    (4, "BC", 1),
    (5, "BC", 2),
    (24, "PC", 1),
    (25, "PC", 2),
    (26, "PR", 1),
    (27, "PR", 2),
)

#: (file column, artery column, timing column, injection column, index) for the
#: five possible balloon inflations. BI4 and BI5 have no injection column — the
#: spreadsheet notes that no tracer was injected during those inflations.
_INFLATION_COLUMNS = (
    (6, 7, 8, 9, 1),
    (10, 11, 12, 13, 2),
    (14, 15, 16, 17, 3),
    (18, 19, 20, None, 4),
    (21, 22, 23, None, 5),
)

#: File numbers in the sheet are unpadded ("7c"); record names are ("007c").
_FILENUM_RE = re.compile(r"^(?P<number>\d+)(?P<suffix>[a-z]+)$")


def artery_territory(artery: str | None) -> str:
    """Map a free-text occluded-artery string onto a coronary territory.

    ``"prox mid LAD"`` and ``"LAD diag"`` both become ``"LAD"``. Returns
    ``UNKNOWN`` for anything unrecognised rather than guessing, so a future
    release adding a vessel name shows up as UNKNOWN instead of being silently
    folded into the wrong territory.
    """
    if not artery or pd.isna(artery):
        return ""
    text = str(artery)
    for needle, territory in _TERRITORY_PATTERNS:
        if needle in text:
            return territory
    logger.warning("Unrecognised occluded artery %r; mapped to %s", text, UNKNOWN)
    return UNKNOWN


def _record_name(filenum: object) -> str | None:
    """Normalise a spreadsheet file number ("7c") to a record name ("007c")."""
    match = _FILENUM_RE.match(str(filenum).strip())
    if not match:
        return None
    return f"{int(match.group('number')):03d}{match.group('suffix')}"


def _clean(value: object) -> str:
    """Spreadsheet cell -> stripped string, with NaN and '?' becoming ''."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"?", "nan"} else text


def read_annotation_sheet(data_path: Path | str) -> pd.DataFrame:
    """Parse the annotation spreadsheet into one row per *inflation or phase*.

    The sheet is one wide row per patient. This unpivots it, so a patient with
    two balloon inflations yields two ``BI`` rows. Nine records genuinely hold
    more than one inflation, which is why the record-level frame built by
    :func:`scan_records` aggregates these rather than assuming one row each.

    Columns: ``record_name``, ``patient``, ``recording_type``,
    ``recording_index``, ``occluded_artery``, ``sheet_timing``,
    ``sheet_injection``.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    xlsx = data_path / ANNOTATION_XLSX
    if not xlsx.exists():
        raise LabelSourceMissingError(
            f"STAFF III labels come from {ANNOTATION_XLSX}, which is not in "
            f"{data_path}. ECGBench publishes fold CSVs only — labels stay with "
            "the source dataset, so point data_path at a full local copy "
            "(https://physionet.org/content/staffiii/1.0.0/)."
        )

    sheet = pd.read_excel(xlsx, sheet_name=0, header=None).iloc[_DATA_START_ROW:]

    rows: list[dict[str, object]] = []
    for _, raw in sheet.iterrows():
        patient = int(raw[_COL_PATIENT])
        for col, code, index in _SIMPLE_PHASE_COLUMNS:
            record = _record_name(raw[col]) if _clean(raw[col]) else None
            if record:
                rows.append(
                    {
                        "record_name": record,
                        "patient": patient,
                        "recording_type": code,
                        "recording_index": index,
                        "occluded_artery": "",
                        "sheet_timing": "",
                        "sheet_injection": "",
                    }
                )
        for col, artery_col, timing_col, injection_col, index in _INFLATION_COLUMNS:
            record = _record_name(raw[col]) if _clean(raw[col]) else None
            if not record:
                continue
            rows.append(
                {
                    "record_name": record,
                    "patient": patient,
                    "recording_type": "BI",
                    "recording_index": index,
                    "occluded_artery": _clean(raw[artery_col]),
                    "sheet_timing": _clean(raw[timing_col]),
                    "sheet_injection": (
                        _clean(raw[injection_col]) if injection_col is not None else ""
                    ),
                }
            )

    df = pd.DataFrame(rows)
    logger.info(
        "Parsed %d protocol entries for %d patients from %s",
        len(df),
        df["patient"].nunique(),
        ANNOTATION_XLSX,
    )
    return df


def read_patient_attributes(data_path: Path | str) -> pd.DataFrame:
    """Per-patient age, sex and prior-MI location, indexed by patient number.

    Only patients that actually have recordings are returned: the sheet has 108
    rows but numbers 28, 67, 78 and 103 are unused, which is the whole reason
    the cohort is 104 and not the 108 some publications quote.
    """
    data_path = Path(data_path)
    sheet = pd.read_excel(data_path / ANNOTATION_XLSX, sheet_name=0, header=None).iloc[
        _DATA_START_ROW:
    ]

    rows = []
    for _, raw in sheet.iterrows():
        prior_mi = _clean(raw[_COL_PRIOR_MI]).lower()
        rows.append(
            {
                "patient": int(raw[_COL_PATIENT]),
                "age": _clean(raw[_COL_AGE]),
                "sex": _clean(raw[_COL_SEX]).upper(),
                "prior_mi_location": prior_mi,
                # "no" is the sheet's own wording for absence, not a missing value.
                "prior_mi": "" if not prior_mi else str(prior_mi != "no"),
            }
        )
    return pd.DataFrame(rows).set_index("patient")


def read_events(data_path: Path | str, record: str) -> dict[str, object]:
    """Summarise one record's ``.event`` annotations.

    Returns inflation start and deflation times in seconds (semicolon-joined
    when a record holds more than one inflation), the contrast-injection times,
    and counts. Records with no ``.event`` file — every non-inflation record —
    come back with empty strings and zero counts rather than NaN, so the column
    dtype stays stable across the whole frame.
    """
    import wfdb

    empty: dict[str, object] = {
        "inflation_start_s": "",
        "deflation_s": "",
        "inflation_duration_s": "",
        "injection_s": "",
        "n_inflations": 0,
        "n_injections": 0,
    }

    path = Path(data_path) / SIGNAL_SUBDIR / record
    if not path.with_suffix(f".{ANNOTATOR}").exists():
        return empty

    try:
        annotation = wfdb.rdann(str(path), ANNOTATOR)
    except Exception as e:  # one unreadable .event must not kill the scan
        logger.warning("Could not read %s.%s: %s", record, ANNOTATOR, e)
        return empty

    rate = annotation.fs or 1000
    inflations: list[float] = []
    deflations: list[float] = []
    injections: list[float] = []
    for sample, aux in zip(annotation.sample, annotation.aux_note):
        note = str(aux).strip("\x00").strip()
        seconds = round(float(sample) / rate, 3)
        if note == "balloon inflation":
            inflations.append(seconds)
        elif note == "balloon deflation":
            deflations.append(seconds)
        elif note == "contrast injection":
            injections.append(seconds)
        else:
            logger.warning("%s: unexpected event note %r", record, note)

    durations = [round(end - start, 3) for start, end in zip(inflations, deflations)]
    join = ";".join
    return {
        "inflation_start_s": join(str(v) for v in inflations),
        "deflation_s": join(str(v) for v in deflations),
        "inflation_duration_s": join(str(v) for v in durations),
        "injection_s": join(str(v) for v in injections),
        "n_inflations": len(inflations),
        "n_injections": len(injections),
    }


def read_header_geometry(hea_path: Path) -> dict[str, object]:
    """Record length and header demographics from one ``.hea``.

    ``n_samples`` is what the header *declares*; it is not always what the
    ``.dat`` holds (see ``089d`` and ``089e`` in the module docstring). The
    validation engine is what catches that, by trying to read the signal.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: dict[str, object] = {
        "n_samples": 0,
        "sampling_rate": 0,
        "header_age": "",
        "header_sex": "",
    }
    if not lines:
        logger.warning("Empty header: %s", hea_path.name)
        return out

    fields = lines[0].split()
    if len(fields) >= 4:
        out["sampling_rate"] = int(float(fields[2]))
        out["n_samples"] = int(fields[3])
    else:
        logger.warning("Unparsed header line in %s: %r", hea_path.name, lines[0])

    for line in lines:
        if line.startswith("# Age:"):
            out["header_age"] = line.split(":", 1)[1].strip()
        elif line.startswith("# Sex:"):
            out["header_sex"] = line.split(":", 1)[1].strip().upper()
    return out


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Build the record-level frame: one row per record, 520 rows.

    Aggregates the unpivoted sheet back to record level, because nine records
    hold two or three balloon inflations. Multi-valued fields are
    semicolon-joined in inflation order.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    headers = sorted((data_path / SIGNAL_SUBDIR).glob("[0-9][0-9][0-9][a-z].hea"))
    if not headers:
        raise LabelSourceMissingError(
            f"No NNNx.hea headers under {data_path / SIGNAL_SUBDIR}. Point "
            "data_path at the STAFF III version directory — the one holding "
            "RECORDS, SHA256SUMS.txt and data/."
        )

    entries = read_annotation_sheet(data_path)
    attributes = read_patient_attributes(data_path)

    on_disk = {h.stem for h in headers}
    in_sheet = set(entries["record_name"])
    if on_disk != in_sheet:
        # Not fatal — the record-level frame is driven by the files on disk —
        # but a disagreement means the sheet no longer describes this release.
        logger.warning(
            "Annotation sheet and signal files disagree: %d only on disk (%s), "
            "%d only in the sheet (%s)",
            len(on_disk - in_sheet),
            sorted(on_disk - in_sheet)[:5],
            len(in_sheet - on_disk),
            sorted(in_sheet - on_disk)[:5],
        )

    by_record = {name: group for name, group in entries.groupby("record_name")}

    rows = []
    for hea in headers:
        record = hea.stem
        patient = int(record[:3])
        group = by_record.get(record)

        if group is None:
            logger.warning("%s is on disk but not in the annotation sheet", record)
            phase, index, arteries = UNKNOWN, "", ""
        else:
            group = group.sort_values("recording_index")
            phases = sorted(set(group["recording_type"]))
            if len(phases) > 1:
                # Never happens in this release; would mean one file was recorded
                # as two different protocol phases, which the label cannot express.
                logger.warning("%s maps to several phases: %s", record, phases)
            phase = phases[0]
            index = ";".join(str(v) for v in group["recording_index"])
            arteries = ";".join(a for a in group["occluded_artery"] if a)

        row: dict[str, object] = {
            "record_name": record,
            "patient_id": f"patient{patient:03d}",
            "patient_number": patient,
            "recording_type": phase,
            "recording_type_label": RECORDING_TYPES.get(phase, ""),
            "recording_index": index,
            "occluded_artery": arteries,
            "artery_territory": ";".join(artery_territory(a) for a in arteries.split(";") if a),
            "suspect_leads": str(patient in SUSPECT_LEAD_PATIENTS),
            "signal_path": f"{SIGNAL_SUBDIR}/{record}",
        }
        row.update(read_header_geometry(hea))
        row.update(read_events(data_path, record))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["duration_seconds"] = (
        df["n_samples"] / df["sampling_rate"].where(df["sampling_rate"] > 0)
    ).round(3)

    df = df.merge(
        attributes.reset_index().rename(columns={"patient": "patient_number"}),
        on="patient_number",
        how="left",
    )
    # The headers and the sheet agree on age and sex for all 520 records; the
    # sheet is used, and header_age/header_sex are kept so a future release that
    # breaks that agreement is visible rather than silently resolved.
    disagree = df[(df["header_sex"] != "") & (df["sex"] != "") & (df["header_sex"] != df["sex"])]
    if len(disagree):
        logger.warning(
            "Header and sheet disagree on sex for %d record(s): %s",
            len(disagree),
            sorted(disagree["record_name"])[:5],
        )

    df = df.sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d STAFF III records from %d patients; %d balloon inflations "
        "across %d inflation records",
        len(df),
        df["patient_id"].nunique(),
        int(df["n_inflations"].sum()),
        int((df["recording_type"] == "BI").sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: the patient's primary occluded-artery territory.

    This is the **only** derivation of the stratification label — the splitter
    reads the column rather than recomputing it, so the exposed label and the
    fold label cannot drift.

    Why the territory and not ``recording_type``: folds are grouped by patient,
    so only patient-level attributes can actually be balanced across folds. Every
    patient contributes roughly the same mix of protocol phases (one inflation
    plus its baselines and recoveries), which makes ``recording_type`` almost
    uniform per patient and therefore useless to stratify on — it comes out
    balanced whatever the split does. The occluded territory is the attribute
    that genuinely varies between patients, so balancing it is what stops one
    fold ending up all-RCA.

    "Primary" is the territory of the patient's **first** inflation. Ten patients
    had inflations in more than one territory; the full list is in
    ``artery_territory``. Do not train on ``stratify_class``.
    """
    out = df.copy()

    first = (
        out[out["recording_type"] == "BI"]
        .sort_values(["patient_number", "record_name", "recording_index"])
        .groupby("patient_number")["artery_territory"]
        .first()
        .str.split(";")
        .str[0]
    )
    labels = out["patient_number"].map(first).fillna("").replace("", UNKNOWN)

    patients_per_class = (
        pd.DataFrame({"label": labels, "patient": out["patient_number"]})
        .groupby("label")["patient"]
        .nunique()
    )
    rare = set(patients_per_class[patients_per_class < MIN_CLASS_PATIENTS].index)
    if rare:
        logger.info(
            "Pooling %d territory class(es) with <%d patients into '%s': %s",
            len(rare),
            MIN_CLASS_PATIENTS,
            OTHER,
            sorted(rare),
        )
        labels = labels.where(~labels.isin(rare), OTHER)

    out["primary_artery_territory"] = out["patient_number"].map(first).fillna("")
    out["stratify_class"] = labels
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return STAFF III labels indexed by record name.

    Columns:

    - ``recording_type`` — ``BR``/``BC``/``BI``/``PC``/``PR``, the protocol phase
      and the dataset's primary label; ``recording_type_label`` spells it out.
      ``BI`` records are the ones during balloon occlusion.
    - ``recording_index`` — which BC/BI/PC/PR of that patient this is (1-5).
      Semicolon-joined for the nine records holding several inflations.
    - ``occluded_artery`` — free text from the spreadsheet ("prox mid LAD"), empty
      for non-inflation records. ``artery_territory`` reduces it to
      ``LAD``/``RCA``/``LCx``/``LM``.
    - ``inflation_start_s``, ``deflation_s``, ``inflation_duration_s``,
      ``injection_s`` — sample-accurate event times in seconds, semicolon-joined,
      from the ``.event`` files. Use these with ``ECGDataset(window=...)`` to cut
      the occluded interval out of a recording.
    - ``n_inflations``, ``n_injections`` — counts for this record.
    - ``patient_id`` — ``patientNNN``. 104 patients over 520 records, a mean of 5
      records each; this is what folds are grouped by.
    - ``age``, ``sex``, ``prior_mi_location``, ``prior_mi`` — patient-level.
      Age is missing for patients 14 and 15.
    - ``suspect_leads`` — ``"True"`` for patients 1, 4, 5, 6 and 89, whose
      recordings the depositors flagged for possible lead or sign reversal.
    - ``n_samples``, ``duration_seconds`` — from the header. Records run 94.5 s to
      960 s, so a fixed ``window=`` must fit the shortest.
    - ``stratify_class`` — pooled primary territory, **for fold construction
      only**. See :func:`attach_stratify_class`.

    Labels are multi-valued only in the semicolon-joined columns; every record
    has exactly one ``recording_type``.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
