"""
PhysioNet/CinC Challenge 2017 labels: the four-class rhythm code, and its history.

The label is one of four codes per record — ``N`` normal, ``A`` atrial
fibrillation, ``O`` other rhythm, ``~`` too noisy to classify — and it ships as a
headerless two-column CSV, ``training/REFERENCE.csv``::

    A00/A00001,N
    A00/A00002,N
    A00/A00004,A

**The interesting part is that four versions of that file ship**, and this loader
exposes all of them. The challenge relabelled its data twice mid-competition,
using the entrants' own disagreement to find the records it trusted least
(Clifford et al. 2017, §2.3), so a record's label history is a usable proxy for
how contentious it is. 412 of the 8,528 training labels changed between the first
and last version.

**The shipped file numbers are off by one from the paper's, and one shipped
version is not in the paper at all.** The paper's Table 2 tabulates training
counts for versions it calls V1, V2 and V3; matching those counts against the
shipped files gives:

============  ===================================  =================================
Shipped file  Training counts (N / A / O / ~)      Paper's name for it
============  ===================================  =================================
``-v0.csv``   5154 / 771 / 2557 / 46               **V1** — unofficial phase, Feb-Apr 2017
``-v1.csv``   5040 / 736 / 2469 / 283              *not tabulated in the paper*
``-v2.csv``   5050 / 738 / 2456 / 284              **V2** — official phase, Apr-Sep 2017
``-v3.csv``   5076 / 758 / 2415 / 279              **V3** — final scoring
============  ===================================  =================================

``REFERENCE.csv`` is byte-identical to ``REFERENCE-v3.csv``, so **v3 is the
label**, and it is what ``class_code`` and ``class_name`` carry. Anyone citing
"the shipped v1 labels" and anyone citing "the paper's V1 labels" is naming a
different file; the columns here are named after the *shipped* files, which is
what a reader can verify on disk.

Everything below was verified against the files.

**There are no demographics, and no patient identifier.** Every one of the 8,528
headers is exactly two lines with no ``#`` comment of any kind — no age, no sex,
no subject id, nothing. The recordings came from members of the public who had
bought an AliveCor handheld device, and nothing in the release says whether one
person contributed more than one recording. So folds are stratified but
**ungrouped**, and that is a limitation of the source, not a choice; see the
config's ``patient_id_column: null``.

**The header timestamp is de-identified and is not a wall clock.** The date is
``1/<month>/2000`` in all 8,528 records — day always 1, year always 2000, only
the month varying (January 421 records to May 1,283). The time field has 4,976
distinct values but its hour runs 0-12 for 8,378 records and 21-23 for the
remaining 150, which is not a 24-hour distribution. ``header_timestamp`` is
exposed verbatim precisely so nobody has to guess; do not read it as a recording
time of day.

**Length varies and is not a nuisance variable.** 2,714 to 18,286 samples at
300 Hz — 9.05 s to 60.95 s, in 1,487 distinct lengths — and it correlates with
the label: noisy records average 24.4 s against 32-34 s for the other three
classes, and 44% of them are the modal 30 s against 73% of normals. A model given
whole records can learn duration instead of rhythm, which is the argument for
``window=``.

**The challenge's own 300-record "validation" set is a subset of these 8,528, not
a held-out split.** All 300 ``validation/*.mat`` files are byte-identical to their
``training/`` counterparts (checked, all 300), and the paper describes it as
"300 records (3.5%) of training set just to ensure the algorithm produced the
expected results". ``in_challenge_validation_subset`` flags them so they can be
excluded from a comparison, never used as a test set.

**The 3,658-record hidden test set is not in this release** and was never
published, so nothing here reproduces the challenge's own scoring split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Directory holding the released recordings, relative to the dataset root.
TRAINING_DIR = "training"

#: Directory holding the 300-record subset the challenge called "validation".
#: Its files duplicate ``training/`` byte for byte; see the module docstring.
VALIDATION_DIR = "validation"

#: The four rhythm classes, in the order the challenge's own tables list them.
CLASS_NAMES = {
    "N": "normal",
    "A": "atrial_fibrillation",
    "O": "other_rhythm",
    "~": "noisy",
}

#: Shipped ``REFERENCE-v<N>.csv`` versions, oldest first. ``REFERENCE.csv`` is
#: byte-identical to ``-v3`` and is the authoritative label.
REFERENCE_VERSIONS = (0, 1, 2, 3)

#: Shipped file -> the name Clifford et al. (2017) Table 2 gives it. Shipped v1
#: has no entry there. Documented because the two numbering schemes are one apart
#: and citing "v1" is ambiguous without it.
PAPER_VERSION_NAMES = {0: "V1", 1: None, 2: "V2", 3: "V3"}


def read_reference(csv_path: Path) -> pd.Series:
    """Read one headerless ``REFERENCE*.csv`` into record_name -> class code.

    The record column carries the ``A00/`` subdirectory prefix in
    ``training/REFERENCE*.csv`` and does not in the copies at the dataset root;
    the prefix is stripped either way so the two are interchangeable.
    """
    df = pd.read_csv(csv_path, header=None, names=["record", "code"], dtype=str)
    names = df["record"].str.rsplit("/", n=1).str[-1]
    return pd.Series(df["code"].to_numpy(), index=names.to_numpy(), name=csv_path.stem)


def read_header(hea_path: Path) -> dict[str, object]:
    """Parse one two-line WFDB header.

    These headers are uniform to a degree that is worth stating rather than
    assuming: all 8,528 declare 1 signal at 300 Hz, format ``16+24`` (WFDB's
    MATLAB v4 wrapper — the 24-byte ``.mat`` preamble is the offset), gain
    ``1000/mV``, 16-bit resolution, ADC zero 0, and channel name ``ECG``. Only
    the sample count, the baseline and the timestamp vary.
    """
    lines = [ln for ln in hea_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{hea_path} has {len(lines)} non-empty lines, expected 2")

    record_line = lines[0].split()
    signal_line = lines[1].split()

    timestamp = " ".join(record_line[4:6]) if len(record_line) >= 6 else ""

    return {
        "record_name": record_line[0],
        "n_leads": int(record_line[1]),
        "sampling_rate": int(float(record_line[2])),
        "n_samples": int(record_line[3]),
        "header_timestamp": timestamp,
        "signal_file": signal_line[0],
        "storage_format": signal_line[1],
        "adc_gain": signal_line[2],
        "adc_resolution": int(signal_line[3]),
        "adc_zero": int(signal_line[4]),
        "baseline": int(signal_line[5]),
        "lead_name": signal_line[-1],
    }


def _read_record_list(path: Path) -> list[str]:
    if not path.exists():
        from ecgbench.labels import LabelSourceMissingError

        raise LabelSourceMissingError(
            f"{path} is missing. The Challenge 2017 labels come from the shipped "
            f"RECORDS and REFERENCE.csv files, so data_path must point at a full "
            f"local copy of the release "
            f"(https://physionet.org/content/challenge-2017/1.0.0/)."
        )
    return [ln.strip() for ln in path.read_text().split() if ln.strip()]


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Build the per-record frame from ``RECORDS``, the headers and every reference.

    One row per record listed in ``training/RECORDS``, in that file's order.
    """
    root = Path(data_path)
    training = root / TRAINING_DIR

    records = _read_record_list(training / "RECORDS")
    logger.info("Scanning %d records listed in %s", len(records), training / "RECORDS")

    references: dict[int, pd.Series] = {}
    for version in REFERENCE_VERSIONS:
        path = training / f"REFERENCE-v{version}.csv"
        if path.exists():
            references[version] = read_reference(path)
        else:
            logger.warning("%s is absent; its column will be empty", path)

    final = read_reference(training / "REFERENCE.csv")

    validation_subset = set()
    validation_records = root / VALIDATION_DIR / "RECORDS"
    if validation_records.exists():
        validation_subset = {r.rsplit("/", 1)[-1] for r in _read_record_list(validation_records)}
    else:
        logger.warning(
            "%s is absent; in_challenge_validation_subset will be all False",
            validation_records,
        )

    rows = []
    for relative in records:
        name = relative.rsplit("/", 1)[-1]
        header = read_header(training / f"{relative}.hea")
        if header["record_name"] != name:
            raise ValueError(
                f"{training / relative}.hea declares record "
                f"{header['record_name']!r} but RECORDS calls it {name!r}"
            )

        code = final.get(name)
        if code is None:
            raise ValueError(f"{name} is in RECORDS but not in training/REFERENCE.csv")

        row: dict[str, object] = {
            "record_name": name,
            # Relative to the dataset root, with the .mat suffix wfdb strips.
            "signal_path": f"{TRAINING_DIR}/{relative}.mat",
            "class_code": code,
            "class_name": CLASS_NAMES[code],
            "is_af": code == "A",
            "is_noisy": code == "~",
        }
        row.update(
            {
                "n_samples": header["n_samples"],
                "duration_seconds": round(header["n_samples"] / header["sampling_rate"], 4),
                "sampling_rate": header["sampling_rate"],
                "n_leads": header["n_leads"],
                "lead_name": header["lead_name"],
                "adc_gain": header["adc_gain"],
                "baseline": header["baseline"],
                "storage_format": header["storage_format"],
                "header_timestamp": header["header_timestamp"],
            }
        )
        for version in REFERENCE_VERSIONS:
            series = references.get(version)
            row[f"class_code_v{version}"] = None if series is None else series.get(name)
        row["in_challenge_validation_subset"] = name in validation_subset
        rows.append(row)

    df = pd.DataFrame(rows)

    version_columns = [f"class_code_v{v}" for v in REFERENCE_VERSIONS if v in references]
    if version_columns:
        # How many distinct labels a record was ever given. 1 means every version
        # agreed; anything above that means the organisers' relabelling touched it,
        # which is the closest thing this release has to an annotation-confidence
        # measure. See the module docstring on how V-numbering differs.
        df["n_distinct_labels"] = df[version_columns].nunique(axis=1)
        first, last = version_columns[0], version_columns[-1]
        df["label_revised"] = df[first] != df[last]
    else:
        df["n_distinct_labels"] = 1
        df["label_revised"] = False

    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: the four-class label, used as-is.

    No reduction or pooling is needed. The label is single-valued (no record
    carries two classes) and the rarest class, ``noisy``, has 279 records — about
    28 per fold over ten folds, comfortably splittable. This function exists so
    the splitter has one named place to read the fold label from rather than
    reaching for ``class_name`` directly.
    """
    out = df.copy()
    out["stratify_class"] = out["class_name"]
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Challenge 2017 labels indexed by record name.

    Columns:

    - ``class_code`` / ``class_name`` — the label, from ``training/REFERENCE.csv``
      (byte-identical to ``REFERENCE-v3.csv``): ``N`` normal 5,076,
      ``O`` other_rhythm 2,415, ``A`` atrial_fibrillation 758, ``~`` noisy 279.
      Single-label — no record carries two classes, and none carries none.
      ``is_af`` and ``is_noisy`` are the two convenience booleans.
    - ``class_code_v0`` … ``class_code_v3`` — the label under each shipped
      reference file. **These numbers are one behind the paper's V1/V2/V3**, and
      shipped v1 is not in the paper's table at all; see the module docstring.
      412 records differ between v0 and v3, 148 between v2 and v3.
    - ``label_revised`` — ``class_code_v0 != class_code_v3``, true for 412
      records. ``n_distinct_labels`` counts how many different labels a record was
      ever given across the four versions (1 for 8,104 records, 2 for 418, 3 for
      6). Records above 1 are the ones the challenge organisers' bootstrap
      relabelling flagged as contentious; the paper measured Fleiss' κ = 0.245
      over the 1,129 most-disputed recordings, so this is not a small effect.
    - ``in_challenge_validation_subset`` — true for the 300 records the challenge
      shipped a duplicate copy of under ``validation/``. Not a held-out split:
      the files are byte-identical to the ``training/`` originals. Use it to
      *exclude*, never to evaluate.
    - ``n_samples``, ``duration_seconds``, ``sampling_rate`` — record length,
      2,714-18,286 samples at 300 Hz (9.05-60.95 s). Length correlates with the
      label; see the module docstring.
    - ``header_timestamp`` — verbatim and **de-identified**: day is always 1 and
      year always 2000, and the hour field is not a 24-hour clock. Do not treat
      it as a recording time.
    - ``n_leads``, ``lead_name``, ``adc_gain``, ``baseline``, ``storage_format``,
      ``signal_path`` — from the header. Uniform across all 8,528 records except
      ``baseline`` (1,781 distinct values) and the sample count.
    - ``stratify_class`` — the fold label, equal to ``class_name``. It is the real
      label here rather than a reduction, so training on it is fine.

    There are **no demographics and no patient identifier** of any kind — the
    headers carry no comment lines at all. See the module docstring.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
