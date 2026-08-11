"""
Apnea-ECG labels: per-minute apnea annotations, the release's A/B/C classes,
polysomnography indices, demographics, and the subject grouping the release
never states.

The ground truth of this database is a **per-minute binary label**. Each record's
``.apn`` file carries one annotation per minute of the recording, ``A`` or ``N``,
and PhysioNet's own ``annotations.html`` is precise about what they mean:

    Each "A" annotation indicates that apnea was in progress at the beginning of
    the associated minute; each "N" annotation indicates that apnea was not in
    progress at the beginning of the associated minute.

Note "at the beginning of" — an earlier, **incorrect** description ("apnea occurs
during the following one-minute interval") was posted with the original release
and is still repeated in some papers. ``apnea_sequence`` here is the raw string
of those annotations in record order, so a user reconstructs the minute-level
target without re-reading the annotation files.

**THE HEADLINE FACT: 70 RECORDS, 30 SUBJECTS, AND THE RELEASE'S OWN
LEARNING/TEST SPLIT LEAKS 18 OF THEM.** Nothing in the release states a subject
identifier, which is why the leak is invisible. It is recoverable, because
``additional-information.txt`` publishes age, sex, height and weight per record,
and those four fields take exactly 32 distinct values over the 70 records —
matching the subject count the database is described by. Two further pairs are
the same *recording* released twice (see :data:`DUPLICATE_RECORDS`), which merges
two more groups, giving **30**. 18 of those 30 subjects — **49 of the 70
records** — have recordings in both the challenge learning set (a/b/c) and the
challenge test set (x). Train on the learning set, evaluate on the test set, and
70% of your test records come from a subject the model has already seen.

ECGBench therefore does **not** adopt the challenge split: folds are grouped on
``subject_id``. ``challenge_set`` is kept as a label so the original challenge
result remains reproducible for anyone who wants to compare against the 2000
literature — with the leak stated rather than inherited silently.

Four more things worth knowing, all verified against the files.

**1. Two pairs of records are the same recording, bit for bit.** ``x35`` is
``x22`` shifted by 40 s and ``c06`` is ``c05`` shifted by 80 s — 100.000% of
2,883,000 and 2,785,000 overlapping samples are identical, against a
maximum-over-lag correlation of 0.003–0.054 for every control pair in the
release. The demographics contradict each other across the ``x22``/``x35`` pair
(27 F, 158 cm, 53 kg against 31 M, 184 cm, 74 kg), so one of those two rows in
``additional-information.txt`` is wrong and there is no way to tell which. Both
records are kept, because they are both official records with official labels,
and ``duplicate_of`` names the relationship; grouping keeps them in one fold.

**2. The A/B/C class is derived, not shipped, and it reproduces exactly.** The
release encodes the class in the record *name* for the learning set only — a01
is class A, b01 class B, c01 class C — and the withheld test records are all
named ``xNN``. Applying PhysioNet's stated criterion to the annotation counts
(``C`` under 5 apnea minutes, ``B`` 5 to 99, ``A`` 100 or more) reproduces all 35
learning-set letters with no exceptions, so :func:`derive_apnea_class` applies it
to the x records too. It lands them on 20 A / 5 B / 10 C — the exact class
composition of the learning set, which is evidently how the test set was built.

**3. The ``.qrs`` beats are machine-generated and unaudited.** PhysioNet states
they came from ``sqrs125`` at per-record thresholds and that "in no case were the
annotations hand-edited". ``mean_hr_bpm``, ``sdnn_ms`` and ``rmssd_ms`` are
computed from them and are descriptive summaries of a whole night, not an HRV
result. Do not use ``.qrs`` as a beat-detection reference.

**4. Two records are shorter than the published table says.** ``c07`` holds
428.2 min of signal against the 454 the table scores, and ``c08`` 514.0 against
535 — 25 and 22 minutes missing. The apnea counts still agree exactly (4 and 0),
so nothing labelled apnea is lost, but a per-record minute total taken from the
paper will not match the file. Every other record agrees to within the one-minute
off-by-one of the annotation convention. ``published_minutes`` and
``n_annotated_minutes`` are both exposed so the discrepancy stays visible.

The 8 ``*r`` records (respiration and SpO2, no ECG) and 8 ``*er`` records (the
same ECG ``.dat`` again, plus those signals) are **not** part of the partition —
see :func:`scan_records`. ``has_respiration`` flags the 8 ECG records whose
companions exist, so a user wanting the respiration channels knows which to open.
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

#: Annotator extensions, per the shipped ANNOTATORS file:
#: "apn  reference apnea annotations (at 1 minute intervals)" and
#: "qrs  unaudited beat annotations".
ANNOTATOR_APNEA = "apn"
ANNOTATOR_QRS = "qrs"

#: The per-record polysomnography table. Not a CSV — a fixed-width ASCII table
#: under a prose header, parsed by :func:`parse_additional_information`.
INFO_FILE = "additional-information.txt"

#: One apnea annotation per minute, at 100 Hz.
SAMPLES_PER_MINUTE = 6000

#: Records that are a second release of another record's waveform, mapped to
#: ``(canonical_record, offset_samples)`` such that
#: ``canonical[i] == duplicate[i + offset]``.
#:
#: Verified, not inferred: after shifting, 100.000% of the 2,883,000 and
#: 2,785,000 overlapping samples are equal, using the integer ADC values so no
#: floating-point tolerance is involved. They were found by counting exact
#: 8-grams of RR intervals shared between every pair of records — these two pairs
#: share 4,152 and 4,006 against a 20–90 background — and confirmed by
#: maximum-over-lag cross-correlation, 0.9998 and 0.9629 against 0.003–0.054 for
#: control pairs including same-subject different-night ones. ``tests/`` re-runs
#: the sample comparison.
#:
#: Both pairs are kept in the partition: each is an official record with its own
#: official annotations, and dropping one would silently diverge from the
#: release. :func:`assign_subject_ids` merges them so they cannot straddle a fold.
DUPLICATE_RECORDS: dict[str, tuple[str, int]] = {
    "x35": ("x22", 4000),
    "c06": ("c05", 8000),
}

#: ECG records documented as having ``*r`` (respiration + SpO2) and ``*er`` (both)
#: companions. PhysioNet: "these contain the same ECG data as a01 - a04, b01, and
#: c01 - c03".
#:
#: ``has_respiration`` is **derived from the companion records actually present**
#: rather than read off this set, so the column describes the copy on disk. This
#: is the documented expectation the derived set is checked against — a partial
#: download that lost the ``*r`` files should say so, not inherit a flag from a
#: constant.
RESPIRATION_RECORDS = frozenset({"a01", "a02", "a03", "a04", "b01", "c01", "c02", "c03"})

#: Apnea-minute thresholds separating the release's own classes. PhysioNet
#: defines class C as fewer than 5 minutes of apnea and class A as at least 100;
#: class B is what lies between. See :func:`derive_apnea_class`.
APNEA_CLASS_MIN_MINUTES_A = 100
APNEA_CLASS_MIN_MINUTES_B = 5

APNEA_CLASS_NAMES = {
    "A": "apnea",
    "B": "borderline",
    "C": "control",
}

#: Conventional clinical AHI cut-points, applied to the ``AHI`` column of
#: ``additional-information.txt``. Reported for description only — ``apnea_class``
#: is what folds are balanced on, because it is the release's own taxonomy.
AHI_SEVERITY_BINS = ((5.0, "normal"), (15.0, "mild"), (30.0, "moderate"))
AHI_SEVERITY_SEVERE = "severe"

#: RR intervals outside this range are dropped before any HRV summary — the
#: ``.qrs`` detections are unaudited, so both dropped beats and double detections
#: occur.
RR_RANGE_SECS = (0.3, 2.0)

#: Record names in this release: one letter of class/set, two digits. The ``*r``
#: and ``*er`` companions listed in RECORDS do not match, which is one of the two
#: independent filters :func:`scan_records` agrees on.
_RECORD_RE = re.compile(r"^[abcx]\d{2}$")

#: A data row of ``additional-information.txt``: the record name followed by
#: eleven whitespace-separated fields.
_INFO_ROW_RE = re.compile(
    r"^(?P<record_name>[abcx]\d{2})\s+"
    r"(?P<published_minutes>\d+)\s+"
    r"(?P<published_nonapnea_minutes>\d+)\s+"
    r"(?P<published_apnea_minutes>\d+)\s+"
    r"(?P<published_hours_with_apnea>\d+)\s+"
    r"(?P<ai>[\d.]+)\s+"
    r"(?P<hi>[\d.]+)\s+"
    r"(?P<ahi>[\d.]+)\s+"
    r"(?P<age>\d+)\s+"
    r"(?P<sex>[MF])\s+"
    r"(?P<height_cm>\d+)\s+"
    r"(?P<weight_kg>\d+)\s*$"
)

_INFO_FLOAT_COLUMNS = ("ai", "hi", "ahi")
_INFO_INT_COLUMNS = (
    "published_minutes",
    "published_nonapnea_minutes",
    "published_apnea_minutes",
    "published_hours_with_apnea",
    "age",
    "height_cm",
    "weight_kg",
)


def parse_additional_information(data_path: Path | str) -> pd.DataFrame:
    """Parse ``additional-information.txt`` into one row per record.

    The file is a fixed-width ASCII table under 20 lines of prose explaining how
    the challenge scored records; the prose is skipped by matching data rows
    rather than by counting lines, so a re-release that rewords the header still
    parses. It is the only source in the database for AHI, the apnea and
    hypopnea indices, and the demographics that recover the subject grouping.
    """
    from ecgbench.labels import LabelSourceMissingError

    path = Path(data_path) / INFO_FILE
    if not path.exists():
        raise LabelSourceMissingError(
            f"Apnea-ECG labels need {INFO_FILE}, which is not in {Path(data_path)}. "
            "It carries the AHI, the apnea/hypopnea indices and the age/sex/height/"
            "weight fields that ECGBench uses to group records by subject — without "
            "it there is no patient grouping and the folds would leak. Get it from "
            "https://physionet.org/content/apnea-ecg/1.0.0/"
        )

    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _INFO_ROW_RE.match(line.strip())
        if match:
            rows.append(match.groupdict())

    if not rows:
        raise ValueError(
            f"No data rows parsed from {path}. Expected lines like "
            "'a01\t490\t20\t470\t9\t12.5\t57.1\t69.6\t51\tM\t175\t102'."
        )

    df = pd.DataFrame(rows)
    for column in _INFO_INT_COLUMNS:
        df[column] = df[column].astype(int)
    for column in _INFO_FLOAT_COLUMNS:
        df[column] = df[column].astype(float)

    logger.info("Parsed %d records from %s", len(df), INFO_FILE)
    return df


def derive_apnea_class(n_apnea_minutes: int) -> str:
    """Return the release's own class letter for a record.

    PhysioNet describes the three groups by minutes of apnea: class C (control)
    has fewer than 5, class A (apnea) has at least 100, class B (borderline) is
    between. Applying that to the ``.apn`` counts reproduces the class letter
    encoded in the *name* of all 35 learning-set records — a01–a20 are all A,
    b01–b05 all B, c01–c10 all C, with no exceptions — which is what licenses
    using it on the 35 ``xNN`` records, whose names carry no class at all.
    """
    if n_apnea_minutes < APNEA_CLASS_MIN_MINUTES_B:
        return "C"
    if n_apnea_minutes < APNEA_CLASS_MIN_MINUTES_A:
        return "B"
    return "A"


def ahi_severity(ahi: float) -> str:
    """Bin an apnea-hypopnea index by the conventional clinical cut-points."""
    if not np.isfinite(ahi):
        return ""
    for threshold, name in AHI_SEVERITY_BINS:
        if ahi < threshold:
            return name
    return AHI_SEVERITY_SEVERE


def _summarise_apnea(record_path: Path) -> dict[str, object]:
    """Read one ``.apn`` file into a minute string and its counts.

    A missing ``.apn`` is called out specifically rather than left to ``wfdb``,
    because there is one likely cause and it is not a broken download: PhysioNet
    withheld the test set's answers for the challenge and only published
    ``x01.apn``–``x35.apn`` on 2020-06-01. A copy fetched before then holds all 70
    signals and only 35 labels, and ``wfdb`` would report that as a bare
    ``FileNotFoundError`` naming a path.
    """
    import wfdb

    from ecgbench.labels import LabelSourceMissingError

    apn_path = record_path.with_suffix(f".{ANNOTATOR_APNEA}")
    if not apn_path.exists():
        raise LabelSourceMissingError(
            f"No {apn_path.name} beside {record_path.name}. The per-minute apnea "
            "annotations are this dataset's ground truth, and for the 35 test records "
            "(x01-x35) PhysioNet published them only on 2020-06-01, after the "
            "challenge closed — a copy downloaded before then has the signals but not "
            "the labels, and the release's 2019 SHA256SUMS.txt does not list them "
            "either. Re-fetch from https://physionet.org/content/apnea-ecg/1.0.0/"
        )

    ann = wfdb.rdann(str(record_path), ANNOTATOR_APNEA)
    symbols = list(ann.symbol)
    sequence = "".join(symbols)

    unexpected = set(symbols) - {"A", "N"}
    if unexpected:
        logger.warning("%s.apn holds symbols beyond A/N: %s", record_path.name, sorted(unexpected))

    # The annotations are documented as one per minute starting at sample 0. Check
    # it rather than assume it: apnea_sequence is indexed by minute downstream, so
    # a gap would shift every label after it without any error being raised.
    expected = np.arange(len(symbols), dtype=np.int64) * SAMPLES_PER_MINUTE
    if not np.array_equal(np.asarray(ann.sample, dtype=np.int64), expected):
        logger.warning(
            "%s.apn annotations are not at exact one-minute intervals from sample 0; "
            "apnea_sequence may not be minute-aligned",
            record_path.name,
        )

    n_apnea = sequence.count("A")
    n_total = len(sequence)
    return {
        "apnea_sequence": sequence,
        "n_annotated_minutes": n_total,
        "n_apnea_minutes": n_apnea,
        "n_nonapnea_minutes": sequence.count("N"),
        "apnea_minute_fraction": (n_apnea / n_total) if n_total else np.nan,
    }


def _summarise_beats(record_path: Path, fs: float) -> dict[str, object]:
    """Whole-record HRV summaries from the unaudited ``.qrs`` detections."""
    import wfdb

    out: dict[str, object] = {
        "n_qrs_beats": 0,
        "n_qrs_artifacts": 0,
        "mean_hr_bpm": np.nan,
        "sdnn_ms": np.nan,
        "rmssd_ms": np.nan,
        "n_rr_rejected": 0,
    }

    ann = wfdb.rdann(str(record_path), ANNOTATOR_QRS)
    symbols = np.asarray(ann.symbol)
    samples = np.asarray(ann.sample, dtype=np.int64)

    # PhysioNet: detected beats are "N", QRS-like artifacts are "|". Only beats
    # may enter an RR interval.
    is_beat = symbols == "N"
    out["n_qrs_beats"] = int(is_beat.sum())
    out["n_qrs_artifacts"] = int((symbols == "|").sum())

    beats = samples[is_beat]
    if beats.size > 2:
        rr = np.diff(beats) / fs
        low, high = RR_RANGE_SECS
        keep = (rr >= low) & (rr <= high)
        out["n_rr_rejected"] = int((~keep).sum())
        rr = rr[keep]
        if rr.size > 1:
            out["mean_hr_bpm"] = float(60.0 / rr.mean())
            out["sdnn_ms"] = float(1000.0 * rr.std(ddof=1))
            out["rmssd_ms"] = float(1000.0 * np.sqrt(np.mean(np.diff(rr) ** 2)))

    return out


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every ECG record's header and annotations into one frame.

    **The partition is the 70 single-channel ECG records and nothing else.** The
    shipped ``RECORDS`` file lists 86 names, and the other 16 are not additional
    recordings:

    - 8 ``*r`` records (``a01r``, …) hold ``Resp C``, ``Resp A``, ``Resp N`` and
      ``SpO2`` and **no ECG at all** — an ECG benchmark cannot use them, and they
      would fail validation as a record with the wrong number of leads.
    - 8 ``*er`` records (``a01er``, …) are a *view*: their headers point at the
      very same ``a01.dat`` as the plain record, plus the ``r`` signals. Their ECG
      is not merely equivalent, it is the same bytes. Including them would put one
      recording in the partition twice.

    Two independent filters have to agree before a record is kept — the header
    declares exactly one signal named ``ECG``, and the name matches
    ``[abcx]NN`` — because either alone would silently admit the wrong set if a
    re-release changed the naming or the channel layout. The record list comes
    from ``RECORDS`` rather than a glob, so a stray file cannot enter the
    partition either way.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. Apnea-ECG labels live in the record "
            "headers and the .apn/.qrs annotation files, so point data_path at the "
            "version directory — the flat directory holding a01.hea, RECORDS, "
            f"ANNOTATORS and {INFO_FILE}. Get it from "
            "https://physionet.org/content/apnea-ecg/1.0.0/"
        )

    import wfdb

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue

        header = wfdb.rdheader(str(hea.with_suffix("")))
        by_name = _RECORD_RE.match(name) is not None
        by_signals = header.n_sig == 1 and list(header.sig_name or []) == ["ECG"]
        if not (by_name and by_signals):
            if by_name != by_signals:
                # The two filters disagreeing means the release changed shape;
                # never resolve it by preferring one of them.
                raise ValueError(
                    f"Record {name!r} is selected by its name ({by_name}) but not by "
                    f"its signals ({by_signals}, n_sig={header.n_sig}, "
                    f"sig_name={header.sig_name}). Apnea-ECG 1.0.0 has 70 records "
                    "where both agree; this copy differs, so the record set cannot be "
                    "determined safely."
                )
            skipped.append(name)
            continue

        sig_len = int(header.sig_len)
        fs = float(header.fs)
        row: dict[str, object] = {
            "record_name": name,
            "n_samples": sig_len,
            "duration_secs": sig_len / fs,
            "duration_hours": sig_len / fs / 3600.0,
            "sampling_rate": fs,
            "lead_names": "|".join(header.sig_name or []),
            # Flat tree: wfdb takes the bare stem, no extension, no subdirectory.
            "signal_path": name,
            "challenge_set": "test" if name.startswith("x") else "learning",
        }
        row.update(_summarise_apnea(hea.with_suffix("")))
        row.update(_summarise_beats(hea.with_suffix(""), fs))
        rows.append(row)

    if skipped:
        logger.info(
            "Excluded %d companion records with no ECG of their own: %s",
            len(skipped),
            ", ".join(skipped),
        )

    if not rows:
        # Otherwise the sort below fails with a KeyError on a frame that has no
        # columns, which says nothing about the actual mistake.
        raise LabelSourceMissingError(
            f"No single-channel ECG records found under {data_path}, though RECORDS "
            f"lists {len(names)} names. Apnea-ECG 1.0.0 holds 70 of them (a01-a20, "
            "b01-b05, c01-c10, x01-x35); point data_path at the version directory "
            "itself, not at the dataset root above it."
        )

    # Which ECG records have respiration alongside them is derived from the
    # companions actually present, not asserted from a constant, so the column
    # describes this copy. RESPIRATION_RECORDS is the documented expectation and
    # only decides whether to warn.
    with_respiration = {name.removesuffix("er").removesuffix("r") for name in skipped}
    if with_respiration != set(RESPIRATION_RECORDS):
        logger.warning(
            "Records with respiration companions (%s) differ from the set PhysioNet "
            "documents (%s); has_respiration follows the files.",
            ", ".join(sorted(with_respiration)) or "none",
            ", ".join(sorted(RESPIRATION_RECORDS)),
        )

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    df["has_respiration"] = df["record_name"].isin(with_respiration)
    logger.info(
        "Parsed %d ECG records: %.1f h of signal, %d annotated minutes " "(%d apnea, %.1f%%)",
        len(df),
        df["duration_secs"].sum() / 3600,
        int(df["n_annotated_minutes"].sum()),
        int(df["n_apnea_minutes"].sum()),
        100 * df["n_apnea_minutes"].sum() / df["n_annotated_minutes"].sum(),
    )
    return df


def assign_subject_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``subject_id``, recovering the grouping the release does not publish.

    **This is the column that stops the folds leaking, and the database ships
    nothing that would let a user notice its absence.** There is no subject
    field anywhere in Apnea-ECG: not in the headers, not in ``RECORDS``, not in
    the annotations. What ``additional-information.txt`` does publish, per
    record, is age, sex, height and weight — and those four fields take exactly
    **32** distinct values across the 70 records, which is the number of subjects
    the database is described as containing. Records sharing all four are treated
    as one subject.

    Two verified duplicate *recordings* (:data:`DUPLICATE_RECORDS`) then merge
    two further pairs of those groups, because a bit-identical waveform is
    stronger evidence of subject identity than a demographics row — leaving
    **30** groups. That merge is transitive and deliberately conservative: it
    unions ``{x17, x22}`` with ``{c01, x35}`` even though the demographics of
    ``x22`` and ``x35`` contradict each other, since with one of the two rows
    known to be wrong there is no way to tell which subject the other records
    belong to. Over-grouping costs a little fold granularity; under-grouping
    leaks, and cannot be detected downstream.

    The resulting id is ``subj_<lowest record name in the group>`` — stable
    across runs, and traceable back to a record by inspection.

    A group is a **subject**, not a night: 27 of the 30 contribute more than one
    recording, up to four. That is the whole reason the challenge's own split
    leaks, and why ``StratifiedGroupKFold`` is not optional here.
    """
    required = {"age", "sex", "height_cm", "weight_kg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"assign_subject_ids needs {sorted(required)} — {sorted(missing)} are "
            f"absent. Merge {INFO_FILE} into the frame first."
        )

    out = df.copy()
    parent = {name: name for name in out["record_name"]}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    demographics = out.groupby(["age", "sex", "height_cm", "weight_kg"])["record_name"]
    n_demographic_groups = demographics.ngroups
    for _, names in demographics:
        names = list(names)
        for name in names[1:]:
            union(names[0], name)

    for duplicate, (canonical, _offset) in DUPLICATE_RECORDS.items():
        if duplicate in parent and canonical in parent:
            union(duplicate, canonical)

    roots = {name: find(name) for name in parent}
    out["subject_id"] = out["record_name"].map(lambda n: f"subj_{roots[n]}")
    out["duplicate_of"] = out["record_name"].map(lambda n: DUPLICATE_RECORDS.get(n, ("", 0))[0])

    n_subjects = out["subject_id"].nunique()
    logger.info(
        "Recovered %d subjects from %d records (%d demographic groups, merged to %d "
        "by %d verified duplicate recordings)",
        n_subjects,
        len(out),
        n_demographic_groups,
        n_subjects,
        len(DUPLICATE_RECORDS),
    )

    if "challenge_set" in out.columns:
        spanning = out.groupby("subject_id")["challenge_set"].nunique()
        n_spanning = int((spanning > 1).sum())
        n_records = int(out["subject_id"].isin(spanning[spanning > 1].index).sum())
        logger.warning(
            "The challenge's own learning/test split is subject-leaky: %d of %d "
            "subjects (%d of %d records) appear on both sides. ECGBench groups folds "
            "on subject_id instead; use challenge_set only to reproduce the original "
            "challenge result.",
            n_spanning,
            n_subjects,
            n_records,
            len(out),
        )

    return out


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class`` — the release's A/B/C class — and say why.

    This is the **only** derivation of the stratification label;
    ``ApneaECGSplitter`` reads the column rather than recomputing it, so the
    exposed label and the fold label cannot drift.

    ``apnea_class`` is the right axis for three reasons. It is the release's own
    taxonomy, so a fold balanced on it is balanced on the quantity the database
    was built around. It survives 70 records over 10 folds — 40 A, 10 B, 20 C, so
    every class has at least ``n_folds`` members and none is spread impossibly
    thin. And it is a *record-level* summary of the per-minute ground truth,
    which means balancing on it also roughly balances the minute-level positive
    rate that a per-minute apnea detector is actually scored on.

    ``ahi_severity`` is exposed alongside it but is **not** used: its four bins
    split 25/3/11/31, and a class of 3 cannot be spread over 10 folds at all.
    """
    if "apnea_class" not in df.columns:
        raise ValueError(
            "'apnea_class' missing — derive it with derive_apnea_class() before "
            "attaching the stratification label."
        )

    out = df.copy()
    out["stratify_class"] = out["apnea_class"].astype(str)
    logger.info(
        "Stratification classes (apnea_class): %s",
        out["stratify_class"].value_counts().sort_index().to_dict(),
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Apnea-ECG labels indexed by record name, one row per ECG record.

    Columns:

    - ``apnea_sequence`` — **the actual ground truth**: one character per minute,
      ``A`` or ``N``, in record order, straight from the ``.apn`` file. ``A``
      means apnea was in progress *at the beginning* of that minute (PhysioNet
      corrected an earlier, wider description; see the module docstring). Index
      it by minute to build a per-minute target, e.g.
      ``np.frombuffer(seq.encode(), "S1") == b"A"``.
    - ``n_annotated_minutes``, ``n_apnea_minutes``, ``n_nonapnea_minutes``,
      ``apnea_minute_fraction`` — counts over that sequence. 34,313 minutes in
      total, 13,064 of them apnea.
    - ``apnea_class`` — ``A``/``B``/``C``, the release's own classes, derived by
      :func:`derive_apnea_class`. Reproduces the letter in the name of all 35
      learning-set records exactly, and supplies the class for the 35 ``xNN``
      records, whose names carry none.
    - ``subject_id`` — the recovered subject grouping, **30 subjects for 70
      records**. See :func:`assign_subject_ids`; this is the fold grouping key.
    - ``challenge_set`` — ``learning`` or ``test``, the PhysioNet/CinC Challenge
      2000 division. **Not** ECGBench's split, and not safe as one: 18 of the 30
      subjects have records on both sides. Kept so the original challenge result
      stays reproducible.
    - ``duplicate_of`` — the record this one duplicates, for the two verified
      cases (``x35``→``x22``, ``c06``→``c05``), otherwise empty.
    - ``ahi``, ``ai``, ``hi``, ``ahi_severity`` — apnea-hypopnea index and its
      components from ``additional-information.txt``, and the conventional
      clinical bin. Descriptive; folds are not balanced on them.
    - ``age``, ``sex``, ``height_cm``, ``weight_kg``, ``bmi`` — subject
      demographics, also from that file. These four fields are what recovers
      ``subject_id``.
    - ``published_minutes``, ``published_apnea_minutes``,
      ``published_nonapnea_minutes``, ``published_hours_with_apnea`` — the same
      quantities as scored for the challenge. Compare with the ``n_*`` columns:
      they agree to the annotation convention's one minute everywhere except
      ``c07`` and ``c08``, whose shipped signals are 25 and 22 minutes shorter
      than the table scores.
    - ``n_qrs_beats``, ``n_qrs_artifacts``, ``mean_hr_bpm``, ``sdnn_ms``,
      ``rmssd_ms``, ``n_rr_rejected`` — from the ``.qrs`` file, which PhysioNet
      states is **machine-generated and unaudited**. Descriptive only; not a
      beat-detection reference.
    - ``n_samples``, ``duration_secs``, ``duration_hours``, ``sampling_rate``,
      ``lead_names`` — record geometry. Duration is **not** uniform: 6.75 h
      (``x17``) to 9.62 h (``a12``).
    - ``has_respiration`` — True for the 8 records with ``*r``/``*er`` companions
      carrying chest, abdominal and nasal respiration and SpO2. Those companions
      are not in the partition (their ECG is the same bytes as this record's).
    - ``stratify_class`` — ``apnea_class``, for fold construction. See
      :func:`attach_stratify_class`.
    """
    df = scan_records(data_path)
    info = parse_additional_information(data_path)

    merged = df.merge(info, on="record_name", how="left", validate="one_to_one")
    unmatched = merged["ahi"].isna()
    if unmatched.any():
        # Every subject-level field, including the grouping key, comes from this
        # join. A record missing from the table would be silently ungrouped.
        raise ValueError(
            f"{int(unmatched.sum())} records have no row in {INFO_FILE}: "
            f"{sorted(merged.loc[unmatched, 'record_name'])}. AHI, demographics and "
            "the subject grouping all come from that file, so the folds would leak."
        )

    merged["apnea_class"] = merged["n_apnea_minutes"].map(derive_apnea_class)
    merged["apnea_class_name"] = merged["apnea_class"].map(APNEA_CLASS_NAMES)
    merged["ahi_severity"] = merged["ahi"].map(ahi_severity)
    merged["bmi"] = merged["weight_kg"] / (merged["height_cm"] / 100.0) ** 2

    merged = assign_subject_ids(merged)
    merged = attach_stratify_class(merged)

    merged = merged.set_index("record_name")
    merged.index.name = config.record_id_column
    return merged
