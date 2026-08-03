"""
Leipzig Heart Center labels: subject info, header layout and the ``.atr`` annotations.

Two per-subject CSVs ship, not one — ``children-subject-info.csv`` (29 rows) and
``adults-subject-info.csv`` (10 rows) — and they do not have the same columns: only
the children's carries the accessory-pathway location. This loader concatenates
them, adds a ``cohort`` column, and joins the per-record channel layout from the
``.hea`` plus a summary of the cardiologist's ``.atr`` annotations.

Quirks worth knowing before using it, all verified against the files (every one of
the 126 files matches the release's own ``SHA256SUMS.txt``, so these are upstream):

- **The channel layout is not what the README describes, and it varies.** The
  README says children carry 12-lead ECG plus 5 coronary-sinus, 1 right-ventricular
  apex and 1 ablation channel (19), and adults the ECG plus RVA and ABL with the CS
  catheter "in some studies". In fact there are **six distinct layouts** and the
  channel count runs 14, 18, 19 or 20. Four children's records carry no ``ABL12``
  at all; ``x0023`` adds an undocumented ``ABL_uni``; ``x0028`` adds an
  undocumented ``ART`` **at index 12**, shifting everything after it; and ``x100``
  puts ``RVA12`` last instead of 14th. **Only channels 0-11 — the 12-lead ECG — are
  the same channel in the same position in every record.** Anything beyond index 11
  must be located by name from that record's own header, which is what
  :func:`channel_index` is for.
- **The annotation total in the README counts only some of the annotations.** It
  claims 113,924 annotated beats. The 39 ``.atr`` files hold **118,214**
  annotations: exactly 113,924 in the beat classes the README tabulates, plus 1,824
  ``Q`` (unclassifiable, not tabulated), 228 ``~`` (signal-quality change, which is
  not a beat at all — the ``ANNOTATORS`` file does mention quality annotations) and
  2,238 ``+`` rhythm markers. So the README figure is right about what it counts;
  ``n_beats`` here reproduces it, and the others are counted separately.
- **``dataset_info.csv`` has one row wrong.** The ``PVC`` row reads
  ``PVC,PVC,V,Premature ventricular beats`` — i.e. symbol ``PVC`` and aux string
  ``V`` — while the README and the data both use symbol ``V`` with no aux string.
  No annotation anywhere carries the symbol ``PVC``.
- **One age is malformed.** ``x007`` has ``age`` = ``.14.3``, which no float parser
  accepts. A single leading ``.`` is stripped (giving 14.3, consistent with every
  other child's age) and the verbatim string is kept in ``age_raw`` so the repair is
  visible rather than assumed.
- **``x005``'s duration disagrees with its header** — ``ecg_duration`` says
  0:30:58.974 but the header holds 1,813,477 samples at 977 Hz, i.e. 1856.169 s,
  2.8 s shorter. The other 38 records agree to within 0.01 s. ``duration_seconds``
  here comes from the CSV and ``header_seconds`` from the header, so the
  disagreement stays visible.
- **The children's CSV column is spelled ``ap_loacation``.** Exposed as
  ``ap_location``; the typo is the source's.

Records are **one per subject** — 29 children and 10 adults, 39 records, 39
subjects — so there is nothing to group folds by. Length varies from 77.7 s
(``x0027``) to 2 h 30 m (``x003``), 18.5 hours in total.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Annotator extension, per the shipped ANNOTATORS file.
ANNOTATOR = "atr"

#: The two per-subject tables, and the cohort each describes.
SUBJECT_CSVS = {
    "child": "children-subject-info.csv",
    "adult": "adults-subject-info.csv",
}

#: The 12 ECG channels, which are channels 0-11 of every record. Nothing beyond
#: index 11 is safe to address positionally — see the module docstring.
ECG_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

#: Intracardiac channels the release documents, with the catheter each comes from.
#: Which of these a record carries, and in what order, differs per record.
IEGM_CHANNELS = {
    "ABL12": "ablation and mapping catheter, distal electrodes",
    "RVA12": "catheter in the right ventricular apex, distal electrodes",
    "CS12": "coronary sinus electrodes 1/2 (distal)",
    "CS34": "coronary sinus electrodes 3/4",
    "CS56": "coronary sinus electrodes 5/6",
    "CS78": "coronary sinus electrodes 7/8",
    "CS90": "coronary sinus electrodes 9/10 (proximal)",
}

#: Channels present in the files that the release documents nowhere. ``ART`` sits at
#: index 12 of ``x0028``, ahead of the intracardiac channels, so it is the clearest
#: reason not to address channels by position.
UNDOCUMENTED_CHANNELS = {
    "ABL_uni": "unipolar ablation channel; x0023 only",
    "ART": "not described in the release; x0028 only, at channel index 12",
}

#: Beat classes the README tabulates. Their counts sum to the 113,924 the README
#: reports, which is why ``Q`` and ``~`` below are counted separately.
#:
#: ``X`` (tachycardia) and ``b`` (AV block) are **not** standard WFDB symbols. Each
#: ``.atr`` declares them as custom labels at stores 42 and 43, which is why
#: ``wfdb.rdann`` returns them verbatim; a ``wfdb.wrann`` round trip without the
#: same ``custom_labels`` rewrites them as ``'"'`` NOTE annotations instead.
BEAT_SYMBOLS = ("N", "X", "/", "R", "A", "V", "J", "F", "a", "f", "L", "b", "j")

#: The custom annotation labels the release declares, as (label_store, symbol,
#: description). Needed to write a ``.atr`` that reads back the way these do.
CUSTOM_ANNOTATION_LABELS = ((42, "X", "Tachycardias"), (43, "b", "AV-Block"))

#: Human-readable names for the beat symbols, for docs and the example script.
BEAT_NAMES = {
    "N": "normal (sinus) beat",
    "X": "tachycardia beat — see the tachy_* columns for which",
    "/": "paced beat",
    "R": "complete right bundle branch block beat",
    "A": "premature atrial beat",
    "V": "premature ventricular beat",
    "J": "premature junctional beat",
    "F": "fusion beat",
    "a": "aberrated premature atrial beat",
    "f": "fusion of ventricular paced and normal beat",
    "L": "complete left bundle branch block beat",
    "b": "AV block 1 degree",
    "j": "junctional escape beat",
}

#: Annotations that are not beats in the README's sense, and the column each goes
#: to. ``~`` is a signal-quality change marker, not a beat of any kind.
NON_BEAT_SYMBOLS = {
    "Q": "n_unclassifiable",
    "~": "n_quality_marks",
    "+": "n_rhythm_changes",
}

#: Aux strings carried by ``X`` beats, naming the tachycardia. Every one of the
#: 29,477 ``X`` annotations carries exactly one of these.
TACHYCARDIA_AUX = {
    "AVRT": "tachy_AVRT",
    "AVNRT": "tachy_AVNRT",
    "avrt": "tachy_aberrated_AVRT",
    "avnrt": "tachy_aberrated_AVNRT",
    "AVNRT+BII": "tachy_AVNRT_with_AVblock2",
    "VT": "tachy_VT",
    "IVR": "tachy_IVR",
    "AFIB": "tachy_AFIB",
    "EAT": "tachy_EAT",
    "AFL": "tachy_AFL",
}

#: Aux strings that qualify an otherwise-standard beat symbol.
BEAT_AUX = {
    "N-Prex": "aux_preexcited_N",
    "A-Prex": "aux_preexcited_A",
    "/A": "aux_paced_atrial",
    "/V": "aux_paced_ventricular",
    "BI": "aux_avblock1",
}

#: Rhythm aux strings, carried by the ``+`` markers. WFDB writes them with a
#: leading ``(``.
RHYTHM_AUX = {
    "(N": "rhythm_sinus",
    "(VT": "rhythm_VT",
    "(IVR": "rhythm_IVR",
    "(AVRT": "rhythm_AVRT",
    "(AVNRT": "rhythm_AVNRT",
    "(AFIB": "rhythm_AFIB",
    "(EAT": "rhythm_EAT",
    "(AFL": "rhythm_AFL",
    "(A": "rhythm_ectopic_atrial",
    "(B": "rhythm_ventricular_bigeminy",
    "(J": "rhythm_junctional",
    "(/A": "rhythm_paced_atrial",
    "(/V": "rhythm_paced_ventricular",
}

#: Stratification classes with fewer than this many records are pooled into OTHER.
#: With the diagnosis *family* below nothing is pooled — the three families hold 16,
#: 13 and 10 records — but a reissue that added a fourth would be caught here.
MIN_CLASS_RECORDS = 10

OTHER = "OTHER"
UNKNOWN = "UNKNOWN"


def _require(data_path: Path) -> None:
    from ecgbench.labels import LabelSourceMissingError

    missing = [name for name in SUBJECT_CSVS.values() if not (data_path / name).exists()]
    if missing:
        raise LabelSourceMissingError(
            f"Leipzig Heart Center labels come from {' and '.join(SUBJECT_CSVS.values())}, "
            f"and {missing} are not in {data_path}. Point data_path at the dataset root "
            "— the flat directory holding x001.hea, RECORDS and the two subject CSVs — "
            "from https://physionet.org/content/leipzig-heart-center-ecg/1.0.0/ ."
        )


def parse_age(value: object) -> float | None:
    """Parse one ``age`` cell, repairing the single malformed value in the release.

    ``x007`` ships ``.14.3``, which no float parser accepts. A single leading ``.``
    is stripped — giving 14.3, in range for a child in this cohort — and a warning
    is logged. Anything else unparseable comes back as None rather than guessed.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith(".") and text.count(".") == 2:
        repaired = text[1:]
        try:
            age = float(repaired)
        except ValueError:
            logger.warning("Unparseable age %r; returning None", text)
            return None
        logger.warning(
            "Malformed age %r in the shipped CSV; read as %s after stripping the "
            "leading '.'. The verbatim string is kept in age_raw.",
            text,
            age,
        )
        return age
    logger.warning("Unparseable age %r; returning None", text)
    return None


def parse_duration(value: object) -> float | None:
    """Parse an ``ecg_duration`` of the form ``H:m:s.SSS`` into seconds."""
    if value is None:
        return None
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        logger.warning("Unparseable ecg_duration %r; returning None", value)
        return None
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        logger.warning("Unparseable ecg_duration %r; returning None", value)
        return None


def diagnosis_family(diagnosis: object) -> str:
    """Reduce a diagnosis to its family: ``AVRT``, ``AVNRT`` or ``TOF``.

    The shipped ``diagnosis`` values are ``AVRT``, ``AVRT-WPW``, ``AVRT-PJRT``,
    ``AVNRT``, ``TOF with VT``, ``TOF without VT`` and ``TOF with nsVT``. The family
    is the leading token, so the accessory-pathway variants fold into ``AVRT`` and
    the three Tetralogy-of-Fallot presentations into ``TOF``.

    This is a **coarsening for fold construction**, not a clinical grouping: it
    discards whether an adult had sustained VT, which is the very thing the adult
    cohort is interesting for. Train on ``diagnosis``.
    """
    text = str(diagnosis or "").strip()
    if not text:
        return UNKNOWN
    return text.replace("-", " ").split()[0]


def channel_index(channel_names: list[str] | tuple[str, ...] | str, channel: str) -> int | None:
    """Position of ``channel`` in one record's channel list, or None if absent.

    Use this rather than a fixed index for anything past the 12-lead ECG. Six
    distinct layouts ship and index 12 alone is ``ABL12``, ``RVA12`` or ``ART``
    depending on the record.

    ``channel_names`` accepts either the list from ``wfdb.rdheader(...).sig_name``
    or the ``channel_names`` column of :func:`load_labels`, which is the same list
    joined with ``'|'`` so it survives a CSV round trip.
    """
    names = channel_names.split("|") if isinstance(channel_names, str) else list(channel_names)
    try:
        return names.index(channel)
    except ValueError:
        return None


def read_header(record_path: Path) -> dict[str, object]:
    """Read one record's channel layout, sampling rate and length from its header."""
    import wfdb

    out: dict[str, object] = {
        "n_signals": 0,
        "sampling_rate": 0,
        "n_samples": 0,
        "header_seconds": None,
        "channel_names": "",
        "n_iegm_channels": 0,
    }
    try:
        header = wfdb.rdheader(str(record_path))
    except Exception as e:  # corrupt_header in validation reports this properly
        logger.warning("Could not read %s.hea: %s", record_path.name, e)
        return out

    names = list(header.sig_name)
    out["n_signals"] = int(header.n_sig)
    out["sampling_rate"] = int(header.fs)
    out["n_samples"] = int(header.sig_len)
    out["header_seconds"] = header.sig_len / header.fs if header.fs else None
    out["channel_names"] = "|".join(names)
    out["n_iegm_channels"] = sum(1 for n in names if n not in ECG_LEADS)
    unknown = [n for n in names if n not in ECG_LEADS and n not in IEGM_CHANNELS]
    if unknown:
        logger.debug(
            "%s carries channel(s) the release does not document: %s", record_path.name, unknown
        )
    return out


def count_annotations(record_path: Path) -> dict[str, int]:
    """Summarise one record's ``.atr`` into per-symbol and per-aux-string counts.

    ``n_beats`` counts only the classes the README tabulates, so summing it over the
    39 records reproduces the published 113,924. ``n_unclassifiable`` (``Q``),
    ``n_quality_marks`` (``~``) and ``n_rhythm_changes`` (``+``) are counted
    separately for that reason, and ``n_annotations`` is the true file total.
    """
    import wfdb

    counts: dict[str, int] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
    counts.update({column: 0 for column in NON_BEAT_SYMBOLS.values()})
    counts.update({column: 0 for column in TACHYCARDIA_AUX.values()})
    counts.update({column: 0 for column in BEAT_AUX.values()})
    counts.update({column: 0 for column in RHYTHM_AUX.values()})
    counts["n_beats"] = 0
    counts["n_annotations"] = 0

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .atr must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    notes = annotation.aux_note or [""] * len(annotation.symbol)
    unexpected_symbols: set[str] = set()
    unexpected_aux: set[str] = set()
    beat_set = set(BEAT_SYMBOLS)

    for symbol, note in zip(annotation.symbol, notes):
        counts["n_annotations"] += 1
        if symbol in beat_set:
            counts[f"beat_{symbol}"] += 1
            counts["n_beats"] += 1
        elif symbol in NON_BEAT_SYMBOLS:
            counts[NON_BEAT_SYMBOLS[symbol]] += 1
        else:
            unexpected_symbols.add(symbol)

        text = str(note or "").strip("\x00").strip()
        if not text:
            continue
        for table in (TACHYCARDIA_AUX, BEAT_AUX, RHYTHM_AUX):
            if text in table:
                counts[table[text]] += 1
                break
        else:
            unexpected_aux.add(text)

    if unexpected_symbols:
        logger.warning(
            "%s: annotation symbols not in BEAT_SYMBOLS, uncounted: %s",
            record_path.name,
            sorted(unexpected_symbols),
        )
    if unexpected_aux:
        logger.warning(
            "%s: aux strings this module does not know, uncounted: %s",
            record_path.name,
            sorted(unexpected_aux),
        )
    return counts


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Join the two subject CSVs with each record's header and annotation summary.

    Adds ``cohort`` (``child``/``adult``), ``signal_path`` (the bare record stem —
    the tree is flat), the parsed ``age`` and ``duration_seconds``, the channel
    layout, and one column per beat symbol, tachycardia and rhythm.
    """
    data_path = Path(data_path)
    _require(data_path)

    frames = []
    for cohort, filename in SUBJECT_CSVS.items():
        # dtype=str throughout: subject_id is zero-padded ('001', '0010') and would
        # lose its padding as an int, and age must survive the malformed '.14.3'.
        part = pd.read_csv(data_path / filename, dtype=str)
        part["cohort"] = cohort
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    if "ap_loacation" in df.columns:
        # The source misspells it; expose the corrected name.
        df = df.rename(columns={"ap_loacation": "ap_location"})
    elif "ap_location" not in df.columns:
        df["ap_location"] = pd.NA

    df["record_name"] = df["file_name"]
    df["signal_path"] = df["file_name"]  # flat tree: wfdb takes the bare stem
    df["age_raw"] = df["age"]
    df["age"] = df["age_raw"].map(parse_age)
    df["duration_seconds"] = df["ecg_duration"].map(parse_duration)
    df["diagnosis_family"] = df["diagnosis"].map(diagnosis_family)

    extra = []
    for record in df["record_name"]:
        row = read_header(data_path / str(record))
        row.update(count_annotations(data_path / str(record)))
        extra.append(row)
    df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

    df["duration_delta_seconds"] = df["header_seconds"] - df["duration_seconds"]

    # Sort numerically, not lexically. The stems are x001..x009 then x0010..x0029
    # then x100..x109, so a string sort puts x0010 before x002.
    order = df["file_name"].str.removeprefix("x").astype(int)
    df = df.iloc[order.argsort(kind="stable")].reset_index(drop=True)
    logger.info(
        "Parsed %d Leipzig records (%d children, %d adults); %d tabulated beats, "
        "%d annotations in total, %d distinct channel layouts",
        len(df),
        int((df["cohort"] == "child").sum()),
        int((df["cohort"] == "adult").sum()),
        int(df["n_beats"].sum()),
        int(df["n_annotations"].sum()),
        df["channel_names"].nunique(),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: the diagnosis family, pooled for folds.

    This is the **only** derivation of the stratification label —
    ``LeipzigHeartCenterSplitter`` reads the column rather than recomputing it, so
    the exposed label and the fold label cannot drift.

    With 39 records over 10 folds the label has to be coarse. The three families
    hold 16 (AVRT), 13 (AVNRT) and 10 (TOF) records, which is just enough for ten
    folds; the full ``diagnosis`` has seven classes, three of them singletons, and
    cannot be spread across folds at all.
    """
    out = df.copy()
    labels = out["diagnosis_family"].fillna(UNKNOWN).replace("", UNKNOWN)

    counts = labels.value_counts()
    rare = set(counts[counts < MIN_CLASS_RECORDS].index)
    if rare:
        logger.info(
            "Pooling %d diagnosis family/families with <%d records into '%s': %s",
            len(rare),
            MIN_CLASS_RECORDS,
            OTHER,
            sorted(rare),
        )
        labels = labels.where(~labels.isin(rare), OTHER)

    out["stratify_class"] = labels
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return Leipzig Heart Center labels indexed by record name.

    Columns:

    - ``diagnosis`` — the subject-level diagnosis, verbatim: ``AVRT``,
      ``AVRT-WPW``, ``AVRT-PJRT``, ``AVNRT`` for the children and ``TOF with VT``,
      ``TOF without VT``, ``TOF with nsVT`` for the adults. **This is the ground
      truth**; it is one label per record, not multi-label.
    - ``diagnosis_family`` — ``AVRT`` / ``AVNRT`` / ``TOF``, and
      ``stratify_class`` — the same thing after pooling. **For fold construction
      only.** It drops the accessory-pathway variant and whether the VT was
      sustained.
    - ``cohort`` — ``child`` (29 records, x001-x0029) or ``adult`` (10, x100-x109).
    - ``ap_location`` — accessory-pathway location, children only and empty for the
      13 with AVNRT, which has no accessory pathway. The source column is
      misspelled ``ap_loacation``.
    - ``gender``, ``age`` (years, float), ``age_raw`` (verbatim — see
      :func:`parse_age` for the one malformed value).
    - ``ecg_duration`` (verbatim ``H:m:s.SSS``), ``duration_seconds`` (parsed),
      ``header_seconds`` (from the header) and ``duration_delta_seconds``. The two
      disagree by 2.8 s on ``x005`` and by under 0.01 s everywhere else.
    - ``n_signals``, ``channel_names`` (``'|'``-joined), ``n_iegm_channels``,
      ``sampling_rate``, ``n_samples``. Address channels past the ECG with
      :func:`channel_index`, never by position.
    - ``beat_N`` … ``beat_j`` per :data:`BEAT_NAMES`, with ``n_beats`` (the classes
      the README tabulates, summing to its 113,924), ``n_unclassifiable``,
      ``n_quality_marks``, ``n_rhythm_changes`` and ``n_annotations``.
    - ``tachy_*`` — which tachycardia each ``X`` beat was, per
      :data:`TACHYCARDIA_AUX`. This is where AVRT/AVNRT/VT/AFIB episodes are
      recorded at beat level, and it is richer than ``diagnosis``.
    - ``aux_*`` — pre-excitation and pacing qualifiers, per :data:`BEAT_AUX`.
    - ``rhythm_*`` — rhythm-segment markers, per :data:`RHYTHM_AUX`.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
