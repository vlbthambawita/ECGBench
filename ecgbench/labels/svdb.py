"""
MIT-BIH Supraventricular Arrhythmia labels: beat counts, SVEB burden, signal quality.

This release ships **no metadata file and no header comments at all** — not age,
not sex, not a subject identifier, not a clinical description. Every header is
four lines of pure signal specification::

    800 2 128 230400
    800.dat 212 200 10 0 -101 -25183 0 ECG1
    800.dat 212 200 10 0 123 10510 0 ECG2

So unlike ``mitdb`` (which carries demographics, medications and a free-text
description in its comments) and ``nsrdb`` (which carries ``# <age> <sex>``),
everything this module returns is derived from the ``.atr`` reference
annotations. That is not a gap in the parsing; it is the whole of what was
released.

**1. The beat labels use ``S``, where MIT-BIH Arrhythmia uses ``A``.** This
database exists to supplement the supraventricular examples in ``mitdb``, and it
annotates all 12,188 of its supraventricular ectopic beats with ``S``
(supraventricular premature), while ``mitdb`` annotates its 2,546 with ``A``
(atrial premature) and uses ``S`` exactly twice. Concatenating the two on the raw
symbol therefore produces a model that has learnt two disjoint label vocabularies
for the same phenomenon. The ``aami_*`` columns exist to prevent that: they are
the AAMI EC57 five-class reduction (see :data:`AAMI_CLASSES`), under which
``A``, ``a``, ``J`` and ``S`` all collapse to class ``S``, and they are directly
comparable across every MIT-BIH database in this catalogue.

**2. Five of the 78 records contain no supraventricular ectopy whatsoever**
(802, 803, 804, 805, 893), and a sixth (811) contains one beat. The database is
not uniformly supraventricular; ``sveb_fraction`` runs from 0.0 to 0.575 and is
the axis folds are stratified on. At the other end, record 865 is the only one in
which ectopic beats outnumber normal ones — 1,818 SVEB and 235 VEB against 1,102
normal beats.

**3. Signal quality is annotated per channel, and the leading span is not always
asserted.** The ``~`` annotations mark quality transitions and their ``subtype``
is a bitmask over the two channels — 0 clean, 1 ECG1 noisy, 2 ECG2 noisy, 3 both.
43 of the 78 records carry at least one. In 39 of those the first ``~`` is a
transition *into* noise, so the span before it is clean by implication; in the
other **four it is a transition into clean** (803, 855, 857, 885), which means
the span before it was never asserted to be anything. For record 803 that span is
1,555.1 s — 86% of the record. This module counts those spans as clean, which is
what WFDB itself does, and reports them separately in
``quality_head_unasserted_secs`` so the assumption is visible rather than buried.

**4. Beat annotation covers the whole record, unlike nsrdb.** The first beat
falls between 0.01 s and 1.27 s and the last between 1798.87 s and 1800.00 s in
every one of the 78 records, so any window inside the record has reference
annotation behind it. ``annotated_fraction`` is reported anyway, so a re-release
that changes this cannot do so silently.

**5. There is effectively no rhythm annotation layer.** The entire release
contains one ``+`` annotation — ``(N`` at sample 112,143 of record 852 — against
1,291 in ``mitdb``. There is no ``dominant_rhythm`` to be had here and this
module does not invent one; the labels are beat labels.

``mean_hr_bpm``, ``sdnn_ms`` and ``rmssd_ms`` are computed over RR intervals in
:data:`RR_RANGE_SECS`, which drops the double detections and the gaps around
artefact. They are whole-record summaries over a rhythm that is by construction
ectopic, so they describe the recording rather than the subject's sinus node.
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

#: Annotator extension, per the shipped ANNOTATORS file. It is the only one.
ANNOTATOR = "atr"

#: Samples per record. Uniform across all 78 (230,400 at 128 Hz = 1800.0 s).
RECORD_SAMPLES = 230400

#: Sampling rate, uniform across all 78 records.
SAMPLING_RATE = 128

#: Beat symbols occurring in this release, descending by frequency. These eight
#: sum to 184,583; the remaining 3,294 annotations are the non-beat markers below.
BEAT_SYMBOLS = ("N", "S", "V", "Q", "F", "J", "a", "B")

BEAT_NAMES = {
    "N": "normal beat",
    "S": "supraventricular premature or ectopic beat",
    "V": "premature ventricular contraction",
    "Q": "unclassifiable beat",
    "F": "fusion of ventricular and normal beat",
    "J": "nodal (junctional) premature beat",
    "a": "aberrated atrial premature beat",
    "B": "bundle branch block beat (unspecified)",
}

#: AAMI EC57 five-class reduction, symbol -> class. This is what makes the beat
#: labels comparable with ``mitdb``, whose supraventricular beats are ``A`` and
#: whose bundle-branch-block beats are ``L``/``R``; only the symbols occurring in
#: one of the MIT-BIH-family databases in this catalogue are listed, because a symbol
#: in none of them would be a silent no-op here and is better raised as "unexpected"
#: by the scanner. ``ecgbench.labels.edb`` imports this rather than keeping a second
#: copy — ``n`` (supraventricular escape) occurs only there, 5 beats of 790,565.
AAMI_CLASSES = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "B": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "n": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

#: The five AAMI classes, in the order EC57 lists them.
AAMI_ORDER = ("N", "S", "V", "F", "Q")

#: Non-beat annotation symbols occurring here, mapped to the column counting them.
#: Never add these to ``n_beats``.
NON_BEAT_COLUMNS = {
    "|": "n_isolated_artifacts",
    "~": "n_quality_changes",
    "+": "n_rhythm_changes",
}

#: ``subtype`` of a ``~`` annotation: a bitmask over the two channels. -1 is
#: WFDB's "signals unreadable"; it does not occur in this release.
QUALITY_SUBTYPES = {
    0: "clean",
    1: "noisy_ECG1",
    2: "noisy_ECG2",
    3: "noisy_both",
    -1: "unreadable",
}

#: RR intervals outside this range are dropped before any HRV summary — double
#: detections below, and the gaps spanning artefact above.
RR_RANGE_SECS = (0.3, 2.0)

#: Upper edges of the SVEB burden bands, as a fraction of all beats in the record.
#: See :func:`attach_stratify_class` for why these four and not quartiles.
SVEB_BURDEN_EDGES = (0.01, 0.03, 0.10)

#: Band names, one more than there are edges.
SVEB_BURDEN_BANDS = ("minimal", "low", "moderate", "high")


def parse_signal_header(hea_path: Path) -> dict[str, object]:
    """Read lead names, sample count and the declared gain out of one header.

    There are no comment lines to parse — see the module docstring. What *is*
    worth reading is the gain, because it is not constant: 37 of the 78 headers
    declare ``0``, WFDB's code for "uncalibrated", and the other 41 declare
    ``200``. Both end up at 200 adu/mV because WFDB substitutes exactly that as
    its default, so nothing needs scaling either way, but which records carry a
    real calibration statement is a fact about the release rather than a detail
    of the reader.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: dict[str, object] = {
        "lead_names": "",
        "n_samples": RECORD_SAMPLES,
        "declared_gain": np.nan,
        "header_declares_uncalibrated": False,
    }

    signal_lines = [ln for ln in lines[1:] if ln.strip() and not ln.startswith("#")]
    if lines:
        fields = lines[0].split()
        if len(fields) >= 4:
            out["n_samples"] = int(fields[3])

    names, gains = [], []
    for line in signal_lines:
        fields = line.split()
        names.append(fields[-1])
        if len(fields) >= 3:
            # The gain field can carry a baseline in parentheses and a unit after
            # a slash; neither occurs here, but splitting is free.
            gains.append(float(fields[2].split("(")[0].split("/")[0]))

    out["lead_names"] = "|".join(names)
    if gains:
        out["declared_gain"] = gains[0]
        out["header_declares_uncalibrated"] = gains[0] == 0.0
    return out


def _quality_seconds(
    events: list[tuple[int, int]], sig_len: int, fs: float
) -> tuple[dict[str, float], float]:
    """Turn ``~`` transitions into seconds per quality state, plus the unasserted head.

    Each event opens an interval running to the next one, or to the end of the
    record. The span before the first event is counted as clean, which is what
    WFDB does — but in four records (803, 855, 857, 885) the first event is a
    transition *into* clean, so nothing ever asserted what that span was. Those
    seconds are returned separately rather than hidden inside ``clean_secs``.
    """
    secs = {name: 0.0 for name in QUALITY_SUBTYPES.values()}
    if not events:
        secs["clean"] = sig_len / fs
        return secs, 0.0

    head_secs = events[0][0] / fs
    secs["clean"] += head_secs
    # subtype 0 means "quality returns to clean", so whatever preceded it was
    # noisy in the annotator's reading and was simply never marked as such.
    unasserted = head_secs if events[0][1] == 0 else 0.0

    for i, (start, subtype) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else sig_len
        name = QUALITY_SUBTYPES.get(subtype)
        if name is None:
            logger.warning("Unknown signal-quality subtype %r, not counted", subtype)
            continue
        secs[name] += (end - start) / fs
    return secs, unasserted


def summarise_annotations(record_path: Path, sig_len: int) -> dict[str, object]:
    """Summarise one record's reference annotations.

    Returns per-symbol beat counts, the AAMI five-class reduction, artefact and
    quality-change counts, seconds in each annotated signal-quality state, the
    annotated span against the record length, and whole-record HRV summaries.
    """
    import wfdb

    counts: dict[str, object] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
    counts.update({f"aami_{cls}": 0 for cls in AAMI_ORDER})
    counts.update({column: 0 for column in NON_BEAT_COLUMNS.values()})
    counts.update({f"{name}_secs": 0.0 for name in QUALITY_SUBTYPES.values()})
    counts.update(
        {
            "n_beats": 0,
            "n_annotations": 0,
            "n_sveb": 0,
            "sveb_fraction": np.nan,
            "sveb_per_hour": np.nan,
            "n_veb": 0,
            "veb_fraction": np.nan,
            "n_ectopic_beats": 0,
            "ectopic_fraction": np.nan,
            "annotated_secs": np.nan,
            "unannotated_head_secs": np.nan,
            "unannotated_tail_secs": np.nan,
            "annotated_fraction": np.nan,
            "noisy_secs": 0.0,
            "noisy_fraction": np.nan,
            "quality_head_unasserted_secs": 0.0,
            "mean_hr_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "n_rr_rejected": 0,
        }
    )

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .atr must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    fs = float(getattr(annotation, "fs", SAMPLING_RATE) or SAMPLING_RATE)
    beat_set = set(BEAT_SYMBOLS)
    unexpected: set[str] = set()
    quality_events: list[tuple[int, int]] = []
    beat_samples: list[int] = []

    counts["n_annotations"] = len(annotation.symbol)
    for symbol, sample, subtype in zip(annotation.symbol, annotation.sample, annotation.subtype):
        if symbol in beat_set:
            counts[f"beat_{symbol}"] = int(counts[f"beat_{symbol}"]) + 1
            counts["n_beats"] = int(counts["n_beats"]) + 1
            beat_samples.append(int(sample))
            aami = AAMI_CLASSES.get(symbol)
            if aami is None:
                # A beat symbol with no AAMI class would silently vanish from the
                # reduction while still counting in n_beats, which is exactly the
                # kind of drift the aami_* columns exist to prevent.
                logger.warning("Beat symbol %r has no AAMI class, not reduced", symbol)
            else:
                counts[f"aami_{aami}"] = int(counts[f"aami_{aami}"]) + 1
        elif symbol in NON_BEAT_COLUMNS:
            column = NON_BEAT_COLUMNS[symbol]
            counts[column] = int(counts[column]) + 1
            if symbol == "~":
                quality_events.append((int(sample), int(subtype)))
        else:
            # Worth seeing rather than silently dropping: it would mean this
            # release uses symbols this module does not know about.
            unexpected.add(symbol)

    if unexpected:
        logger.warning(
            "%s: annotation symbols outside BEAT_SYMBOLS and NON_BEAT_COLUMNS, not counted: %s",
            record_path.name,
            sorted(unexpected),
        )

    n_beats = int(counts["n_beats"])
    counts["n_sveb"] = int(counts["aami_S"])
    counts["n_veb"] = int(counts["aami_V"])
    counts["n_ectopic_beats"] = n_beats - int(counts["aami_N"])
    if n_beats > 0:
        counts["sveb_fraction"] = int(counts["n_sveb"]) / n_beats
        counts["veb_fraction"] = int(counts["n_veb"]) / n_beats
        counts["ectopic_fraction"] = int(counts["n_ectopic_beats"]) / n_beats
    if sig_len > 0:
        counts["sveb_per_hour"] = int(counts["n_sveb"]) / (sig_len / fs / 3600.0)

    quality, unasserted = _quality_seconds(quality_events, sig_len, fs)
    for name, secs in quality.items():
        counts[f"{name}_secs"] = secs
    counts["quality_head_unasserted_secs"] = unasserted
    counts["noisy_secs"] = (
        float(counts["noisy_ECG1_secs"])
        + float(counts["noisy_ECG2_secs"])
        + float(counts["noisy_both_secs"])
        + float(counts["unreadable_secs"])
    )
    if sig_len > 0:
        counts["noisy_fraction"] = float(counts["noisy_secs"]) / (sig_len / fs)

    if beat_samples:
        first, last = beat_samples[0], beat_samples[-1]
        counts["annotated_secs"] = (last - first) / fs
        counts["unannotated_head_secs"] = first / fs
        counts["unannotated_tail_secs"] = (sig_len - last) / fs
        if sig_len > 0:
            counts["annotated_fraction"] = (last - first) / sig_len

    if len(beat_samples) > 2:
        rr = np.diff(np.asarray(beat_samples, dtype=np.int64)) / fs
        low, high = RR_RANGE_SECS
        keep = (rr >= low) & (rr <= high)
        counts["n_rr_rejected"] = int((~keep).sum())
        rr = rr[keep]
        if rr.size > 1:
            counts["mean_hr_bpm"] = float(60.0 / rr.mean())
            counts["sdnn_ms"] = float(1000.0 * rr.std(ddof=1))
            counts["rmssd_ms"] = float(1000.0 * np.sqrt(np.mean(np.diff(rr) ** 2)))

    return counts


def scan_records(data_path: Path | str) -> pd.DataFrame:
    """Parse every header and annotation file into one frame, one row per record.

    The record list comes from the shipped ``RECORDS`` file rather than a glob, so
    a stray file in the directory cannot enter the partition. That matters here:
    the release ships 78 ``.hea-`` and 24 ``.atr-`` files alongside the current
    ones, which are PhysioNet's superseded revisions, not extra records.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. MIT-BIH Supraventricular Arrhythmia "
            "labels live in the .atr annotation files, so point data_path at the "
            "dataset root — the flat directory holding 800.hea, RECORDS and "
            "ANNOTATORS. Get it from https://physionet.org/content/svdb/1.0.0/"
        )

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue
        row: dict[str, object] = {"record_name": name}
        header = parse_signal_header(hea)
        row.update(header)
        row["duration_secs"] = int(header["n_samples"]) / SAMPLING_RATE
        row.update(summarise_annotations(hea.with_suffix(""), int(header["n_samples"])))
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d MIT-BIH Supraventricular records; %d beats (%d SVEB, %d VEB), "
        "%.2f%% of %.1f h annotated clean",
        len(df),
        int(df["n_beats"].sum()),
        int(df["n_sveb"].sum()),
        int(df["n_veb"].sum()),
        100 * df["clean_secs"].sum() / df["duration_secs"].sum(),
        df["duration_secs"].sum() / 3600,
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sveb_burden`` and use it as ``stratify_class``.

    This is the **only** derivation of the stratification label — ``SVDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **The burden of supraventricular ectopy is the one axis this release
    describes.** There are no demographics, no subject identifiers, no rhythm
    annotations and no diagnoses — the headers are four lines of signal
    specification and the ``.atr`` files are beat labels. What differs between
    records, and what the database was assembled to vary, is how much
    supraventricular ectopy each one holds: ``sveb_fraction`` runs from 0.000 (five
    records) to 0.575 (record 865), a spread of nearly three orders of magnitude
    among the non-zero values.

    **The band edges are fixed fractions, not quantiles.** 1%, 3% and 10% of beats
    give 21 / 20 / 23 / 14 records, so every band clears the 10 members
    ``StratifiedKFold`` needs and no fold can end up holding only zero-ectopy
    records. Quantile edges would balance the bands slightly better and would move
    every record's label the next time a record is added or an annotation revised;
    fixed edges are reproducible against a re-release, which matters more for a
    partition that gets published. The values are conventional clinical
    granularity for ectopic burden rather than anything this release states.

    It is a *fold-construction* label, and a coarsening of a continuous quantity at
    that. Train on ``aami_S``/``aami_V``/``aami_N``, the ``beat_*`` counts, or
    ``sveb_fraction`` itself.
    """
    out = df.copy()
    # -inf/inf edges so 0.0 lands in the first band and 1.0 in the last; pd.cut is
    # right-closed, so the edges read as "up to and including 1% of beats".
    out["sveb_burden"] = pd.cut(
        out["sveb_fraction"],
        bins=[-np.inf, *SVEB_BURDEN_EDGES, np.inf],
        labels=list(SVEB_BURDEN_BANDS),
    ).astype(str)
    out["stratify_class"] = out["sveb_burden"]
    logger.info(
        "SVEB burden bands: %s",
        out["sveb_burden"].value_counts().reindex(SVEB_BURDEN_BANDS).to_dict(),
    )
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return MIT-BIH Supraventricular Arrhythmia labels indexed by record name.

    Columns:

    - ``beat_N`` … ``beat_B`` — reference beat counts per symbol (see
      :data:`BEAT_NAMES`), with ``n_beats``. **Multi-class per record**: every
      record carries several beat types, so there is no single record-level beat
      label.
    - ``aami_N``, ``aami_S``, ``aami_V``, ``aami_F``, ``aami_Q`` — the same beats
      under the AAMI EC57 five-class reduction (:data:`AAMI_CLASSES`). **Use these
      to combine this database with ``mitdb``**, whose supraventricular beats are
      annotated ``A`` rather than ``S``; the raw symbols are not comparable.
    - ``n_sveb``, ``sveb_fraction``, ``sveb_per_hour``, ``n_veb``,
      ``veb_fraction``, ``n_ectopic_beats``, ``ectopic_fraction`` — ectopy burden,
      all derived from the AAMI classes.
    - ``sveb_burden`` / ``stratify_class`` — the burden band, **for fold
      construction** (see :func:`attach_stratify_class`). A coarsening of
      ``sveb_fraction``; train on the continuous column instead.
    - ``clean_secs``, ``noisy_ECG1_secs``, ``noisy_ECG2_secs``, ``noisy_both_secs``,
      ``unreadable_secs``, ``noisy_secs``, ``noisy_fraction`` — time in each
      annotated signal-quality state, from the ``~`` transitions.
      ``quality_head_unasserted_secs`` is the leading span of the four records
      whose first ``~`` marks a return *to* clean, counted as clean but never
      asserted to be.
    - ``n_isolated_artifacts``, ``n_quality_changes``, ``n_rhythm_changes`` —
      annotation markers that are **not** beats and are excluded from ``n_beats``.
      The release holds exactly one rhythm change, in record 852.
    - ``annotated_secs``, ``unannotated_head_secs``, ``unannotated_tail_secs``,
      ``annotated_fraction`` — the beat-annotated span against the record length.
      Effectively complete here, unlike ``nsrdb``.
    - ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, ``n_rr_rejected`` — whole-record
      rate and variability over RR intervals in :data:`RR_RANGE_SECS`. Computed
      over *all* beats including ectopic ones, so on a record like 865 they
      describe the recording and not a sinus rhythm.
    - ``lead_names``, ``declared_gain``, ``header_declares_uncalibrated``,
      ``n_samples``, ``duration_secs`` — from the header. 37 of the 78 records
      declare an uncalibrated gain of ``0``; WFDB substitutes 200 adu/mV for all
      of them, so the amplitudes are millivolts either way.

    There is no patient identifier column, no age and no sex: this release ships
    no header comments at all, so nothing links or describes its subjects.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
