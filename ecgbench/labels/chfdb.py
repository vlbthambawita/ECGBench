"""
BIDMC Congestive Heart Failure labels: demographics, NYHA class, beat counts, HRV.

Nothing machine-readable ships with this dataset. Each record's ``.hea`` carries a
single comment line::

    chf01 2 250 17994491 10:00:00
    chf01.dat 212 0 12 0 127 17579 0 ECG1
    chf01.dat 212 0 12 0 -128 21162 0 ECG2
    #Age: 71  Sex: M  NYHA class: III-IV

That ``#Age: <n>  Sex: <X>  NYHA class: <c>`` line is the whole of the shipped
metadata — no medications (despite the cohort being defined by a drug trial), no
subject identifier, no clinical description. Everything else here is derived from
the companion ``.ecg`` annotation file.

**1. THE BEAT ANNOTATIONS ARE UNAUDITED MACHINE OUTPUT, AND THAT IS THE FIRST
THING TO KNOW.** PhysioNet states it plainly: "Annotation files (with the suffix
.ecg) were prepared using an automated detector and **have not been corrected
manually**", and the shipped ``ANNOTATORS`` file calls them "unaudited beat
annotations from an automated detector". This is the sharpest difference from
``mitdb``, ``nsrdb``, ``svdb`` and ``edb``, whose annotations are cardiologist
reference standards. Every ``beat_*``, ``aami_*``, ``veb_*`` and HRV column in
this module is therefore a *description of what one 1980s detector reported*, not
ground truth. Do not train a beat classifier on these labels and do not quote
their ectopy rates as clinical fact; use them to describe the recording, to
stratify, or as weak supervision that a human would have to confirm.

**2. There is exactly one clinical label and it is a constant.** All 15 subjects
have severe congestive heart failure, NYHA class III–IV, so ``nyha_class`` is
``"III-IV"`` and ``cohort_label`` is ``"severe_chf"`` for every record. This
database is a positive class or a severity-matched cohort, not a classification
task in itself. Folds are stratified on **sex** (11 M / 4 F) because it is the
only axis with a class large enough to survive ten folds — see
:func:`attach_stratify_class`, which shows why the clinically interesting axis
(ventricular ectopy burden) cannot be used.

**3. Ventricular ectopy is heavy and enormously uneven — the real signal here.**
1,622,282 beats, of which 38,524 are ventricular (``V``, ``r`` or ``E``): 2.37%
against NSRDB's 0.0015%, which is what severe heart failure looks like. But the
spread across records is three orders of magnitude: **chf02 is 20.52%
ventricular** (23,510 ``V`` beats) while **chf12 is 0.017%** (19 beats). Nine of
the 15 records carry more ``r`` (R-on-T premature ventricular contraction) than
plain ``V``, so a pipeline that counts only ``V`` undercounts ventricular ectopy
in most of this database — the ``aami_V`` and ``veb_fraction`` columns are the
ones to use, and ``r`` is why ``ecgbench.labels.svdb.AAMI_CLASSES`` now carries it.
There are no fusion beats at all (``aami_F`` is 0 for every record).

**4. Rhythm annotation exists, in 4 of 15 records, and its absence is not a
negative.** 258 ``+`` annotations carry ``(AF`` and ``(N``: chf06 is **80.46%
atrial fibrillation** across 125 episodes, while chf01 (0.23%), chf10 (0.86%) and
chf14 (0.024%) hold one or two brief runs. **The other 11 records carry no ``+`` at
all**, so their ``af_secs`` of 0.0 means "never assessed", not "no AF" — read
``has_rhythm_annotation`` before treating a zero as evidence. chf06 additionally
opens with 1,757.0 s before its first ``+``, and that first marker is ``(N``,
which implies the span before it was AF without ever saying so;
``rhythm_head_unasserted_secs`` reports it rather than folding it into either
total.

**5. Beat annotation covers the whole record, unlike nsrdb.** The first beat falls
between 0.05 s and 0.60 s and the last within 0.65 s of the end, in every one of
the 15 records — ``annotated_fraction`` is at least 0.99998 everywhere — so any
window has annotation behind it. That is worth stating
because the sibling long-term databases are the opposite: NSRDB leaves one to five
hours of every record unannotated. ``annotated_fraction`` is reported anyway so a
re-release cannot change it silently.

**6. There is no signal-quality annotation layer at all.** The release contains no
``~`` and no ``|`` annotations — zero, in all 15 files — where ``nsrdb``, ``svdb``
and ``mitdb`` all carry per-channel quality bitmasks. ``n_quality_changes`` and
``n_isolated_artifacts`` are exposed as 0 so that a re-release adding them is
visible, but there are deliberately **no** ``clean_secs``/``noisy_secs`` columns:
inventing them would assert 100% clean time for 298.9 hours that nobody assessed.
Judge quality from the signal.

``mean_hr_bpm``, ``sdnn_ms`` and ``rmssd_ms`` are computed over RR intervals in
:data:`RR_RANGE_SECS`. Two caveats compound here: the beats are unaudited, and
these are whole-record summaries over ~20 h of mixed activity and sleep in
subjects with severe heart failure and heavy ectopy. A real HRV analysis would
segment, exclude ectopic couplings, and not use a machine's uncorrected beat
labels at all. Take them as a description of the record.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ecgbench.labels.svdb import AAMI_CLASSES, AAMI_ORDER

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Annotator extension, per the shipped ANNOTATORS file. It is the only one, and
#: it is ``ecg`` rather than the ``atr`` every audited MIT-BIH database uses —
#: PhysioNet's own convention for "prepared by a detector, never corrected".
ANNOTATOR = "ecg"

#: The one clinical fact this release asserts about every subject, and the value of
#: ``cohort_label`` for all 15 records: "15 subjects with severe congestive heart
#: failure (NYHA class 3-4)".
COHORT_LABEL = "severe_chf"

#: Sampling rate, uniform across all 15 records.
SAMPLING_RATE = 250

#: Beat symbols occurring in this release, descending by frequency. These six sum
#: to 1,622,282; the remaining 258 annotations are the ``+`` rhythm markers.
BEAT_SYMBOLS = ("N", "V", "r", "S", "Q", "E")

BEAT_NAMES = {
    "N": "normal beat",
    "V": "premature ventricular contraction",
    "r": "R-on-T premature ventricular contraction",
    "S": "supraventricular premature or ectopic beat",
    "Q": "unclassifiable beat",
    "E": "ventricular escape beat",
}

#: Non-beat annotation symbols, mapped to the column counting them. Only ``+``
#: occurs here — ``~`` and ``|`` are absent from all 15 files, and are listed so
#: that a re-release introducing a quality layer is counted rather than warned
#: about. Never add these to ``n_beats``.
NON_BEAT_COLUMNS = {
    "+": "n_rhythm_changes",
    "~": "n_quality_changes",
    "|": "n_isolated_artifacts",
}

#: ``aux_note`` of a ``+`` annotation, as this release spells them. Both occur.
AF_RHYTHM = "(AF"
NORMAL_RHYTHM = "(N"

#: RR intervals outside this range are dropped before any HRV summary — double
#: detections below, and the gaps spanning artefact above.
RR_RANGE_SECS = (0.3, 2.0)

#: Header comment: ``#Age: 71  Sex: M  NYHA class: III-IV``. Age is ``?`` in
#: chf06, which is why it is not simply ``\d+``.
_DEMOGRAPHICS_RE = re.compile(
    r"^#\s*Age:\s*(?P<age>\d+|\?)\s+Sex:\s*(?P<sex>[MF?])\s+NYHA\s+class:\s*(?P<nyha>\S+)\s*$"
)


def parse_header_comments(hea_path: Path) -> dict[str, object]:
    """Parse the one comment line into age, sex and NYHA class.

    A header without a parseable comment comes back with NaN/empty rather than
    raising, so one malformed file cannot fail the whole scan — genuinely broken
    records are what ``corrupt_header`` is for.
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    comments = [line for line in lines if line.startswith("#")]

    out: dict[str, object] = {"age": np.nan, "sex": "", "nyha_class": ""}
    if not comments:
        logger.warning("%s carries no comment line, so no demographics", hea_path.name)
        return out

    match = _DEMOGRAPHICS_RE.match(comments[0])
    if match:
        age = match.group("age")
        # chf06 records "?" — an unknown age, not a zero and not a sentinel.
        out["age"] = float(age) if age != "?" else np.nan
        out["sex"] = match.group("sex") if match.group("sex") != "?" else ""
        out["nyha_class"] = match.group("nyha")
    else:
        logger.warning("Unparsed demographics comment in %s: %r", hea_path.name, comments[0])
    return out


def _rhythm_seconds(
    events: list[tuple[int, str]], sig_len: int, fs: float
) -> dict[str, float | int]:
    """Turn ``+`` rhythm markers into seconds of annotated atrial fibrillation.

    Each marker opens an interval running to the next marker, or to the end of the
    record. The span *before* the first marker is not counted as anything: in 3 of
    the 4 annotated records the first marker is ``(AF``, so what preceded it was
    non-AF by implication, but in chf06 it is ``(N``, which implies the preceding
    1,757.0 s was AF and was simply never marked. That span is returned as
    ``rhythm_head_unasserted_secs`` rather than being folded into either total.

    A record with no markers at all returns zeros — see
    ``has_rhythm_annotation`` in :func:`load_labels` for why that is not the same
    as "no atrial fibrillation".
    """
    out: dict[str, float | int] = {
        "af_secs": 0.0,
        "n_af_episodes": 0,
        "rhythm_asserted_secs": 0.0,
        "rhythm_head_unasserted_secs": 0.0,
    }
    if not events:
        return out

    head_secs = events[0][0] / fs
    # "(N" first means the annotator's opening statement is a *return* to normal,
    # so the span before it was AF in its reading and was never marked as such.
    if events[0][1] == NORMAL_RHYTHM:
        out["rhythm_head_unasserted_secs"] = head_secs
    out["rhythm_asserted_secs"] = (sig_len - events[0][0]) / fs

    unexpected: set[str] = set()
    for i, (start, note) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else sig_len
        if note == AF_RHYTHM:
            out["af_secs"] = float(out["af_secs"]) + (end - start) / fs
            out["n_af_episodes"] = int(out["n_af_episodes"]) + 1
        elif note != NORMAL_RHYTHM:
            unexpected.add(note)
    if unexpected:
        logger.warning("Rhythm notes outside {(AF, (N}, not counted: %s", sorted(unexpected))
    return out


def summarise_annotations(record_path: Path, sig_len: int) -> dict[str, object]:
    """Summarise one record's ``.ecg`` annotations.

    Returns per-symbol beat counts, the AAMI five-class reduction, ventricular
    ectopy burden, annotated atrial fibrillation time, the annotated span against
    the record length, and whole-record HRV summaries. Everything here is
    unaudited detector output — see the module docstring.
    """
    import wfdb

    counts: dict[str, object] = {f"beat_{symbol}": 0 for symbol in BEAT_SYMBOLS}
    counts.update({f"aami_{cls}": 0 for cls in AAMI_ORDER})
    counts.update({column: 0 for column in NON_BEAT_COLUMNS.values()})
    counts.update(
        {
            "n_beats": 0,
            "n_annotations": 0,
            "n_veb": 0,
            "veb_fraction": np.nan,
            "veb_per_hour": np.nan,
            "n_sveb": 0,
            "sveb_fraction": np.nan,
            "n_ectopic_beats": 0,
            "ectopic_fraction": np.nan,
            "af_secs": 0.0,
            "af_fraction": np.nan,
            "n_af_episodes": 0,
            "has_rhythm_annotation": False,
            "rhythm_asserted_secs": 0.0,
            "rhythm_head_unasserted_secs": 0.0,
            "annotated_secs": np.nan,
            "unannotated_head_secs": np.nan,
            "unannotated_tail_secs": np.nan,
            "annotated_fraction": np.nan,
            "mean_hr_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "n_rr_rejected": 0,
        }
    )

    try:
        annotation = wfdb.rdann(str(record_path), ANNOTATOR)
    except Exception as e:  # a missing or unreadable .ecg must not kill the scan
        logger.warning("Could not read %s.%s: %s", record_path.name, ANNOTATOR, e)
        return counts

    fs = float(getattr(annotation, "fs", SAMPLING_RATE) or SAMPLING_RATE)
    beat_set = set(BEAT_SYMBOLS)
    unexpected: set[str] = set()
    rhythm_events: list[tuple[int, str]] = []
    beat_samples: list[int] = []

    counts["n_annotations"] = len(annotation.symbol)
    for symbol, sample, note in zip(annotation.symbol, annotation.sample, annotation.aux_note):
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
            if symbol == "+":
                rhythm_events.append((int(sample), str(note or "").strip()))
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
    counts["n_veb"] = int(counts["aami_V"])
    counts["n_sveb"] = int(counts["aami_S"])
    counts["n_ectopic_beats"] = n_beats - int(counts["aami_N"])
    if n_beats > 0:
        counts["veb_fraction"] = int(counts["n_veb"]) / n_beats
        counts["sveb_fraction"] = int(counts["n_sveb"]) / n_beats
        counts["ectopic_fraction"] = int(counts["n_ectopic_beats"]) / n_beats
    if sig_len > 0:
        counts["veb_per_hour"] = int(counts["n_veb"]) / (sig_len / fs / 3600.0)

    for key, value in _rhythm_seconds(rhythm_events, sig_len, fs).items():
        counts[key] = value
    counts["has_rhythm_annotation"] = bool(rhythm_events)
    if sig_len > 0:
        counts["af_fraction"] = float(counts["af_secs"]) / (sig_len / fs)

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
    the superseded ``.hea-`` and ``.ecg-`` backup copies PhysioNet keeps beside
    the current revisions cannot enter the partition. (Both are present in this
    release and both are listed in its ``SHA256SUMS.txt``: 15 ``.hea-`` predating
    the 2012 revision that added the ``ECG1``/``ECG2`` signal descriptions, and 2
    ``.ecg-`` for chf02 and chf04, whose annotations were regenerated in 2003.)
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_file = data_path / "RECORDS"
    if not records_file.exists():
        raise LabelSourceMissingError(
            f"No RECORDS file under {data_path}. BIDMC Congestive Heart Failure labels "
            "live in the record headers and .ecg annotation files, so point data_path "
            "at the dataset root — the flat directory holding chf01.hea, RECORDS and "
            "ANNOTATORS. Get it from https://physionet.org/content/chfdb/1.0.0/"
        )

    import wfdb

    names = [line.strip() for line in records_file.read_text().split() if line.strip()]
    rows = []
    for name in names:
        hea = data_path / f"{name}.hea"
        if not hea.exists():
            logger.warning("RECORDS names %s but %s is missing", name, hea.name)
            continue

        row: dict[str, object] = {"record_name": name}
        row.update(parse_header_comments(hea))

        header = wfdb.rdheader(str(hea.with_suffix("")))
        sig_len = int(header.sig_len)
        row["n_samples"] = sig_len
        row["duration_secs"] = sig_len / float(header.fs)
        row["lead_names"] = "|".join(header.sig_name or [])
        # Time of day the Holter tape started. No date ships, here or anywhere
        # else in the release.
        row["start_time"] = str(header.base_time) if header.base_time else ""

        row.update(summarise_annotations(hea.with_suffix(""), sig_len))
        # The whole database is one class, and it is asserted by the release
        # rather than derived from anything in the files.
        row["cohort_label"] = COHORT_LABEL
        # Flat tree: wfdb takes the stem, with no extension and no subdirectory.
        row["signal_path"] = name
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info(
        "Parsed %d records: %.1f h of signal, %d unaudited beats (%d ventricular, "
        "%.2f%%), %.1f h annotated AF in %d records",
        len(df),
        df["duration_secs"].sum() / 3600,
        int(df["n_beats"].sum()),
        int(df["n_veb"].sum()),
        100 * df["n_veb"].sum() / max(int(df["n_beats"].sum()), 1),
        df["af_secs"].sum() / 3600,
        int(df["has_rhythm_annotation"].sum()),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``, the subject's sex, and explain why it is that.

    This is the **only** derivation of the stratification label — ``CHFDBSplitter``
    reads the column rather than recomputing it, so the exposed label and the fold
    label cannot drift.

    **There is no clinical label to stratify on.** Every subject here has severe
    congestive heart failure, NYHA class III–IV, so ``cohort_label`` and
    ``nyha_class`` are one value across all 15 records and carry no information a
    fold could be balanced on. What is left is the cohort itself, which PhysioNet
    describes in exactly these terms: "11 men, aged 22 to 71, and 4 women, aged 54
    to 63". Sex is the one documented axis on which these 15 subjects differ.

    **The clinically interesting axis cannot be used, and it is worth saying why.**
    Ventricular ectopy burden is the real per-record signal in this database —
    ``veb_fraction`` runs from 0.017% (chf12) to 20.52% (chf02) — and it is what
    ``svdb`` stratifies on, in bands. Here the arithmetic forbids it.
    ``StratifiedKFold`` raises when *every* class holds fewer members than there
    are folds, so with 15 records over 10 folds a usable split needs one class of
    10 or more. That leaves only a 10/5 or more lopsided cut:

    - four equal-count bands (quartiles of ``veb_fraction``) give 4/4/3/4 —
      **raises**;
    - ``svdb``'s own burden edges (1%, 3%, 10%) give 8/6/0/1 — **raises**, and two
      of its four bands are empty here besides;
    - a natural cut at 1% gives 7/8 — **raises**;
    - the one cut that survives is 2%, giving 5/10, which is a threshold chosen to
      fit these 15 numbers rather than anything clinical.

    Sex gives 11/4 and clears the requirement with margin. It also does not rest on
    the unaudited detector output that every ectopy figure here comes from. Use
    ``veb_fraction``, ``af_fraction``, ``mean_hr_bpm``, ``sdnn_ms`` or the
    ``aami_*`` counts as targets; never ``stratify_class``.
    """
    out = df.copy()
    # Missing sex would be its own class and would break the fold balance
    # silently; there is none in this release, and "U" makes it visible if a
    # re-release introduces one.
    out["stratify_class"] = out["sex"].replace("", "U")
    logger.info("Stratification classes (sex): %s", out["stratify_class"].value_counts().to_dict())
    return out


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return BIDMC Congestive Heart Failure labels indexed by record name.

    **Every annotation-derived column below is unaudited machine output.** The
    ``.ecg`` files were produced by an automated detector and never manually
    corrected — see the module docstring. Only ``age``, ``sex``, ``nyha_class``,
    ``cohort_label`` and the record geometry are independent of it.

    Columns:

    - ``cohort_label`` — ``"severe_chf"`` for **all 15 records**, and
      ``nyha_class`` — ``"III-IV"`` for all 15. The release asserts both of the
      cohort. This database is a positive class or a severity-matched comparison
      group, not a classification task.
    - ``age``, ``sex`` — the rest of the shipped metadata, from the one header
      comment line. 11 men aged 22–71 and 4 women aged 54–63, which reproduces
      PhysioNet's description exactly. chf06's age is ``?`` in the header and NaN
      here.
    - ``beat_N``, ``beat_V``, ``beat_r``, ``beat_S``, ``beat_Q``, ``beat_E`` —
      detector beat counts per symbol (see :data:`BEAT_NAMES`), with ``n_beats``,
      ``n_ectopic_beats`` and ``ectopic_fraction``.
    - ``aami_N``/``aami_S``/``aami_V``/``aami_F``/``aami_Q`` — the AAMI EC57
      five-class reduction, comparable with every other MIT-BIH-family database in
      this catalogue. **Prefer these to the raw symbols:** ``r`` (R-on-T PVC) is an
      AAMI ``V`` and outnumbers plain ``V`` in nine of the 15 records, so counting
      only ``beat_V`` undercounts ventricular ectopy across most of the release.
    - ``n_veb``, ``veb_fraction``, ``veb_per_hour``, ``n_sveb``, ``sveb_fraction``
      — ectopy burden. ``veb_fraction`` spans 0.00017 (chf12) to 0.2052 (chf02) and
      is the most informative per-record quantity here; ``sveb_fraction`` is far
      narrower, 0.000026 to 0.026.
    - ``af_secs``, ``af_fraction``, ``n_af_episodes``, ``has_rhythm_annotation``,
      ``rhythm_asserted_secs``, ``rhythm_head_unasserted_secs`` — annotated atrial
      fibrillation. **Check ``has_rhythm_annotation`` first:** only 4 records carry
      any ``+`` marker, so for the other 11 an ``af_secs`` of 0.0 means the rhythm
      was never assessed, not that there was no AF. chf06 is 80.46% AF.
    - ``annotated_secs``, ``unannotated_head_secs``, ``unannotated_tail_secs``,
      ``annotated_fraction`` — beat annotation covers essentially the whole record
      in all 15 (head ≤ 0.60 s, tail ≤ 0.65 s), unlike ``nsrdb``. Reported so a
      re-release cannot change that quietly.
    - ``n_rhythm_changes``, ``n_quality_changes``, ``n_isolated_artifacts`` —
      non-beat annotation markers, excluded from ``n_beats``. The last two are
      **0 for every record**: this release ships no signal-quality layer, which is
      why there are no ``clean_secs``/``noisy_secs`` columns to go with them.
    - ``mean_hr_bpm``, ``sdnn_ms``, ``rmssd_ms``, ``n_rr_rejected`` — whole-record
      HRV summaries over RR intervals in :data:`RR_RANGE_SECS`. Descriptive only:
      unaudited beats, ~20 h spans, and heavy ectopy all argue against reading them
      as an HRV result.
    - ``n_samples``, ``duration_secs``, ``start_time``, ``lead_names`` — record
      geometry. Duration is near-uniform at 19.77–20.00 h.
    - ``stratify_class`` — the subject's sex, **for fold construction only**. See
      :func:`attach_stratify_class`.

    There is no patient identifier column: PhysioNet describes 15 recordings from
    15 subjects, and the headers carry nothing that would group them further.
    """
    df = scan_records(data_path)
    df = attach_stratify_class(df)
    df = df.set_index("record_name")
    df.index.name = config.record_id_column
    return df
