"""
tOLIet labels: demographics, electrode texture, and whether the channel recorded anything.

**One ECGBench record is one electrode channel, not one sitting.** The release
ships 145 OpenSignals files, each holding four differential channels taken
simultaneously from four electrode pairs moulded into the same toilet seat and
differing only in surface texture — ``A1`` flat, ``A2`` sinusoidal, ``A3``
pyramidal, ``A4`` trapezoidal. ECGBench exposes each of those channels as its own
single-lead record, ``<source>_<channel>``: ``15_1_A2`` is the sinusoidal channel
of subject 15's second sitting. 145 files x 4 textures = **580 records from 86
subjects**. :func:`load_labels` carries ``source_record`` and
``electrode_texture`` so the grouping is never lost.

Why not one record per file with four leads, which is what the files look like?
Because three quarters of the channels are not ECG. Sit a person on the seat and
some of the four electrode pairs make contact and some do not; a pair that did
not reads a constant ADC code, or oscillates between the converter's two rails,
for the whole sitting. Keeping the file as a 4-lead record makes every one of
those a flat lead inside an otherwise good record, and ECGBench's ``flat_line``
check then rejects the whole record: **only 5 of the 145 files have all four
textures live**, so the ``clean`` version — the one ``ECGDataset`` loads by
default — would hold 5 records and two empty folds. Split per channel instead and
``clean`` is the 342 channels that actually recorded, which is the dataset a user
wants.

**Six things worth knowing, all verified against the files.**

**1. 342 of the 580 channels carry a signal; the other 238 are dead.** That is
not a judgement call made here — :func:`scan_signal` runs ECGBench's own
``check_flat_line`` on each channel and records the verdict as
``signal_active``, so the label column and the validation report cannot
disagree, and ``clean`` is exactly the ``signal_active`` records. Per texture:
``A1`` flat is live in 140 of its 145 recordings, ``A2`` sinusoidal in 127,
``A4`` trapezoidal in 68, and **``A3`` pyramidal in 7**. The pyramidal electrode
essentially never worked, which is a result about electrode geometry rather than
a defect in the release, and it is why ``A3`` cannot be a stratification class.

**2. A dead channel is not a zero channel, and its variance is not always
small.** An unconnected pair reads a constant ADC code of 0 — ``+1.5 mV`` after
conversion, so ``check_missing_leads`` (which looks for all-NaN or all-zero)
never sees it and ``check_flat_line`` does. Others sit railed at the bottom code
with a few counts of dither. A third kind oscillates *between* both rails, which
gives it a large variance and lets it pass ``flat_line`` while carrying no ECG:
``58_1_A4`` has variance 0.028 mV^2 and is at a rail for 99.7% of its samples.
``clipped_fraction`` is what identifies those — see point 3 — and no check in
``CHECK_REGISTRY`` measures it, so **``signal_active`` is a floor, not a
guarantee**. Filter on ``clipped_fraction`` as well for anything that depends on
morphology.

**3. Clipping is pervasive and is the main quality axis here.** The front end is
+/-1.5 mV full scale into a 10-bit converter, so every sample is inside the
config's ``amplitude_range_mv`` by construction and ``amplitude_outlier`` is a
no-op for this dataset. What actually goes wrong is saturation *at* the rail:
66 of the 145 sittings drive at least one *live* channel into one, and 12 live
channels spend more than half their samples there. ``clipped_fraction``,
``min_mv`` and ``max_mv`` are per record. The highest ADC code occurring anywhere
in the release is 1022, not 1023.

**4. Records are 14.4 s to 197.2 s, not "up to 5 minutes".** The landing page
says up to five minutes per session; the longest file is 197,250 samples at
1 kHz, i.e. 3 min 17 s. Median 126.3 s. Length varies, so ``expected_samples`` is
deliberately unset in the config and any ``window=`` has to fit the
14,400-sample shortest record.

**5. The release says 149 sittings and ships 145.** ``DataSet.csv`` lists 149
IDs; ``ECG_EXP/`` holds 145 ``.txt`` files. ``12_1``, ``13_1``, ``14_1`` and
``41_1`` are tabulated and absent. :func:`scan_records` drops them with a warning
rather than emitting records that would fail ``corrupt_header``. The subject
count is unaffected — each of the four is a *later* sitting of a subject whose
first is present, so 86 subjects either way.

**6. Only 23 sittings have the clinical reference the abstract mentions.**
``ECG_REF/`` holds 23 Sapphire-format ``.XML`` files in microvolts, one per
sitting, covering subjects 58, 59, 60, 67 and 68-86, and every one names a
sitting that exists. ``has_reference_ecg`` and ``reference_path`` point at them.
ECGBench does **not** load them: a 10-second clinical 12-lead ECG is a different
modality from a two-minute thigh recording, and folding them into the same record
set would put both in the same folds. Parse them with the release's own
``Script/read_ref_data.py``.

**The four channels of one sitting are the same beats.** They are simultaneous
measurements of one thigh-to-thigh derivation, so 580 records are not 580
independent observations — they are 145, and 86 at the subject level. Folds group
by subject, so nothing leaks across a fold boundary, but a model evaluated as
though it saw 580 samples is overstating its evidence. ``source_record`` is there
to let you collapse them.

Signals are read here, because points 1 and 3 cannot be answered without them:
:func:`scan_records` decodes all four channels of all 145 files — 17.9 million
samples, about 20 seconds. The splitter caches the result as
``ecgbench_metadata.csv``.
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

#: Subject demographics, one row per sitting. Semicolon-separated, UTF-8 with a
#: BOM, and padded out to 22 columns with 16 unnamed empty ones.
SOURCE_METADATA = "DataSet.csv"

#: Directory holding the sittings, one OpenSignals ``.txt`` per source record.
SIGNAL_DIR = "ECG_EXP"

#: Directory holding the 23 clinical 12-lead reference ECGs, ``<ID>.XML``.
REFERENCE_DIR = "ECG_REF"

#: The four electrode channels, in the order the files store them.
CHANNELS = ("A1", "A2", "A3", "A4")

#: Channel -> electrode surface texture, from the README's "File Descriptions".
TEXTURES = {"A1": "flat", "A2": "sinusoidal", "A3": "pyramidal", "A4": "trapezoidal"}

#: Millivolts at each end of the 10-bit converter's +/-1.5 mV span: code 0 is
#: exactly +1.5 and code 1023 exactly -1.4970703125. A sample at either end is
#: saturated, which is what ``clipped_fraction`` counts.
RAIL_MV = (-1.4970703125, 1.5)


def _read_source_metadata(data_path: Path) -> pd.DataFrame:
    """Read DataSet.csv into ID, age, weight, height, sex and observations.

    Three things about the file need undoing: it is semicolon-separated, it
    starts with a UTF-8 BOM (so the first column reads as ``\\ufeffID`` unless the
    encoding says otherwise), and every row is padded to 22 fields, leaving 16
    unnamed all-empty columns behind.
    """
    csv_path = data_path / SOURCE_METADATA
    if not csv_path.exists():
        from ecgbench.labels import LabelSourceMissingError

        raise LabelSourceMissingError(
            f"tOLIet demographics come from {SOURCE_METADATA}, which is not in "
            f"{data_path}. ECGBench publishes fold CSVs only — labels stay with "
            "the source dataset, so point data_path at a full local copy "
            "(https://physionet.org/content/tollet/1.0.1/)."
        )

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [column.strip() for column in df.columns]

    required = ("ID", "Age", "Weight", "Height", "Gender")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing column(s) {missing}. Found: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "source_record": df["ID"].astype(str).str.strip(),
            "age": pd.to_numeric(df["Age"], errors="coerce"),
            "weight_kg": pd.to_numeric(df["Weight"], errors="coerce"),
            "height_cm": pd.to_numeric(df["Height"], errors="coerce"),
            "sex": df["Gender"].astype(str).str.strip().str.lower(),
            "observations": df.get("Observations field", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .str.strip(),
        }
    )
    out["bmi"] = (out["weight_kg"] / (out["height_cm"] / 100.0) ** 2).round(2)
    return out


def scan_signal(txt_path: Path, config: DatasetConfig) -> list[dict[str, object]]:
    """Summarise one sitting: one row per electrode channel.

    Reads all four channels in full — there is no way to tell a disconnected
    electrode from a quiet one without seeing every sample.

    ``signal_active`` is the negation of ECGBench's own ``check_flat_line`` run on
    that channel, not a threshold invented here, so the label and the validation
    report are the same verdict by construction. It is a floor and not a
    guarantee: a channel oscillating between both converter rails has a large
    variance and passes. See point 2 of the module docstring, and
    ``clipped_fraction``.
    """
    from ecgbench.dataset import _load_signal
    from ecgbench.validation.checks import check_flat_line

    source = txt_path.stem
    reference = f"{SIGNAL_DIR}/{source}.txt"
    signal = _load_signal(
        f"{txt_path}:{','.join(CHANNELS)}", "opensignals", config.signal_unit_scale
    )

    rows: list[dict[str, object]] = []
    for position, channel in enumerate(CHANNELS):
        lead = signal[position]
        n_samples = int(lead.shape[0])
        clipped = int(np.count_nonzero((lead <= RAIL_MV[0]) | (lead >= RAIL_MV[1])))
        rows.append(
            {
                "record_id": f"{source}_{channel}",
                "source_record": source,
                "channel": channel,
                "electrode_texture": TEXTURES[channel],
                "signal_path": f"{reference}:{channel}",
                "n_samples": n_samples,
                "duration_secs": round(n_samples / config.default_sampling_rate, 3),
                "sampling_rate": config.default_sampling_rate,
                # check_flat_line takes (leads, samples); hand it this one lead.
                "signal_active": not check_flat_line(lead[np.newaxis, :], config),
                "variance_mv2": float(np.nanvar(lead)),
                "clipped_fraction": round(clipped / n_samples, 6) if n_samples else 0.0,
                "min_mv": round(float(lead.min()), 4),
                "max_mv": round(float(lead.max()), 4),
            }
        )
    return rows


def scan_records(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Build the per-channel frame: each sitting's metadata row, exploded x4.

    Rows of ``DataSet.csv`` with no ``.txt`` on disk are dropped with a warning —
    see point 5 in the module docstring. The four channels of every remaining file
    are decoded in full, which takes about twenty seconds for the release.
    """
    data_path = Path(data_path)
    signal_dir = data_path / SIGNAL_DIR
    if not signal_dir.is_dir():
        raise FileNotFoundError(
            f"Expected the signal directory {signal_dir}. Point data_path at the "
            f"version directory holding {SIGNAL_DIR}/, {REFERENCE_DIR}/ and "
            f"{SOURCE_METADATA} — not at {SIGNAL_DIR}/ itself."
        )

    sittings = _read_source_metadata(data_path)

    has_signal = sittings["source_record"].map(
        lambda name: (signal_dir / f"{name}.txt").exists()
    )
    if not has_signal.all():
        absent = sittings.loc[~has_signal, "source_record"].tolist()
        logger.warning(
            "%d of %d %s rows have no %s/<ID>.txt and are dropped: %s. This is the "
            "whole of the difference between the release's stated 149 recordings "
            "and the 145 it ships.",
            len(absent), len(sittings), SOURCE_METADATA, SIGNAL_DIR, ", ".join(absent),
        )
        sittings = sittings[has_signal].reset_index(drop=True)

    channels = pd.DataFrame(
        [
            row
            for name in sittings["source_record"]
            for row in scan_signal(signal_dir / f"{name}.txt", config)
        ]
    )
    df = channels.merge(sittings, on="source_record", how="left", validate="many_to_one")

    # "15_1" is subject 15's second sitting; a bare "15" is the first. The suffix
    # is the only session marker in the release — no timestamps ship.
    parts = df["source_record"].str.split("_", n=1)
    df["subject_id"] = parts.str[0]
    df["session_index"] = parts.str[1].fillna("0").astype(int)
    df["n_sittings_for_subject"] = df.groupby("subject_id")["source_record"].transform(
        "nunique"
    )

    reference = df["source_record"].map(lambda name: f"{REFERENCE_DIR}/{name}.XML")
    df["has_reference_ecg"] = reference.map(lambda rel: (data_path / rel).exists())
    df["reference_path"] = reference.where(df["has_reference_ecg"], "")

    df = df.sort_values(
        ["subject_id", "session_index", "channel"],
        key=lambda column: (
            column.astype(int) if column.name == "subject_id" else column
        ),
    ).reset_index(drop=True)

    logger.info(
        "tOLIet: %d channel records from %d sittings by %d subjects; %d channels "
        "carry a signal, %d are flat; %d sittings have a 12-lead reference",
        len(df), df["source_record"].nunique(), df["subject_id"].nunique(),
        int(df["signal_active"].sum()), int((~df["signal_active"]).sum()),
        df.loc[df["has_reference_ecg"], "source_record"].nunique(),
    )
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: sex crossed with whether the channel recorded.

    **Why this cross.** Two quantities have to stay balanced across folds. Sex is
    the demographic one — 50 female and 36 male subjects, both far above the ten a
    class needs to reach every fold. ``signal_active`` is the other, and it is the
    one that decides how big a fold is in the version most people load: ``clean``
    is the 342 live channels of the 580, so stratifying on sex alone lets the
    clean folds run 28 to 39 records while stratifying on liveness alone lets the
    female fraction swing from 0.33 to 0.93. Measured over the shipped files at
    ``random_state=42``:

    ==========================  ==================  ================
    stratification              ``clean`` per fold  female fraction
    ==========================  ==================  ================
    sex only                    28 – 39             0.57 – 0.60
    ``signal_active`` only      31 – 37             0.33 – 0.93
    **sex x signal_active**     **32 – 37**         **0.57 – 0.60**
    ==========================  ==================  ================

    **Electrode texture is deliberately not in the cross, and does not need to
    be.** Every sitting contributes all four textures, so any partition of
    sittings splits them evenly by construction — the measured spread is 13 to 15
    live ``A1`` channels per fold under the cross above. Putting texture in
    explicitly would also fail: ``A3_active`` has 7 records from 7 subjects, and a
    class needs ten subjects to appear in ten folds.

    The four cells hold ``F_active`` 192 records from 50 subjects, ``M_active``
    150/36, ``F_flat`` 148/47 and ``M_flat`` 90/34. They overlap, since a subject with
    one live and one dead channel is in two of them. That is fine under
    ``StratifiedGroupKFold``, which places each group once and balances over
    records.
    """
    liveness = np.where(df["signal_active"], "active", "flat")
    df = df.copy()
    df["stratify_class"] = df["sex"].str[0].str.upper() + "_" + liveness
    return df


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return tOLIet labels indexed by ``record_id``.

    **No diagnosis ships for all but 16 of the 580 records.** ``observations`` is
    the release's free-text clinical note, populated for 4 of the 145 sittings —
    all paroxysmal atrial fibrillation — and blank for the rest, so this is a
    demographic and signal-quality table, not a classification target. The
    dataset's own stated purposes are electrode-texture comparison and biometric
    identification. For the latter the label is ``subject_id``, which is also the
    config's ``patient_id_column``, so ECGBench's subject-grouped folds put a
    person wholly inside one fold and cannot be used to train an identifier —
    split within subject on ``session_index`` for that.

    Columns:

    - ``source_record``, ``channel``, ``electrode_texture`` — **which sitting and
      which electrode this record is.** ``record_id`` is ``<source>_<channel>``.
      The four channels of a sitting are simultaneous measurements of the same
      beats, so group on ``source_record`` before counting independent samples.
    - ``subject_id``, ``session_index``, ``n_sittings_for_subject`` — 86 subjects
      over 145 sittings, 1 to 3 each (33 subjects have 1, 47 have 2, 6 have 3).
      ``session_index`` is the ``_N`` suffix of the source record, 0 for the
      first; no timestamps ship.
    - ``age``, ``sex``, ``weight_kg``, ``height_cm``, ``bmi`` — from
      ``DataSet.csv``, constant within a subject in every case. 50 female and 36
      male subjects; age 18 to 83 (subject mean 31.73 ± 13.11), weight 49 to 95 kg
      (66.89 ± 10.70), height 150 to 185 cm (166.83 ± 6.07), BMI 16.6 to 34.1
      (24.08 ± 3.91). Those three published means are per *subject*, not per
      recording — the per-sitting age mean is 29.99 ± 10.59.
    - ``observations`` — the free-text clinical note, non-empty for 4 sittings (16
      records): three variants of paroxysmal AF, one of them a month after
      ablation and one with left bundle branch block and cardiomyopathy under
      study, plus one plain "Paroxysmal AF".
    - ``signal_active`` — **whether this electrode recorded anything**, taken from
      ECGBench's own ``check_flat_line``. True for 342 of 580; per texture 140
      (A1, flat), 127 (A2, sinusoidal), 7 (A3, pyramidal), 68 (A4, trapezoidal).
      This is exactly what decides membership of the ``clean`` version.
    - ``clipped_fraction``, ``min_mv``, ``max_mv``, ``variance_mv2`` — saturation
      and range. **Check ``clipped_fraction`` as well as ``signal_active``**: a
      channel oscillating between both converter rails passes ``flat_line`` and is
      not ECG. 12 live channels are at a rail for more than half their samples.
    - ``n_samples``, ``duration_secs``, ``sampling_rate`` — 14.4 s to 197.2 s at
      1000 Hz, median 126.3 s.
    - ``has_reference_ecg``, ``reference_path`` — the 23 sittings (92 records) with
      a simultaneous clinical 12-lead ECG, which ECGBench does not load.
    - ``signal_path`` — ``ECG_EXP/<source>.txt:<channel>``, the OpenSignals
      reference the loader resolves to one column of one file.
    - ``stratify_class`` — for fold construction only. See
      :func:`attach_stratify_class`.
    """
    df = scan_records(data_path, config)
    df = attach_stratify_class(df)
    df = df.set_index("record_id")
    df.index.name = config.record_id_column
    return df
