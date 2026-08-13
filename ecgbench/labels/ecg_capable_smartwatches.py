"""ECG-capable smartwatches labels: the simulator setting each record was measured at.

**There is no patient here, and that is the point.** Every one of the 915 records
is a METRON PS-440 patient simulator output, recorded simultaneously by a
reference 12-lead electrocardiograph and by four consumer smartwatches, under the
IEC 60601-2-25:2011 protocol. The release's own Ethics section is one line: "Data
collected from synthetic sources". So the ground truth is not a diagnosis — it is
the **knob the simulator was set to**, and this module's job is to recover it,
because the release records it nowhere but in the directory names.

Three columns carry it, one per experiment family, and each is NaN outside its own
family: ``nominal_rate_bpm`` (30-300 bpm, 15 settings), ``nominal_r_amplitude_uv``
(500-2000 µV, 4 settings) and ``nominal_st_offset_uv`` (-800 to +800 µV in 100 µV
steps, 16 settings). The 36th setting is a 2 Hz square wave with no ECG parameter
at all. What the dataset is *for* is measuring how far each device's reading
departs from those numbers, so they are handed over as the labels and nothing here
estimates a measured heart rate or ST level on the user's behalf.

**Nine things worth knowing, all verified against the shipped files** (all 1,833
of which match the release's own ``SHA256SUMS.txt``).

**1. THE FOUR SMARTWATCH RECORDS ARE LEAD I, AND THEIR HEADERS SAY ``II``.** The
Methods are explicit: the simulator's R output (right arm) went to the watch crown
and its L output (left arm) to the caseback, and the watches' own PDF exports are
"formatted as single-lead (Lead I) electrocardiograms". LA minus RA is lead I. All
720 smartwatch headers nonetheless name the channel ``II``, and ECGBench's
``lead_names`` follows the files because that is what ``leads=`` resolves against.
The consequence is a trap with no error message: ``ECGDataset(leads=["II"])``
returns the Philips' genuine lead II (LL minus RA) for 195 records and an
arm-to-arm lead I for the other 720. ``derivation`` is here so that is filterable;
filter on it, or on ``device``, before comparing morphology across devices.

**2. THE PHILIPS REFERENCE IS 12-LEAD; ONLY THE WATCHES ARE SINGLE-LEAD.** 195
records store 12 channels at 500 Hz, the other 720 store one. ``n_leads`` and
``lead_names_stored`` say which, and the config declares the second layout in
``alternate_lead_names`` so a lead selected by name is re-resolved per record.

**3. EVERY SAMSUNG RECORD ENDS IN AN INVALID SAMPLE, AND THAT COSTS THE WHOLE
DEVICE ITS PLACE IN ``clean``.** All 179 Samsung Galaxy Watch 6 records are 15,001
samples long where 15,000 is 30.000 s at 500 Hz, and the extra final sample is
digital ``-32768`` — WFDB's invalid-sample marker for format 16 — which ``wfdb``
returns as NaN. ``check_nan_values`` has no threshold, so all 179 fail it and the
``clean`` version holds 736 records **and not one Samsung record**. The signal
before that sample is fine: ``window=(0, 15000)`` reads it with no NaN at all.
``trailing_invalid_sample`` and ``nan_samples`` mark it per record, so the
exclusion is explainable from the labels rather than only from the validation
report. It is exactly the ``sddb`` situation one format up.

**4. "ALL EXPERIMENTS IN QUINTUPLICATE" IS TRUE OF 895 RECORDS AND FALSE OF 20.**
36 settings x 5 devices x 5 repetitions is 900, and the release ships 915.
Seventeen settings carry a **sixth** repetition — every Philips ``freq_test``
setting, plus Fitbit's ``f80`` and ``ST-m6`` — and two carry only **four**:
Samsung's ``st-p8`` and Fitbit's ``ST-p8``. ``replicate`` and
``is_extra_replicate`` expose it. No two records in the release hold an identical
signal, so the sixth repetitions are genuine extra acquisitions and are kept; the
Philips ``_5`` records correlate with their own siblings at 0.49 to 0.996.

**5. FITBIT'S ST DIRECTORIES AND RECORD NAMES ARE UPPERCASE, AND EVERY FITBIT
HEADER NAMES THE WRONG DEVICE.** Fitbit stores ``st-segment/ST-m1/ST-m1_0`` where
the other four devices store ``st-segment/st-m1/st-m1_0``, so a setting key taken
verbatim splits the ST ladder into 32 settings instead of 16 — which would put the
same simulator condition in two different folds. ``setting_id`` is lowercased for
exactly that reason. Separately, all 181 Fitbit headers carry the comment
"Withings Scanwatch reading METRON PS-440 patient simulator"; they are not Withings
records (250 Hz against Withings' 300 Hz, and no two records share a signal), the
comment is a copy-paste error, and ``device_model`` is taken from the directory
rather than from it.

**6. ``RECORDS`` NAMES 75 FILES THAT DO NOT EXIST.** The shipped ``RECORDS`` index
lists Withings' ``freq_test`` entries under ``WithingsScanwatch/`` while the
directory on disk is ``withingsscanwatch/``. On a case-sensitive filesystem 75 of
its 915 lines resolve to nothing, so building paths from ``RECORDS`` yields 75
records that all fail ``corrupt_header``. :func:`scan_records` enumerates the
headers from disk instead and checks ``RECORDS`` for agreement, reporting the
case-only mismatches as such rather than as missing data.

**7. THE APPLE DIRECTORY AND HEADERS SAY SERIES 8; THE RELEASE SAYS SERIES 9.**
The abstract, the Data Description and the paper all name the Apple Watch Series
9, and the Data Description maps ``applewatch_serie8`` to it explicitly. The
directory name and all 180 header comments say "Serie 8". ``device`` keeps the
directory name so paths stay traceable; ``device_model`` carries the release's own
prose answer.

**8. EVERY RECORD WAS RESCALED TO FILL INT16, SO THE HEADER GAIN IS PER RECORD AND
MEANS NOTHING ACROSS RECORDS.** 914 of the 915 reach digital ``+32767`` and the
same number reach ``-32767`` or ``-32768``; gains run from 15,082 to 207,386
adu/mV and differ record by record. Physical millivolts are therefore recovered
correctly by ``wfdb`` — do not apply a unit scale — but the 16-bit resolution is
a per-record quantity and the rail is not a clipping artefact: it is where the
record's own extremes were mapped. The one exception is
``withingsscanwatch/st-segment/st-p1/st-p1_1``, whose maximum code is 32,766.

**9. LENGTHS ARE NOT UNIFORM AND NEITHER IS THE RATE.** Philips is 5,500 samples
at 500 Hz (11.0 s, not the 10 s the Methods state), Apple 15,360 at 512 Hz (30.0 s)
except ``amp1000_0`` at 13,968 (27.3 s), Samsung 15,001 at 500 Hz, Fitbit 7,500 at
250 Hz, Withings 9,000 at 300 Hz except six ``freq_test`` records at 8,999. So
``expected_samples`` is unset in the config, and the largest window fitting every
record is ``window=(0, 5500)`` — which is 11.0 s of Philips and 22.0 s of Fitbit.
A window in samples is not a window in time here.

Signals are read here, because points 3, 8 and 9 cannot be answered from the
directory tree: :func:`scan_records` decodes all 915 records once, about 15
seconds. The splitter caches the result as ``ecgbench_metadata.csv``.
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

#: Device directory -> the model the release's own prose names, and the derivation
#: it recorded. The directory names are not trustworthy as model names: point 7
#: of the module docstring, ``applewatch_serie8`` is a Series 9.
DEVICES: dict[str, dict[str, str]] = {
    "philips_tc30": {
        "model": "Philips TC30",
        "role": "reference",
        "derivation": "standard 12-lead",
    },
    "applewatch_serie8": {
        "model": "Apple Watch Series 9",
        "role": "smartwatch",
        "derivation": "lead I (LA-RA)",
    },
    "samsunggalaxy6": {
        "model": "Samsung Galaxy Watch 6",
        "role": "smartwatch",
        "derivation": "lead I (LA-RA)",
    },
    "fitbitsense2": {
        "model": "Fitbit Sense 2",
        "role": "smartwatch",
        "derivation": "lead I (LA-RA)",
    },
    "withingsscanwatch": {
        "model": "Withings ScanWatch",
        "role": "smartwatch",
        "derivation": "lead I (LA-RA)",
    },
}

#: The release's own index of records. Read only to be checked against the tree —
#: 75 of its lines name a directory that does not exist. See point 6.
RECORDS_INDEX = "RECORDS"

#: Experiment family directories. ``sqr-2hz`` holds its records directly, the
#: other three hold one directory per setting.
FAMILIES = ("amp_test", "freq_test", "st-segment", "sqr-2hz")

_AMP_RE = re.compile(r"^amp(\d+)$")
_FREQ_RE = re.compile(r"^f(\d+)$")
_ST_RE = re.compile(r"^st-([mp])(\d+)$")


def parse_setting(setting_id: str) -> dict[str, object]:
    """Decode a lowercased setting key into the simulator knob it names.

    ``setting_id`` is the directory the record sits in, lowercased —
    ``amp1500``, ``f220``, ``st-m3``, ``sqr-2hz``. Lowercasing is load-bearing:
    Fitbit spells its sixteen ST directories ``ST-m1``..``ST-p8`` and the other
    four devices spell them ``st-m1``..``st-p8``, so a verbatim key would describe
    one simulator condition as two (point 5).

    Returns the family plus the one nominal parameter that family varies; the
    other two come back as ``None``, because there is no meaningful ST offset for
    a heart-rate sweep and pretending otherwise would put a number in a column a
    user might well train on.
    """
    out: dict[str, object] = {
        "family": None,
        "nominal_rate_bpm": None,
        "nominal_r_amplitude_uv": None,
        "nominal_st_offset_uv": None,
    }
    if setting_id == "sqr-2hz":
        out["family"] = "sqr-2hz"
        return out
    if m := _AMP_RE.match(setting_id):
        out["family"] = "amp_test"
        out["nominal_r_amplitude_uv"] = int(m.group(1))
        return out
    if m := _FREQ_RE.match(setting_id):
        out["family"] = "freq_test"
        out["nominal_rate_bpm"] = int(m.group(1))
        return out
    if m := _ST_RE.match(setting_id):
        sign = -1 if m.group(1) == "m" else 1
        out["family"] = "st-segment"
        # st-m8 is -800 µV and st-p8 is +800 µV, so the index is hundreds of
        # microvolts. There is no zero-offset setting: the ladder runs -800..-100,
        # +100..+800, which is 16 settings and no baseline control.
        out["nominal_st_offset_uv"] = sign * 100 * int(m.group(2))
        return out
    raise ValueError(
        f"Unrecognised simulator setting {setting_id!r}. Expected one of "
        "amp<uV>, f<bpm>, st-m<n>, st-p<n> or sqr-2hz."
    )


def _check_records_index(data_path: Path, found: set[str]) -> None:
    """Compare the shipped ``RECORDS`` index against the headers on disk.

    Point 6 of the module docstring: 75 of its 915 lines name
    ``WithingsScanwatch/`` where the directory is ``withingsscanwatch/``, so they
    resolve to nothing on a case-sensitive filesystem. Separating the case-only
    mismatches from genuine absences is the whole reason this is a function and
    not an assertion — "75 records missing" would send someone re-downloading a
    complete copy.
    """
    index_path = data_path / RECORDS_INDEX
    if not index_path.exists():
        logger.warning("No %s in %s; skipping the index cross-check.", RECORDS_INDEX, data_path)
        return

    listed = [line.strip() for line in index_path.read_text().splitlines() if line.strip()]
    lower = {name.lower(): name for name in found}
    case_only = [name for name in listed if name not in found and name.lower() in lower]
    absent = [name for name in listed if name.lower() not in lower]

    if case_only:
        logger.warning(
            "%d of the %d lines in %s differ from the tree only in case (e.g. %s vs %s) "
            "— an upstream defect in the release, not missing data. Paths here come "
            "from the headers on disk.",
            len(case_only), len(listed), RECORDS_INDEX, case_only[0], lower[case_only[0].lower()],
        )
    if absent:
        logger.warning(
            "%d records listed in %s have no header on disk: %s. Your copy may be "
            "incomplete — check it against SHA256SUMS.txt.",
            len(absent), RECORDS_INDEX, absent[:5],
        )
    extra = sorted(found - set(listed) - {lower[n.lower()] for n in case_only})
    if extra:
        logger.warning("%d headers on disk are absent from %s: %s", len(extra),
                       RECORDS_INDEX, extra[:5])


def scan_records(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Build the per-record frame by walking the five device directories.

    Every record is decoded in full — roughly 13.7 million samples, about fifteen
    seconds — because the trailing invalid sample of point 3 and the per-record
    rescaling of point 8 are properties of the samples and of nothing else.
    """
    import wfdb

    from ecgbench.dataset import _load_signal

    data_path = Path(data_path)
    missing = [d for d in DEVICES if not (data_path / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Expected one directory per device under {data_path}; {missing} are "
            f"absent. Point data_path at the version directory holding "
            f"{sorted(DEVICES)} — not at the dataset root above it."
        )

    rows: list[dict[str, object]] = []
    found: set[str] = set()
    for device, info in DEVICES.items():
        for header in sorted((data_path / device).rglob("*.hea")):
            relative = header.relative_to(data_path).with_suffix("")
            found.add(str(relative))
            parts = relative.parts
            # <device>/<family>/<setting>/<name> for three families and
            # <device>/sqr-2hz/<name> for the fourth, which has a single setting.
            setting_dir = parts[2] if len(parts) == 4 else parts[1]
            setting = parse_setting(setting_dir.lower())
            name = relative.name

            signal = _load_signal(str(data_path / relative), config.signal_format)
            n_leads, n_samples = signal.shape
            nan_samples = int(np.isnan(signal).sum())
            finite = signal[~np.isnan(signal)]
            # A single trailing NaN is the Samsung case (point 3) and is the
            # difference between "unusable record" and "read one sample less".
            trailing = bool(
                nan_samples and np.all(np.isnan(signal[:, -1])) and nan_samples == n_leads
            )

            head = wfdb.rdheader(str(data_path / relative))
            rows.append(
                {
                    "record_id": f"{device}_{name}",
                    "device": device,
                    "device_model": info["model"],
                    "device_role": info["role"],
                    "derivation": info["derivation"],
                    "setting_id": setting_dir.lower(),
                    "setting_dir": setting_dir,
                    "family": setting["family"],
                    "nominal_rate_bpm": setting["nominal_rate_bpm"],
                    "nominal_r_amplitude_uv": setting["nominal_r_amplitude_uv"],
                    "nominal_st_offset_uv": setting["nominal_st_offset_uv"],
                    "replicate": int(name.rsplit("_", 1)[-1]),
                    "signal_path": str(relative),
                    "n_leads": n_leads,
                    "lead_names_stored": "|".join(head.sig_name),
                    "sampling_rate": int(head.fs),
                    "n_samples": n_samples,
                    "duration_secs": round(n_samples / float(head.fs), 4),
                    "nan_samples": nan_samples,
                    "trailing_invalid_sample": trailing,
                    "min_mv": round(float(finite.min()), 4) if finite.size else None,
                    "max_mv": round(float(finite.max()), 4) if finite.size else None,
                    "span_mv": round(float(finite.max() - finite.min()), 4)
                    if finite.size
                    else None,
                    "variance_mv2": round(float(np.nanvar(signal[0])), 6),
                    "header_comment": head.comments[0] if head.comments else "",
                }
            )

    if not rows:
        raise FileNotFoundError(f"No WFDB headers found under {data_path}.")

    _check_records_index(data_path, found)

    df = pd.DataFrame(rows)
    # Record ids are <device>_<name>, and no device name prefixes another, so a
    # collision would mean two headers with the same path — but the ids are the
    # join key for every label lookup, so this is checked rather than argued.
    if df["record_id"].duplicated().any():
        clashes = df.loc[df["record_id"].duplicated(keep=False), "record_id"].tolist()
        raise ValueError(f"Record ids are not unique: {sorted(set(clashes))[:5]}")

    # The 17 sixth repetitions of point 4. Flagged rather than dropped: no two
    # records in the release share a signal, so they are real acquisitions.
    df["is_extra_replicate"] = df["replicate"] >= 5
    return df


def attach_stratify_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stratify_class``: the experiment family.

    **What has to stay balanced, and what cannot.** Folds group on ``setting_id``
    (see :mod:`ecgbench.splitting.strategies.ecg_capable_smartwatches` for why),
    which leaves 36 groups to spread over ten folds. A stratification class needs
    at least ``n_folds`` *groups* to reach every fold, and only two of the four
    families have that many: ``st-segment`` has 16 settings and ``freq_test`` 15,
    while ``amp_test`` has 4 and ``sqr-2hz`` has 1. So no axis can put all four
    families in all ten folds — that is arithmetic, not a tuning failure, and it
    holds at every fold count above four.

    Measured over the shipped files at ``random_state=42``, family against the
    alternatives:

    ========================  ===========  ==================  ===================
    stratification            fold sizes   folds with ``amp``  folds with ``st``
    ========================  ===========  ==================  ===================
    a constant                76 – 102     3                   8
    ``st`` vs everything      77 – 102     4                   10
    **family (4 classes)**    **77 – 102** **4**               **10**
    ========================  ===========  ==================  ===================

    Family and the binary cut tie on those two counts; family is used because it
    also keeps ``freq_test`` in all ten folds and says what it is doing. A
    constant leaves folds 2 and 9 with no ``st-segment`` record and no
    ``amp_test`` record at all, which is what ruled it out.

    **Device is deliberately not in the cross, and does not need to be.** Every
    setting was recorded on all five devices, so any partition of settings splits
    the devices evenly by construction: the measured spread is 15 to 22 records
    per device per fold. It could not be crossed in anyway — device is not
    constant within a group, and a stratification label has to be.

    **What this leaves the caller.** ``amp_test`` lands in 4 of the 10 folds and
    ``sqr-2hz`` in exactly 1, so the default ``test`` fold (fold 10) holds
    ``freq_test`` and ``st-segment`` records only. For R-amplitude or square-wave
    work use ``split=None`` with ``fold_numbers=[...]`` and select the folds that
    hold the family, or hold out settings by hand.
    """
    df = df.copy()
    df["stratify_class"] = df["family"].astype(str)
    return df


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return the simulator settings and signal properties, indexed by ``record_id``.

    **The label is the knob, not a diagnosis.** No human was recorded, so there is
    no sex, no age and no clinical finding to expose; what a user wants is the
    simulator parameter the record was made at, which the release states only in
    its directory names. See the module docstring for the nine properties of the
    files that the columns below encode.

    Columns:

    - ``device``, ``device_model``, ``device_role``, ``derivation`` — **which
      instrument this is.** ``device`` is the directory name, kept verbatim so
      paths stay traceable; ``device_model`` is the model the release's prose
      names, which differs for Apple (``applewatch_serie8`` is a Series 9, point
      7). ``device_role`` is ``reference`` for the 195 Philips records and
      ``smartwatch`` for the other 720. ``derivation`` is "standard 12-lead" or
      "lead I (LA-RA)" — **read point 1 before comparing leads across devices**,
      because every smartwatch header calls its lead I channel ``II``.
    - ``setting_id``, ``family``, ``replicate``, ``is_extra_replicate`` — **which
      simulator condition, and which repetition of it.** 36 settings across four
      families: 15 ``freq_test``, 16 ``st-segment``, 4 ``amp_test`` and one
      ``sqr-2hz``. ``setting_id`` is lowercased so Fitbit's ``ST-m1`` and the
      other devices' ``st-m1`` are one condition (point 5); ``setting_dir`` keeps
      the directory's own spelling. ``setting_id`` is also the config's grouping
      column, so a setting lies wholly inside one fold.
    - ``nominal_rate_bpm``, ``nominal_r_amplitude_uv``, ``nominal_st_offset_uv`` —
      **the ground truth, and each is NaN outside its own family.** 30-300 bpm in
      15 steps; 500, 1000, 1500, 2000 µV; -800 to +800 µV in 100 µV steps with no
      zero-offset control. Nothing here is measured from the signal: the whole
      question the dataset was published to answer is how far a device's reading
      departs from these, so estimating them for the user would be answering it
      for them, badly.
    - ``n_leads``, ``lead_names_stored``, ``sampling_rate`` — 12 leads at 500 Hz
      for Philips, 1 lead at 512 (Apple), 500 (Samsung), 250 (Fitbit) or 300 Hz
      (Withings). Rate is a per-record property, not a choice of representation.
    - ``n_samples``, ``duration_secs`` — 5,500 to 15,360 samples, 11.0 to 30.0 s.
      Lengths are not uniform (point 9); ``window=(0, 5500)`` is the largest
      window that fits every record.
    - ``nan_samples``, ``trailing_invalid_sample`` — **why ``clean`` has no Samsung
      records.** Every Samsung record's final sample is digital ``-32768``, WFDB's
      invalid marker, so all 179 fail ``check_nan_values`` (point 3). No other
      record in the release contains a NaN. ``window=(0, 15000)`` reads a Samsung
      record without one.
    - ``min_mv``, ``max_mv``, ``span_mv``, ``variance_mv2`` — range over the finite
      samples. Every record was rescaled to fill int16 (point 8), so the extremes
      are the record's own and not a converter rail; ``span_mv`` runs 0.32 to 5.14
      mV and is *not* the nominal amplitude.
    - ``header_comment`` — the header's free-text device line, exposed because it
      is wrong for two devices: all 181 Fitbit records claim to be Withings, and
      all 180 Apple records claim Series 8 (points 5 and 7).
    - ``signal_path`` — the record stem relative to the dataset root, e.g.
      ``philips_tc30/amp_test/amp1000/amp1000_0``. Taken from the headers on disk,
      not from ``RECORDS``, 75 of whose lines name a directory that does not exist
      (point 6).
    - ``stratify_class`` — for fold construction only. See
      :func:`attach_stratify_class`.
    """
    df = scan_records(data_path, config)
    df = attach_stratify_class(df)
    df = df.set_index("record_id")
    df.index.name = config.record_id_column
    return df
