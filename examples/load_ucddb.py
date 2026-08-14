#!/usr/bin/env python3
"""
Example: St. Vincent's / UCD Sleep Apnea Database (UCDDB), with labels.

25 overnight sleep studies from Dublin — 203.4 hours of three-channel Holter ECG
at 128 Hz, 7.52 h to 8.68 h per record. **The first EDF dataset in ECGBench**, and
the one whose annotations do not line up with its own ECG until they are moved.

Six things to demonstrate:

1. **The record-level label is one number per subject.** 25 subjects, one AHI
   each. Nothing trained on that at the record level should be believed; what is
   learnable here is an event *within* a recording.
2. **The annotations are stamped in polysomnogram time and the Holter's clock is
   a placeholder**, so a respiratory event's time of day is not a position in the
   ECG. `psg_offset_secs` is the recovered difference, and
   `respiratory_events()` / `sleep_stages()` apply it for you as `holter_secs`.
3. **`ucddb028`'s Holter file is a byte-identical copy of `ucddb014`'s.** Two
   different men, one ECG recording, and nothing upstream says so. Drop it for
   record-level supervised work.
4. **These are 8-hour records**, so `window=` is not an optimisation — a whole
   record is ~40 MB and the whole database ~1 GB in memory.
5. **The channels are V5, CC5 and V5R**, none of which is a standard 12-lead
   name, so `leads=` by name is the only safe way to pick one.
6. **The ECG rides a ~5 mV pedestal** by the files' own EDF calibration.
7. **Every record opens with 67-119 s of calibration square wave**, so
   `window=(0, n)` returns no ECG at all.

Labels come from `SubjectDetails.xls` and the annotation text files, so this works
without running the split pipeline first. The fold CSVs come from the Hub by
default.

Prerequisites:
  - pip install ecgbench[torch,xls]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_ucddb.py --data-path /path/to/ucddb/1.0.0/
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.ucddb import (
    CALIBRATION_SAMPLES,
    ECG_STARTS_AT_SAMPLE,
    HOLTER_DUPLICATES,
    OSA_AHI_THRESHOLD,
    respiratory_events,
    sleep_stages,
)

#: 30 s at 128 Hz, STARTING AFTER THE CALIBRATION BLOCK. window=(0, 3840) would
#: return the 1 mV calibration square wave for every record — byte-identical
#: across all 25 of them — rather than anybody's ECG. ECG_STARTS_AT_SAMPLE is
#: 15,232 (119.0 s), the first sample past the longest block in the database, and
#: the shortest record is 3,463,680 samples, so this fits everywhere.
WINDOW = (ECG_STARTS_AT_SAMPLE, 3840)

#: ucddb014's length in samples — the ceiling on `start + length` for a window
#: that must work on every record.
SHORTEST_RECORD_SAMPLES = 3_463_680

#: Seconds of ECG to take around a respiratory event. Scored events run 6 s to
#: 56 s, so a minute of context comfortably contains one.
EVENT_CONTEXT_SECS = 60


def main():
    parser = argparse.ArgumentParser(description="Load ucddb with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("ucddb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Format:   {config.signal_format}  <- the first EDF dataset in ECGBench")
    print(f"Leads:    {config.lead_names}  <- from the landing page, not the headers")
    print(f"Rate:     {config.default_sampling_rate} Hz")
    print(f"Duration: median {config.duration_seconds / 3600:.2f} h, 7.52-8.68 h in fact")
    print(f"Folds:    {config.n_folds}, grouped on {config.patient_id_column!r}")
    print()
    print("!! THIS IS 25 PEOPLE FROM ONE SLEEP CLINIC, and everybody was already suspected")
    print("!! of sleep-disordered breathing — one subject has an AHI below 5, so there is")
    print("!! effectively no healthy control group. The record-level label is one number")
    print("!! per subject. Use the 3,428 scored respiratory events, not the 25 AHIs.")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
    )

    try:
        dataset = ECGDataset("ucddb", labels=True, window=WINDOW, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")
    data_path = dataset.data_path

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    print(f"  record_id             {sample['record_id']!r}")
    print(f"  recording_group       {labels['recording_group']!r}")
    print(f"  age / sex / BMI       {int(labels['age'])} / {labels['sex']} / {labels['bmi']}")
    print(f"  psg_ahi               {labels['psg_ahi']:.0f}  ({labels['ahi_severity']})")
    print(f"  n_apnea_hypopnea      {int(labels['n_apnea_hypopnea'])}")
    print(f"  n_epochs              {int(labels['n_epochs'])} scored 30 s epochs")
    print(
        f"  duration_secs         {labels['duration_secs']:.0f}"
        f"  ({labels['duration_secs'] / 3600:.2f} h)"
    )
    print(f"  psg_offset_secs       {labels['psg_offset_secs']:.0f}  <- see section 2")
    print(f"  psg_offset_r          {labels['psg_offset_r']:.3f}")
    print(f"  n_distinct_leads      {int(labels['n_distinct_leads'])}")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. One label per subject ---------------------------------------------
    print("\n" + "=" * 78)
    print("1. THE RECORD-LEVEL LABEL IS ONE AHI PER SUBJECT")
    print("=" * 78)
    print(f"  {'record':10} {'sex':>3} {'age':>4} {'BMI':>5} {'AHI':>5} {'grade':>9}"
          f" {'hours':>6} {'events':>7} {'stratify':>20}")
    for record, row in df.sort_values("psg_ahi").iterrows():
        print(
            f"  {record:10} {row['sex']:>3} {int(row['age']):4d} {row['bmi']:5.1f}"
            f" {row['psg_ahi']:5.0f} {row['ahi_severity']:>9}"
            f" {row['duration_secs'] / 3600:6.2f} {int(row['n_apnea_hypopnea']):7d}"
            f" {row['stratify_class']:>20}"
        )
    print(f"\n  Grades in this split: {df['ahi_severity'].value_counts().to_dict()}")
    print(f"  Folds are stratified on the pooled class instead (AHI >= {OSA_AHI_THRESHOLD:.0f}):")
    print(f"    {df['stratify_class'].value_counts().to_dict()}")
    print("  The four-class grade has a class of ONE subject over the whole database, which")
    print("  ten folds cannot carry. Train on psg_ahi or ahi_severity; stratify_class exists")
    print("  to build the partition.")

    # --- 2. The recovered clock -----------------------------------------------
    print("\n" + "=" * 78)
    print("2. AN ANNOTATION'S TIME OF DAY IS NOT A POSITION IN THE ECG")
    print("=" * 78)
    print("  The respiratory events and sleep epochs are stamped against the POLYSOMNOGRAM,")
    print("  whose header start time matches SubjectDetails.xls. The Holter EDF headers all")
    print("  read 09:0x on 01.01.06 — archive timestamps, and the landing page says so:")
    print('  "The recording dates and times are not available."')
    print()
    print("  ECGBench recovers the offset for 24 of the 25 records by cross-correlating")
    print("  heart rate between the two recordings of the same night:")
    print(f"\n  {'record':10} {'offset':>8} {'as h:mm':>9} {'r':>6} {'spread':>7} {'usable':>7}")
    for record, row in df.iterrows():
        offset = row["psg_offset_secs"]
        if not np.isfinite(offset):
            print(f"  {record:10} {'--':>8} {'--':>9} {'--':>6} {'--':>7} {'no':>7}")
            continue
        print(
            f"  {record:10} {offset:8.0f} {int(offset) // 3600}:{int(offset) % 3600 // 60:02d}"
            f"{'':>5} {row['psg_offset_r']:6.3f} {row['psg_offset_spread_secs']:7.0f}"
            f" {'yes' if row['psg_offset_reliable'] else 'NO':>7}"
        )
    print("\n  'spread' is how far the offset moves when it is refitted on the first, middle")
    print("  and last third of the night — a real offset does not move, a spurious")
    print("  correlation peak does. Recompute the whole table with:")
    print("    from ecgbench.labels.ucddb import verify_psg_alignment")
    print("    verify_psg_alignment('/path/to/ucddb/1.0.0/')")

    # --- 3. Reading the ECG at a scored respiratory event ----------------------
    print("\n" + "=" * 78)
    print("3. READING THE ECG AT A SCORED APNEA")
    print("=" * 78)
    usable = df[df["psg_offset_reliable"].astype(bool)]
    record = usable.index[0]
    events = respiratory_events(data_path, record)
    apneas = events[events["event_type"].str.startswith("APNEA")]
    chosen = (apneas if len(apneas) else events).iloc[0]
    fs = config.default_sampling_rate
    start = max(0, int((chosen["holter_secs"] - EVENT_CONTEXT_SECS / 2) * fs))
    length = EVENT_CONTEXT_SECS * fs
    print(
        f"  {record}: a {chosen['event_type']} of {chosen['duration_secs']:.0f} s at"
        f" {chosen['holter_secs']:.0f} s into the Holter"
        f" (SpO2 nadir {chosen['spo2_low_pct']:.1f}%)"
    )
    print(f"  -> window=({start}, {length})")

    event_ds = ECGDataset("ucddb", window=(start, length), **common)
    index = list(event_ds.metadata_df[config.record_id_column]).index(record)
    signal = event_ds[index]["signal"]
    print(
        f"  -> signal {tuple(signal.shape)}, {float(signal.min()):.2f} to "
        f"{float(signal.max()):.2f} mV.  Only these samples were decoded."
    )

    stages = sleep_stages(data_path, record)
    epoch = stages[
        (stages["holter_secs"] <= chosen["holter_secs"])
        & (stages["holter_secs"] + 30 > chosen["holter_secs"])
    ]
    if len(epoch):
        print(f"  -> the subject was in stage {epoch.iloc[0]['stage_name']!r} at that moment")
    counts = events["event_type"].value_counts().to_dict()
    print(f"  -> this record's scored events: {counts}")

    raw = sample["signal"]

    # --- 4. Long records need a window ----------------------------------------
    print("\n" + "=" * 78)
    print("4. THESE ARE EIGHT-HOUR RECORDS")
    print("=" * 78)
    whole = float(df["n_samples"].max()) * 3 * 4 / 1e6
    print(f"  Longest record: {int(df['n_samples'].max())} samples x 3 leads = {whole:.0f} MB")
    print(f"  as float32. The whole database is {df['n_samples'].sum() * 3 * 4 / 1e9:.1f} GB.")
    print(f"  window={WINDOW} decodes {WINDOW[1] / fs:.0f} s and seeks past the rest.")
    print(f"  Any window inside {SHORTEST_RECORD_SAMPLES} samples fits EVERY record;")
    print("  beyond that, WindowOutOfRangeError names the record and its true length.")

    # --- 4b. The calibration block --------------------------------------------
    print("\n" + "=" * 78)
    print("4b. THE FIRST 67-119 SECONDS OF EVERY RECORD ARE NOT ECG")
    print("=" * 78)
    calibration = ECGDataset("ucddb", window=(0, 3840), **common)[0]["signal"]
    print(f"  window=(0, 3840) on {sample['record_id']}: "
          f"{float(calibration.min()):.4f} to {float(calibration.max()):.4f} mV, "
          f"{int(calibration.unique().numel())} distinct values")
    print("  -> a 2 Hz square wave at 4.5006/5.5018 mV: the instrument's 1.0012 mV")
    print("     calibration pulse, and BYTE-IDENTICAL across all 25 records.")
    print(f"  This split's block lengths (samples): "
          f"{ {r: CALIBRATION_SAMPLES[r] for r in list(df.index)[:4]} } ...")
    print(f"  Longest in the database: {ECG_STARTS_AT_SAMPLE} samples "
          f"({ECG_STARTS_AT_SAMPLE / fs:.1f} s, ucddb027), which is why WINDOW starts there.")
    print(f"  window={WINDOW} on the same record: "
          f"{float(raw.min()):.4f} to {float(raw.max()):.4f} mV, "
          f"{int(raw.unique().numel())} distinct values  <- an ECG")
    print("  Nothing in the release documents the block. Recompute it with:")
    print("    from ecgbench.labels.ucddb import verify_calibration_block")
    print("    verify_calibration_block('/path/to/ucddb/1.0.0/')")

    # --- 5. Lead selection by name --------------------------------------------
    print("\n" + "=" * 78)
    print("5. V5, CC5, V5R — NONE OF THEM A STANDARD 12-LEAD NAME")
    print("=" * 78)
    one_lead = ECGDataset("ucddb", window=WINDOW, leads=["CC5"], **common)[0]["signal"]
    print(f"  leads=['CC5'] -> {tuple(one_lead.shape)}, stored channel 1")
    print("  Selecting by index would be guesswork; the EDF headers name the channels")
    print("  'chan 1', 'chan 2' and 'chan 3', and the electrode names come from the")
    print("  landing page. In ucddb002 the third channel is a COPY of the second, so its")
    print("  'V5R' is not a third electrode — check n_distinct_leads before assuming.")

    # --- 6. The 5 mV pedestal --------------------------------------------------
    print("\n" + "=" * 78)
    print("6. THE ECG RIDES A ~5 mV PEDESTAL")
    print("=" * 78)
    print(f"  {sample['record_id']}: mean {float(raw.mean()):.3f} mV, "
          f"range {float(raw.min()):.3f} to {float(raw.max()):.3f} mV")
    print("  Every Holter channel declares digital 0-4095 mapping to physical 0-10 mV, so")
    print("  the baseline sits at mid-scale. ECGBench applies the declared calibration")
    print("  verbatim, as any EDF reader does. Centre it yourself if you want zero-mean:")
    centred = raw - raw.median(dim=1, keepdim=True).values
    print(f"  raw - median -> mean {float(centred.mean()):.3f} mV, "
          f"range {float(centred.min()):.3f} to {float(centred.max()):.3f} mV")

    # --- 7. The duplicated Holter ---------------------------------------------
    print("\n" + "=" * 78)
    print("7. ONE RECORD'S ECG BELONGS TO ANOTHER SUBJECT")
    print("=" * 78)
    duplicated = df[~df["waveform_matches_subject"].astype(bool)]
    if len(duplicated):
        for record, row in duplicated.iterrows():
            print(f"  {record}: its _lifecard.edf is {row['holter_duplicate_of']}'s, "
                  f"bit for bit.")
            print(f"    Its labels are its own (AHI {row['psg_ahi']:.0f}); the waveform "
                  "belongs to a different man.")
    else:
        print(f"  Not in this split — {sorted(HOLTER_DUPLICATES)} is in another fold.")
    print(f"  Affected records across the database: {sorted(HOLTER_DUPLICATES)}")
    print("  Both are kept, because each is an official record with its own official")
    print("  annotations, and they share a recording_group so they cannot straddle a fold.")
    print("  For record-level supervised work, filter on waveform_matches_subject:")
    print("    df = df[df['waveform_matches_subject']]")

    # --- 8. A batch ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("8. A BATCH, AND A TARGET TENSOR")
    print("=" * 78)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"  signals   {tuple(batch['signal'].shape)}  (batch, leads, samples)")
    print(f"  record_id {batch['record_id']}")

    # ecg_collate_fn keeps label dicts as a LIST of dicts, not a dict of lists.
    ahi = torch.tensor(
        [float(row["psg_ahi"]) for row in batch["labels"]], dtype=torch.float32
    )
    target = (ahi >= OSA_AHI_THRESHOLD).long()
    print(f"  psg_ahi   {ahi.tolist()}")
    print(f"  target    {target.tolist()}  (1 = moderate-or-severe OSA)")
    print("\n  window= is plain data, so it survives DataLoader(num_workers>0) under the")
    print("  'spawn' start method, where a lambda transform would raise PicklingError.")


if __name__ == "__main__":
    main()
