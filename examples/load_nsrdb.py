#!/usr/bin/env python3
"""
Example: MIT-BIH Normal Sinus Rhythm Database with labels.

18 full-day two-lead Holter recordings of subjects with no significant
arrhythmias — the reference *normal* cohort. Five things to demonstrate:

1. **There is no clinical label, and that is the point.** `cohort_label` is
   `normal_sinus_rhythm` for all 18 records, because the release ships no rhythm
   annotations at all. Folds are stratified on the subject's sex (13 F, 5 M), the
   one axis PhysioNet documents about this cohort. Use the database as a control
   or a pretraining corpus, not as a classification task.
2. **The beat annotations stop long before the signal does.** They cover 79.5% to
   95.7% of each record; the last one to five hours carry waveform with no
   reference behind it, and nothing announces it. This script prints the
   annotated span per record so a window can be kept inside it.
3. **Records are a full day (10.7M-12.0M samples, 85-96 MB of float32).**
   `window=(start, length)` is needed to batch at all, and because it is pushed
   into the reader it avoids decoding the other 24 hours. Length is *not*
   uniform, so a window sized for one record need not fit another.
4. **Signal quality is annotated per channel.** The `~` transitions give seconds
   of clean, ECG1-noisy, ECG2-noisy and both-noisy time. 98.6% of the release is
   clean, but that spans 0.23% (16786) to 9.60% (16272) noisy per record.
5. **The two channels are not named leads.** The headers call them `ECG1` and
   `ECG2` and the release never says which anatomical leads they are, so `leads=`
   selects a channel position — as in `afdb`, and unlike `mitdb`.

Labels come straight from the headers and annotation files, so this works without
running the split pipeline first. The fold CSVs do come from the pipeline (or the
Hub) — pass --metadata-source local after copying
output/nsrdb/{clean,original}/ into the dataset root.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_nsrdb.py --data-path /path/to/nsrdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.nsrdb import BEAT_NAMES, COHORT_LABEL, RR_RANGE_SECS

#: 10 s at 128 Hz. Records are 10.7M-12.0M samples, so a window is required to
#: batch at all — and because window= pushes down into the reader, it also avoids
#: decoding the other ~24 h.
WINDOW = (0, 1280)

#: The shortest record: 17052 holds 10,659,840 samples (23.13 h). Any window has
#: to end at or before this, or it raises on that one record.
SHORTEST_RECORD_SAMPLES = 10_659_840


def main():
    parser = argparse.ArgumentParser(description="Load MIT-BIH NSRDB with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("nsrdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- channel positions; the release names no leads")
    print(f"Duration: nominal {config.duration_seconds:.0f} s, but 83,280-93,440 s in fact")
    print(f"Patients: {config.patient_id_column}  <- no subject id ships, so folds are ungrouped")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("nsrdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}"
          f"  (window {WINDOW} of {labels['n_samples']})")
    print(f"  record_id          {sample['record_id']!r}")
    print(f"  lead_names         {labels['lead_names']}")
    print(f"  cohort_label       {labels['cohort_label']}   <- the same for all 18 records")
    print(f"  age / sex          {labels['age']:.0f} {labels['sex']}")
    print(f"  duration_secs      {labels['duration_secs']:.0f}"
          f"  ({labels['duration_secs'] / 3600:.2f} h)")
    print(f"  n_beats            {labels['n_beats']}  ({labels['n_ectopic_beats']} not normal)")
    print(f"  mean_hr_bpm        {labels['mean_hr_bpm']:.1f}")
    print(f"  sdnn_ms            {labels['sdnn_ms']:.1f}")
    print(f"  rmssd_ms           {labels['rmssd_ms']:.1f}")
    print(f"  annotated_fraction {labels['annotated_fraction']:.3f}")
    print(f"  unannotated tail   {labels['unannotated_tail_secs']:.0f} s at the end")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. There is no clinical label ---------------------------------------
    print(f"\ncohort_label over this split: {df['cohort_label'].value_counts().to_dict()}")
    print(f"  One class ({COHORT_LABEL}) for every record, asserted by the release rather")
    print("  than derived from anything in the files — there are no rhythm annotations here.")
    print(f"  Folds are stratified on sex instead: {df['sex'].value_counts().to_dict()}"
          " in this split")
    print("  (13 F / 5 M across the whole release). That is for fold construction only;")
    print("  do not train on stratify_class.")

    print("\nEctopy exists, but only just — this is what 'no significant arrhythmias' means:")
    for symbol, name in BEAT_NAMES.items():
        total = int(df[f"beat_{symbol}"].sum())
        if total:
            print(f"  beat_{symbol:2s} {total:9d}  {name}")
    beats, ectopic = int(df["n_beats"].sum()), int(df["n_ectopic_beats"].sum())
    print(f"  {ectopic} of {beats} beats are not normal "
          f"({1e5 * ectopic / beats:.1f} per 100,000)")
    none = df.index[df["n_ectopic_beats"] == 0].tolist()
    print(f"  records with no ectopic beat at all: {none or 'none in this split'}")

    # --- 2. The annotations stop before the signal does -----------------------
    print("\nBeat annotation does NOT cover the whole record:")
    for record, row in df.sort_values("annotated_fraction").iterrows():
        bar = "#" * round(40 * row["annotated_fraction"])
        print(
            f"  {record}  {100 * row['annotated_fraction']:5.1f}% annotated "
            f" tail {row['unannotated_tail_secs'] / 3600:4.1f} h"
            f" head {row['unannotated_head_secs']:5.1f} s  {bar}"
        )
    unannotated = float((df["duration_secs"] - df["annotated_secs"]).sum())
    print(f"  {unannotated / 3600:.1f} h of this split's "
          f"{df['duration_secs'].sum() / 3600:.1f} h carries no beat reference.")
    print("  A window into the tail returns waveform with nothing to score it against —")
    print("  fine for self-supervised work, wrong for evaluating a detector.")

    # --- 3. Signal quality, per channel ---------------------------------------
    print("\nAnnotated signal quality (seconds, from the `~` transitions):")
    print(f"  {'record':8} {'clean h':>8} {'ECG1 noisy':>11} {'ECG2 noisy':>11} "
          f"{'both':>8} {'noisy %':>8} {'artifacts':>10}")
    for record, row in df.sort_values("noisy_fraction", ascending=False).iterrows():
        print(
            f"  {record:8} {row['clean_secs'] / 3600:8.2f} "
            f"{row['noisy_ECG1_secs']:11.0f} {row['noisy_ECG2_secs']:11.0f} "
            f"{row['noisy_both_secs']:8.0f} {100 * row['noisy_fraction']:8.2f} "
            f"{int(row['n_isolated_artifacts']):10d}"
        )
    print("  n_isolated_artifacts spans three orders of magnitude across the release")
    print("  (52 in 16273, 30,782 in 16773), so per-record metrics are not comparable")
    print("  without controlling for it.")

    # --- 4. HRV, the reason most people come here -----------------------------
    print(f"\nWhole-record HRV, over RR intervals in {RR_RANGE_SECS} s:")
    print(f"  mean_hr_bpm  {df['mean_hr_bpm'].min():.1f} - {df['mean_hr_bpm'].max():.1f}"
          f"  (mean {df['mean_hr_bpm'].mean():.1f})")
    print(f"  sdnn_ms      {df['sdnn_ms'].min():.1f} - {df['sdnn_ms'].max():.1f}")
    print(f"  rmssd_ms     {df['rmssd_ms'].min():.1f} - {df['rmssd_ms'].max():.1f}")
    print(f"  RR intervals rejected by that filter: {int(df['n_rr_rejected'].sum())}")
    print("  The filter is load-bearing: without it the multi-hour unannotated gaps")
    print("  above enter as single enormous 'RR intervals'. These are whole-record")
    print("  descriptive summaries over ~24 h of activity and sleep, not a segmented")
    print("  HRV analysis.")

    # --- 5. The two channels are unnamed --------------------------------------
    print(f"\nLead layouts in this split: {df['lead_names'].value_counts().to_dict()}")
    one = ECGDataset("nsrdb", leads=["ECG2"], **common)
    print(f"leads=['ECG2'] -> {one.lead_names}, shape {tuple(one[0]['signal'].shape)}")
    print("  These are channel positions. The release states no electrode placement,")
    print("  so do NOT read ECG1/ECG2 as MLII/V1 by analogy with mitdb.")

    # A window that fits the longest record but not the shortest.
    print(f"\nThe shortest record holds {SHORTEST_RECORD_SAMPLES} samples (17052), so a")
    print("window must fit inside that rather than inside the longest record's 11960320:")
    far = ECGDataset("nsrdb", **{**common, "window": (10_700_000, 128_000)})
    raised = 0
    for i in range(len(far)):
        try:
            far[i]
        except WindowOutOfRangeError as e:
            raised += 1
            if raised == 1:
                print(f"  window=(10700000, 128000) raises: {e}")
    print(f"  {raised} of {len(far)} records in this split are too short for it")

    # --- Batching -------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    mb = 2 * 11_960_320 * 4 / 1e6
    print(f"  Without window= the longest record is 2 x 11960320 float32 (~{mb:.0f} MB), so a")
    print(f"  batch of {args.batch_size} would be ~{mb * args.batch_size:.0f} MB,"
          " every byte decoded.")

    # There is no class to predict here, so the useful target is a per-record
    # continuous quantity. Mean heart rate is the obvious one.
    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["mean_hr_bpm"]], dtype=torch.float32
    )
    print(f"\nmean_hr_bpm target tensor: {tuple(target.shape)}  mean {target.mean():.1f} bpm")
    print("  (a regression target; there is no classification target in this database)")


if __name__ == "__main__":
    main()
