#!/usr/bin/env python3
"""
Example: Brno University of Technology ECG Quality Database (BUT QDB).

18 single-lead wearable recordings, each longer than 24 hours, from 15 people going
about ordinary life with a Bittium Faros 180. What makes it unlike every other
dataset in ECGBench is the ground truth: three ECG experts graded the SIGNAL QUALITY
sample by sample, and the release ships all three opinions plus their consensus.

Six things to demonstrate:

1. **The label is per sample, and the fold CSVs cannot carry it.** The per-record
   columns from `labels=True` are summaries. The real ground truth comes from
   `ecgbench.labels.butqdb.quality_vector(data_path, record_id, start=, length=)`,
   which takes the same `(start, length)` pair as `ECGDataset(window=...)` and
   returns one class per sample. This script windows a record and lines the two up.
2. **Only 20.8% of the recorded time is annotated, and it is concentrated in three
   records.** 100001, 105001 and 111001 are graded end to end and hold 88.6% of all
   annotated time; the other 15 get 40-80 minutes each out of 24+ hours. A window
   placed outside those blocks returns signal with no label behind it, so
   `annotated_blocks()` is what tells you where a window may go.
3. **The experts disagree a great deal**, which is the ceiling on any result
   measured against the consensus. All three agree on 69.4% of graded samples, and
   expert 1 is systematically stricter than the other two.
4. **Records are 24.0 to 38.7 hours**, so `window=` is needed to batch at all —
   86.4M to 139.1M samples is 346-556 MB of float32 per record.
5. **Every record saturates the converter**, so `amplitude_outlier` cannot fire and
   `clipped_fraction` is the column that measures it. `clean` is all 18 records by
   design: excluding noisy recordings would destroy what the database is for.
6. **The signal is stored in microvolts** with a per-record gain and baseline, so
   `units="uV"` shows the source scale that `signal_unit_scale: 0.001` converts.

Labels come straight from the shipped annotation files, so this works without
running the split pipeline first. The fold CSVs come from the Hub by default.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_butqdb.py --data-path /path/to/butqdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.butqdb import (
    ANNOTATORS,
    CLASS3_HIGH_THRESHOLD,
    QUALITY_CLASSES,
    annotated_blocks,
    quality_vector,
)

#: 10 s at 1 kHz. Records are 86.4M-139.1M samples, so a window is required to batch
#: at all — and because window= pushes down into the reader, it also avoids decoding
#: the other 24+ hours.
WINDOW = (0, 10_000)

#: The shortest record: 103003 holds 86,420,000 samples (24.01 h). Any window has to
#: end at or before this to fit every record — which is a whole day of headroom, so
#: unlike ptbdb or cpsc_2018 the window is never the binding constraint here.
SHORTEST_RECORD_SAMPLES = 86_420_000

#: Start samples of the two standard 20-minute graded segments. Identical in all 15
#: partially-annotated records — 8 h 00 m and 16 h 00 m into the recording — which the
#: release does not state and this script recomputes. `window=(0, n)` therefore has no
#: labels behind it for 15 of the 18 records, however small n is.
FIRST_STANDARD_SEGMENT = 28_800_000
SECOND_STANDARD_SEGMENT = 57_600_000


def main():
    parser = argparse.ArgumentParser(description="Load BUT QDB with quality labels")
    parser.add_argument("--data-path", default=None, help="Path to the version directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("butqdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- one channel; the release names no anatomy")
    print(f"Duration: {config.duration_seconds:.0f} s nominal (the SHORTEST record);"
          " 24.0-38.7 h in fact")
    print(f"Patients: {config.patient_id_column}  <- 15 subjects for 18 records")
    print(f"Label:    {config.label_column}  <- A REDUCTION. The label is per sample;"
          " see below")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("butqdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}"
          f"  (window {WINDOW} of {labels['n_samples']})")
    # int64, not str: record ids run 100001-126001 with no leading zero, so the fold
    # CSV round trip is lossless and the config leaves zero_padded_identifiers off.
    # Anything taking a record id must therefore str() it — quality_vector does.
    print(f"  record_id                  {sample['record_id']}"
          f"  ({type(sample['record_id']).__name__})")
    print(f"  subject_id / session       {labels['subject_id']} / {labels['session_index']}")
    print(f"  sex / age / bmi            {labels['sex']} {labels['age']:.0f}"
          f" {labels['bmi']:.1f}")
    print(f"  duration_secs              {labels['duration_secs']:.0f}"
          f"  ({labels['duration_secs'] / 3600:.2f} h)")
    print(f"  annotated_secs             {labels['annotated_secs']:.0f}"
          f"  ({100 * labels['annotated_fraction']:.1f}% of the record)")
    print(f"  fully_annotated            {bool(labels['fully_annotated'])}")
    print(f"  n_annotated_blocks         {int(labels['n_annotated_blocks'])}")
    for k in (1, 2, 3):
        print(f"  consensus_class{k}_fraction  {labels[f'consensus_class{k}_fraction']:.4f}"
              f"  ({labels[f'consensus_class{k}_secs'] / 60:.1f} min)")
    print(f"  dominant_consensus_class   {int(labels['dominant_consensus_class'])}"
          "   <- a reduction, do not train on it")
    print(f"  clipped_fraction           {labels['clipped_fraction']:.6f}")
    print(f"  min_mv / max_mv            {labels['min_mv']:.3f} / {labels['max_mv']:.3f}"
          "   <- this record's ADC rails")
    print(f"  acc_path                   {labels['acc_path']}"
          "   <- 100 Hz accelerometer, not an ECGBench record")
    print(f"  stratify_class             {labels['stratify_class']}"
          "   <- for fold construction only")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].astype(str).to_numpy()

    print("\nThe three quality classes, in the release's own words:")
    for value, meaning in QUALITY_CLASSES.items():
        print(f"  {value}  {meaning}")

    # --- 1. The label is per sample ------------------------------------------
    record = sample["record_id"]
    print(f"\n--- The actual ground truth, for {record} -------------------------------")
    print("quality_vector() takes the same (start, length) as ECGDataset(window=...),")
    print("so the label array lines up with the signal tensor sample for sample:")
    vector = quality_vector(dataset.data_path, record, start=WINDOW[0], length=WINDOW[1])
    print(f"  signal {tuple(sample['signal'].shape)}   labels {vector.shape}"
          f"   dtype {vector.dtype}")
    assert vector.shape[0] == sample["signal"].shape[1]
    counts = {int(k): int((vector == k).sum()) for k in (0, 1, 2, 3)}
    print(f"  classes in this window: {counts}   (0 = never annotated)")
    print("  Every one of the four annotators is reachable, and they differ:")
    for annotator in ANNOTATORS:
        v = quality_vector(dataset.data_path, record, annotator=annotator,
                           start=WINDOW[0], length=WINDOW[1])
        mix = {int(k): int((v == k).sum()) for k in (1, 2, 3) if (v == k).any()}
        print(f"    {annotator:10} {mix}")

    # The trap: window=(0, n) has no labels behind it for 15 of the 18 records.
    partial = df.index[~df["fully_annotated"].astype(bool)]
    if len(partial):
        other = partial[0]
        naive = quality_vector(dataset.data_path, other, start=0, length=WINDOW[1])
        good = quality_vector(dataset.data_path, other,
                              start=FIRST_STANDARD_SEGMENT, length=WINDOW[1])
        print(f"\n  THE TRAP, on {other} (graded for"
              f" {df.loc[other, 'annotated_secs'] / 60:.0f} min of"
              f" {df.loc[other, 'duration_secs'] / 3600:.1f} h):")
        print(f"    window=(0, {WINDOW[1]})           -> classes"
              f" {sorted(set(naive.tolist()))}  <- all 0: NEVER ANNOTATED")
        print(f"    window=({FIRST_STANDARD_SEGMENT}, {WINDOW[1]}) -> classes"
              f" {sorted(set(good.tolist()))}  <- inside the 8-hour segment")
        print("    The two standard 20-minute segments are at the SAME offsets in all")
        print(f"    15 partially-graded records: {FIRST_STANDARD_SEGMENT} and"
              f" {SECOND_STANDARD_SEGMENT} (8 h and 16 h in).")
        print("    Nothing in the release mentions that; it is recomputed here.")

    # --- 2. Where a window may legitimately go --------------------------------
    print("\n--- Where the annotations are ------------------------------------------")
    print("15 of the 18 records are graded for 40-80 minutes of a 24+ hour recording,")
    print("so a window outside those blocks returns signal with no label behind it:")
    print(f"  {'record':8} {'hours':>7} {'graded min':>11} {'blocks':>7} {'graded %':>9}"
          "  block bounds (samples)")
    for name, row in df.sort_values("annotated_fraction", ascending=False).iterrows():
        blocks = annotated_blocks(dataset.data_path, name)
        bounds = ", ".join(f"{int(b.start)}-{int(b.end)}" for b in blocks.itertuples())
        if len(bounds) > 46:
            bounds = bounds[:43] + "..."
        print(f"  {name:8} {row['duration_secs'] / 3600:7.2f}"
              f" {row['annotated_secs'] / 60:11.1f} {int(row['n_annotated_blocks']):7d}"
              f" {100 * row['annotated_fraction']:9.2f}  {bounds}")
    graded = float(df["annotated_secs"].sum())
    total = float(df["duration_secs"].sum())
    print(f"  {graded / 3600:.1f} h of this split's {total / 3600:.1f} h is graded"
          f" ({100 * graded / total:.1f}%).")
    full = df.index[df["fully_annotated"].astype(bool)].tolist()
    if full:
        share = 100 * df.loc[full, "annotated_secs"].sum() / graded
        print(f"  {len(full)} record(s) graded end to end ({full}) hold"
              f" {share:.1f}% of that.")

    # --- 3. How much the experts agreed --------------------------------------
    print("\n--- Inter-annotator agreement, which bounds any result ------------------")
    print(f"  {'record':8} {'unanimous':>10} {'mean pairwise':>14} {'maj. exists':>12}"
          f" {'consensus=maj':>14}")
    for name, row in df.sort_values("expert_unanimous_fraction").iterrows():
        print(f"  {name:8} {row['expert_unanimous_fraction']:10.4f}"
              f" {row['mean_expert_agreement']:14.4f}"
              f" {row['expert_majority_fraction']:12.6f}"
              f" {row['consensus_matches_majority']:14.6f}")
    print("  `consensus=maj` is the evidence that the fourth column triple is a")
    print("  majority vote of the other three, measured rather than assumed — the")
    print("  release states the layout but not the rule.")
    print("\n  Class 1 called by each annotator (they disagree most about 1 vs 2):")
    print(f"  {'record':8} {'expert_1':>9} {'expert_2':>9} {'expert_3':>9} {'consensus':>10}")
    for name, row in df.iterrows():
        print(f"  {name:8} {row['expert_1_class1_fraction']:9.3f}"
              f" {row['expert_2_class1_fraction']:9.3f}"
              f" {row['expert_3_class1_fraction']:9.3f}"
              f" {row['consensus_class1_fraction']:10.3f}")

    # --- 4. Long records, so window= is mandatory -----------------------------
    print("\n--- Record length ------------------------------------------------------")
    print(f"  {int(df['n_samples'].min())} to {int(df['n_samples'].max())} samples"
          f" ({df['duration_secs'].min() / 3600:.2f}-{df['duration_secs'].max() / 3600:.2f} h)")
    mb = float(df["n_samples"].max()) * 4 / 1e6
    print(f"  The longest is 1 x {int(df['n_samples'].max())} float32 (~{mb:.0f} MB), so a")
    print(f"  batch of {args.batch_size} without window= would be ~{mb * args.batch_size:.0f}"
          " MB, every byte decoded.")
    print(f"  Every record holds at least {SHORTEST_RECORD_SAMPLES} samples, so unlike")
    print("  ptbdb or cpsc_2018 the shortest record is never the binding constraint.")

    # --- 5. Saturation, which no check catches --------------------------------
    print("\n--- Converter saturation ----------------------------------------------")
    print("Every record in the release attains both 16-bit ADC rails, so the configured")
    print(f"amplitude_range_mv {config.validation.amplitude_range_mv} has to be their union")
    print("and amplitude_outlier cannot fail anything. clipped_fraction is the measure:")
    for name, row in df.sort_values("clipped_fraction", ascending=False).head(5).iterrows():
        print(f"  {name}  {row['clipped_fraction']:.6f}"
              f"  ({int(row['clipped_samples']):6d} samples at a rail)"
              f"  span [{row['min_mv']:.2f}, {row['max_mv']:.2f}] mV")
    print(f"  invalid (-32768) samples anywhere in this split:"
          f" {int(df['n_invalid_samples'].sum())}")
    print("  `clean` is all 18 records by design: this is the one dataset here whose")
    print("  subject IS signal quality, so excluding noisy recordings would destroy it.")
    print("  Filter on the annotations instead:")
    print("    usable = ds.labels_df[ds.labels_df['consensus_class3_fraction'] < 0.05]")

    # --- 6. Microvolts at source ---------------------------------------------
    print("\n--- Units --------------------------------------------------------------")
    uv = ECGDataset("butqdb", units="uV", **common)
    print(f"  mV (default): max {sample['signal'].max():.3f}")
    print(f"  uV (source):  max {uv[0]['signal'].max():.1f}"
          "   <- signal_unit_scale = 0.001")
    print("  Gain and baseline differ per record (0.99998-1.996 ADC units per uV,")
    print("  baseline -18289 to +11462), so each record's physical span differs too.")

    # --- Stratification ------------------------------------------------------
    print("\n--- Folds --------------------------------------------------------------")
    print(f"stratify_class in this split: {df['stratify_class'].value_counts().to_dict()}")
    print(f"  Whether more than {100 * CLASS3_HIGH_THRESHOLD:.0f}% of the record's graded"
          " time is class 3 (unusable).")
    print("  Balancing unusable signal is what matters here, because rejecting it is the")
    print("  task: only six records carry an appreciable class-3 burden, so at most six")
    print("  of the ten folds can hold one — and this axis achieves exactly six. Sex")
    print("  reaches five. For fold construction only; do not train on it.")
    print(f"  Subjects in this split: {df['subject_id'].nunique()} for {len(df)} records")

    # --- Batching -------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")

    # The per-record target worth having is a continuous one: how much of the graded
    # time is unusable. There is no per-record classification target here.
    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["consensus_class3_fraction"]],
        dtype=torch.float32,
    )
    print(f"\nconsensus_class3_fraction target: {tuple(target.shape)}"
          f"  mean {target.mean():.4f}  max {target.max():.4f}")
    print("  (a regression target over records; the real task is per-sample")
    print("   classification, for which the target is quality_vector())")


if __name__ == "__main__":
    main()
