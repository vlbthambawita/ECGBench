#!/usr/bin/env python3
"""
Example: MIT-BIH ST Change Database with labels.

28 short recordings — mostly exercise stress tests — selected because they show
transient ST change. Five things to demonstrate, and the first two are the ones
that will otherwise cost you an afternoon:

1. **The ST Change Database annotates no ST change.** The name promises a
   delineation dataset and the release is not one: all 76,181 annotations in the
   28 `.atr` files are beat labels (76,175 beats plus six quality markers), with
   no ST episode, boundary or deviation anywhere. PhysioNet says so on the landing
   page. Use `edb` or `ltstdb` for annotated ST episodes; use this for 28
   recordings *selected* for ST change, with reference beats.
2. **Ten of the 28 records hold ONE channel, not two.** Nothing on the landing
   page says so and the catalogue's "2 leads" is true of 18 of them. A batch
   mixing layouts raises in collation, so this script shows the failure and then
   the fix — `leads=["ECG1"]`, which every record has.
3. **Records are 13.1 to 67.2 minutes and no two lengths match.**
   `window=(start, length)` is needed to batch at all, and it must fit the
   *shortest* record (784.3 s), not the longest.
4. **The exercise/long-term grouping is transcribed from a web page.**
   `st_change_type` is not measured; `group_source` says so on every row. The
   heart-rate profile is the per-record evidence, and for record 323 it disagrees.
5. **Six of these recordings are also in `qtdb`.** Records 301, 302, 306, 307,
   308 and 310 appear there as 15-minute excerpts. Training on one and evaluating
   on the other is testing on training data.

Labels come straight from the headers and annotation files, so this works without
running the split pipeline first. The fold CSVs come from the Hub by default.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_stdb.py --data-path /path/to/stdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.stdb import BEAT_NAMES, LONG_TERM_EXCERPTS, RR_RANGE_SECS

#: 30 s at 360 Hz. Records run 282,341 to 1,451,857 samples, so a window is needed
#: to batch at all — and because window= pushes down into the reader, it also
#: avoids decoding the other 13 to 67 minutes.
WINDOW = (0, 10_800)

#: The shortest record: 305 holds 282,341 samples (784.3 s). Any window has to end
#: at or before this, or it raises on that one record.
SHORTEST_RECORD_SAMPLES = 282_341

#: The six recordings that also appear in qtdb, as 15-minute excerpts resampled to
#: 250 Hz. Each qtdb header names its stdb parent.
ALSO_IN_QTDB = ("301", "302", "306", "307", "308", "310")


def main():
    parser = argparse.ArgumentParser(description="Load MIT-BIH ST Change with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("stdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- channel positions; the release names no leads")
    print(f"          alternate layouts: {config.alternate_lead_names}  <- ten records")
    print(f"Duration: nominal {config.duration_seconds:.0f} s, but 784-4,033 s in fact")
    print(f"Patients: {config.patient_id_column}  <- NOTHING identifies subjects here")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        # leads= is needed even for a single sample-by-sample walk if you intend to
        # stack anything; see section 2. Start without it to show why.
        dataset = ECGDataset("stdb", labels=True, **common)
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
    print(f"  n_channels         {labels['n_channels']}  <- 1 or 2, per record")
    print(f"  record_group       {labels['record_group']}")
    print(f"  st_change_type     {labels['st_change_type']}"
          f"   (source: {labels['group_source']})")
    print(f"  duration_secs      {labels['duration_secs']:.1f}"
          f"  ({labels['duration_secs'] / 60:.1f} min)")
    print(f"  n_beats            {labels['n_beats']}  ({labels['n_ectopic_beats']} not normal)")
    print(f"  baseline/peak HR   {labels['baseline_hr_bpm']:.1f} -> {labels['peak_hr_bpm']:.1f}"
          f" bpm  (rise {labels['hr_rise_bpm']:.1f})")
    print(f"  mean_hr_bpm        {labels['mean_hr_bpm']:.1f}")
    print(f"  annotated_fraction {labels['annotated_fraction']:.4f}")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number. They come
    # back as ints, because record names are "300".."327" with no leading zero and
    # so zero_padded_identifiers is False; str() before comparing with the string
    # constants this module exports.
    df = dataset.labels_df.copy()
    df.index = [str(r) for r in dataset.metadata_df[config.record_id_column]]

    # --- 1. The database named for ST change annotates none of it -------------
    print("\nThere are no ST annotations in this release, despite the name:")
    print(f"  n_rhythm_changes (`+`) over this split: {int(df['n_rhythm_changes'].sum())}")
    print(f"  n_quality_changes (`~`):                {int(df['n_quality_changes'].sum())}")
    print(f"  beats:                                  {int(df['n_beats'].sum())}")
    print("  Every annotation is a beat label bar the `~`. PhysioNet: the files")
    print("  'do not include ST change annotations, as in the European ST-T Database'.")
    print("  st_change_type below is a web page transcribed, NOT a measurement.")

    print("\nBeat vocabulary — the whole release uses three symbols:")
    for symbol, name in BEAT_NAMES.items():
        total = int(df[f"beat_{symbol}"].sum())
        print(f"  beat_{symbol:2s} {total:8d}  {name}")
    beats, ectopic = int(df["n_beats"].sum()), int(df["n_ectopic_beats"].sum())
    print(f"  {ectopic} of {beats} beats are not normal "
          f"({1e5 * ectopic / beats:.0f} per 100,000)")
    print("  Ectopy is concentrated, not spread: over the whole release 305 holds 265")
    print("  of the 322 V, and 324 and 326 hold 699 of the 815 S.")

    # --- 2. Two channel layouts, and what that does to a batch ----------------
    print(f"\nChannel layouts in this split: {df['n_channels'].value_counts().to_dict()}")
    single = sorted(df.index[df["n_channels"] == 1].tolist())
    print(f"  single-channel records here: {single}")
    print("  (release-wide: 313, 314, 315, 316, 317, 319, 320, 321, 322, 323)")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    try:
        batch = next(iter(loader))
        print(f"  A batch of {args.batch_size} happened to be uniform: "
              f"{tuple(batch['signal'].shape)}")
    except RuntimeError as e:
        print(f"  DataLoader without leads= raises: {type(e).__name__}: "
              f"{str(e).splitlines()[0]}")
    print("  ecg_collate_fn stacks with torch's default_collate, so a batch holding")
    print("  both layouts cannot be stacked. The fix is to ask for a lead every")
    print("  record has:")

    one = ECGDataset("stdb", leads=["ECG1"], **common)
    uniform = DataLoader(
        one, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(uniform))
    print(f"  leads=['ECG1'] -> {one.lead_names}, batch {tuple(batch['signal'].shape)}"
          f"  ids {batch['record_id']}")
    print("  Asking for ECG2 raises on the ten single-channel records instead of")
    print("  silently returning ECG1 — that is what alternate_lead_names buys.")
    print("  These are channel POSITIONS: the release states no electrode placement,")
    print("  so do not read ECG1/ECG2 as MLII/V1 by analogy with mitdb, whose 360 Hz")
    print("  sampling rate this release shares.")

    # --- 3. Length varies by a factor of five ---------------------------------
    print("\nAll 28 lengths in the release differ; this split holds:")
    for record, row in df.sort_values("duration_secs").iterrows():
        bar = "#" * round(30 * row["duration_secs"] / 4032.9)
        print(f"  {record}  {row['duration_secs']:7.1f} s "
              f"({row['duration_secs'] / 60:5.1f} min)  {int(row['n_channels'])}ch  {bar}")
    print(f"\nThe shortest record holds {SHORTEST_RECORD_SAMPLES} samples (305, 784.3 s), so a")
    print("window must fit inside that rather than inside record 306's 1,451,857:")
    far = ECGDataset("stdb", **{**common, "window": (300_000, 10_800)})
    raised = 0
    for i in range(len(far)):
        try:
            far[i]
        except WindowOutOfRangeError as e:
            raised += 1
            if raised == 1:
                print(f"  window=(300000, 10800) raises: {e}")
    print(f"  {raised} of {len(far)} records in this split are too short for it")

    # --- 4. The group label, and the measurement that checks it ---------------
    print("\nst_change_type is transcribed; hr_rise_bpm is measured. Compare them:")
    print(f"  {'record':8} {'group':18} {'ST':11} {'base':>6} {'peak':>6} {'rise':>6}")
    for record, row in df.sort_values("hr_rise_bpm").iterrows():
        flag = "  <- named a long-term excerpt" if record in LONG_TERM_EXCERPTS else ""
        print(f"  {record:8} {row['record_group']:18} {row['st_change_type']:11} "
              f"{row['baseline_hr_bpm']:6.1f} {row['peak_hr_bpm']:6.1f} "
              f"{row['hr_rise_bpm']:6.1f}{flag}")
    print("  Release-wide, 324/325/326 rise 0.0/3.1/7.9 bpm — flat, as an ambulatory")
    print("  excerpt should be — but 323 ramps 84 -> 172 bpm and is still at 117 bpm")
    print("  in its final minute. The grouping still follows the landing page; this")
    print("  column is how you check it.")

    print(f"\nWhole-record HRV, over RR intervals in {RR_RANGE_SECS} s:")
    print(f"  mean_hr_bpm  {df['mean_hr_bpm'].min():.1f} - {df['mean_hr_bpm'].max():.1f}"
          f"  (mean {df['mean_hr_bpm'].mean():.1f})")
    print(f"  sdnn_ms      {df['sdnn_ms'].min():.1f} - {df['sdnn_ms'].max():.1f}")
    print("  On a stress test these summarise a deliberately non-stationary recording,")
    print("  so they describe the record rather than measuring autonomic tone.")

    # --- 5. The overlap with qtdb ---------------------------------------------
    here = [r for r in ALSO_IN_QTDB if r in set(df.index)]  # df.index is str
    print(f"\nAlso in qtdb as 15-minute excerpts: {list(ALSO_IN_QTDB)}")
    print(f"  in this split: {here or 'none'}")
    print("  qtdb's sel301/302/306/307/308/310 are these recordings resampled to")
    print("  250 Hz. Do not train on one and evaluate on the other.")

    # A per-record continuous target is the honest one here: the group label has
    # only two values and 23 of 28 records carry the same one.
    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["peak_hr_bpm"]], dtype=torch.float32
    )
    print(f"\npeak_hr_bpm target tensor: {tuple(target.shape)}  mean {target.mean():.1f} bpm")
    classes = sorted(set(df["st_change_type"]))
    index = {name: i for i, name in enumerate(classes)}
    y = torch.tensor([index[v] for v in dataset.labels_df["st_change_type"]])
    print(f"st_change_type target:     {tuple(y.shape)}  classes {classes}"
          f"  counts {torch.bincount(y).tolist()}")
    print("  (23 of 28 records release-wide carry 'depression', so treat that as a")
    print("   descriptor of the cohort rather than a balanced classification task)")


if __name__ == "__main__":
    main()
