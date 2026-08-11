#!/usr/bin/env python3
"""
Example: tOLIet — thigh-based ECG from an instrumented toilet seat.

145 sittings by 86 volunteers, each recorded simultaneously by four dry electrode
pairs moulded into the seat that differ only in surface texture. Five things to
demonstrate, and the first two decide how you use the dataset at all:

1. **One record is one electrode channel, not one sitting.** ECGBench splits each
   file into its four channels, so `15_1_A2` is the sinusoidal-electrode channel
   of subject 15's second sitting: 145 x 4 = 580 records, tensor shape
   (1, samples). `source_record` and `electrode_texture` say which is which.
2. **238 of the 580 channels are electrodes that made no contact**, and they are
   what separates `original` from `clean`. Sit on a seat and some pairs touch skin
   and some do not; a pair that did not reads a constant ADC code for the whole
   sitting. The default `version="clean"` gives you the 342 that recorded. Per
   texture, live in: 140 of 145 for the flat electrode, 127 sinusoidal, 68
   trapezoidal — and 7 pyramidal, which is a result about electrode geometry
   rather than a defect.
3. **`signal_active` is a floor, not a guarantee — check `clipped_fraction` too.**
   The front end is ±1.5 mV into a 10-bit converter, so `amplitude_outlier` cannot
   fire, and a channel oscillating between both rails has a *large* variance and
   passes `flat_line` while carrying no ECG. This script ranks the split by
   clipping so the trap is visible rather than described.
4. **Length varies fourteenfold**, 14.4 s to 197.2 s, so a fixed `window=` has to
   fit the shortest record. `window=` is pushed into the reader, so on a 197 s
   record it decodes 14 s instead of all of it.
5. **The samples are inverted and offset in the source**, not merely scaled. The
   config's `signal_unit_scale: -3.0` is the amplifier's full-scale span in
   millivolts, negative because the seat's front end inverts; the opensignals
   reader returns fractions of full scale. Together they reproduce the release's
   own `Script/read_ecg_data.py` bit for bit.

Labels come from `DataSet.csv` plus a scan of the signals, so the first call takes
about 20 seconds and this works without running the split pipeline first. The fold
CSVs come from the Hub by default.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_tollet.py --data-path /path/to/tollet/1.0.1/
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.tollet import CHANNELS, TEXTURES

#: The first 14.4 seconds. The shortest record in the release holds exactly 14,400
#: samples, so this is the largest fixed window that fits every one of them; a
#: single sample more raises WindowOutOfRangeError on record 79. Pushed into the
#: reader's skiprows/nrows, so a 197-second record never decodes its other 183 s.
WINDOW = (0, 14_400)

#: Shortest and longest records in the release, in samples at 1000 Hz.
SHORTEST_RECORD = 14_400
LONGEST_RECORD = 197_250


def main():
    parser = argparse.ArgumentParser(description="Load tOLIet with labels")
    parser.add_argument("--data-path", default=None, help="Path to the version directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("tollet")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- ONE channel; the electrode is in the labels")
    print(f"Length:   varies, {SHORTEST_RECORD} to {LONGEST_RECORD} samples")
    print(f"Patients: {config.patient_id_column}  <- 86 subjects over 145 sittings")
    print(f"Scale:    {config.signal_unit_scale}  <- mV per full scale, NEGATIVE (inverted)")
    print()
    print("Electrode textures: " + ", ".join(f"{c}={TEXTURES[c]}" for c in CHANNELS))
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
    )

    try:
        dataset = ECGDataset("tollet", labels=True, window=WINDOW, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(
        f"Signal shape:  {tuple(sample['signal'].shape)}"
        f"  (window {WINDOW}, record holds {int(labels['n_samples'])} samples)"
    )
    print(f"  record_id            {sample['record_id']!r}")
    print(f"  source_record        {labels['source_record']!r}  <- the sitting")
    print(
        f"  channel / texture    {labels['channel']} = {labels['electrode_texture']}"
    )
    print(
        f"  subject_id           {labels['subject_id']}"
        f"  (session {int(labels['session_index'])})"
    )
    print(
        f"  age / sex / BMI      {labels['age']:.0f} / {labels['sex']} / {labels['bmi']:.1f}"
    )
    print(f"  duration             {labels['duration_secs']:.1f} s")
    print(
        f"  signal_active        {bool(labels['signal_active'])}   "
        f"clipped_fraction {labels['clipped_fraction']:.4f}"
    )
    print(f"  amplitude            {labels['min_mv']:.3f} to {labels['max_mv']:.3f} mV")
    print(f"  has_reference_ecg    {bool(labels['has_reference_ecg'])}")

    # labels_df is aligned POSITIONALLY with metadata_df, not indexed by record id,
    # so attach the ids before reporting anything per record.
    frame = dataset.labels_df.copy()
    frame.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- The dataset's own variable: which electrode texture recorded ---------
    print(f"\nElectrode texture over this split ({args.version}, no signals decoded):")
    for channel in CHANNELS:
        rows = frame[frame["channel"] == channel]
        if rows.empty:
            continue
        active = int(rows["signal_active"].sum())
        print(
            f"  {channel} {TEXTURES[channel]:<12} {len(rows):>4} records"
            f" ({len(rows) / len(frame):>4.0%} of split)"
            f"   {active:>4} active"
            f"   median clipped {rows['clipped_fraction'].median():.4f}"
        )
    if args.version == "clean":
        print(
            "  ('active' is 100% by construction here — clean IS the active records."
        )
        print(
            "   Run with --version original to see the drop-out: 140/127/7/68 of 145.)"
        )

    # --- The trap: passing flat_line is not the same as carrying an ECG -------
    active = frame[frame["signal_active"]]
    railed = active[active["clipped_fraction"] > 0.5]
    print(
        f"\n{len(railed)} of the {len(active)} active records in this split are at a "
        "converter rail for"
    )
    print("more than half their samples, and every one of them passed flat_line:")
    for record_id, row in railed.nlargest(3, "clipped_fraction").iterrows():
        print(
            f"  {record_id:<10} {row['electrode_texture']:<12}"
            f" clipped {row['clipped_fraction']:.3f}"
            f"  variance {row['variance_mv2']:.4f} mV^2"
        )
    print("  -> filter on clipped_fraction for anything that depends on morphology.")

    # --- 580 records are not 580 independent observations --------------------
    print(
        f"\n{len(frame)} records come from {frame['source_record'].nunique()} sittings "
        f"by {frame['subject_id'].nunique()} subjects."
    )
    print(
        "The four channels of a sitting are the SAME BEATS from four electrodes, so "
        "group on"
    )
    print("source_record before counting independent samples.")

    # --- Batching ------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nBatch signal:  {tuple(batch['signal'].shape)}  (window= makes this stackable)")
    print(f"Batch records: {batch['record_id']}")

    # A target tensor: the electrode texture, which is what this release is about.
    texture_index = {TEXTURES[channel]: i for i, channel in enumerate(CHANNELS)}
    targets = torch.tensor(
        [texture_index[row["electrode_texture"]] for row in batch["labels"]],
        dtype=torch.long,
    )
    print(f"Targets:       {targets.tolist()}  (index into {list(texture_index)})")

    # --- Units ---------------------------------------------------------------
    microvolts = ECGDataset("tollet", window=(0, 1000), units="uV", **common)[0]["signal"]
    millivolts = ECGDataset("tollet", window=(0, 1000), **common)[0]["signal"]
    print(
        f"\nUnits: max {millivolts.max():.4f} mV == {microvolts.max():.1f} uV"
        "  (source is 10-bit codes; signal_unit_scale does the rest)"
    )


if __name__ == "__main__":
    main()
