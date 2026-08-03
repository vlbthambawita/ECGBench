#!/usr/bin/env python3
"""
Example: ECGs recorded inside MRI scanners, distorted by the MHD effect.

53 records from 26 healthy volunteers at 1024 Hz, acquired in 1T, 3T and 7T MRI
scanners. Inside the bore the magnetohydrodynamic effect — blood ions moving
through the static B0 field — superimposes a voltage large enough to bury the P
wave, ST segment and T wave. 10 records are reference ECGs taken outside the
scanner for the same subjects. Every QRS complex is manually annotated.

There is **no diagnosis to predict**: all subjects were healthy and the QRS marks
carry no beat classification. The label is the acquisition condition, and the
task is signal separation.

Four things this script demonstrates, because each one bites:

1. **Records are not all 12-lead.** 39 carry the diagnostic 12; 14 carry only
   I, II, III from an MRI-conditional monitor. Batch without filtering and you
   get mixed (12, N) and (3, N) tensors.
2. **Length varies 24.4 s to 722.7 s**, so batching needs a fixed `window=`.
   The window must fit the SHORTEST record: 25,000 samples at 1024 Hz.
3. **Amplitudes reach -31 mV**, far past the devices' nominal +/-6 mV and
   +/-2.4 mV input ranges. That is the MHD effect, not corruption.
4. **The subject key is derived, not shipped.** Filename subject numbers are
   scoped per scanner, so 3T01 and 1T01 are different people while 3T01 and
   7T04 are the same one.

Usage:
  python examples/load_mhd_effect_ecg_mri.py --data-path /path/to/mhd-effect-ecg-mri/1.0.0/
"""

import argparse
from collections import Counter

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config

#: The shortest record is 25,000 samples (24.4 s), so this fits every record.
#: A larger window raises WindowOutOfRangeError on ECGMRI3T02Ff/Out.
SAFE_WINDOW = (0, 25000)


def main():
    parser = argparse.ArgumentParser(
        description="Load the MHD-effect ECG-in-MRI dataset via ECGBench"
    )
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("mhd_effect_ecg_mri")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {', '.join(config.lead_names)}")
    print("          ^ the 12-lead layout — but 14 of 53 records hold only I, II, III")
    print()

    dataset = ECGDataset(
        "mhd_effect_ecg_mri", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source, labels=True,
    )
    print(f"Records:  {len(dataset)}  (of 53)")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nLabel fields: {list(labels)}")
    print("\nFirst record:")
    print(f"  ID:              {sample['record_id']}")
    print(f"  signal:          {tuple(sample['signal'].shape)}  (leads, samples)")
    print(f"  condition:       {labels['condition']!r}   "
          f"(field_strength_T={labels['field_strength_T']}, "
          f"scanner={labels['scanner_field_T']}T)")
    print(f"  position:        {labels['position']!r}")
    print(f"  scanner:         {labels['mr_scanner']!r}, B0 {labels['b0_orientation']!r}")
    print(f"  ECG device:      {labels['ecg_recorder']!r}")
    print(f"  lead_config:     {labels['lead_config']!r}  ({labels['n_signals']} channels)")
    print(f"  duration:        {labels['duration_seconds']} s, {labels['n_qrs']} QRS "
          f"(~{labels['mean_hr_bpm']} bpm)")
    print(f"  subject_key:     {labels['subject_key']!r}")
    print(f"  filename slot:   {labels['scanner_subject_slot']!r}  <- per SCANNER, "
          "not a patient ID")

    frame = dataset.labels_df.copy()
    frame["record_id"] = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. Acquisition conditions ------------------------------------------
    print("\nAcquisition condition over this split (records / subjects):")
    for condition, group in frame.groupby("condition"):
        print(f"  {condition:10s} {len(group):3d} records  "
              f"{group['subject_key'].nunique():2d} subjects")
    print("  'reference' = recorded outside the bore, standing in for the in-bore")
    print("  ground truth that cannot be measured. It is a stationarity assumption,")
    print("  not a simultaneous recording — heart rate and morphology do differ.")

    # --- 2. The two channel layouts -----------------------------------------
    print("\nChannel layouts in this split:")
    for names, group in frame.groupby("channel_names"):
        print(f"  {len(group):3d} records  {len(names.split('|')):2d} ch  {names}")
    shapes = Counter(tuple(dataset[i]["signal"].shape) for i in range(min(len(dataset), 12)))
    print(f"  Raw signal shapes in the first {sum(shapes.values())} records: {dict(shapes)}")
    print("  -> pass leads= to get a homogeneous batch; I, II and III are the only")
    print("     channels present in every record.")

    # --- 3. Batching a variable-length dataset ------------------------------
    print(f"\nDurations here: {frame['duration_seconds'].min()} s to "
          f"{frame['duration_seconds'].max()} s — so a fixed window is required.")
    windowed = ECGDataset(
        "mhd_effect_ecg_mri", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source,
        leads=["I", "II", "III"], window=SAFE_WINDOW,
    )
    loader = DataLoader(windowed, batch_size=args.batch_size, shuffle=False,
                        collate_fn=ecg_collate_fn)
    batch = next(iter(loader))
    print(f"leads=['I','II','III'], window={SAFE_WINDOW} -> batch "
          f"{tuple(batch['signal'].shape)} {batch['signal'].dtype}")
    print("  window= is pushed into the reader (sampfrom/sampto), so a 722 s record")
    print("  decodes only the requested 24.4 s. It also pickles, unlike a lambda")
    print("  transform, so DataLoader(num_workers>0) works under 'spawn'.")

    # --- 4. Amplitude: the MHD effect is the signal --------------------------
    print("\nAmplitude by condition (mV, over the windowed records):")
    # labels_df is aligned positionally with the dataset, so one pass suffices.
    extremes: dict[str, list[float]] = {}
    for position, condition in enumerate(frame["condition"]):
        signal = windowed[position]["signal"]
        low, high = extremes.setdefault(condition, [float("inf"), float("-inf")])
        extremes[condition] = [
            min(low, float(signal.min())), max(high, float(signal.max()))
        ]
    for condition in sorted(extremes):
        low, high = extremes[condition]
        print(f"  {condition:10s} {low:+7.2f} .. {high:+7.2f}")
    print("  Excursions past the devices' +/-6 mV and +/-2.4 mV nominal ranges are")
    print("  the MHD distortion plus per-channel baseline offset — the phenomenon")
    print("  under study, which is why amplitude_range_mv is +/-35 and no record is")
    print("  excluded for it.")

    # --- 5. Subject grouping ------------------------------------------------
    print(f"\nSubjects in this split: {frame['subject_key'].nunique()} "
          f"over {frame['scanner_subject_slot'].nunique()} filename slots")
    multi = (
        frame.groupby("subject_key")["scanner_subject_slot"].nunique().loc[lambda s: s > 1]
    )
    if len(multi):
        print("  Subjects recorded in more than one scanner (same person, different")
        print("  filename subject number — folds keep them together):")
        for key in multi.index:
            slots = sorted(set(frame.loc[frame["subject_key"] == key,
                                         "scanner_subject_slot"]))
            print(f"    {key}  ->  {slots}")
    else:
        print("  (none in this split — they are in whichever fold holds that subject)")

    disagree = frame[frame["position_disagrees"].astype(bool)]
    if len(disagree):
        print("\nRecords whose filename and header positions disagree (both kept):")
        for _, row in disagree.iterrows():
            print(f"  {row['record_id']}: filename {row['position']!r} vs "
                  f"header {row['position_header']!r}")

    print("\nNote: the README, PhysioNet page and CinC paper all say 43 records /")
    print("      23 subjects / 203 min. The shipped release has 53 records and")
    print("      226.6 min, and 26 subjects by demographics. Every figure ECGBench")
    print("      reports is recomputed from the checksum-verified files.")


if __name__ == "__main__":
    main()
