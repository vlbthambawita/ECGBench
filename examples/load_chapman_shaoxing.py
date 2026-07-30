#!/usr/bin/env python3
"""
Example: Chapman-Shaoxing (figshare release) with labels.

10,646 twelve-lead records, one per patient, as CSV rather than WFDB — and stored
in microvolts, which the config converts via signal_unit_scale so everything
downstream sees millivolts.

Labels are unusually rich for an open ECG dataset: a single-label rhythm class,
a space-separated multi-label beat/condition annotation, demographics, and eleven
automated measurements.

Prerequisites:
  - pip install ecgbench[torch]
  - Fetch the dataset:
      python examples/download_chapman_figshare.py --dest /path/to/chapman-figshare/

Usage:
  python examples/load_chapman_shaoxing.py --data-path /path/to/chapman-figshare/
"""

import argparse
from collections import Counter

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError


def main():
    parser = argparse.ArgumentParser(description="Load Chapman-Shaoxing with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("chapman_shaoxing")
    print(f"Dataset:  {config.name} (figshare v{config.version})")
    print(f"Format:   {config.signal_format}, scaled by {config.signal_unit_scale} to mV")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print()

    try:
        dataset = ECGDataset(
            "chapman_shaoxing",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nLabel fields: {list(labels)}")
    print("\nFirst record:")
    print(f"  record:   {sample['record_id']}")
    print(f"  signal:   {tuple(sample['signal'].shape)}  "
          f"min {sample['signal'].min():.2f} max {sample['signal'].max():.2f} mV")
    print(f"  Rhythm:   {labels['Rhythm']}   (single label, stratification target)")
    print(f"  Beat:     {labels['Beat']!r}   (space-separated, multi-label)")
    print(f"  age/sex:  {labels['PatientAge']} / {labels['Gender']}")
    print(f"  measured: rate={labels['VentricularRate']} QRS={labels['QRSDuration']}ms "
          f"QTc={labels['QTCorrected']}ms axis R={labels['RAxis']}")

    rhythms = dataset.labels_df["Rhythm"]
    print("\nRhythm distribution over this split (single label, so these sum):")
    for name, n in rhythms.value_counts().items():
        print(f"  {name:6s} {n:6d}")

    beats = dataset.labels_df["Beat"].fillna("NONE").map(
        lambda s: [c for c in str(s).split() if c and c != "NONE"]
    )
    counts = Counter(code for row in beats for code in row)
    print(f"\nBeat annotations (multi-label, mean {beats.map(len).mean():.2f} per record, "
          f"{int((beats.map(len) == 0).sum())} with none):")
    for code, n in counts.most_common(8):
        print(f"  {code:8s} {n:6d}")
    print(f"  ... {len(counts)} distinct codes")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print("\nFirst batch:")
    print(f"  signal: {tuple(batch['signal'].shape)} {batch['signal'].dtype}")
    print(f"  labels: list of {len(batch['labels'])} dicts")
    for record_id, row in zip(batch["record_id"], batch["labels"], strict=True):
        print(f"    {record_id}: {row['Rhythm']:6s} {row['Beat']}")

    print("\nNote: leads are in the standard order here (I, II, III, aVR, aVL, aVF,")
    print("      V1-V6), unlike MIMIC-IV-ECG which transposes aVF and aVL. The same")
    print("      recordings also appear in the PhysioNet ecg_arrhythmia dataset in")
    print("      WFDB form — do not train on one and evaluate on the other.")


if __name__ == "__main__":
    main()
