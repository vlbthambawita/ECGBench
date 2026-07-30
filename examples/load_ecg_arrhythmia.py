#!/usr/bin/env python3
"""
Example: PhysioNet ecg-arrhythmia (Chapman-Shaoxing + Ningbo) with labels.

45,152 twelve-lead records, one per patient. Diagnoses are SNOMED-CT codes and
are **multi-label** — a mean of 2.0 codes per record.

Labels here come from `ecgbench_metadata.csv`, which ECGBench generates by
scanning all 45,152 WFDB headers (the dataset ships no metadata CSV of its own).
That means you must run the split pipeline once before labels=True works:

    ecgbench splits --dataset ecg_arrhythmia --data-path /path/to/1.0.0/

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_ecg_arrhythmia.py --data-path /path/to/ecg-arrhythmia/1.0.0/
"""

import argparse
from collections import Counter

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError


def main():
    parser = argparse.ArgumentParser(description="Load ecg-arrhythmia with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top", type=int, default=10, help="How many diagnoses to list")
    args = parser.parse_args()

    config = load_config("ecg_arrhythmia")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print()

    try:
        dataset = ECGDataset(
            "ecg_arrhythmia",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            labels=True,
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        print("\nRun the split pipeline once to generate the metadata CSV:")
        print(f"  ecgbench splits --dataset ecg_arrhythmia --data-path {args.data_path}")
        return

    print(f"Records:  {len(dataset)}")

    # --- One sample ---------------------------------------------------------
    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nLabel fields: {list(labels)}")
    print("\nFirst record:")
    print(f"  record:             {sample['record_id']}")
    print(f"  signal:             {tuple(sample['signal'].shape)}")
    print(f"  dx (SNOMED codes):  {labels['dx']}")
    print(f"  dx_acronyms:        {labels['dx_acronyms']}")
    print(f"  primary_dx_acronym: {labels['primary_dx_acronym']}  (stratification label)")
    print(f"  age / sex:          {labels['age']} / {labels['sex']}")

    # --- Distribution over the split ----------------------------------------
    # dx is a comma-separated code list, so split it to count properly.
    acronyms = dataset.labels_df["dx_acronyms"].fillna("")
    per_record = acronyms.map(lambda s: [c for c in str(s).split(",") if c])
    counts = Counter(code for row in per_record for code in row)

    print(f"\nDiagnoses in this split (multi-label, mean "
          f"{per_record.map(len).mean():.2f} per record):")
    for code, n in counts.most_common(args.top):
        print(f"  {code:12s} {n:6d}")
    print(f"  ... {len(counts)} distinct codes in total")
    print("\nNote: codes absent from the dataset's ConditionNames_SNOMED-CT.csv keep")
    print("      their raw numeric SNOMED code as the acronym.")

    # --- Batching -----------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print("\nFirst batch:")
    print(f"  signal: {tuple(batch['signal'].shape)} {batch['signal'].dtype}")
    print(f"  labels: list of {len(batch['labels'])} dicts")
    for record_id, row in zip(batch["record_id"], batch["labels"], strict=True):
        print(f"    {record_id}: {row['dx_acronyms']}")


if __name__ == "__main__":
    main()
