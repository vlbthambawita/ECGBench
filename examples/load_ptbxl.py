#!/usr/bin/env python3
"""
Example: PTB-XL with labels.

PTB-XL is the richest label source ECGBench ships: SCP-ECG statements, the
diagnostic superclass and subclass above them, plus demographics and the
cardiologist's report text. `labels=True` attaches all of it.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of PTB-XL. Labels are NOT on the HuggingFace Hub — only fold
    CSVs are — so labels=True needs the real dataset directory.

Usage:
  python examples/load_ptbxl.py --data-path /path/to/ptb-xl/1.0.3/
"""

import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels.ptbxl import SUPERCLASSES, multi_hot


def main():
    parser = argparse.ArgumentParser(description="Load PTB-XL with labels via ECGBench")
    parser.add_argument("--data-path", default=None, help="Path to PTB-XL dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--sampling-rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("ptbxl")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Leads:    {config.leads}")
    print(f"Rates:    {config.sampling_rates} Hz")
    print(f"Split:    {args.split} ({args.version}) @ {args.sampling_rate} Hz")
    print()

    dataset = ECGDataset(
        "ptbxl",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        sampling_rate=args.sampling_rate,
        labels=True,
    )
    print(f"Records:  {len(dataset)}")

    # --- What a single sample carries ---------------------------------------
    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:  {sorted(sample.keys())}")
    print(f"Label fields: {list(labels)}")
    print("\nFirst record:")
    print(f"  ecg_id:             {sample['record_id']}")
    print(f"  signal:             {tuple(sample['signal'].shape)}")
    print(f"  scp_codes:          {labels['scp_codes']}")
    print(f"  superclasses:       {labels['superclasses']}")
    print(f"  subclasses:         {labels['subclasses']}")
    print(f"  primary_superclass: {labels['primary_superclass']}  (stratification only)")
    print(f"  age / sex:          {labels['age']} / {labels['sex']}  (sex: 0=male, 1=female)")
    print(f"  report:             {labels['report']!r}")

    # --- Distribution over the split ----------------------------------------
    # dataset.labels_df is the whole split's labels, aligned row-for-row with
    # dataset.metadata_df — use it instead of iterating the Dataset.
    split_labels = dataset.labels_df["superclasses"]
    counts = Counter(name for row in split_labels for name in row)
    print("\nSuperclass distribution (multi-label, so these overlap):")
    for name in SUPERCLASSES:
        print(f"  {name:5s} {counts[name]:6d}")
    print(f"  no superclass:     {int((split_labels.map(len) == 0).sum())}")
    print(f"  >1 superclass:     {int((split_labels.map(len) > 1).sum())}")

    # --- Batching, and turning labels into training targets -----------------
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
    )
    batch = next(iter(loader))

    # ecg_collate_fn keeps dicts uncollated, so batch["labels"] is a list of
    # per-record dicts. Encode the field you actually want to train on.
    targets = torch.from_numpy(
        multi_hot([r["superclasses"] for r in batch["labels"]])
    )

    print("\nFirst batch:")
    print(f"  signal:  {tuple(batch['signal'].shape)} {batch['signal'].dtype}")
    print(f"  labels:  list of {len(batch['labels'])} dicts")
    print(f"  targets: {tuple(targets.shape)} over {SUPERCLASSES}")
    for ecg_id, row, target in zip(
        batch["record_id"].tolist(), batch["labels"], targets, strict=True
    ):
        shown = ", ".join(row["superclasses"]) or "-"
        print(f"    ecg_id {ecg_id:>6}: {shown:22s} {target.tolist()}")

    # A record with several superclasses, to show the task really is multi-label
    multi = next(
        (
            (i, row)
            for i, row in zip(dataset.metadata_df["ecg_id"], dataset.labels_df.itertuples(),
                              strict=True)
            if len(row.superclasses) > 1
        ),
        None,
    )
    if multi:
        ecg_id, row = multi
        print(f"\n  e.g. multi-label: ecg_id {ecg_id} -> {', '.join(row.superclasses)}"
              f"  (subclasses: {', '.join(row.subclasses)})")


if __name__ == "__main__":
    main()
