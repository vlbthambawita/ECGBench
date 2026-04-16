#!/usr/bin/env python3
"""
Example: Load PTB-XL dataset using ECGBench.

Prerequisites:
  - pip install ecgbench[torch]
  - PTB-XL data at a local path, or let ECGBench auto-download

Usage:
  python examples/load_ptbxl.py --data-path /path/to/ptb-xl/1.0.3/
  python examples/load_ptbxl.py  # auto-download
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config


def main():
    parser = argparse.ArgumentParser(description="Load and inspect PTB-XL via ECGBench")
    parser.add_argument("--data-path", default=None, help="Path to PTB-XL dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Show config info
    config = load_config("ptbxl")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Leads:    {config.leads}")
    print(f"Rates:    {config.sampling_rates} Hz")
    print(f"Split:    {args.split}")
    print(f"Version:  {args.version}")
    print()

    # Load dataset
    dataset = ECGDataset(
        "ptbxl",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        sampling_rate=500,
    )
    print(f"Records:  {len(dataset)}")

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
    )

    # Inspect first batch
    batch = next(iter(loader))
    print("\nFirst batch:")
    print(f"  signal shape: {batch['signal'].shape}")  # (B, 12, 5000)
    print(f"  signal dtype: {batch['signal'].dtype}")
    print(f"  record_ids:   {batch['record_id'][:3]}...")
    print(f"  keys:         {list(batch.keys())}")

    # Signal stats
    signal = batch["signal"]
    print("\nSignal stats:")
    print(f"  min:  {signal.min().item():.4f}")
    print(f"  max:  {signal.max().item():.4f}")
    print(f"  mean: {signal.mean().item():.4f}")
    print(f"  std:  {signal.std().item():.4f}")


if __name__ == "__main__":
    main()
