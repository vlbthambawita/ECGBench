#!/usr/bin/env python3
"""
Example: MIMIC-IV-ECG Demo — a dataset with no labels.

Worth its own example precisely because `labels=True` cannot work here. The demo
subset ships `record_list.csv` with identifiers and timestamps only; machine
measurements and report text belong to the full credentialed release. ECGBench
says so explicitly rather than returning empty columns.

The splits are still useful: 659 records from 92 patients, grouped so no patient
spans folds — which matters, because records per patient run from 1 to 52.

Prerequisites:
  - pip install ecgbench[torch]

Usage:
  python examples/load_mimic_iv_ecg_demo.py --data-path /path/to/mimic-iv-ecg-demo/0.1/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelsUnavailableError


def main():
    parser = argparse.ArgumentParser(description="Load MIMIC-IV-ECG Demo via ECGBench")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("mimic_iv_ecg_demo")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print()

    # --- What asking for labels does here -----------------------------------
    print("Asking for labels=True:")
    try:
        ECGDataset(
            "mimic_iv_ecg_demo",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            labels=True,
        )
    except LabelsUnavailableError as e:
        print(f"  LabelsUnavailableError: {e}\n")

    # --- Signals and splits work fine ---------------------------------------
    dataset = ECGDataset(
        "mimic_iv_ecg_demo",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
    )
    print(f"Records:  {len(dataset)}")

    meta = dataset.metadata_df
    per_patient = meta.groupby("subject_id").size()
    print(f"Patients: {meta.subject_id.nunique()}")
    print(f"Records per patient in this split: min {per_patient.min()}, "
          f"max {per_patient.max()}, median {int(per_patient.median())}")

    # The guarantee this dataset's split actually provides
    spanning = meta.groupby("subject_id").fold.nunique().gt(1).sum()
    print(f"Patients spanning more than one fold: {spanning}  (must be 0)")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print("\nFirst batch:")
    print(f"  signal:  {tuple(batch['signal'].shape)} {batch['signal'].dtype}")
    print(f"  keys:    {sorted(batch.keys())}")
    print(f"  studies: {batch['record_id'][:4].tolist()}...")

    print("\nNote: every header in this dataset stores leads in a non-standard order —")
    print("      I, II, III, aVR, aVF, aVL, V1-V6, with aVF and aVL transposed.")
    print("      ECGBench does not reorder them, so signal[4] is aVF here and aVL")
    print("      in PTB-XL. Reorder yourself before training across both.")


if __name__ == "__main__":
    main()
