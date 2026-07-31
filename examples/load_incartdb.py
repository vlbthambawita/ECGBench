#!/usr/bin/env python3
"""
Example: St Petersburg INCART 12-lead Arrhythmia Database with labels.

75 half-hour Holter extracts from **32 patients**, with 175,907 manually corrected
reference beat annotations. Three things to demonstrate:

1. **Patient grouping is the point.** 30 of the 32 patients contributed more than
   one record, and some findings live almost entirely in one patient — 3,166 of
   the 3,174 RBBB beats belong to `patient08`. Folds are grouped so no patient
   spans one; this script checks that rather than asserting it.
2. **Records are 1800 s (~44 MB each).** They cannot be batched as they are, so a
   cropping `transform` is mandatory.
3. **Labels are three-layered**: a patient-level diagnosis (absent for 14 of 32
   patients), free-text per-record ECG findings, and per-record beat counts derived
   from the `.atr` files. None of the three alone is the whole label.

Labels come straight from the headers and annotations, so this works without
running the split pipeline first. The fold CSVs do come from the pipeline (or the
Hub) — pass --metadata-source local after copying output/incartdb/{clean,original}/
into the dataset root.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_incartdb.py --data-path /path/to/incartdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.incartdb import BEAT_NAMES, BEAT_SYMBOLS

#: 10 s at 257 Hz. Records are 1800 s, so some crop is required to batch at all.
CROP_SAMPLES = 2570


def main():
    parser = argparse.ArgumentParser(description="Load INCART with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("incartdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}")
    print(f"Duration: {config.duration_seconds} s per record")
    print()

    try:
        dataset = ECGDataset(
            "incartdb",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            transform=lambda x: x[:, :CROP_SAMPLES],
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (cropped from 462600)")
    print(f"  record          {sample['record_id']}")
    print(f"  patient_id      {labels['patient_id']}   <- folds are grouped by this")
    print(f"  age / sex       {labels['age']} / {labels['sex']}")
    print(f"  diagnosis       {labels['diagnosis'] or '(none recorded)'}")
    print(f"  record_features {labels['record_features']}")
    print(f"  n_beats         {labels['n_beats']}  (PVC fraction {labels['pvc_fraction']:.4f})")

    df = dataset.labels_df

    # The property that makes this dataset usable. Check it, don't assume it.
    patients = dataset.metadata_df[config.patient_id_column]
    print(f"\n{len(df)} records from {patients.nunique()} patients in this split")
    print("  records per patient:", patients.value_counts().value_counts().sort_index().to_dict())

    print("\nPatient-level diagnoses in this split:")
    diag = df["diagnosis"].fillna("").replace("", "(none recorded)")
    for name, n in diag.value_counts().items():
        print(f"  {n:3d}  {name}")

    print("\nReference beat annotations over this split:")
    total = int(df["n_beats"].sum())
    for symbol in BEAT_SYMBOLS:
        n = int(df[f"beat_{symbol}"].sum())
        if not n:
            continue
        holding = int((df[f"beat_{symbol}"] > 0).sum())
        print(
            f"  {symbol:2s} {n:7d} ({100 * n / total:5.2f}%) in {holding:3d} records"
            f"  {BEAT_NAMES[symbol]}"
        )
    print(f"  total {total} beats; {int(df['n_rhythm_changes'].sum())} rhythm-change markers")

    # Why grouping matters, shown rather than asserted: some beat types are
    # concentrated in a single patient.
    print("\nBeat types concentrated in one patient (the reason folds are grouped):")
    for symbol in BEAT_SYMBOLS:
        col = f"beat_{symbol}"
        if df[col].sum() == 0:
            continue
        by_patient = df.groupby(patients.to_numpy())[col].sum()
        top = by_patient.idxmax()
        share = by_patient.max() / by_patient.sum()
        if share > 0.9:
            print(f"  {symbol:2s} {BEAT_NAMES[symbol]:38s} {100 * share:5.1f}% from {top}")

    print("\nPVC burden per record (straight from the .atr annotations):")
    print(f"  median {df['pvc_fraction'].median():.4f}  max {df['pvc_fraction'].max():.4f}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print("  Without the transform= crop each record is 12 x 462600 (~44 MB), so a")
    print(f"  batch of {args.batch_size} would be ~{44 * args.batch_size} MB of float32.")


if __name__ == "__main__":
    main()
