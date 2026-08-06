#!/usr/bin/env python3
"""
Example: IKEM — 98,130 Prague hospital ECGs, 8 leads, and no diagnoses.

The second dataset in the catalogue whose fold CSVs are **not published**, and
the first withheld by its licence rather than by a data use agreement: the
release ships under CC BY-NC-ND 4.0, whose NoDerivatives term makes republishing
a derived fold table on a public repo legally unclear. So `metadata_source="hf"`
raises `SplitsNotPublishedError` and this example defaults to `"local"`. The
split is a recipe — see the three steps below.

Four things worth seeing rather than reading:

1. **Eight leads, and the order is the most unusual in the catalogue.** Only the
   independent leads are stored, as `V1-V6, II, I` — precordial first, and II
   before I. `signal[0]` is V1, not lead I. III/aVR/aVL/aVF are exact linear
   combinations of II and I and are not stored; ECGBench does not synthesise
   them. Deriving one is shown below.
2. **No diagnoses ship.** The release carries demographics and two cart-measured
   rates and nothing else. It looks like a 98,130-record classification corpus
   and is not one.
3. **-1 is a missing-value sentinel in every numeric column**, and 89.6% of
   weights are missing. Read literally the cohort's mean weight is negative.
4. **It is 8.192 s, not the 10 s the release states.** 4,096 samples at 500 Hz.

Three-step recipe (this is how the split is distributed):

    ecgbench splits --dataset ikem --data-path /path/to/IKEM_dataset_v1.0.0/
    python -c "from ecgbench import verify_splits; verify_splits('ikem', 'output/ikem')"
    cp -r output/ikem/clean output/ikem/original /path/to/IKEM_dataset_v1.0.0/

The last step is what lets `metadata_source="local"` find the fold CSVs: in local
mode `data_path` serves as both the signal root and the splits root, so the fold
tree has to sit inside the dataset directory.

Prerequisites:
  - pip install ecgbench[torch,hdf5]
  - A local copy of the release (exams.csv + three exams_part_*.hdf5).
  - The three steps above, run once.

Usage:
  python examples/load_ikem.py --data-path /path/to/IKEM_dataset_v1.0.0/
"""

import argparse

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError

#: Every record is stored as 4,096 samples, but 48 hold only 2,500 real samples
#: (5.0 s) zero-padded into that. Sized to the shortest so no record returns pure
#: padding; real_length_samples is what tells you which ones are short.
WINDOW = (0, 2500)


def main():
    parser = argparse.ArgumentParser(description="Load IKEM with its metadata")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    # Defaults to local, unlike every other example: the Hub has no IKEM folds.
    parser.add_argument("--metadata-source", default="local", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("ikem")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Licence:  {config.license}")
    print(f"Split:    {args.split} ({args.version}) from {args.metadata_source}")
    print(f"Leads:    {config.leads} — {config.lead_names}")
    print(
        f"Rate:     {config.default_sampling_rate} Hz, {config.duration_seconds} s "
        f"({config.validation.expected_samples[500]} samples)"
    )
    print(f"Publish:  publish_fold_csvs={config.publish_fold_csvs}")
    print()

    if args.metadata_source == "hf":
        print("Note: this will raise SplitsNotPublishedError — the Hub has no IKEM folds.")

    try:
        dataset = ECGDataset(
            "ikem",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            window=WINDOW,
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return
    except FileNotFoundError as e:
        print(f"Fold CSVs not found: {e}")
        print("\nRun the three-step recipe in this file's docstring first.")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    print("               ^ EIGHT leads, not twelve")
    labels = sample["labels"]
    print(f"  record        {sample['record_id']}")
    print(f"  patient       {labels['patient_id'][:12]}...  (SHA-1 surrogate)")
    print(f"  age / sex     {labels['age']} / {labels['sex']}")
    print(f"  weight/height {labels['weight']} / {labels['height']}   <- NaN means -1 in source")
    print(
        f"  rates         ventricular {labels['ventricular_rate']}, "
        f"atrial {labels['atrial_rate']}"
    )
    print(f"  acquired      {labels['acquisition_date']}")
    print(
        f"  real length   {labels['real_length_samples']} samples "
        f"({labels['real_duration_seconds']:.3f} s)"
    )
    print(f"  stratify      {labels['stratify_class']}   <- a rate band, not a diagnosis")

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print("\nTHE LEAD ORDER — the most unusual in the catalogue:")
    for position, name in enumerate(config.lead_names):
        print(f"  signal[{position}] = {name}")
    print("  signal[0] is V1. Reaching for signal[0] as lead I gives a precordial lead.")
    print("  III/aVR/aVL/aVF are NOT stored — they are linear combinations of II and I.")

    # Derive a missing augmented lead, since the maths is exact.
    raw = dataset[0]["signal"].numpy()
    lead_ii = raw[config.lead_names.index("II")]
    lead_i = raw[config.lead_names.index("I")]
    print("\nDeriving the four leads the release drops (exact, not approximate):")
    print(f"  III  = II - I        -> peak {np.abs(lead_ii - lead_i).max():.3f} mV")
    print(f"  aVR  = -(I + II) / 2 -> peak {np.abs(-(lead_i + lead_ii) / 2).max():.3f} mV")
    print(f"  aVL  = I - II / 2    -> peak {np.abs(lead_i - lead_ii / 2).max():.3f} mV")
    print(f"  aVF  = II - I / 2    -> peak {np.abs(lead_ii - lead_i / 2).max():.3f} mV")

    print("\nNO DIAGNOSES SHIP — this is not a classification dataset:")
    print(f"  columns available: {sorted(df.columns)}")
    print("  The paper's diagnostic labels are not part of the Zenodo release. What")
    print("  this supports is age/sex estimation, rate regression and pretraining.")

    print("\nTHE SENTINEL TRAP — -1 means missing, in every numeric column:")
    for column in ("age", "weight", "height", "ventricular_rate", "atrial_rate"):
        n_missing = int(df[column].isna().sum())
        print(f"  {column:17s} missing {n_missing:6d} ({100 * n_missing / len(df):5.1f}%)")
    print(f"  sex               missing {int(df['is_male'].isna().sum()):6d}")
    print("  Every one of these is a literal -1 in the source, so notna() on the raw")
    print("  CSV reports 100% complete and every mean comes out wrong.")
    weight = pd.to_numeric(df["weight"])
    print(
        f"  mean weight, sentinels as NaN: {weight.mean():.1f} kg "
        f"(n={int(weight.notna().sum())})"
    )
    print("  mean weight, read literally:   about -76 kg")

    print("\nValues that survive the sentinel filter still deserve suspicion:")
    age = pd.to_numeric(df["age"])
    print(f"  age 0:      {int((age == 0).sum())} records")
    print(f"  age >= 100: {int((age >= 100).sum())} records")
    print(
        f"  full non-missing range:   {age.min():.0f}-{age.max():.0f}, "
        f"median {age.median():.0f}"
    )
    print("  Those extremes are left as they are — they are not sentinels, and")
    print("  guessing which are real is not the loader's job.")

    print("\nStratification is the cart's ventricular rate, banded:")
    for name, n in df["stratify_class"].value_counts().items():
        print(f"  {name:8s} {n:6d}")
    print("  A rate band is a MEASUREMENT: 75 bpm is as compatible with atrial")
    print("  fibrillation as with sinus rhythm. It exists to keep the folds balanced.")

    print("\nPatients repeat heavily, which is why folds are grouped:")
    repeats = df["patient_id"].value_counts()
    print(f"  {df['patient_id'].nunique()} patients for {len(df)} records")
    print(f"  with more than one record: {int((repeats > 1).sum())}")
    print(f"  most from one patient: {int(repeats.max())}")
    print("  Across the whole release 88.6% of records share a patient with another.")

    print("\nAcquisition is concentrated in two years:")
    by_year = df["acquisition_year"].value_counts().sort_index()
    for year, n in by_year.items():
        if n > len(df) * 0.01:
            print(f"  {int(year)}: {n:6d}")
    print(f"  full range {int(by_year.index.min())}-{int(by_year.index.max())}; the tail is tiny")

    print("\n48 records in the release are zero-padded from 2,500 real samples:")
    short = df["real_length_samples"] < 4096
    print(f"  in this split: {int(short.sum())}")
    print(f"  WINDOW={WINDOW} is sized to them, so no record returns pure padding.")

    # Age regression is what this release actually supports.
    target = age[age.notna()]
    print(
        f"\nAge-regression target: n={len(target)} of {len(df)} "
        f"({100 * len(target) / len(df):.1f}% usable)"
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  (8 leads)")

    print("\nSelecting leads by name is the only safe way to combine IKEM with anything:")
    by_name = ECGDataset(
        "ikem",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=["I", "II", "V1"],
    )
    print(f"  leads=['I','II','V1'] -> {tuple(by_name[0]['signal'].shape)}")
    print("  Asking for a lead IKEM does not store raises rather than guessing:")
    try:
        ECGDataset(
            "ikem",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            window=WINDOW,
            leads=["aVR"],
        )
        print("    leads=['aVR'] was accepted — THIS WOULD BE THE BUG")
    except ValueError as e:
        print(f"    leads=['aVR'] -> {str(e)[:80]}...")

    print("\nNote on the records the clean version drops: they fail amplitude_outlier")
    print("(a lead beyond +-10 mV) or missing_leads. The worst peaks land exactly on")
    print("32.767 mV, which is int16 full scale — a railed ADC, not a real amplitude.")
    print("Use version='original' to get them back, with is_valid and quality_issues.")


if __name__ == "__main__":
    main()
