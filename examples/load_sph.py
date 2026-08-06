#!/usr/bin/env python3
"""
Example: SPH (Shandong Provincial Hospital) with AHA diagnostic statements.

25,770 twelve-lead ECGs from 24,666 patients, and the largest single-source ECG
dataset in the catalogue. Four things about it are worth seeing rather than
reading, and this script shows all four:

1. **It is the only HDF5 dataset in ECGBench.** One .h5 per record holding a
   single (12, N) float16 array named `ecg`, already in millivolts. Needs
   `pip install ecgbench[hdf5]`; nothing else changes at the API surface.
2. **The labels are AHA/ACC/HRS standardised statements, not a bespoke
   vocabulary** — 44 primary statements in 11 categories, each optionally
   qualified by one or more of 15 modifiers. `aha_primary_codes` is the ground
   truth; the multi-hot target below is how to use it.
3. **There is no primary diagnosis.** The release does not rank a record's
   statements, so `stratify_code` exists only to make folds stratifiable — never
   train on it.
4. **Record length varies from 10 s to 56 s** in 39 distinct lengths, so records
   cannot be batched without a window.

Labels come from the two shipped CSVs (metadata.csv joined against code.csv), so
this works without running the split pipeline first. The fold CSVs come from the
Hub (or from a local run with --metadata-source local).

Prerequisites:
  - pip install ecgbench[torch,hdf5]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.
  - records.tar.gz extracted: `tar -xf records.tar.gz` (it is an UNCOMPRESSED
    tar despite the name, so -xzf fails).

Usage:
  python examples/load_sph.py --data-path /path/to/SPH/
"""

import argparse
from collections import Counter

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.sph import LIST_SEPARATOR, load_code_table

#: Records are variable length, so every signal is windowed to this many samples.
#: 5000 is 10 s at 500 Hz — the length of the SHORTEST records in the dataset
#: (18,842 of them), so this window fits all 25,770. Anything longer raises
#: WindowOutOfRangeError on the shortest ones.
WINDOW = (0, 5000)


def main():
    parser = argparse.ArgumentParser(description="Load SPH with AHA labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("sph")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Format:   {config.signal_format}  <- the only one in ECGBench")
    print(f"Leads:    {config.lead_names}")
    print(f"Rate:     {config.default_sampling_rate} Hz (constant)")
    print()

    try:
        dataset = ECGDataset(
            "sph",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            # window=(start, length) is pushed into h5py's own slicing rather than
            # cropped afterwards, so a 56 s record decodes 10 s. Unlike a lambda
            # transform it also survives DataLoader(num_workers>0) under "spawn".
            window=WINDOW,
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    labels = sample["labels"]
    print(f"  record        {sample['record_id']}")
    print(f"  patient       {labels['patient_id']}")
    print(f"  aha_code      {labels['aha_code']}          <- as shipped")
    print(f"  primary codes {labels['aha_primary_codes']}          <- ground truth")
    print(f"  descriptions  {labels['aha_primary_descriptions']}")
    print(f"  categories    {labels['aha_primary_categories']}")
    print(f"  modifiers     {labels['aha_modifier_codes'] or '(none)'}")
    print(f"  stratify_code {labels['stratify_code']}   <- folds only, never train on this")
    print(f"  age / sex     {labels['age']} / {labels['sex']}")
    print(
        f"  duration      {labels['duration_seconds']} s  <- the FULL record, "
        f"not the {WINDOW[1] / config.default_sampling_rate:.0f} s window"
    )

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print(f"\nRecord length in this split ({len(df)} records):")
    dur = pd.to_numeric(df["duration_seconds"])
    print(f"  min {dur.min():.0f} s   median {dur.median():.0f} s   max {dur.max():.0f} s")
    print(f"  exactly 10 s: {int((dur == 10).sum())}   longer than 15 s: {int((dur > 15).sum())}")
    print("  n_samples comes from the metadata, which agrees with every HDF5 array,")
    print("  so nothing has to open a signal file to learn a length.")

    print(f"\nPatients: {df['patient_id'].nunique()} for {len(df)} records")
    repeats = df["patient_id"].value_counts()
    print(f"  with more than one record in this split: {int((repeats > 1).sum())}")
    print("  Folds are grouped on patient_id, so no patient spans two folds.")

    print(
        f"\nMulti-label: {int((df['n_primary_codes'] > 1).sum())} records carry more "
        f"than one statement (max {int(df['n_primary_codes'].max())})"
    )
    print(f"  strictly normal (code 1 and nothing else): {int(df['is_normal'].sum())}")

    # Category-level view first: 11 categories are readable, 44 codes are not.
    print("\nAHA category distribution (records carrying each; sums past the total):")
    codes = load_code_table(dataset.data_path)
    cat_counts = Counter(
        c
        for s in df["aha_primary_categories"].fillna("")
        for c in str(s).split(LIST_SEPARATOR)
        if c
    )
    for category, n in sorted(cat_counts.items()):
        example = codes[(codes["category"] == category) & ~codes["is_modifier"]]
        label = example["description"].iloc[0] if len(example) else ""
        print(f"  {category:9s} {n:6d}   e.g. {label}")
    print(f"  {'total':9s} {sum(cat_counts.values()):6d}   statement-record pairs")

    counts = Counter(
        c for s in df["aha_primary_codes"].fillna("") for c in str(s).split(LIST_SEPARATOR) if c
    )
    print("\nTop primary statements:")
    for code, n in counts.most_common(10):
        print(f"  {code:>4s} {n:6d}  {codes['description'].get(code, '?')}")
    print(f"  ... {len(counts)} distinct primary codes in this split")

    print("\nThe fold label is a rarest-code reduction, NOT the distribution above:")
    strat = df["stratify_code"].value_counts()
    print(f"  {len(strat)} classes, largest {strat.iloc[0]}, smallest {strat.iloc[-1]}")
    print("  (SPHSplitter pools codes under 10 records into OTHER before splitting)")

    age = pd.to_numeric(df["age"])
    print(f"\nAge {age.min():.0f}-{age.max():.0f}, mean {age.mean():.1f} — no sentinels,")
    print(f"no blanks, adults only. Sex: {df['sex'].value_counts().to_dict()}")

    # Multi-hot target over the 44 primary statements — the release's own task.
    primaries = [c for c in codes.index[~codes["is_modifier"]]]
    code_lists = df["aha_primary_codes"].fillna("").astype(str).str.split(LIST_SEPARATOR)
    targets = pd.DataFrame(
        {c: code_lists.apply(lambda lst, c=c: int(c in lst)) for c in primaries},
        index=df.index,
    )
    print(f"\nMulti-hot target over the {len(primaries)} primary statements: {targets.shape}")
    print(f"  positives per record: mean {targets.sum(axis=1).mean():.2f}")
    print(f"  all-zero rows: {int((targets.sum(axis=1) == 0).sum())} (every record is labelled)")

    # Batching only works because of the window above.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print(
        f"  Without window= this raises — this split alone mixes {dur.min():.0f} s to "
        f"{dur.max():.0f} s records,\n"
        "  and torch cannot stack differing widths."
    )

    print("\nNote on the 323 records the clean version drops: every one of them fails")
    print("amplitude_outlier and nothing else — a railed or artefact-dominated lead,")
    print("against a median per-record peak of 1.74 mV. Use version='original' to")
    print("get them back, with is_valid and quality_issues attached.")


if __name__ == "__main__":
    main()
