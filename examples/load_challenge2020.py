#!/usr/bin/env python3
"""
Example: PhysioNet/CinC Challenge 2020 with labels.

43,101 twelve-lead records pooled from six source cohorts. Four things make this
dataset different from most of ECGBench, and this script demonstrates all four:

1. **It is contained in Challenge 2021.** All 43,101 records are in that release
   too, bit-identical. Never train on one year and evaluate on the other.
2. **It is a meta-dataset.** PTB-XL, PTBDB, INCART and CPSC-2018 are all inside
   it, renamed. The `source` label says which cohort a record came from — use it
   to exclude a cohort you plan to evaluate on.
3. **Sampling rate and record length vary per record** (257/500/1000 Hz, 5 s to
   1800 s). Records must be cropped before they can be batched, and rate is a
   label to filter on rather than a parameter to pass.
4. **Labels are multi-label SNOMED-CT codes** with no clinically meaningful
   primary, and 631 records ship a repeated code inside their own `#Dx` list.
   The loader deduplicates; `stratify_dx` exists only so folds can be stratified
   — train on `dx`.

Labels come straight from the WFDB headers, so unlike ecg_arrhythmia this works
without running the split pipeline first. The fold CSVs still come from the Hub
(or from --version/--split of a local run).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_challenge2020.py --data-path /path/to/challenge-2020/1.0.2/
"""

import argparse
from collections import Counter

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.challenge2020 import AGE_SENTINELS, load_dx_mapping

#: Records are variable length, so every signal is windowed to this many samples.
#: 2500 is 5 s at the nominal 500 Hz — and the length of the shortest record in
#: the dataset, so this window fits all 43,101.
WINDOW = (0, 2500)


def main():
    parser = argparse.ArgumentParser(description="Load Challenge 2020 with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top", type=int, default=10, help="How many diagnoses to list")
    args = parser.parse_args()

    config = load_config("challenge2020")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Leads:    {config.lead_names}")
    print(f"Rates:    {config.sampling_rates} (per record — see below)")
    print()

    try:
        dataset = ECGDataset(
            "challenge2020",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            # window=(start, length) is read at load time rather than cropped
            # afterwards, and unlike a lambda transform it survives a DataLoader
            # with num_workers>0 under the "spawn" start method.
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
    print(f"  record       {sample['record_id']}")
    print(f"  source       {labels['source']}")
    print(f"  dx           {labels['dx']}          <- ground truth, multi-label")
    print(f"  dx_abbrev    {labels['dx_abbreviations']}")
    print(f"  scored_dx    {labels['scored_dx']}")
    strat = labels["stratify_dx_abbreviation"]
    print(f"  stratify_dx  {strat}   <- folds only, never train on this")
    print(f"  age / sex    {labels['age']} / {labels['sex']}")
    print(f"  rate         {labels['sampling_rate']} Hz, {labels['n_samples']} samples")

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print(f"\nSource cohorts in this split ({len(df)} records):")
    for source, n in df["source"].value_counts().items():
        print(f"  {source:22s} {n:6d}  ({100 * n / len(df):5.2f}%)")

    print("\nSampling rate is a PER-RECORD property, not a parameter:")
    for rate, n in df["sampling_rate"].value_counts().items():
        print(f"  {rate:5d} Hz  {n:6d}")
    print("  ECGDataset(sampling_rate=257) raises — filter labels_df instead:")
    # labels_df carries a RangeIndex aligned row-for-row with metadata_df, so the
    # record IDs for a boolean mask come from metadata_df, not from the index.
    slow = dataset.metadata_df.loc[(df["sampling_rate"] == 257).to_numpy(), config.record_id_column]
    print(f"  {len(slow)} records at 257 Hz, e.g. {list(slow[:3])}")

    print(f"\nMulti-label: mean {df['n_dx'].mean():.2f} codes/record, max {df['n_dx'].max()}")
    print(f"  records with no challenge-scored code: {(df['n_scored_dx'] == 0).sum()}")

    # Age is shipped with sentinels, which the loader deliberately does not hide.
    age = pd.to_numeric(df["age"], errors="coerce")
    real = age[~df["age"].astype(str).isin(AGE_SENTINELS) & age.notna()]
    print(f"\nAge needs the sentinels {AGE_SENTINELS} filtered out first:")
    print(f"  raw min/max      {age.min():.0f} / {age.max():.0f}   <- 300 means 'over 89'")
    print(f"  after filtering  {real.min():.0f} / {real.max():.0f}, mean {real.mean():.1f}")
    print(f"  sentinel rows {int(df['age'].astype(str).isin(AGE_SENTINELS).sum())}, "
          f"missing {int(age.isna().sum())}")

    counts = Counter(code for dx in df["dx"].fillna("") for code in str(dx).split(",") if code)
    mapping = load_dx_mapping()
    print(f"\nTop {args.top} diagnoses (records carrying each; sums past {len(df)}):")
    for code, n in counts.most_common(args.top):
        row = mapping.loc[code] if code in mapping.index else None
        abbrev = row["abbreviation"] if row is not None else "?"
        name = row["dx_name"] if row is not None else "?"
        flag = "scored" if row is not None and row["scored"] else "      "
        print(f"  {abbrev:8s} {flag}  {n:6d}  {name}")

    # Multi-hot target over the 27 scored classes — the challenge's own task.
    # NOTE: 2021 scored 30 classes, a different subset. Use the year you report.
    scored_codes = list(mapping.index[mapping["scored"]])
    code_lists = df["dx"].fillna("").astype(str).str.split(",")
    targets = pd.DataFrame(
        {c: code_lists.apply(lambda lst, c=c: int(c in lst)) for c in scored_codes},
        index=df.index,
    )
    targets.columns = [mapping.loc[c, "abbreviation"] for c in scored_codes]
    print(f"\nMulti-hot target over the {len(scored_codes)} scored classes: {targets.shape}")
    print(f"  positives per record: mean {targets.sum(axis=1).mean():.2f}")
    print(f"  all-zero rows: {(targets.sum(axis=1) == 0).sum()} (no scored code)")

    # Batching only works because of the crop above.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print("  Without window= this raises — the split mixes 5 s, 10 s, 120 s and")
    print("  1800 s records, and torch cannot stack tensors of differing width.")

    print("\nLeakage warning: every record here is also in challenge2021, and this")
    print("dataset CONTAINS PTB-XL, PTBDB, INCART and CPSC-2018. To evaluate on one")
    print("of those, drop its cohort:")
    kept = df[~df["source"].isin(["ptb-xl"])]
    print(f"  df[df.source != 'ptb-xl'] -> {len(kept)} of {len(df)} records")


if __name__ == "__main__":
    main()
