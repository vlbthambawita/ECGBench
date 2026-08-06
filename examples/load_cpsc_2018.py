#!/usr/bin/env python3
"""
Example: CPSC 2018 with labels.

The 6,877-record public training set of the China Physiological Signal Challenge
2018. Four things about it are worth seeing rather than reading, and this script
shows all four:

1. **Record length varies by a factor of 24** — 6 s to 144 s, 1,650 distinct
   lengths. Records cannot be batched without a window, and the window has to fit
   the shortest one.
2. **It is multi-label.** 476 of 6,877 records carry two or three of the nine
   classes. `dx` is the ground truth; the multi-hot target below is how to use it.
3. **There is no primary diagnosis.** CPSC's REFERENCE.csv distinguished
   First/Second/Third label, but the WFDB copy everyone uses sorted each record's
   codes by class index and dropped the distinction. `stratify_dx` exists only so
   folds can be stratified — never train on it.
4. **It is contained whole in Challenge 2020 and 2021**, byte-identically. If you
   trained on either, you cannot evaluate here.

Labels come straight from the WFDB headers, so this works without running the
split pipeline first. The fold CSVs still come from the Hub (or from a local run
with --metadata-source local).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_cpsc_2018.py --data-path /path/to/CPSC_2018/
"""

import argparse
from collections import Counter

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.cpsc_2018 import AGE_SENTINELS, CPSC_CLASSES

#: Records are variable length, so every signal is windowed to this many samples.
#: 3000 is 6 s at 500 Hz — the length of the SHORTEST record in the dataset
#: (A5277), so this window fits all 6,877. Anything longer raises
#: WindowOutOfRangeError on the short tail.
WINDOW = (0, 3000)


def main():
    parser = argparse.ArgumentParser(description="Load CPSC 2018 with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("cpsc_2018")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Leads:    {config.lead_names}")
    print(f"Rate:     {config.default_sampling_rate} Hz (constant, unlike the challenges)")
    print()

    try:
        dataset = ECGDataset(
            "cpsc_2018",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            # window=(start, length) is pushed into the wfdb reader rather than
            # cropped afterwards, so a 144 s record decodes 6 s. Unlike a lambda
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
    print(f"  record       {sample['record_id']}")
    print(f"  dx           {labels['dx']}          <- ground truth, multi-label")
    print(f"  dx_abbrev    {labels['dx_abbreviations']}")
    print(f"  dx_names     {labels['dx_names']}")
    strat = labels["stratify_dx_abbreviation"]
    print(f"  stratify_dx  {strat}   <- folds only, never train on this")
    print(f"  age / sex    {labels['age']} / {labels['sex']}")
    print(
        f"  duration     {labels['duration_seconds']} s  <- the FULL record, "
        f"not the {WINDOW[1] / config.default_sampling_rate:.0f} s window"
    )

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print(f"\nRecord length in this split ({len(df)} records):")
    dur = pd.to_numeric(df["duration_seconds"])
    print(f"  min {dur.min():.0f} s   median {dur.median():.0f} s   max {dur.max():.0f} s")
    print(f"  exactly 10 s: {int((dur == 10).sum())}   longer than 60 s: {int((dur > 60).sum())}")
    print("  The challenge page claims a 60 s maximum; the shipped files disagree.")

    print(
        f"\nMulti-label: {int((df['n_dx'] > 1).sum())} records carry more than one "
        f"class (max {int(df['n_dx'].max())})"
    )

    counts = Counter(a for s in df["dx_abbreviations"].fillna("") for a in str(s).split(",") if a)
    print(f"\nClass distribution (records carrying each; sums past {len(df)}):")
    for _, code, abbrev, name in CPSC_CLASSES:
        print(f"  {abbrev:5s} {code:>10s}  {counts.get(abbrev, 0):5d}  {name}")
    print(f"  {'':5s} {'':>10s}  {sum(counts.values()):5d}  total class-record pairs")

    print("\nThe fold label is a rarest-class reduction, NOT the distribution above:")
    for abbrev, n in df["stratify_dx_abbreviation"].value_counts().items():
        print(f"  {abbrev:5s} {n:5d}")

    # Age is shipped with a sentinel, which the loader deliberately does not hide.
    age = pd.to_numeric(df["age"], errors="coerce")
    real = age[~df["age"].astype(str).isin(AGE_SENTINELS) & age.notna()]
    print(f"\nAge needs the sentinel {AGE_SENTINELS} filtered out first:")
    print(f"  raw min/max      {age.min():.0f} / {age.max():.0f}")
    print(f"  after filtering  {real.min():.0f} / {real.max():.0f}, mean {real.mean():.1f}")
    print(
        f"  sentinel rows {int(df['age'].astype(str).isin(AGE_SENTINELS).sum())}, "
        f"missing {int(age.isna().sum())}"
    )
    print("  NB: ages over 89 are exact here; PhysioNet's copy rails them to 92.")

    # Multi-hot target over the nine classes — the challenge's own task.
    names = [abbrev for _, _, abbrev, _ in CPSC_CLASSES]
    abbrev_lists = df["dx_abbreviations"].fillna("").astype(str).str.split(",")
    targets = pd.DataFrame(
        {n: abbrev_lists.apply(lambda lst, n=n: int(n in lst)) for n in names},
        index=df.index,
    )
    print(f"\nMulti-hot target over the {len(names)} classes: {targets.shape}")
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
        f"{dur.max():.0f} s records\n"
        "  (6-144 s across the dataset), and torch cannot stack differing widths."
    )

    print("\nLeakage warning: all 6,877 of these records are also in challenge2020")
    print("and challenge2021, byte-identical and under the same A#### names. Do not")
    print("evaluate here after training on either challenge — filter their")
    print("cpsc_2018 cohort out instead:")
    print("  df[df.source != 'cpsc_2018']   # on the challenge datasets' labels_df")


if __name__ == "__main__":
    main()
