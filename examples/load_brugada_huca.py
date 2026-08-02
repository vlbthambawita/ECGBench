#!/usr/bin/env python3
"""
Example: Brugada-HUCA — 363 twelve-lead ECGs screened for Brugada syndrome.

The smallest and cleanest dataset in ECGBench, and the only one sampled at
100 Hz alone. Every record is 12 leads x 1200 samples (12 s), no record contains
a NaN sample or a flat lead, and all 363 pass validation.

Three things worth demonstrating:

1. **The labels are bare integers.** `brugada` is 0/1/2 with no accompanying
   string anywhere in the CSV; the meanings live only in the release's
   README.md. This script prints them so a reader is not left guessing.
2. **It is a screening cohort, not a case-control study.** "0" means
   "investigated for Brugada syndrome and not diagnosed", not "healthy member of
   the public" — which changes what a classifier trained here is estimating.
3. **The rare class stays unpooled.** `brugada == 2` (other/atypical) has 7
   records, fewer than the 10 folds, so it cannot appear in every fold. Folding
   it into either other class would be clinically wrong, so ECGBench keeps it.

Labels come from the shipped metadata.csv, so `labels=True` works without
running the split pipeline first.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_brugada_huca.py --data-path /path/to/brugada-huca/1.0.0/
"""

import argparse

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.splitting.strategies.brugada_huca import BRUGADA_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Load Brugada-HUCA with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("brugada_huca")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}")
    print(
        f"Duration: {config.duration_seconds} s "
        f"({config.duration_seconds * config.default_sampling_rate:.0f} samples)"
    )
    print()

    try:
        dataset = ECGDataset(
            "brugada_huca",
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
    print(f"\nSample keys:  {sorted(sample.keys())}")
    print(f"Signal shape: {tuple(sample['signal'].shape)}")
    print(f"  patient_id     {sample['record_id']}")
    print(
        f"  brugada        {labels['brugada']}  " f"({BRUGADA_CLASSES.get(labels['brugada'], '?')})"
    )
    print(
        f"  basal_pattern  {labels['basal_pattern']}  "
        f"({'pathological baseline' if labels['basal_pattern'] else 'normal baseline'})"
    )
    print(f"  sudden_death   {labels['sudden_death']}")

    df = dataset.labels_df

    print("\nbrugada — the integer codes have no string form in the CSV:")
    for code, n in df["brugada"].value_counts().sort_index().items():
        print(
            f"  {code} = {BRUGADA_CLASSES.get(code, '?'):28s} {n:4d}  "
            f"({100 * n / len(df):5.1f}%)"
        )
    print("  NB: 0 means 'investigated and not diagnosed', not a general-population")
    print("      control — this is a screening cohort from one referral hospital.")

    for column, meaning in (
        ("basal_pattern", "pathological baseline ECG"),
        ("sudden_death", "experienced sudden death"),
    ):
        n = int(df[column].sum())
        print(f"\n{column}: {n} of {len(df)} ({100 * n / len(df):.1f}%) — {meaning}")

    print("\nThe three labels are not interchangeable — basal_pattern is documented as")
    print("independent of the diagnosis. Cross-tabulated over this split:")
    print(
        pd.crosstab(
            df["brugada"], df["basal_pattern"], rownames=["brugada"], colnames=["basal_pattern"]
        ).to_string()
    )

    # A binary target: confirmed Brugada versus everything else. Note this folds
    # the 'other/atypical' class in with the undiagnosed, which is a modelling
    # choice, not something the dataset says.
    target = (df["brugada"] == 1).astype(int)
    print(
        f"\nBinary target (brugada == 1): {int(target.sum())} positive of {len(target)}"
        f" ({100 * target.mean():.1f}%)"
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)} {batch['signal'].dtype}")
    print("  Records are a uniform 1200 samples, so no window= or crop is needed")
    print("  here — unlike ptbdb, incartdb or challenge2021.")


if __name__ == "__main__":
    main()
