#!/usr/bin/env python3
"""
Example: PhysioNet/CinC Challenge 2017 with labels.

The 8,528-record public training set of the 2017 AF challenge — the reference
benchmark for atrial fibrillation detection from *consumer* single-lead ECG
rather than clinical 12-lead. Five things about it are worth seeing rather than
reading, and this script shows all five:

1. **One channel, and it is called "ECG", not "I".** The AliveCor device gives a
   nominal lead I (LA-RA) equivalent, but it does not enforce orientation, so an
   unknown number of traces are inverted. Do not stack this with 12-lead data
   under the name "I".
2. **Record length varies** — 9.05 s to 60.95 s, 1,487 distinct lengths — so
   records cannot be batched without a window, and the window has to fit the
   shortest one. Length also *correlates with the label*, which is a shortcut a
   model will happily take.
3. **The labels were revised twice**, and all four shipped versions are exposed.
   412 of 8,528 changed between the first and the last, and the shipped file
   numbers are one behind the paper's V1/V2/V3.
4. **The shipped `validation/` directory is not a held-out split** — its 300
   records are byte-identical copies of training records. Evaluating on it means
   evaluating on training data.
5. **There are no demographics and no patient identifier**, so folds are
   stratified but ungrouped.

Labels come from the release's own RECORDS/REFERENCE files and the WFDB headers,
so this works without running the split pipeline first. The fold CSVs still come
from the Hub (or from a local run with --metadata-source local).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_challenge2017.py --data-path /path/to/challenge-2017/1.0.0/
"""

import argparse

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.challenge2017 import CLASS_NAMES, PAPER_VERSION_NAMES

#: Records are variable length, so every signal is windowed to this many samples.
#: 2700 is 9 s at 300 Hz, which fits inside the SHORTEST record in the dataset
#: (A05493, 2,714 samples), so this window fits all 8,528. Anything longer raises
#: WindowOutOfRangeError on the short tail.
WINDOW = (0, 2700)


def main():
    parser = argparse.ArgumentParser(description="Load Challenge 2017 with labels")
    parser.add_argument("--data-path", default=None, help="Path to the version directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("challenge2017")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Leads:    {config.lead_names}  <- the header's own name, not 'I'")
    print(f"Rate:     {config.default_sampling_rate} Hz (constant across all records)")
    print()

    try:
        dataset = ECGDataset(
            "challenge2017",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            # window=(start, length) is pushed into the wfdb reader rather than
            # cropped afterwards, so a 61 s record decodes 9 s. Unlike a lambda
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
    print(f"  class_code   {labels['class_code']}   <- ground truth, single-label")
    print(f"  class_name   {labels['class_name']}")
    print(f"  is_af        {labels['is_af']}")
    print(
        f"  duration     {labels['duration_seconds']} s  <- the FULL record, "
        f"not the {WINDOW[1] / config.default_sampling_rate:.0f} s window"
    )
    history = " -> ".join(str(labels[f"class_code_v{v}"]) for v in (0, 1, 2, 3))
    print(f"  label hist   {history}   (shipped v0..v3)")
    print(f"  timestamp    {labels['header_timestamp']}   <- de-identified, not a clock")

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print(f"\nClass distribution in this split ({len(df)} records):")
    counts = df["class_name"].value_counts()
    for code, name in CLASS_NAMES.items():
        n = int(counts.get(name, 0))
        print(f"  {code:2s} {name:20s} {n:5d}  ({n / len(df):6.2%})")
    print(f"     {'total':20s} {int(counts.sum()):5d}  (single-label: no record has two)")

    print("\nRecord length varies, and it is not independent of the label:")
    dur = pd.to_numeric(df["duration_seconds"])
    print(f"  min {dur.min():.2f} s   median {dur.median():.2f} s   max {dur.max():.2f} s")
    print(f"  exactly 30 s: {int((dur == 30).sum())}   exactly 60 s: {int((dur == 60).sum())}")
    print("  mean duration per class:")
    for name, mean in dur.groupby(df["class_name"]).mean().sort_values().items():
        print(f"    {name:20s} {mean:5.1f} s")
    print("  A model fed whole records can learn duration instead of rhythm.")

    print("\nThe labels were revised twice during the competition:")
    print("  shipped file  paper's name  changed vs shipped v0")
    for v in (0, 1, 2, 3):
        paper = PAPER_VERSION_NAMES[v] or "(not in the paper)"
        changed = int((df[f"class_code_v{v}"] != df["class_code_v0"]).sum())
        print(f"  REFERENCE-v{v}    {paper:18s}  {changed:5d}")
    print(
        f"  {int(df['label_revised'].sum())} records in this split were relabelled "
        f"between v0 and v3;"
    )
    print("  n_distinct_labels counts how many labels a record was ever given:")
    for n, count in df["n_distinct_labels"].value_counts().sort_index().items():
        print(f"    {n} label(s) ever: {count:5d}")
    print("  Records above 1 are the ones the organisers' bootstrap relabelling")
    print("  flagged as contentious (Fleiss' kappa 0.245 over the 1,129 worst).")

    # A single-label target: the challenge's own four-class task.
    classes = list(CLASS_NAMES.values())
    target = df["class_name"].map({n: i for i, n in enumerate(classes)})
    print(f"\nFour-class target over {classes}:")
    print(f"  shape {target.shape}, unlabelled rows {int(target.isna().sum())}")
    weights = ", ".join(f"{n}: {len(df) / max(int(counts.get(n, 0)), 1):.2f}" for n in classes)
    print(f"  class weights (inverse frequency): {{{weights}}}")

    # Batching only works because of the window above.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print(
        f"  Without window= this raises — this split alone mixes {dur.min():.2f} s to "
        f"{dur.max():.2f} s records,\n"
        "  and torch cannot stack differing widths."
    )

    n_dup = int(df["in_challenge_validation_subset"].sum())
    print(f"\nThe challenge's own 'validation' set is inside this data: {n_dup} records")
    print("of this split are among the 300 that ship a byte-identical duplicate copy")
    print("under validation/. It is not held-out data — the paper calls it '300")
    print("records (3.5%) of training set just to ensure the algorithm produced the")
    print("expected results'. To compare against published challenge numbers, drop it:")
    print("  df[~df.in_challenge_validation_subset]")
    print("\nThe real 3,658-record test set was never released, so nothing here")
    print("reproduces the challenge's own scoring split.")


if __name__ == "__main__":
    main()
