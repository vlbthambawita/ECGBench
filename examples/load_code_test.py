#!/usr/bin/env python3
"""
Example: CODE-test — 827 ECGs read by six annotator groups and a neural network.

Small, and the most heavily annotated dataset in the catalogue. Four things
worth seeing rather than reading:

1. **It is an evaluation set that ECGBench nevertheless gives ten folds.** That
   is the framework's uniform convention, not advice. The intended use is all
   827 records as one hold-out set, which is the `split=None` call at the end of
   this script. Train on CODE-15%, not on 8/10 of this.
2. **Nothing in the release has an identifier.** The waveform file holds one
   `(827, 4096, 12)` array and no ids; the eight tables are aligned to it by row
   position alone. So `record_id` is the row index, 0-826.
3. **Seven annotation sets ship, not one.** Two cardiologists, a gold standard
   adjudicated from them, two cardiology residents, two emergency residents, two
   medical students, and the paper's DNN. The loader exposes all of them, which
   is what makes reader-agreement analysis possible.
4. **Its lead order is NOT standard** — aVL, aVF, aVR — and differs from
   CODE-15%'s, which is. Select leads by name when using both.

Labels come from the shipped CSVs, so this works without running the split
pipeline first. The fold CSVs come from the Hub (or from a local run with
--metadata-source local).

Prerequisites:
  - pip install ecgbench[torch,hdf5]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.
  - data.zip extracted; --data-path must be the `data/` directory it creates,
    the one holding ecg_tracings.hdf5, attributes.csv and annotations/.

Usage:
  python examples/load_code_test.py --data-path /path/to/code-test/data/
"""

import argparse

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.code_test import ABNORMALITIES, ANNOTATORS, LIST_SEPARATOR

#: All 827 records are exactly 4,096 samples, so a window is optional. 4,000
#: samples is the 10 s of real signal inside the symmetric zero padding — but
#: note that records whose acquisition was 7 s are padded by 648 samples on each
#: side, so this window still includes padding for them.
WINDOW = (48, 4000)

#: The three leads whose position differs between this release and CODE-15%.
CROSS_RELEASE_LEADS = ["aVR", "aVL", "aVF"]


def main():
    parser = argparse.ArgumentParser(description="Load CODE-test with all annotators")
    parser.add_argument("--data-path", default=None, help="Path to the data/ directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("code_test")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Format:   {config.signal_format}  <- a row of one shared 3-D array")
    print(f"Leads:    {config.lead_names}")
    print("          ^ aVL, aVF, aVR — NOT the standard order")
    print(f"Rate:     {config.default_sampling_rate} Hz (constant)")
    print()

    try:
        dataset = ECGDataset(
            "code_test",
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

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    labels = sample["labels"]
    print(f"  record_id     {sample['record_id']}   <- the row index, not an id the")
    print("                    release assigns; it has none")
    print(f"  gold standard {labels['abnormality_codes'] or '(none of the six)'}")
    print(f"  cardiologist1 {labels['cardiologist1_abnormality_codes'] or '(none)'}")
    print(f"  cardiologist2 {labels['cardiologist2_abnormality_codes'] or '(none)'}")
    print(f"  dnn           {labels['dnn_abnormality_codes'] or '(none)'}")
    print(f"  stratify      {labels['stratify_class']}   <- folds only, never train on it")
    print(f"  age / sex     {labels['age']} / {labels['sex']}")

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print(f"\nGold-standard distribution over this split ({len(df)} records):")
    for code in ABNORMALITIES:
        print(f"  {code:6s} {int(df[code].astype(bool).sum()):4d}")
    n_none = int((df["n_abnormalities"] == 0).sum())
    print(f"  none   {n_none:4d}   <- 'none of these six', NOT 'normal':")
    print("                 this release publishes no normal flag at all")
    print(f"  multi  {int((df['n_abnormalities'] > 1).sum()):4d}   records carry more than one")

    print("\nEvery annotator's total, side by side — this is what the release is for:")
    print(f"  {'annotator':22s} " + " ".join(f"{c:>6s}" for c in ABNORMALITIES) + "   any")
    for name in ANNOTATORS:
        per_code = [int(df[f"{name}_{c}"].astype(bool).sum()) for c in ABNORMALITIES]
        any_flag = int((df[f"{name}_n_abnormalities"] > 0).sum())
        print(f"  {name:22s} " + " ".join(f"{n:6d}" for n in per_code) + f"  {any_flag:4d}")
    print("  Note cardiologist1 and cardiologist2 are the two reads the gold standard")
    print("  was adjudicated from, so they are not independent of it. The three")
    print("  non-expert rows are each two people who annotated half the set each.")

    print("\nAgreement with the gold standard, per annotator (exact match on all six):")
    gold = df[[f"gold_standard_{c}" for c in ABNORMALITIES]].astype(bool).to_numpy()
    for name in ANNOTATORS:
        other = df[[f"{name}_{c}" for c in ABNORMALITIES]].astype(bool).to_numpy()
        exact = (gold == other).all(axis=1).mean()
        print(f"  {name:22s} {100 * exact:5.1f}%")

    age = pd.to_numeric(df["age"])
    sex_counts = df["sex"].value_counts().to_dict()
    print(f"\nAge {age.min():.0f}-{age.max():.0f}, mean {age.mean():.1f}. Sex: {sex_counts}")
    print("  (The parent CODE-15% release is 40.3% men; this sample is not.)")

    # Multi-hot target over the six flags — the release's own task.
    code_lists = df["abnormality_codes"].fillna("").astype(str).str.split(LIST_SEPARATOR)
    targets = pd.DataFrame(
        {c: code_lists.apply(lambda lst, c=c: int(c in lst)) for c in ABNORMALITIES},
        index=df.index,
    )
    print(f"\nMulti-hot target over the six flags: {targets.shape}")
    print(f"  positives per record: mean {targets.sum(axis=1).mean():.3f}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")

    # The cross-release lead trap, shown rather than described.
    print(f"\nLead order here:   {config.lead_names[:6]}")
    print(f"Lead order code15: {load_config('code15').lead_names[:6]}")
    print("signal[3] is aVL here and aVR there, from the same cohort. Select by name:")
    by_name = ECGDataset(
        "code_test",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=CROSS_RELEASE_LEADS,
    )
    shape = tuple(by_name[0]["signal"].shape)
    print(f"  ECGDataset(leads={CROSS_RELEASE_LEADS}) -> {shape}")

    # What this release is actually for: the whole thing as one hold-out set.
    print("\nUSING IT AS INTENDED — all 827 records as a single hold-out set,")
    print("rather than the train/val/test division above:")
    whole = ECGDataset(
        "code_test",
        split=None,
        fold_numbers=list(range(1, 11)),
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        labels=True,
        window=WINDOW,
    )
    print(f"  ECGDataset(split=None, fold_numbers=range(1, 11)) -> {len(whole)} records")
    first_split = whole[0]["split"]
    print(f"  each sample's ['split'] reports its own default split, e.g. {first_split!r}")
    print(f"  ({args.version} drops the records failing validation; 'original' keeps all 827)")


if __name__ == "__main__":
    main()
