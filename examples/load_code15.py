#!/usr/bin/env python3
"""
Example: CODE-15% — 345,779 telehealth ECGs, six abnormality flags.

The largest dataset in the ECGBench catalogue, and the one with the sharpest
label trap. Four things worth seeing rather than reading:

1. **A record is a row of a shared HDF5 array, not a file.** 18 parts each hold
   one `(N, 4096, 12)` array, so a signal path reads
   `exams_part0.hdf5:tracings:417`. Nothing changes at the API surface — but see
   `load_code_test.py` for why the row index cannot be taken from the CSV order.
2. **"No abnormality" is not "normal", and the gap is 173,347 records.** Only
   134,657 of the 308,004 records with no flag are actually flagged normal. A
   model trained on the six flags alone treats the other 173,347 as negative
   examples of everything.
3. **Its lead order is standard and its sibling CODE-test's is not.** Same
   cohort, same rate — `signal[3]` is aVR here and aVL there. The `leads=`
   selection below is how to cross the two safely.
4. **Mortality follow-up is missing for 112,132 records**, and missing means
   "not followed up", not "survived".

Labels come from the shipped `exams.csv`, so this works without running the
split pipeline first. The fold CSVs come from the Hub (or from a local run with
--metadata-source local).

Prerequisites:
  - pip install ecgbench[torch,hdf5]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.
  - All 18 exams_part*.hdf5 files extracted alongside exams.csv.

Usage:
  python examples/load_code15.py --data-path /path/to/code-15/
"""

import argparse
from collections import Counter

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.code15 import ABNORMALITIES, LIST_SEPARATOR

#: Every record is exactly 4,096 samples, so a window is not required here the
#: way it is for a variable-length dataset. It is used anyway to show the shape:
#: 4,000 samples is the 10 s of real signal inside the symmetric zero padding.
WINDOW = (48, 4000)

#: The three leads whose position differs between CODE-15% and CODE-test.
CROSS_RELEASE_LEADS = ["aVR", "aVL", "aVF"]


def main():
    parser = argparse.ArgumentParser(description="Load CODE-15% with its labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("code15")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Format:   {config.signal_format}  <- a row of a shared 3-D array")
    print(f"Leads:    {config.lead_names}")
    print(f"Rate:     {config.default_sampling_rate} Hz (constant)")
    print()

    try:
        dataset = ECGDataset(
            "code15",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            # Pushed into h5py's own slicing rather than cropped afterwards, and
            # unlike a lambda transform it survives DataLoader(num_workers>0)
            # under the "spawn" start method.
            window=WINDOW,
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    print(f"Signal path:   {sample.get('signal_path', '(see folds.csv)')}")
    labels = sample["labels"]
    print(f"  record        {sample['record_id']}")
    print(f"  patient       {labels['patient_id']}")
    print(f"  codes         {labels['abnormality_codes'] or '(none of the six)'}")
    print(f"  normal_ecg    {labels['normal_ecg']}   <- read this WITH the codes")
    print(f"  stratify      {labels['stratify_class']}   <- folds only, never train on it")
    print(f"  age / sex     {labels['age']} / {labels['sex']}")
    print(f"  nn_pred_age   {labels['nn_predicted_age']}   <- a model output, not an observation")
    print(f"  death         {labels['death']}  (follow-up: {labels['has_followup']})")

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print(f"\nPatients: {df['patient_id'].nunique()} for {len(df)} records")
    repeats = df["patient_id"].value_counts()
    print(f"  with more than one record in this split: {int((repeats > 1).sum())}")
    print(f"  most records from one patient: {int(repeats.max())}")
    print("  Folds are grouped on patient_id, so no patient spans two folds.")

    print("\nTHE TRAP — an empty label list is not a normal ECG:")
    n_none = int((df["n_abnormalities"] == 0).sum())
    n_normal = int(df["normal_ecg"].sum())
    print(f"  records with none of the six flags: {n_none}")
    print(f"  of those, actually flagged normal:  {n_normal}")
    print(f"  neither flagged nor normal:         {n_none - n_normal}")
    print("  Those last ones have some finding outside this six-class vocabulary.")
    print("  Train on abnormality_codes together with normal_ecg, not on the flags alone.")

    print("\nAbnormality distribution (records carrying each; sums past the total):")
    counts = Counter(
        c for s in df["abnormality_codes"].fillna("") for c in str(s).split(LIST_SEPARATOR) if c
    )
    for code in ABNORMALITIES:
        print(f"  {code:6s} {counts.get(code, 0):7d}")
    print(f"  {'total':6s} {sum(counts.values()):7d}   flag-record pairs")
    print(
        f"  multi-label: {int((df['n_abnormalities'] > 1).sum())} records carry more "
        f"than one (max {int(df['n_abnormalities'].max())})"
    )

    print("\nThe fold label is a rarest-flag reduction, NOT the distribution above:")
    strat = df["stratify_class"].value_counts()
    for name, n in strat.items():
        print(f"  {name:8s} {n:7d}")

    age = pd.to_numeric(df["age"])
    sex_counts = df["sex"].value_counts().to_dict()
    print(f"\nAge {age.min():.0f}-{age.max():.0f}, mean {age.mean():.1f}. Sex: {sex_counts}")

    followed = df["has_followup"]
    print(f"\nMortality follow-up: {int(followed.sum())} of {len(df)} records have any.")
    print(f"  died: {int(df.loc[followed, 'death'].astype(bool).sum())}")
    print(f"  no follow-up (NOT 'survived'): {int((~followed).sum())}")

    # Multi-hot target over the six flags — the release's own task.
    code_lists = df["abnormality_codes"].fillna("").astype(str).str.split(LIST_SEPARATOR)
    targets = pd.DataFrame(
        {c: code_lists.apply(lambda lst, c=c: int(c in lst)) for c in ABNORMALITIES},
        index=df.index,
    )
    print(f"\nMulti-hot target over the six flags: {targets.shape}")
    print(f"  positives per record: mean {targets.sum(axis=1).mean():.3f}")
    n_zero = int((targets.sum(axis=1) == 0).sum())
    print(f"  all-zero rows: {n_zero} — of which {n_normal} are genuinely normal")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")

    # The cross-release lead trap, shown rather than described.
    print(f"\nLead order here:      {config.lead_names[:6]}  (standard)")
    print(f"Lead order code_test: {load_config('code_test').lead_names[:6]}  (NOT standard)")
    print("So signal[3] is a different physical lead in each release. Select by name:")
    by_name = ECGDataset(
        "code15",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=CROSS_RELEASE_LEADS,
    )
    shape = tuple(by_name[0]["signal"].shape)
    print(f"  ECGDataset(leads={CROSS_RELEASE_LEADS}) -> {shape}")
    print("  The same call against code_test returns the same three physical leads,")
    print("  in the same order, despite the differing storage order.")

    print("\nNote on the records the clean version drops: they fail amplitude_outlier")
    print("(a lead beyond +-20 mV) or missing_leads (a lead recorded as exactly zero).")
    print("The range is wider than ECGBench's usual +-10 because these are telehealth")
    print("recordings with a median per-record peak of 4.27 mV; +-10 would drop 11.8%")
    print("of the release. Use version='original' to get them back, with is_valid and")
    print("quality_issues attached.")


if __name__ == "__main__":
    main()
