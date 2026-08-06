#!/usr/bin/env python3
"""
Example: SaMi-Trop — 1,631 Chagas cardiomyopathy ECGs with mortality follow-up.

The third TNMG release in the catalogue after CODE-15% and CODE-test, and the
one that is not a classification dataset. Four things worth seeing rather than
reading:

1. **There is no diagnostic vocabulary.** No abnormality flags at all — the only
   ECG label is a binary `normal_ecg`. What this release is for is the *outcome*:
   every one of the 1,631 patients has complete mortality follow-up.
2. **"Normal ECG" does not mean healthy.** Every patient here already has
   chronic Chagas cardiomyopathy, so a `normal_ecg` record is a normal tracing in
   a diseased patient. These are not usable as healthy controls, which is the
   mistake to avoid when pooling this with other datasets.
3. **Follow-up is complete, unlike CODE-15%'s.** CODE-15% needs a nullable
   boolean because 112,132 of its records have no outcome; here all 1,631 do, so
   `death` is a plain bool and there is no missingness to reason about.
4. **One record per patient.** The release is each patient's *first* exam, so
   there is no patient grouping — the rare case in this catalogue where an
   ungrouped split is genuinely safe rather than merely unchecked.

Labels come from the shipped `exams.csv`, so this works without running the
split pipeline first. The fold CSVs come from the Hub.

Prerequisites:
  - pip install ecgbench[torch,hdf5]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.
  - exams.hdf5 extracted from exams.zip, alongside exams.csv.

Usage:
  python examples/load_sami_trop.py --data-path /path/to/SaMi-Trop/
"""

import argparse

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError

#: Every record is exactly 4,096 samples, so a window is not needed the way it is
#: for a variable-length dataset. Used anyway to show the shape: the shortest real
#: signal in the release is 1,568 samples, so anything wider than that risks
#: reading only zero padding on the shortest records.
WINDOW = (0, 1568)


def main():
    parser = argparse.ArgumentParser(description="Load SaMi-Trop with its labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("sami_trop")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Format:   {config.signal_format}  <- a row of one shared 3-D array")
    print(f"Leads:    {config.lead_names}")
    print(f"Rate:     {config.default_sampling_rate} Hz, {config.duration_seconds} s")
    print(f"Patients: patient_id_column={config.patient_id_column}  <- one ECG per patient")
    print()

    try:
        dataset = ECGDataset(
            "sami_trop",
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
    labels = sample["labels"]
    print(f"  record        {sample['record_id']}")
    print(f"  row           {labels['row']}   <- its index in the tracings array")
    print(f"  age / sex     {labels['age']} / {labels['sex']}")
    print(f"  normal_ecg    {labels['normal_ecg']}   <- the ONLY ECG label here")
    print(f"  death         {labels['death']}")
    print(f"  followup_yrs  {labels['followup_years']:.2f}")
    print(
        f"  nn_pred_age   {labels['nn_predicted_age']:.1f}   <- a model output, not an observation"
    )
    print(f"  stratify      {labels['stratify_class']}   <- folds only, never train on it")

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print("\nWHAT THIS DATASET IS FOR — the mortality endpoint:")
    n_death = int(df["death"].sum())
    print(f"  deaths: {n_death} of {len(df)} ({100 * n_death / len(df):.1f}%)")
    years = pd.to_numeric(df["followup_years"])
    print(
        f"  follow-up years: median {years.median():.2f}, range "
        f"{years.min():.2f}-{years.max():.2f}"
    )
    print(f"  every record has an outcome: {bool(df['death'].notna().all())}")
    print("  A survivor is only known to have survived those few years, not longer.")

    print("\nTHE TRAP — normal_ecg is not a healthy control:")
    n_normal = int(df["normal_ecg"].sum())
    print(f"  flagged normal: {n_normal} of {len(df)} ({100 * n_normal / len(df):.1f}%)")
    print("  All of these patients have chronic Chagas cardiomyopathy. A normal")
    print("  tracing in a diseased patient is not a healthy subject, so do not pool")
    print("  these with another dataset's normals as if they were.")
    print("  And a record that is NOT normal carries no statement of what is wrong:")
    print(f"  {len(df) - n_normal} records are only known to be abnormal.")

    print("\nStratification is mortality-first, so the rare outcome balances:")
    for name, n in df["stratify_class"].value_counts().items():
        print(f"  {name:16s} {n:5d}")
    print("  Note it is 3 classes, not the death x normal_ecg cross: only 3 records")
    print("  are both dead and normal, and a 3-member class cannot span 10 folds.")

    age = pd.to_numeric(df["age"])
    print(
        f"\nAge {age.min():.0f}-{age.max():.0f}, median {age.median():.0f}. "
        f"Sex: {df['sex'].value_counts().to_dict()}"
    )
    print("  The cohort skews female and old — it is a disease cohort, not a sample.")

    # The release's own task: predict the DNN age, or use it as a risk marker.
    nn_age = pd.to_numeric(df["nn_predicted_age"])
    gap = nn_age - age
    print(f"\nECG-age gap (nn_predicted_age - age): mean {gap.mean():+.1f} years")
    died = df["death"].astype(bool)
    print(f"  among those who died:     {gap[died].mean():+.1f}")
    print(f"  among those who did not:  {gap[~died].mean():+.1f}")
    print("  The gap is larger in those who died, but a difference of a few tenths of")
    print("  a year between two group means is NOT the paper's result — that came from")
    print("  a survival model over the whole cohort, not from comparing these means.")
    print("  Do not read this printout as a reproduction of it.")

    # A survival target, which is what this dataset supports.
    target = pd.DataFrame({"event": died.astype(int), "duration": years}, index=df.index)
    print(f"\nSurvival target: {target.shape}  columns {list(target.columns)}")
    print(f"  events {int(target['event'].sum())}, censored {int((1 - target['event']).sum())}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")

    # Lead order was verified from the arrays, because a sibling release differs.
    print(f"\nLead order here:      {config.lead_names[:6]}  (standard)")
    print(f"Lead order code_test: {load_config('code_test').lead_names[:6]}  (NOT standard)")
    print("Same telehealth network, so select by name when combining them:")
    by_name = ECGDataset(
        "sami_trop",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=["aVR", "aVL", "aVF"],
    )
    print(f"  ECGDataset(leads=['aVR','aVL','aVF']) -> " f"{tuple(by_name[0]['signal'].shape)}")

    print("\nNote on the records the clean version drops: 14 fail amplitude_outlier")
    print("(a lead beyond +-20 mV) and 2 fail missing_leads (a lead recorded as")
    print("exactly zero throughout). The +-20 range matches CODE-15% because these")
    print("are the same telehealth instruments; +-10 would drop 180 records. Use")
    print("version='original' to get them back, with is_valid and quality_issues.")


if __name__ == "__main__":
    main()
