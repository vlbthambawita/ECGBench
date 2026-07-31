#!/usr/bin/env python3
"""
Example: MIMIC-IV-ECG — the full 800,035-record credentialed release.

Not to be confused with `load_mimic_iv_ecg_demo.py`: that is the open 659-record
demo, which ships no labels at all. This one has them.

Four things to demonstrate:

1. **The stored lead order transposes aVF and aVL.** `signal[4]` is aVF here and
   aVL in every other 12-lead dataset in ECGBench. Selecting `leads=[...]` by name
   is the fix, and this script shows the two orders side by side.
2. **Labels are free-text machine reports**, up to 18 lines per study, joined into
   `report_text`. `primary_report` is only the *first* line — usually the rhythm,
   sometimes a data-quality warning — so it is not a rhythm label.
3. **Numeric measurements use integer sentinels, not NaN.** 29999 / 32767 / 65535
   mean "not measurable". The loader converts them; this script shows what a
   user who read the CSV directly would compute instead.
4. **It is big.** 800,035 records from 161,352 subjects. `fold_numbers=[1]` is how
   you work on a tenth of it, and folds are grouped by `subject_id` because 64.5%
   of subjects contributed more than one study.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the credentialed release. Labels are never on the HuggingFace
    Hub: MIMIC-IV is under a DUA, so its report text is not redistributed.

Usage:
  python examples/load_mimic_iv_ecg.py --data-path /path/to/mimic-iv-ecg/1.0/
"""

import argparse

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.mimic_iv_ecg import (
    AXIS_COLUMNS,
    SENTINELS,
    SOURCE_CSV,
    TIMING_COLUMNS,
)

#: The standard 12-lead order. MIMIC-IV-ECG does not store this order.
STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

#: Findings to match in the free-text report, for a multi-hot training target.
#: Use the ECG cart's vocabulary, not clinical prose: the machine writes
#: "infarct" (179,588 records) and almost never "myocardial infarction" (211), so
#: searching for the latter suggests this hospital population has no infarcts.
FINDINGS = [
    "atrial fibrillation",
    "sinus bradycardia",
    "sinus tachycardia",
    "left bundle branch block",
    "right bundle branch block",
    "infarct",
    "hypertrophy",
]


def main():
    parser = argparse.ArgumentParser(description="Load MIMIC-IV-ECG with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument(
        "--fold", type=int, default=1, help="Single fold to load (this dataset is large)"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    config = load_config("mimic_iv_ecg")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}), fold {args.fold}")
    print(f"Stored leads: {config.lead_names}")
    print(f"Standard:     {STANDARD_12}")
    bad = [i for i, (a, b) in enumerate(zip(config.lead_names, STANDARD_12)) if a != b]
    print(
        f"  -> positions {bad} differ: signal[4] is "
        f"{config.lead_names[4]!r} here, {STANDARD_12[4]!r} elsewhere"
    )
    print()

    try:
        dataset = ECGDataset(
            "mimic_iv_ecg",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            fold_numbers=[args.fold],
            leads=STANDARD_12,  # reorder to the conventional order, by name
            labels=True,
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records in this fold: {len(dataset):,}")
    print(f"Subjects in this fold: " f"{dataset.metadata_df[config.patient_id_column].nunique():,}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:  {sorted(sample.keys())}")
    print(f"Signal shape: {tuple(sample['signal'].shape)}  (reordered to standard leads)")
    print(f"  study_id       {sample['record_id']}")
    print(f"  report_text    {labels['report_text'][:88]}")
    print(f"  primary_report {labels['primary_report']!r}   <- first line only")
    print(f"  rr_interval    {labels['rr_interval']}  qrs_duration {labels['qrs_duration']}")
    print(f"  axes P/QRS/T   {labels['p_axis']} / {labels['qrs_axis']} / {labels['t_axis']}")

    df = dataset.labels_df

    print(f"\nTop {args.top} first report lines in this fold (primary_report):")
    for value, n in df["primary_report"].value_counts().head(args.top).items():
        print(f"  {n:6,d}  ({100 * n / len(df):5.2f}%)  {value[:60]}")

    print("\nWhy primary_report is not a rhythm label — some first lines are not rhythms:")
    for needle in ("warning", "age not entered", "consider acute"):
        n = int(df["primary_report"].str.contains(needle, na=False).sum())
        if n:
            example = df.loc[
                df["primary_report"].str.contains(needle, na=False), "primary_report"
            ].iloc[0]
            print(f"  {n:6,d} records lead with: {example[:66]}")

    print("\nThe sentinel trap. Raw CSV values against what the loader returns:")
    print(f"  (reading {SOURCE_CSV} directly would keep {SENTINELS} as numbers)")
    for column in TIMING_COLUMNS + AXIS_COLUMNS:
        n_missing = int(df[column].isna().sum())
        unit = "deg" if column in AXIS_COLUMNS else "ms"
        print(
            f"  {column:12s} median {df[column].median():7.1f} {unit:3s} "
            f"| not measurable in {n_missing:6,d} ({100 * n_missing / len(df):5.2f}%)"
        )
    p_missing = df["p_onset"].isna()
    af = df["primary_report"].str.contains("atrial fibrillation", na=False)
    if af.any():
        print(
            f"  p_onset is unmeasurable in {100 * p_missing[af].mean():.1f}% of atrial "
            f"fibrillation records vs {100 * p_missing[~af].mean():.1f}% of the rest —"
        )
        print("  the sentinel means 'no organised P wave', not 'data missing'.")

    # A target tensor: multi-hot over the findings mentioned anywhere in the report.
    text = df["report_text"].fillna("").str.lower()
    targets = pd.DataFrame({f: text.str.contains(f).astype(int) for f in FINDINGS})
    print(f"\nMulti-hot over {len(FINDINGS)} findings matched in report_text: {targets.shape}")
    for f in FINDINGS:
        print(f"  {targets[f].sum():6,d}  {f}")
    print(f"  records matching none of them: {(targets.sum(axis=1) == 0).sum():,}")
    print("  NB: matching is literal, so check terms against the text — 'myocardial")
    print("      infarction' is in 211 records of 800,035, 'infarct' in 179,588.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)} {batch['signal'].dtype}")
    print(f"  lead_names {dataset.lead_names}")
    print("\nFolds are grouped by subject_id: no subject appears in two folds, which")
    print("matters because one subject contributed 260 studies in the full release.")


if __name__ == "__main__":
    main()
