#!/usr/bin/env python3
"""
Example: MIMIC-IV-ECG-Ext-ICD — ICD-10 diagnoses for MIMIC-IV-ECG's records.

This release ships **no waveforms**. Its 800,035 rows are exactly the 800,035
studies of MIMIC-IV-ECG, keyed by that dataset's own `study_id`. So there is
deliberately no `mimic_iv_ecg_ext_icd` config and no
`ecgbench splits --dataset mimic_iv_ecg_ext_icd`: a second ten-fold partition over
records `mimic_iv_ecg` already partitions would let someone train on one split and
evaluate on the other without noticing.

Instead you load MIMIC-IV-ECG on ECGBench's folds and join these labels onto it,
which is what this script does.

Four things worth seeing in the output:

1. **Only 58.5% of records carry any diagnosis.** The rest were not part of an ED
   or hospital stay MIMIC-IV holds a discharge diagnosis for, and their diagnosis
   columns are empty *lists*, not nulls.
2. **The release ships its own 20-fold split, which is not ECGBench's.** The two
   are independent, so mixing them puts the same patient on both sides.
3. **The published 1,076-code label set is reproducible** — but only if you strip
   trailing ICD-10 placeholder Xs before propagating superclasses.
4. **`gender` encodes missing as the string `"missing"`,** and ages over 89 are a
   de-identification artefact of MIMIC-IV's age capping.

Prerequisites:
  - pip install ecgbench[torch]
  - Local copies of BOTH datasets, each credentialed on PhysioNet. Ext-ICD has no
    waveforms; MIMIC-IV-ECG has no ICD codes.
  - ECGBench's fold CSVs for `mimic_iv_ecg` are never published to the Hub, so
    generate them and copy them into the MIMIC-IV-ECG root first:

      ecgbench splits --dataset mimic_iv_ecg --data-path /path/to/mimic-iv-ecg/1.0/
      cp -r output/mimic_iv_ecg/clean output/mimic_iv_ecg/original \\
            /path/to/mimic-iv-ecg/1.0/

Usage:
  python examples/load_mimic_iv_ecg_ext_icd.py \\
      --mimic-path   /path/to/mimic-iv-ecg/1.0/ \\
      --ext-icd-path /path/to/mimic-iv-ecg-ext-icd-labels/1.0.1/
"""

import argparse
from collections import Counter

from ecgbench import ECGDataset
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.mimic_iv_ecg_ext_icd import (
    DIAGNOSIS_COLUMNS,
    ECG_SUBSETS,
    ecg_subset,
    label_set,
    load_ext_icd,
    multi_hot,
    propagate_superclasses,
    upstream_fold_split,
)

STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    parser = argparse.ArgumentParser(description="Join Ext-ICD onto MIMIC-IV-ECG's folds")
    parser.add_argument("--mimic-path", required=True, help="MIMIC-IV-ECG root (waveforms)")
    parser.add_argument("--ext-icd-path", required=True, help="Ext-ICD root (labels)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--fold", type=int, default=1, help="one ECGBench fold; 800k records total")
    args = parser.parse_args()

    print("Ext-ICD is a label layer, so it is consumed through MIMIC-IV-ECG's splits.")
    print("There is no mimic_iv_ecg_ext_icd config and no separate fold assignment.\n")

    # metadata_source="local": MIMIC-IV-ECG's fold CSVs are credentialed and never
    # published to the Hub, so they are the ones you generated yourself.
    ds = ECGDataset(
        "mimic_iv_ecg",
        split=args.split,
        data_path=args.mimic_path,
        metadata_source="local",
        fold_numbers=[args.fold],
        leads=STANDARD_12,
    )
    print(f"MIMIC-IV-ECG {args.split} fold {args.fold}: {len(ds):,} records")

    try:
        # prefix= because MIMIC-IV-ECG's own label frame also carries ecg_time.
        icd = load_ext_icd(args.ext_icd_path, prefix="icd_")
    except LabelSourceMissingError as e:
        print(f"Ext-ICD unavailable: {e}")
        return

    print(f"Ext-ICD frame:               {icd.shape[0]:,} records x {icd.shape[1]} columns")

    study_ids = ds.metadata_df["study_id"]
    joined = icd.reindex(study_ids.values)
    matched = int(joined.notna().any(axis=1).sum())
    print(
        f"joined on study_id:          {matched:,} of {len(study_ids):,} "
        f"({100 * matched / len(study_ids):.2f}%)"
    )

    def n_with_codes(column):
        cell = joined[f"icd_{column}"]
        return int(cell.map(lambda v: bool(v) if isinstance(v, list) else False).sum())

    print("\nFive diagnosis columns, and how many of this fold's records carry each:")
    for column, description in DIAGNOSIS_COLUMNS.items():
        n = n_with_codes(column)
        print(f"  {column:15s} {n:>6,} ({100 * n / len(joined):5.1f}%)  {description}")
    print("  -> train on all_diag_all. An empty list means 'no linked discharge")
    print("     diagnosis', which is a real value here, not a parse failure.")

    print("\nThe published label set, reproduced from the full table:")
    # prefix= is how every helper finds the columns load_ext_icd renamed.
    codes = label_set(icd, prefix="icd_")
    by_length = Counter(len(c) for c in codes)
    print(f"  {len(codes)} codes with >=2000 records; by length {dict(sorted(by_length.items()))}")
    print(f"  most common: {codes[:8]}")
    print("  The trailing-X strip is what makes this exact:")
    print(f"    ['I2510', 'W19XXXA'] -> {propagate_superclasses(['I2510', 'W19XXXA'])}")
    print("    without it 'W19XXXA' would contribute 'W19XX' and the set comes out 1089.")

    print("\nMulti-hot targets for this fold (superclasses propagated):")
    targets = multi_hot(joined, codes, prefix="icd_")
    labelled = targets.sum(axis=1) > 0
    has_codes = n_with_codes("all_diag_all")
    print(
        f"  {targets.shape[0]:,} x {targets.shape[1]} matrix; "
        f"{int(labelled.sum()):,} records positive for at least one code"
    )
    print(f"  mean codes per labelled record: {float(targets[labelled].sum(axis=1).mean()):.1f}")
    print(f"  {has_codes - int(labelled.sum()):,} records carry codes that ALL fell below")
    print("  the 2,000-record threshold, so they come back all-zero — check the")
    print("  positive count, not the number of records carrying codes.")

    row = joined.iloc[0]
    print(f"\nOne record — study_id {study_ids.iloc[0]}:")
    print(f"  all_diag_all (raw)  {row['icd_all_diag_all'][:6]}")
    print(f"  in the label set    {[c for c in codes if targets.iloc[0][c] == 1][:8]}")
    print(f"  gender / age        {row['icd_gender']} / {row['icd_age']}")
    print(f"  signal              {tuple(ds[0]['signal'].shape)} (from MIMIC-IV-ECG)")

    print("\nThe upstream benchmark's own subsets and folds are also shipped:")
    for name in ECG_SUBSETS:
        print(f"  subset {name:5s} {len(ecg_subset(icd, name, prefix='icd_')):>7,} records")
    for split in ("train", "val", "test"):
        n = len(upstream_fold_split(icd, split, prefix="icd_"))
        print(f"  upstream {split:5s} {n:>7,} records")

    print("\nThose 20 folds are NOT ECGBench's 10 — they are an independent partition:")
    upstream_test = set(upstream_fold_split(icd, "test", prefix="icd_").index)
    overlap = sum(1 for sid in study_ids if sid in upstream_test)
    print(f"  {overlap:,} of this ECGBench {args.split} fold's {len(study_ids):,} records")
    print(f"  ({100 * overlap / len(study_ids):.1f}%) sit in the upstream test fold.")
    print("  Pick one partition and stay inside it; mixing them leaks patients.")


if __name__ == "__main__":
    main()
