#!/usr/bin/env python3
"""
Example: PTB-XL+ — derived features and annotations for PTB-XL's records.

PTB-XL+ ships **no raw ECGs**. It annotates the same 21,799 recordings PTB-XL
holds, keyed by PTB-XL's own `ecg_id`. So there is deliberately no
`ptbxl_plus` dataset config and no `ecgbench splits --dataset ptbxl_plus`:
generating a second ten-fold partition over records `ptbxl` already partitions
would let a user train on one split and evaluate on the other without noticing.

Instead you load PTB-XL as usual — on PTB-XL's own official folds — and join
PTB-XL+ onto it, which is what this script demonstrates.

Three release defects worth seeing in the output:

1. `12sl_features.csv` keeps `ecg_id` at **column 145 of 783**, not at the front,
   and neither 12SL table is sorted by id. Eyeballing the header suggests there is
   no key; assuming ascending order attaches rows to the wrong records.
2. The `median_beats/12sl` headers are unreadable by wfdb (a stale
   `ge_median_beats_wfdb/` prefix in the record line), and the `unig` amplitudes
   are ~1000x their declared `/mV` gain. ECGBench therefore exposes median beats
   as paths only, never as decoded signals.
3. `unig_features.csv` is missing 4 records that the other tables have.

Prerequisites:
  - pip install ecgbench[torch]
  - Local copies of BOTH PTB-XL and PTB-XL+ (PTB-XL+ has no waveforms).

Usage:
  python examples/load_ptbxl_plus.py \\
      --ptbxl-path /path/to/ptb-xl/1.0.3/ \\
      --plus-path  /path/to/ptb-xl-plus/1.0.1/
"""

import argparse

from ecgbench import ECGDataset
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.ptbxl_plus import (
    load_feature_description,
    load_features,
    load_ptbxl_plus,
    load_snomed_description,
    median_beat_path,
)


def main():
    parser = argparse.ArgumentParser(description="Join PTB-XL+ onto PTB-XL's folds")
    parser.add_argument("--ptbxl-path", required=True, help="PTB-XL root (waveforms)")
    parser.add_argument("--plus-path", required=True, help="PTB-XL+ root (annotations)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    args = parser.parse_args()

    print("PTB-XL+ is an annotation layer, so it is consumed through PTB-XL's splits.")
    print("There is no ptbxl_plus config and no separate fold assignment.\n")

    ds = ECGDataset(
        "ptbxl",
        split=args.split,
        data_path=args.ptbxl_path,
        metadata_source=args.metadata_source,
        labels=True,
    )
    print(f"PTB-XL {args.split} split: {len(ds):,} records on PTB-XL's official folds")

    try:
        plus = load_ptbxl_plus(args.plus_path, features=("unig",))
    except LabelSourceMissingError as e:
        print(f"PTB-XL+ unavailable: {e}")
        return

    print(f"PTB-XL+ frame:            {plus.shape[0]:,} records x {plus.shape[1]} columns")

    ecg_ids = ds.metadata_df["ecg_id"]
    joined = plus.reindex(ecg_ids.values)
    matched = int(joined.notna().any(axis=1).sum())
    print(
        f"joined on ecg_id:         {matched:,} of {len(ecg_ids):,} "
        f"({100 * matched / len(ecg_ids):.2f}%)"
    )

    print("\nTwo independent opinions on the same recording:")
    row = joined.iloc[0]
    print(f"  ecg_id            {ecg_ids.iloc[0]}")
    print(f"  PTB-XL  (human)   {row['ptbxl_scp_codes']}")
    print(f"  12SL (algorithm)  {row['12sl_statements']}")
    print("  -> PTB-XL's statements are cardiologist-assigned; 12SL's are a")
    print("     commercial algorithm's. Comparing them is much of the point of PTB-XL+.")

    print("\nMeasured features for that record (Uni-G):")
    for column in ("unig_QRS_Dur_Global", "unig_QT_Int_Global", "unig_P_On_I"):
        if column in joined.columns:
            print(f"  {column:24s} {row[column]}")

    print("\nThe hidden-key trap, shown rather than described:")
    import pandas as pd

    raw_cols = list(pd.read_csv(f"{args.plus_path}/features/12sl_features.csv", nrows=0).columns)
    position = raw_cols.index("ecg_id") + 1
    print(f"  12sl_features.csv has {len(raw_cols)} columns; ecg_id is column {position},")
    print(f"  between {raw_cols[position - 2]!r} and {raw_cols[position]!r} — not at the front,")
    print("  so checking the first/last few columns suggests there is no key at all.")
    f12 = load_features(args.plus_path, "12sl")
    print(f"  Keyed index order starts: {list(f12.index[:5])}")
    print("  Not ascending — so a positional join assuming sorted ids is also wrong.")

    print("\nCoverage differs between providers:")
    for provider in ("ecgdeli", "unig"):
        n = len(load_features(args.plus_path, provider))
        print(f"  {provider:8s} {n:,} records" + ("" if n == 21799 else f"  ({21799 - n} missing)"))

    print("\nSNOMED is the shared vocabulary the two statement sets map into:")
    snomed = load_snomed_description(args.plus_path)
    print(f"  {len(snomed)} concepts; in both sets: {int(snomed['in_both'].sum())}")

    fd = load_feature_description(args.plus_path)
    print(f"\nFeature dictionary: {len(fd)} rows mapping equivalent columns across providers,")
    print("  e.g. P wave amplitude is unig 'P_Amp_X' / 12sl 'P_PeakAmpl_X' / ecgdeli 'PWa_X'.")

    print("\nMedian beats are exposed as paths only, never decoded:")
    for provider in ("unig", "12sl"):
        path = median_beat_path(args.plus_path, int(ecg_ids.iloc[0]), provider)
        print(f"  {provider:5s} -> {path.name if path else '(absent)'}")
    print("  12sl headers are unreadable by wfdb (stale 'ge_median_beats_wfdb/' prefix)")
    print("  and unig amplitudes are ~1000x their declared /mV gain, so ECGBench will")
    print("  not present either as a millivolt signal.")

    print("\nWaveforms still come from PTB-XL:")
    print(f"  ds[0]['signal'].shape = {tuple(ds[0]['signal'].shape)}")


if __name__ == "__main__":
    main()
