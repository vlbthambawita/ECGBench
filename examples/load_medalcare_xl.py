#!/usr/bin/env python3
"""
Example: MedalCare-XL — 16,842 ECGs that no heart ever produced.

Every record here is the output of a multi-scale electrophysiological simulation,
not a recording. That makes it unlike anything else in the catalogue in four ways
worth seeing rather than reading:

1. **The label is exact, and that is a warning as much as a feature.** There is no
   reader disagreement, no comorbidity, no borderline case — the label is the
   condition the simulator was told to produce. A model that separates these
   classes perfectly has learned to separate simulator settings.
2. **The CSVs are transposed.** 12 rows x 5000 columns and no header, which is why
   `signal_format` is `csv_lead_rows` rather than `csv`. Reading one with the
   ordinary CSV reader returns a plausibly-shaped array of the wrong thing rather
   than raising.
3. **Each record ships three times** — raw simulator output, the same with noise,
   and a filtered version. ECGBench wires up the filtered one. This script shows
   all three side by side, because the difference is the whole point of the
   release for anyone studying robustness to noise.
4. **The split is the authors' own, and its guarantee has one hole.** Ventricular
   model S64 is test-side for five pathologies and train-side for three others.
   `model_id` is in every fold CSV so you can see it; this script prints it.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset (https://doi.org/10.5281/zenodo.8068944). It
    extracts to a MedalCare-XL/MedalCare-XL/ nesting — point --data-path at the
    inner one, the directory holding WP2_largeDataset_Noise/.
  - `ecgbench splits --dataset medalcare_xl --data-path ...` must have been run
    once against a WRITABLE copy: the release ships no metadata table, so the
    labels come from a CSV that run generates.

Usage:
  python examples/load_medalcare_xl.py --data-path /path/to/MedalCare-XL/MedalCare-XL/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.medalcare_xl import load_simulation_parameters

#: Every record is exactly 5000 samples (10 s at 500 Hz), so a window is optional
#: here — unlike the variable-length datasets. This one takes the middle 4 s to
#: show the mechanism and to keep the batch small.
WINDOW = (3000, 2000)

#: The lead the amplitude blowups land on — see the excluded-records section.
BLOWUP_LEAD = "V1"


def main():
    parser = argparse.ArgumentParser(description="Load MedalCare-XL with its labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("medalcare_xl")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Format:   {config.signal_format}  <- 12 rows x 5000 cols, NO header")
    print(f"Leads:    {config.lead_names}")
    print(f"Rate:     {config.default_sampling_rate} Hz, {config.duration_seconds} s, uniform")
    print(f"Folds:    {config.predefined_splits.fold_mapping}  <- the authors' own split")
    print()

    try:
        dataset = ECGDataset(
            "medalcare_xl",
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
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    print(f"  record        {sample['record_id']}")
    print(f"  pathology     {labels['pathology']}  ({labels['pathology_name']})")
    print(f"  subclass      {labels['pathology_subclass']}")
    print(f"  model_id      {labels['model_id']}   <- the simulation model, not a patient")
    print(f"  source_split  {labels['source_split']}")
    print(f"  signal        {labels['signal_path']}")

    # labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print("\nTHE LABEL IS A SIMULATOR SETTING, NOT A DIAGNOSIS:")
    print(f"  8 pathologies, {df['pathology_subclass'].nunique()} subclasses in this split")
    for name, n in df["pathology"].value_counts().items():
        print(f"  {name:10s} {n:6d}  ({100 * n / len(df):5.1f}%)")
    print("  Single-label: every record has exactly one condition, none unlabelled.")
    print(f"  unlabelled: {int(df['pathology'].isna().sum())}")

    print("\n  Myocardial infarction decomposes into occlusion site x transmurality:")
    mi = df[df["pathology"] == "mi"]
    for name, n in mi["mi_subclass"].value_counts().sort_index().items():
        print(f"    {name:16s} {n:5d}")
    print("    ant/post is resolved for LCX only — the one site the release splits.")

    print("\nTHE SPLIT GUARANTEE HAS ONE HOLE — see model_id:")
    print(f"  models in this split: {sorted(df['model_id'].unique())}")
    print("  S64 is the test-side model for sinus/avblock/lbbb/rbbb/mi and a")
    print("  train-side model for fam/iab/lae; S67 likewise for val vs train.")
    print("  Verified at the parameter level: the two agree on all 84 ventricular")
    print("  parameters constant within a run directory. No RECORDS are shared, so")
    print("  this is shared anatomy, not duplicate rows — but a classifier spanning")
    print("  the atrial and ventricular pathology arms sees S64 on both sides.")
    straddling = df[df["model_id"].isin(["S64", "S67"])]
    print(f"  records on a straddling model in this split: {len(straddling)}")

    print("\nONE RECORD, THREE RENDERINGS (the config wires up `filtered`):")
    root = Path(dataset.data_path)
    row = df.iloc[0]
    for name, column in (
        ("raw", "signal_path_raw"),
        ("noise", "signal_path_noise"),
        ("filtered", "signal_path"),
    ):
        signal = np.loadtxt(root / row[column], delimiter=",", dtype=np.float32)
        v1 = signal[config.lead_names.index(BLOWUP_LEAD)]
        print(
            f"  {name:9s} shape {signal.shape}  {BLOWUP_LEAD} range "
            f"{v1.min():+.3f} to {v1.max():+.3f} mV   sd {v1.std():.4f}"
        )
    print("  Same simulation; the noise and filter stages are the only difference.")
    print("  Swap `signal_path` for either of the other two to train on them.")

    print("\n32 RECORDS ARE EXCLUDED FROM `clean`, AND IT IS NOT A UNITS BUG:")
    print("  All 32 fail amplitude_outlier, 30 of them on V1 — up to 879 mV where")
    print("  the release's median record peaks at 2.09 mV and its 99th percentile")
    print("  at 5.30 mV. The blowup is already in the RAW simulator output, so it")
    print("  is a forward-solution artefact, not the noise or filtering stage.")
    print("  Use version='original' to get them back, with is_valid marking them.")

    print("\nTHE SIMULATION PARAMETERS ARE THE REAL GROUND TRUTH (opt-in):")
    # labels_df is reindexed positionally against the split, so record ids come
    # from metadata_df — the two are aligned row for row. Take some MI records
    # deliberately: the isch[0].* block only exists for those.
    record_ids = dataset.metadata_df[config.record_id_column]
    is_mi = (df["pathology"] == "mi").to_numpy()
    ids = list(record_ids[~is_mi][:100]) + list(record_ids[is_mi][:100])
    params = load_simulation_parameters(root, config, record_ids=ids)
    print(f"  {params.shape[0]} records x {params.shape[1]} parameters")
    interesting = [
        "atrial.im.name",
        "atrial.geo.atria",
        "atrial.geo.torso",
        "ventricular.im.name",
        "ventricular.APD.max",
    ]
    for column in interesting:
        if column in params.columns:
            print(f"  {column:24s} {params[column].iloc[0]}")
    ischaemic = [c for c in params.columns if c.startswith("ventricular.isch")]
    populated = params[ischaemic].notna().any(axis=1).sum() if ischaemic else 0
    print(
        f"  {len(ischaemic)} isch[0].* columns, populated for {populated} of "
        f"{len(params)} records — the MI ones."
    )
    print("  A NaN there means the pathology has no such parameter, not missing data.")
    print("  This reads 2 text files per record, so pass record_ids= for a split.")

    print("\nOne batch through DataLoader + ecg_collate_fn:")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
    )
    batch = next(iter(loader))
    print(f"  signal batch: {tuple(batch['signal'].shape)}  dtype {batch['signal'].dtype}")
    print(f"  record_ids:   {batch['record_id'][:3]} ...")

    print("\nTurning the labels into a target tensor:")
    classes = sorted(df["pathology_subclass"].unique())
    index = {name: i for i, name in enumerate(classes)}
    # ecg_collate_fn keeps label dicts as a LIST of dicts, not a dict of lists.
    targets = torch.tensor(
        [index[sample["pathology_subclass"]] for sample in batch["labels"]],
        dtype=torch.long,
    )
    print(f"  {len(classes)} classes: {classes}")
    print(f"  targets: {targets.tolist()}")
    print("  Single-label, so a plain LongTensor and cross-entropy — no multi-hot.")


if __name__ == "__main__":
    main()
