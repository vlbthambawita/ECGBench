#!/usr/bin/env python3
"""
Example: STAFF III Database — ECGs recorded during acutely induced ischaemia.

520 recordings from 104 patients undergoing elective coronary angioplasty. Unlike
most of this catalogue it is a **protocol** dataset: the label is not a diagnosis
but where in the procedure a recording sits, and the interesting records are the
142 taken while a balloon was inflated inside a coronary artery.

Four things to demonstrate, all of which will bite someone who assumes otherwise:

1. **There are 9 leads, not 12, and V1-V6 come FIRST.** `signal[0]` is V1, not
   lead I. aVR/aVL/aVF are not stored at all — they are exact linear combinations
   of I and II. This script selects by name to show the fix.
2. **Records run 94.5 s to 960 s.** A fixed `window=` must fit the *shortest*
   record, so 90 s is the largest round window that loads every record.
3. **Every patient is their own control.** Each contributed a baseline, one or
   more occlusions and a recovery, which is why folds are grouped by patient —
   and why a record-level split would score patient identity, not ischaemia.
4. **The event annotations are sample-accurate.** `inflation_start_s` and
   `inflation_duration_s` come from the `.event` files, so you can window
   straight onto the occluded interval instead of guessing.

Labels come from the annotation spreadsheet, the headers and the event files, so
this works without running the split pipeline first. The fold CSVs come from the
Hub (or `--metadata-source local` after copying `output/staffiii/{clean,original}/`
into the dataset root).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_staffiii.py --data-path /path/to/staffiii/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.staffiii import RECORDING_TYPES

#: 90 s at 1000 Hz. The shortest record in the dataset is 94.514 s (006a), so
#: this is the largest round window that fits every record — anything longer
#: raises WindowOutOfRangeError on the short ones. window= is pushed into the
#: wfdb reader, so the 960 s records decode 90 s rather than all 16 minutes.
WINDOW = (0, 90_000)


def main():
    parser = argparse.ArgumentParser(description="Load STAFF III with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("staffiii")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.leads} — {config.lead_names}")
    print("          ^ precordials FIRST; aVR/aVL/aVF are not stored")
    print()

    try:
        dataset = ECGDataset(
            "staffiii",
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
    print(f"\nSample keys:  {sorted(sample.keys())}")
    print(f"Signal shape: {tuple(sample['signal'].shape)}  (window {WINDOW})")
    print(f"  record            {sample['record_id']}")
    print(f"  patient_id        {labels['patient_id']}   <- folds are grouped by this")
    print(f"  recording_type    {labels['recording_type']} ({labels['recording_type_label']})")
    print(f"  duration          {labels['duration_seconds']} s")
    print(f"  age / sex         {labels['age'] or '(not recorded)'} / {labels['sex']}")
    print(f"  prior MI          {labels['prior_mi_location']}")
    if labels["recording_type"] == "BI":
        print(f"  occluded artery   {labels['occluded_artery']} ({labels['artery_territory']})")
        print(f"  inflation at      {labels['inflation_start_s']} s")
        print(f"  for               {labels['inflation_duration_s']} s")

    df = dataset.labels_df

    # --- the dataset's own label -------------------------------------------
    print("\nProtocol phase over this split (the label you train on):")
    for code, n in df["recording_type"].value_counts().items():
        print(f"  {code:3s} {n:4d}  {RECORDING_TYPES.get(code, '')}")

    # --- patient grouping: check it, don't assume it -----------------------
    patients = dataset.metadata_df[config.patient_id_column]
    print(f"\n{len(df)} records from {patients.nunique()} patients in this split")
    print(
        "  records per patient:",
        patients.value_counts().value_counts().sort_index().to_dict(),
    )

    # --- occlusion detail ---------------------------------------------------
    inflations = df[df["recording_type"] == "BI"]
    print(f"\n{len(inflations)} balloon-inflation records in this split:")
    territory = inflations["artery_territory"].str.split(";").explode().value_counts()
    for name, n in territory.items():
        print(f"  {n:3d} inflations in {name}")
    durations = inflations["inflation_duration_s"].str.split(";").explode().astype(float)
    print(
        f"  occlusion lasted {durations.min():.0f}-{durations.max():.0f} s "
        f"(median {durations.median():.0f} s)"
    )

    # --- record length is NOT constant, and it leaks the label -------------
    print("\nRecord length by phase — note this correlates with the label:")
    for code, group in df.groupby("recording_type")["duration_seconds"]:
        print(
            f"  {code:3s} {group.min():6.1f} - {group.max():6.1f} s "
            f"(median {group.median():6.1f})"
        )
    print("  A model given the raw length can read the phase off it. Window first.")

    # --- selecting leads by name, which is the only safe way here ----------
    print("\nLead order is non-standard, so select by NAME:")
    limb_only = ECGDataset(
        "staffiii",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=["I", "II", "III"],
    )
    print(f"  full   {tuple(sample['signal'].shape)}  {config.lead_names}")
    print(f"  leads= {tuple(limb_only[0]['signal'].shape)}  ['I', 'II', 'III']")
    print("  By index those three would be signal[6:9], not signal[0:3].")

    # --- windowing onto the occluded interval ------------------------------
    # labels_df is positional and row-aligned with metadata_df, not indexed by
    # record name — so take the row number and read the id from metadata_df.
    if len(inflations):
        row = int(inflations.index[0])
        record = dataset.metadata_df[config.record_id_column].iloc[row]
        start = float(str(df.loc[row, "inflation_start_s"]).split(";")[0])
        held = float(str(df.loc[row, "inflation_duration_s"]).split(";")[0])
        window = (int(start * 1000), int(min(held, 60) * 1000))
        print(f"\nWindowing onto the occlusion itself, using {record}:")
        print(f"  balloon up at {start:.0f} s, held {held:.0f} s")
        print(f"  window={window} reads the ischaemic interval and nothing else")
        occluded = ECGDataset(
            "staffiii",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            window=window,
        )
        print(f"  -> {tuple(occluded[row]['signal'].shape)}")

    # --- a batch through DataLoader ----------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print(f"  records {batch['record_id']}")

    # --- turning the label into a target tensor ----------------------------
    import torch

    # ecg_collate_fn keeps dicts as a LIST of per-sample dicts, so index the
    # sample first and the field second.
    codes = list(RECORDING_TYPES)
    index = {code: i for i, code in enumerate(codes)}
    phases = [sample["recording_type"] for sample in batch["labels"]]
    targets = torch.tensor([index[p] for p in phases], dtype=torch.long)
    print(f"  phases  {phases}")
    print(f"  targets {targets.tolist()}  over classes {codes}")

    # The binary task most papers on this dataset actually run.
    ischaemic = torch.tensor([p == "BI" for p in phases], dtype=torch.float32)
    print(f"  ischaemic (BI vs rest) {ischaemic.tolist()}")

    print("\nCaveats worth carrying into any experiment:")
    suspect = df[df["suspect_leads"] == "True"]
    print(f"  - {len(suspect)} records from patients 1, 4, 5, 6 and 89 are flagged by")
    print("    the depositors for possible lead or sign reversal (which leads is")
    print("    not recorded). Filter on labels['suspect_leads'] if that matters.")
    print("  - 089d is unreadable in the published release and is excluded from")
    print("    clean/; it appears in original/ with a corrupt_header flag.")
    print("  - Ischaemia here is induced and controlled. It does not resemble the")
    print("    onset dynamics of spontaneous coronary occlusion.")


if __name__ == "__main__":
    main()
