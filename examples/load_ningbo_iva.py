#!/usr/bin/env python3
"""
Example: Ningbo First Hospital IVA — predicting an arrhythmia's origin.

334 twelve-lead ECGs recorded *during* catheter ablation, each labelled with the
outflow tract the ablation proved the arrhythmia came from. The label is invasive
ground truth rather than an ECG reading, which is what makes 334 records worth
having. Four things about it are worth seeing rather than reading, and this script
shows all four:

1. **The lead order is ALPHABETICAL** — aVF, aVL, aVR, I, II, III, V1..V6. So
   `signal[0]` is aVF, not lead I, and `signal[4]` is lead II, not aVL. The
   `leads=[...]` selection below is the fix.
2. **The samples carry no declared unit.** The release ships bare integers and
   neither the paper nor figshare states a gain, so ECGBench supplies an estimated
   one (1 mV = 16384 counts, measured against `sph`). `units="uV"` below shows the
   conversion, and the raw counts are one division away.
3. **The sampling rate is 2000 Hz** — the highest in the catalogue, and four times
   the usual 500 Hz. A 10 s window is 20,000 samples here.
4. **Record length varies from 2.9 s to 59 s** in 317 distinct lengths over 334
   records, so records cannot be batched without a window.

Labels come from the shipped Diagnosis.xlsx, so this works without running the
split pipeline first. The fold CSVs come from the Hub (or from a local run with
--metadata-source local).

Prerequisites:
  - pip install ecgbench[torch] openpyxl
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_ningbo_iva.py --data-path /path/to/Ningbo_IVA/
"""

import argparse

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError

#: Records are variable length, so every signal is windowed to this many samples.
#: 5000 is 2.5 s at 2000 Hz, and the SHORTEST record in the dataset is 5,791
#: samples (2.9 s), so this window fits all 334. Anything longer raises
#: WindowOutOfRangeError on that record.
WINDOW = (0, 5000)

#: Standard 12-lead order. The files store the leads alphabetically, so asking by
#: name is the only way to get the order a model trained elsewhere expects.
STANDARD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    parser = argparse.ArgumentParser(description="Load Ningbo IVA with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("ningbo_iva")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Rate:     {config.default_sampling_rate} Hz  <- 4x the usual 500 Hz")
    print(f"Leads:    {config.lead_names}")
    print("          ^ ALPHABETICAL. signal[0] is aVF, and lead I is signal[3].")
    print(f"Unit:     x{config.signal_unit_scale:g} mV/count  <- ESTIMATED, not declared")
    print()

    try:
        dataset = ECGDataset(
            "ningbo_iva",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            # window=(start, length) is pushed into the csv reader's skiprows/
            # max_rows rather than cropped afterwards, so a 59 s record at 2000 Hz
            # parses 5,000 rows instead of 118,642. Unlike a lambda transform it
            # also survives DataLoader(num_workers>0) under "spawn".
            window=WINDOW,
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}  (one per patient — HospitalID is both ids)")

    sample = dataset[0]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    labels = sample["labels"]
    print(f"  record          {sample['record_id']}")
    print(f"  left_right      {labels['left_right']}          <- ground truth, ablation-proven")
    print(f"  sublocation     {labels['sublocation']}")
    print(f"  arrhythmia_type {labels['arrhythmia_type']}")
    print(f"  sex             {labels['sex']} ({labels['sex_code']})   <- no age ships at all")

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print(f"\nOutflow tract in this split ({len(df)} records):")
    for tract, n in df["left_right"].value_counts().items():
        print(f"  {tract:5s} {n:4d}  ({100 * n / len(df):.1f}%)")
    print("  Strongly imbalanced by design — the cohort is 257 RVOT to 77 LVOT.")

    print("\nSub-site — the finer label, and NOT the split target:")
    counts = df["sublocation"].value_counts(dropna=False)
    for site, n in counts.items():
        name = "(blank)" if pd.isna(site) else str(site)
        print(f"  {name:18s} {n:4d}")
    print("  Blanks are left blank: the paper's Table 2 calls 45 RVOT patients")
    print("  'RVOTOther' where the file has 6 explicit plus 39 blanks, but that")
    print("  inference is ours, not the providers'.")

    print("\nPresentation type — the file disagrees with the paper here:")
    print(f"  {df['arrhythmia_type'].value_counts().to_dict()}")
    print("  Table 1 reports 325 PVC / 9 VT for the whole cohort; the shipped")
    print("  spreadsheet reads 329 / 5. The file is what ECGBench reports.")

    print("\nSex is skewed differently per class, so it leaks the label:")
    print(pd.crosstab(df["left_right"], df["sex"]).to_string())

    # --- the two read-time quirks, demonstrated rather than described ---

    print(f"\nLead order: files store {config.lead_names[:6]}")
    reordered = ECGDataset(
        "ningbo_iva",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=STANDARD_ORDER,
    )
    print(f"  leads={STANDARD_ORDER[:6]}... gives {tuple(reordered[0]['signal'].shape)}")
    shipped = sample["signal"]
    standard = reordered[0]["signal"]
    # aVF is index 0 as shipped and index 5 in the standard order.
    print(
        f"  shipped signal[0] == standard signal[5] (both aVF): {bool((shipped[0] == standard[5]).all())}"
    )
    print(
        f"  shipped signal[3] == standard signal[0] (both I):   {bool((shipped[3] == standard[0]).all())}"
    )

    raw_counts = ECGDataset(
        "ningbo_iva",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        units="uV",
    )[0]["signal"]
    print("\nUnits: the CSVs hold integers with no declared meaning.")
    print(f"  default (mV)  peak {shipped.abs().max().item():.3f}")
    print(f"  units='uV'    peak {raw_counts.abs().max().item():.1f}")
    print(f"  raw counts    peak {shipped.abs().max().item() / config.signal_unit_scale:.0f}")
    print(f"  Divide the mV values by {config.signal_unit_scale:g} to recover the shipped")
    print("  integers exactly, if you would rather calibrate them yourself.")

    # Binary target — the release's own task.
    target = (df["left_right"] == "LVOT").astype(int)
    print(f"\nBinary target (1 = LVOT): {int(target.sum())} positive of {len(target)}")
    print(f"  base rate {target.mean():.3f} — quote balanced accuracy, not accuracy")

    # Batching only works because of the window above.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print("  Without window= this raises — record length is essentially unique per")
    print("  record here (317 distinct lengths over 334 records, 2.9 s to 59.3 s),")
    print("  and torch cannot stack differing widths.")

    print("\nThe denoised copy: PVCVTECGData/ holds wavelet-denoised versions of the")
    print("same recordings under the same filenames, and load_labels exposes the path")
    print("as signal_path_denoised. It is NOT a drop-in substitute — the denoiser ran")
    print("per lead, so III = II - I no longer holds, and 106 of the 334 files are")
    print("shorter than their raw counterparts, so the two are not sample-aligned.")


if __name__ == "__main__":
    main()
