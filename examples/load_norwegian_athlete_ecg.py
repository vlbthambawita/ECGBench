#!/usr/bin/env python3
"""
Example: Norwegian Endurance Athlete ECG Database with both interpretations.

28 twelve-lead records, one per elite endurance athlete, 10 s at 500 Hz. The
dataset is tiny, and that is not the point of it: every record carries **two**
readings, appended to its WFDB header as comment lines — the GE Marquette SL12
algorithm's and a cardiologist's. Comparing them is the intended use, and the
disagreement is the finding: SL12 raises a critical ACUTE MI/STEMI alert on 4
athletes the cardiologist reads as normal or borderline.

Two things this script goes out of its way to show, because both are easy to
get wrong and neither is documented upstream:

1. **The amplitudes are not calibrated.** Every lead of every record was
   independently min-max normalised to the full int16 range, so each one spans
   exactly +/-0.6553 mV. Voltage criteria are not computable from this data and
   `units=` cannot rescue it — there is no per-lead factor to undo.
2. **The human labels are degenerate.** The cardiologist calls 26 of 28 records
   "Normal ECG" and 2 "Borderline ECG". There is no abnormal class at all, so
   `cardiologist_primary_rhythm` (16/7/5) is the only usable target — and it is a
   single-label reduction of a multi-statement reading.

Usage:
  python examples/load_norwegian_athlete_ecg.py --data-path /path/to/norwegian-athlete-ecg/1.0.0/
"""

import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config


def as_statements(value):
    """Normalise a findings cell to a list, whether it came from the label loader
    (a real list) or the generated metadata CSV (';'-joined — NOT comma-joined,
    because the statements contain commas themselves)."""
    if isinstance(value, list):
        return value
    text = "" if value is None else str(value)
    if text in ("", "nan"):
        return []
    return [part for part in text.split(";") if part]


def main():
    parser = argparse.ArgumentParser(
        description="Load the Norwegian athlete ECG database via ECGBench"
    )
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("norwegian_athlete_ecg")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {', '.join(config.lead_names)}")
    print("          ^ uppercase AVR/AVL/AVF, as PTB-XL spells them — not aVR/aVL/aVF")
    print()

    dataset = ECGDataset(
        "norwegian_athlete_ecg", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source, labels=True,
    )
    print(f"Records:  {len(dataset)}  (of 28 — folds hold 2-3 records each)")

    sample = dataset[0]
    labels = sample["labels"]
    signal = sample["signal"]
    print(f"\nLabel fields: {list(labels)}")
    print("\nFirst record:")
    print(f"  ID:                {sample['record_id']}")
    print(f"  signal:            {tuple(signal.shape)}  (leads, samples)")
    print(f"  SL12 findings:     {as_statements(labels['sl12_findings'])}")
    print(f"  SL12 verdict:      {labels['sl12_verdict']!r}")
    print(f"  cardiologist:      {as_statements(labels['cardiologist_findings'])}")
    print(f"  card. verdict:     {labels['cardiologist_verdict']!r}")
    print(f"  primary rhythm:    {labels['cardiologist_primary_rhythm']!r}")

    # --- 1. The amplitudes are normalised, not calibrated ----------------------
    print("\nPer-lead amplitude range of this record (mV as the header declares them):")
    for index, name in enumerate(config.lead_names[:4]):
        lead = signal[index]
        print(f"  {name:4s} min {lead.min():+.4f}  max {lead.max():+.4f}  "
              f"span {(lead.max() - lead.min()):.4f}")
    print("  ...")
    spans = (signal.max(dim=1).values - signal.min(dim=1).values)
    print(f"  All 12 leads span {spans.min():.4f}-{spans.max():.4f} mV — every lead was")
    print("  independently min-max normalised to the full int16 range. Amplitude is")
    print("  meaningless in absolute terms AND incomparable between leads, so LVH")
    print("  voltage criteria and ST elevation in mm cannot be computed here.")
    print("  Morphology and timing are unaffected. units= cannot undo this.")

    # --- 2. Selecting leads by name -------------------------------------------
    by_name = ECGDataset(
        "norwegian_athlete_ecg", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source,
        leads=["II", "V5"],
    )
    print(f"\nleads=['II','V5'] -> {tuple(by_name[0]['signal'].shape)}  "
          "(always ask by name; index 4 is AVL here but aVF in MIMIC-IV-ECG)")

    # --- 3. The two interpretations disagree ----------------------------------
    # labels_df is aligned POSITIONALLY with metadata_df and carries a RangeIndex,
    # not record IDs — attach them explicitly before reporting anything per record.
    frame = dataset.labels_df.copy()
    frame["record_id"] = dataset.metadata_df[config.record_id_column].to_numpy()

    print("\nVerdicts over this split (SL12 is the system under test, not the truth):")
    for source in ("sl12", "cardiologist"):
        counts = frame[f"{source}_verdict"].value_counts()
        print(f"  {source:13s} " + ", ".join(f"{k} ({v})" for k, v in counts.items()))

    overcalls = frame[frame["sl12_overcalls"].astype(bool)]
    print(f"\nSL12 flagged {len(overcalls)} of {len(frame)} records borderline/abnormal "
          "where the cardiologist read normal.")

    critical = frame[frame["sl12_critical_test_result"].notna()]
    if len(critical):
        print("\nCritical SL12 alerts — the dataset's headline result:")
        for _, row in critical.iterrows():
            print(f"  {row['record_id']}: {row['sl12_critical_test_result']} / "
                  f"{row['sl12_acute_alert']}  ->  cardiologist said "
                  f"{row['cardiologist_verdict']!r}")

    # --- 4. Findings and the stratification label ------------------------------
    print("\nMost common findings over this split (multi-label, so these do not sum "
          "to the record count):")
    for source in ("sl12", "cardiologist"):
        counter = Counter(
            statement
            for row in frame[f"{source}_findings"].map(as_statements)
            for statement in row
        )
        top = ", ".join(f"{k} ({v})" for k, v in counter.most_common(3)) or "-"
        print(f"  {source:13s} {len(counter):3d} distinct   {top}")

    print("\nStratification label (cardiologist_primary_rhythm) over this split:")
    for name, n in frame["cardiologist_primary_rhythm"].value_counts().items():
        print(f"  {name:22s} {n:3d}")
    print("  This is a single-label reduction of a multi-statement reading, used to")
    print("  balance the folds. Train on cardiologist_findings instead.")

    # --- 5. A batch, and a target tensor --------------------------------------
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=ecg_collate_fn)
    batch = next(iter(loader))
    print(f"\nFirst batch: signal {tuple(batch['signal'].shape)} "
          f"{batch['signal'].dtype}, {len(batch['labels'])} label dicts")

    classes = sorted(frame["cardiologist_primary_rhythm"].unique())
    targets = torch.tensor([
        classes.index(row["cardiologist_primary_rhythm"]) for row in batch["labels"]
    ])
    print(f"Targets:     {targets.tolist()}  over {classes}")

    print("\nNote: with 28 records in 10 folds, val and test hold 2 records each and")
    print("      only Normal sinus rhythm — no fold assignment avoids that. Use")
    print("      ECGDataset(split=None, fold_numbers=[...]) and rotate the held-out")
    print("      folds rather than training on the default train/val/test mapping.")


if __name__ == "__main__":
    main()
