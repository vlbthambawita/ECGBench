#!/usr/bin/env python3
"""
Example: LUDB (Lobachevsky University ECG Database) with labels.

200 twelve-lead records, one per patient, 10 s at 500 Hz. The dataset's real value
is its manual P/QRS/T delineation — 12 annotation files per record, one per lead —
which ECGBench does not model: it splits and loads signals, and the annotations
stay where they are for you to read with wfdb.rdann.

Diagnoses come from ludb.csv, whose cells all carry trailing newlines and whose
multi-value cells are newline-joined. ecgbench.labels.ludb normalises both.

Usage:
  python examples/load_ludb.py --data-path /path/to/ludb/1.0.1/
"""

import argparse
from collections import Counter

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config

MULTI_LABEL = [
    "rhythms", "conduction_abnormalities", "extrasystolies", "hypertrophies",
    "cardiac_pacing", "ischemia", "repolarization_abnormalities", "other_states",
]


def as_labels(value):
    """Normalise a multi-label cell to a list, whether it came from the loader
    (a real list) or from the generated metadata CSV (';'-joined)."""
    if isinstance(value, list):
        return value
    text = "" if value is None else str(value)
    if text in ("", "nan"):
        return []
    return [part for part in text.split(";") if part]


def main():
    parser = argparse.ArgumentParser(description="Load LUDB with labels via ECGBench")
    parser.add_argument("--data-path", default=None, help="Path to the LUDB root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("ludb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {', '.join(config.lead_names)}  (lowercase in the headers)")
    print()

    dataset = ECGDataset(
        "ludb", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, labels=True,
    )
    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nLabel fields: {list(labels)}")
    print("\nFirst record:")
    print(f"  ID:             {sample['record_id']}")
    print(f"  signal:         {tuple(sample['signal'].shape)}  "
          f"min {sample['signal'].min():.3f} max {sample['signal'].max():.3f} mV")
    print(f"  rhythms:        {labels['rhythms']}")
    print(f"  hypertrophies:  {labels['hypertrophies']}")
    print(f"  electric_axis:  {labels['electric_axis']!r}")
    print(f"  age / sex:      {labels['age_raw']} / {labels['sex']}")

    print("\nDiagnosis categories over this split "
          "(multi-label; most are sparse by design):")
    for column in MULTI_LABEL:
        # labels_df comes from the label loader, so these are real lists. (The
        # generated metadata CSV joins them with ';' to survive a CSV round-trip.)
        values = dataset.labels_df[column].map(as_labels)
        counts = Counter(x for row in values for x in row)
        populated = int((values.map(len) > 0).sum())
        top = ", ".join(f"{k} ({v})" for k, v in counts.most_common(2)) or "-"
        print(f"  {column:30s} {populated:4d} records, {len(counts):3d} labels   {top}")

    print("\nStratification label (primary_rhythm, rare rhythms pooled):")
    for name, n in dataset.labels_df["primary_rhythm"].value_counts().items():
        print(f"  {name:22s} {n:4d}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=ecg_collate_fn)
    batch = next(iter(loader))
    print(f"\nFirst batch: signal {tuple(batch['signal'].shape)} "
          f"{batch['signal'].dtype}, {len(batch['labels'])} label dicts")

    print("\nNote: the 12 per-lead annotation files (.i .ii ... .v6) hold manually")
    print("      marked P/QRS/T onsets, peaks and offsets — not diagnoses. Read them")
    print("      directly, e.g. wfdb.rdann(f'{data_path}/data/1', 'ii').")


if __name__ == "__main__":
    main()
