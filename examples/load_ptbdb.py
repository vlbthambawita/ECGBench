#!/usr/bin/env python3
"""
Example: PTB Diagnostic ECG Database — 15 leads and variable-length records.

Two things make this dataset different from every other one in ECGBench:

  1. **15 signals**, not 12 — the conventional leads plus the three Frank
     vectorcardiography leads vx, vy, vz. Use leads=[...] to take the standard 12.
  2. **Variable-length records** — 11 distinct lengths between 32 s and 120 s. They
     cannot be batched as-is: torch cannot stack tensors of different widths. Take
     a fixed window=(start, length), or use batch_size=1.

Labels live only in the .hea comment blocks: 47 clinical fields per record,
including a full haemodynamics panel. ecgbench.labels.ptbdb parses them.

Usage:
  python examples/load_ptbdb.py --data-path /path/to/ptbdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config

STANDARD_12 = ["i", "ii", "iii", "avr", "avl", "avf",
               "v1", "v2", "v3", "v4", "v5", "v6"]


def main():
    parser = argparse.ArgumentParser(description="Load PTBDB with labels via ECGBench")
    parser.add_argument("--data-path", default=None, help="Path to the PTBDB root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--seconds", type=int, default=10,
                        help="Crop length in seconds (records are variable length)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("ptbdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Signals:  {config.leads} — {', '.join(config.lead_names)}")
    print(f"Rate:     {config.default_sampling_rate} Hz, records 32-120 s (NOT uniform)")
    print()

    # No crop: shows the real, unequal lengths.
    raw = ECGDataset("ptbdb", split=args.split, version=args.version,
                     data_path=args.data_path, metadata_source=args.metadata_source,
                     labels=True)
    print(f"Records:  {len(raw)}  from {raw.metadata_df.patient_id.nunique()} patients")
    print("\nLengths differ between records:")
    for i in range(4):
        s = raw[i]["signal"]
        print(f"  {raw.metadata_df.record_name[i]:12s} {tuple(s.shape)}  "
              f"{s.shape[1] / config.default_sampling_rate:6.1f} s")

    sample = raw[0]
    labels = sample["labels"]
    print(f"\nFirst record ({sample['record_id']}):")
    print(f"  diagnosis:   {labels['diagnosis']!r}")
    print(f"  age / sex:   {labels['age']} / {labels['sex']}")
    print(f"  acute MI:    {labels['Acute infarction (localization)']!r}")
    print(f"  stenosis:    {labels['Left coronary artery stenoses (RIVA)']!r}")
    print(f"  {len(labels)} label fields in total — the headers carry a full "
          "haemodynamics and therapy panel")

    print("\nDiagnosis distribution over this split:")
    for name, n in raw.labels_df["diagnosis"].replace("", "(none recorded)") \
            .value_counts().items():
        print(f"  {name:28s} {n:4d}")

    # Batching needs equal widths: window, and take only the standard 12 leads.
    # window= is read at load time, so the other 22-110 s are never decoded.
    n = args.seconds * config.default_sampling_rate
    cropped = ECGDataset("ptbdb", split=args.split, version=args.version,
                         data_path=args.data_path, metadata_source=args.metadata_source,
                         leads=STANDARD_12, window=(0, n))
    loader = DataLoader(cropped, batch_size=args.batch_size, shuffle=False,
                        collate_fn=ecg_collate_fn)
    batch = next(iter(loader))
    print(f"\nWith leads=STANDARD_12 and a {args.seconds}s window, batching works:")
    print(f"  signal {tuple(batch['signal'].shape)} {batch['signal'].dtype}")
    print(f"  lead_names {cropped.lead_names}")

    print("\nNote: without a window, DataLoader raises \"stack expects each tensor to")
    print("      be equal size\" as soon as a batch mixes two record lengths. window= is")
    print("      applied first, so leads= then selects from the windowed signal.")


if __name__ == "__main__":
    main()
