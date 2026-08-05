#!/usr/bin/env python3
"""
Example: EchoNext — structural heart disease labels from a matched echocardiogram.

100,000 12-lead ECGs from Columbia, each paired with an echo. Four things about
this dataset are unlike anything else in ECGBench, and this script demonstrates
all of them:

1. **Signals are rows of a shared array, not files.** `signal_format: npy`, and a
   record's path is `EchoNext_test_waveforms.npy:417`. Reads are memory-mapped, so
   pulling one record out of the 17 GB training array costs a few milliseconds.
2. **Samples are z-scores, not millivolts.** The publisher median-filtered,
   clipped and standardised the waveforms and did not release the mean/SD, so
   `units="uV"` raises `UnitConversionError` instead of silently multiplying by
   1000, and `amplitude_outlier` validation is skipped.
3. **Splits are the publisher's own, and `no_split` is dropped.** Its 17,457
   records share patients with val and test, so folding them into train — which is
   what an unmapped fold does by default — would leak. Excluded, the three splits
   are patient-disjoint.
4. **A 0 label can mean "not measured".** Every flag is 0/1 with no nulls, but the
   underlying echo measurement is often absent, and then the flag reads 0. Use the
   `<flag>_measured` masks.

The fold CSVs are NOT on the HuggingFace Hub — EchoNext is under the PhysioNet
Restricted Health Data License, so you regenerate them:

    ecgbench splits --dataset echonext --data-path /path/to/echonext/1.1.0/

then point `--splits-path` here at `output/echonext/`, or copy that tree into the
data directory. `metadata_source="local"` is required either way.

Prerequisites:
  - pip install ecgbench[torch]
  - A credentialed copy: https://physionet.org/content/echonext/1.1.0/

Usage:
  python examples/load_echonext.py --data-path /path/to/echonext/1.1.0/
"""

import argparse

from ecgbench import ECGDataset
from ecgbench.dataset import UnitConversionError
from ecgbench.labels.echonext import (
    COMPOSITE_FLAG,
    FLAG_SOURCES,
    README_TABULAR_FEATURE_COLUMNS,
    TABULAR_FEATURE_COLUMNS,
    load_labels,
    load_tabular_features,
)


def main():
    parser = argparse.ArgumentParser(description="Load EchoNext with ECGBench")
    parser.add_argument("--data-path", required=True, help="EchoNext 1.1.0 root")
    parser.add_argument(
        "--splits-path",
        default=None,
        help="Directory holding clean/ and original/ (default: --data-path)",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    splits_path = args.splits_path or args.data_path

    print("EchoNext splits are not published — they are regenerated locally, so")
    print("metadata_source='local' is required. See the module docstring.\n")

    ds = ECGDataset(
        "echonext",
        split=args.split,
        version="clean",
        data_path=splits_path,
        metadata_source="local",
        labels=True,
    )

    # ------------------------------------------------------------------ signals
    print("=" * 72)
    print("SIGNALS -- rows of a shared memory-mapped array")
    print("=" * 72)
    sample = ds[0]
    print(f"split={args.split!r}  n={len(ds):,}  signal={tuple(sample['signal'].shape)}")
    print(f"record_id={sample['record_id']}  fold={sample['fold']}")
    print(f"dataset units: {ds.units!r}  <- NOT millivolts")
    print(
        f"sample range: [{sample['signal'].min():.3f}, {sample['signal'].max():.3f}]"
        "  (z-scored, so ~N(0,1))"
    )

    # The scale is unrecoverable, so the loader refuses rather than guessing.
    try:
        ECGDataset(
            "echonext", split=args.split, data_path=splits_path, metadata_source="local", units="uV"
        )
    except UnitConversionError as exc:
        print(f"\nunits='uV' -> UnitConversionError: {str(exc).splitlines()[0]}")

    # leads= works by name because lead_names was inferred from Einthoven's law.
    two = ECGDataset(
        "echonext",
        split=args.split,
        data_path=splits_path,
        metadata_source="local",
        leads=["II", "V5"],
    )
    print(f"\nleads=['II','V5'] -> {tuple(two[0]['signal'].shape)}")
    half = ECGDataset(
        "echonext",
        split=args.split,
        data_path=splits_path,
        metadata_source="local",
        window=(0, 1250),
    )
    print(f"window=(0,1250)   -> {tuple(half[0]['signal'].shape)}  (first 5 s of 10)")

    # ------------------------------------------------------------------- labels
    print("\n" + "=" * 72)
    print("LABELS -- and why a 0 is not always a negative")
    print("=" * 72)
    labels = load_labels(args.data_path)
    print(
        f"{len(labels):,} records x {labels.shape[1]} columns "
        "(all 100,000, including the excluded no_split rows)\n"
    )

    print(f"{'flag':52}{'all':>8}{'measured':>10}{'n measured':>12}")
    for flag in (*FLAG_SOURCES, COMPOSITE_FLAG):
        mask = f"{flag}_measured"
        if mask in labels.columns:
            measured = labels[labels[mask]]
            print(
                f"  {flag:50}{labels[flag].mean():8.3f}"
                f"{measured[flag].mean():10.3f}{int(labels[mask].sum()):12,}"
            )
        else:
            print(f"  {flag:50}{labels[flag].mean():8.3f}{'--':>10}{'composite':>12}")

    print("\nThe two rightmost columns differ most where the echo measurement is")
    print("often missing: an unmeasured value is recorded as a 0 flag, never a null.")

    # ------------------------------------------------- the README's column order
    print("\n" + "=" * 72)
    print("TABULAR FEATURES -- the release's README lists them in the wrong order")
    print("=" * 72)
    tabular = load_tabular_features(args.data_path, args.split)
    print(f"shape {tabular.shape}")
    print(f"  README order: {list(README_TABULAR_FEATURE_COLUMNS)}")
    print(f"  TRUE order:   {list(TABULAR_FEATURE_COLUMNS)}")
    print("  age_at_ecg is column 1, not column 6 -- verified by rank-correlating")
    print("  every array column against the metadata (Spearman 1.000 on all splits).")

    # ------------------------------------------------------------------ batching
    print("\n" + "=" * 72)
    print("ONE BATCH")
    print("=" * 72)
    from torch.utils.data import DataLoader

    from ecgbench import ecg_collate_fn

    batch = next(iter(DataLoader(ds, batch_size=8, collate_fn=ecg_collate_fn)))
    print(f"signal batch: {tuple(batch['signal'].shape)}")
    targets = [row[COMPOSITE_FLAG] for row in batch["labels"]]
    print(f"{COMPOSITE_FLAG}: {targets}")
    print("\nTrain on the composite, or on a per-condition flag masked to its")
    print("_measured subset -- but do not mix the two definitions of a negative.")


if __name__ == "__main__":
    main()
