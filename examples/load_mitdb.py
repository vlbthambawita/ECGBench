#!/usr/bin/env python3
"""
Example: MIT-BIH Arrhythmia Database with labels.

48 half-hour two-channel Holter excerpts from **47 subjects**, with 109,494
consensus-reviewed reference beat annotations. Four things to demonstrate:

1. **The two leads are not the same two in every record.** 40 records store
   MLII/V1; the rest store MLII/V5, MLII/V2, MLII/V4 or V5/V2, and record 114
   stores the predominant pair *reversed*. Every record holds exactly 2 leads, so
   nothing about the shape gives it away — `signal[0]` is a limb-type lead in 46
   records and a chest lead in 2. `leads=["MLII"]` is the fix, and this script
   shows both what it returns and where it correctly refuses.
2. **Records are 650,000 samples (~5 MB each).** `window=(start, length)` is
   needed to batch at all, and because it is pushed into the reader it avoids
   decoding the other 1795 s.
3. **Labels are annotation-derived, not diagnostic.** Beat counts across 15
   types, seconds spent in each of 15 annotated rhythms, and artefact markers.
   There is no record-level diagnosis; the free-text `description` is the
   nearest thing.
4. **Folds are grouped by analog tape.** Records 201 and 202 came off the same
   tape, so they are the same subject and must not be split apart.

Labels come straight from the headers and annotations, so this works without
running the split pipeline first. The fold CSVs do come from the pipeline (or the
Hub) — pass --metadata-source local after copying output/mitdb/{clean,original}/
into the dataset root.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_mitdb.py --data-path /path/to/mitdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.mitdb import BEAT_NAMES, BEAT_SYMBOLS, RHYTHM_NAMES

#: 10 s at 360 Hz. Records are 650,000 samples, so a window is required to batch
#: at all — and because window= pushes down into the reader, it also avoids
#: decoding the other 1795 s.
WINDOW = (0, 3600)


def main():
    parser = argparse.ArgumentParser(description="Load MIT-BIH Arrhythmia with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("mitdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- the PREDOMINANT layout, not the only one")
    print(f"Layouts:  {config.record_lead_layouts}")
    print(f"Duration: {config.duration_seconds} s per record")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("mitdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW} of 650000)")
    print(f"  record            {sample['record_id']}")
    print(f"  lead_names        {labels['lead_names']}   <- this record's own layout")
    print(f"  patient_id        {labels['patient_id']}   <- folds are grouped by this")
    print(f"  age / sex         {labels['age']} / {labels['sex']}")
    print(f"  medications       {labels['medications'] or '(none)'}")
    print(f"  description       {labels['description'] or '(none)'}")
    rhythm = labels["dominant_rhythm"]
    print(f"  dominant_rhythm   {rhythm} ({RHYTHM_NAMES.get(rhythm, '?')})")
    print(f"  rhythms           {labels['rhythms']}")
    print(f"  n_beats           {labels['n_beats']}  (PVC fraction {labels['pvc_fraction']:.4f})")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. The lead layout, and why an index is not a lead -------------------
    print("\nLead layouts in this split:")
    for layout, n in df["lead_names"].value_counts().items():
        holding = list(df.index[df["lead_names"] == layout][:4])
        print(f"  {n:3d}  {layout:10s}  e.g. records {holding}")

    odd = df[df["lead_names"] != "|".join(config.lead_names)]
    if len(odd):
        print(
            f"\n  {len(odd)} of {len(df)} records do NOT store {config.lead_names}. "
            "Selecting by index crosses leads for them:"
        )
        for record, layout in odd["lead_names"].items():
            stored = layout.split("|")
            note = (
                f"MLII at position {stored.index('MLII')}"
                if "MLII" in stored
                else "no MLII at all"
            )
            print(f"    {record}: {layout:10s}  ({note})")

    # Select by NAME, and ECGBench re-resolves it against each record's header.
    mlii = ECGDataset("mitdb", leads=["MLII"], **common)
    print(f"\nleads=['MLII'] -> {mlii.lead_names}, shape {tuple(mlii[0]['signal'].shape)}")
    refused = 0
    for i in range(len(mlii)):
        try:
            mlii[i]["signal"]
        except ValueError as e:
            refused += 1
            if refused == 1:
                print(f"  and it REFUSES rather than substituting: {e}")
    print(f"  {refused} record(s) in this split store no MLII and raise.")

    # --- 2. Subject grouping --------------------------------------------------
    subjects = dataset.metadata_df[config.patient_id_column]
    print(f"\n{len(df)} records from {subjects.nunique()} subjects in this split")
    shared = subjects.value_counts()
    shared = shared[shared > 1]
    print(f"  subjects with more than one record: {shared.to_dict() or '(none in this split)'}")

    # --- 3. Annotation-derived labels ----------------------------------------
    print("\nReference beat annotations over this split:")
    total = int(df["n_beats"].sum())
    for symbol in BEAT_SYMBOLS:
        n = int(df[f"beat_{symbol}"].sum())
        if not n:
            continue
        holding = int((df[f"beat_{symbol}"] > 0).sum())
        print(
            f"  {symbol:2s} {n:7d} ({100 * n / total:5.2f}%) in {holding:3d} records"
            f"  {BEAT_NAMES[symbol]}"
        )
    print(f"  total {total} beats")
    print(
        "  non-beat markers: "
        f"{int(df['n_rhythm_changes'].sum())} rhythm changes, "
        f"{int(df['n_signal_quality_changes'].sum())} quality changes, "
        f"{int(df['n_isolated_artifacts'].sum())} artefacts"
    )

    print("\nTime spent in each annotated rhythm, over this split:")
    seconds = {
        code: float(df[f"rhythm_secs_{code}"].sum())
        for code in RHYTHM_NAMES
        if float(df[f"rhythm_secs_{code}"].sum()) > 0
    }
    grand = sum(seconds.values())
    for code in sorted(seconds, key=lambda c: -seconds[c]):
        holding = int((df[f"rhythm_secs_{code}"] > 0).sum())
        print(
            f"  {code:5s} {seconds[code] / 60:8.1f} min ({100 * seconds[code] / grand:5.2f}%)"
            f" in {holding:3d} records  {RHYTHM_NAMES[code]}"
        )

    print("\nBeat types concentrated in one record (why a per-record split is not enough):")
    for symbol in BEAT_SYMBOLS:
        col = f"beat_{symbol}"
        if df[col].sum() == 0:
            continue
        share = df[col].max() / df[col].sum()
        if share > 0.9:
            print(
                f"  {symbol:2s} {BEAT_NAMES[symbol]:38s} "
                f"{100 * share:5.1f}% from record {df[col].idxmax()}"
            )

    # --- 4. Batching ----------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print("  Without window= each record is 2 x 650000 (~5 MB), so a batch of")
    print(f"  {args.batch_size} would be ~{5 * args.batch_size} MB, every byte decoded.")


if __name__ == "__main__":
    main()
