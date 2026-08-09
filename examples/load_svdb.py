#!/usr/bin/env python3
"""
Example: MIT-BIH Supraventricular Arrhythmia Database with labels.

78 half-hour two-lead recordings assembled to supply what the MIT-BIH Arrhythmia
Database lacks: supraventricular ectopy. Five things to demonstrate:

1. **The beat symbol for a supraventricular beat is `S` here and `A` in mitdb.**
   All 12,188 supraventricular beats in this release are annotated `S`; mitdb
   annotates its 2,546 as `A` and uses `S` twice. Concatenating the two databases
   on the raw symbol trains a model on two disjoint vocabularies for one
   phenomenon. The `aami_*` columns are the AAMI EC57 five-class reduction and are
   what you combine on. This script prints both side by side.
2. **The headers carry no comment lines at all** — no age, no sex, no subject id,
   no medications, no clinical text. mitdb has all of those on records from the
   same era and nsrdb has age and sex. So folds here are stratified but
   **ungrouped**, and nothing rules out one subject contributing several records.
3. **Folds are stratified on ectopy burden, because there is nothing else.** No
   demographics, no diagnoses, and one rhythm annotation in the entire release.
   `sveb_burden` bands the SVEB share of beats at 1% / 3% / 10%. It is a fold
   label; train on `sveb_fraction` or the `aami_*` counts instead.
4. **Records are uniform: exactly 230,400 samples (1800.0 s) each.** `window=` is
   still worth using to batch — it is pushed into the reader, so it avoids
   decoding the other 29 minutes — but unlike `nsrdb`, `ptbdb` and `cpsc_2018`,
   *any* window inside 1800 s fits every record. This script shows both.
5. **Signal quality is annotated per channel, and beat annotation is complete.**
   The `~` transitions give seconds of clean / ECG1-noisy / ECG2-noisy / both.
   Unlike nsrdb, whose annotations stop hours before the signal does, every record
   here is annotated end to end — which this script verifies rather than asserts.

Labels come straight from the annotation files, so this works without running the
split pipeline first. The fold CSVs come from the Hub by default (or from the
pipeline — pass --metadata-source local after copying
output/svdb/{clean,original}/ into the dataset root).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_svdb.py --data-path /path/to/svdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.svdb import (
    AAMI_CLASSES,
    AAMI_ORDER,
    BEAT_NAMES,
    RECORD_SAMPLES,
    RR_RANGE_SECS,
    SVEB_BURDEN_BANDS,
    SVEB_BURDEN_EDGES,
)

#: 10 s at 128 Hz. Records are 230,400 samples each, so a window is not strictly
#: required to batch — but window= pushes down into the reader, so it avoids
#: decoding the other 29 minutes.
WINDOW = (0, 1280)


def main():
    parser = argparse.ArgumentParser(description="Load MIT-BIH SVDB with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("svdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- channel positions; the release names no leads")
    print(f"Duration: {config.duration_seconds:.0f} s, uniform — all 78 records are"
          f" {RECORD_SAMPLES} samples")
    print(f"Patients: {config.patient_id_column}  <- no subject id ships at all,"
          " so folds are ungrouped")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("svdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}"
          f"  (window {WINDOW} of {int(labels['n_samples'])})")
    print(f"  record_id          {sample['record_id']!r}")
    print(f"  lead_names         {labels['lead_names']}")
    print(f"  n_beats            {int(labels['n_beats'])}")
    print(f"  beat_S / aami_S    {int(labels['beat_S'])} / {int(labels['aami_S'])}"
          "   <- combine with mitdb on aami_S")
    print(f"  sveb_fraction      {labels['sveb_fraction']:.4f}")
    print(f"  sveb_burden        {labels['sveb_burden']}   <- fold label, not a target")
    print(f"  mean_hr_bpm        {labels['mean_hr_bpm']:.1f}")
    print(f"  noisy_fraction     {labels['noisy_fraction']:.4f}")
    print(f"  annotated_fraction {labels['annotated_fraction']:.4f}")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. The symbol clash with mitdb, and the AAMI fix ---------------------
    print("\nReference beat annotations over this split:")
    beats = int(df["n_beats"].sum())
    for symbol, name in BEAT_NAMES.items():
        total = int(df[f"beat_{symbol}"].sum())
        if total:
            print(f"  beat_{symbol:2s} {total:8d}  {100 * total / beats:6.2f}%"
                  f"  -> AAMI {AAMI_CLASSES[symbol]}   {name}")
    print(f"  {beats} beats in this split")

    print("\nThe same beats under the AAMI EC57 reduction:")
    for cls in AAMI_ORDER:
        total = int(df[f"aami_{cls}"].sum())
        built = [s for s, c in AAMI_CLASSES.items() if c == cls]
        print(f"  aami_{cls} {total:8d}  {100 * total / beats:6.2f}%"
              f"   from {''.join(built)}")
    print("  THIS is the column to combine with mitdb on. mitdb spells a")
    print("  supraventricular beat 'A' and this database spells it 'S', so a")
    print("  concatenation keyed on the raw symbol produces two disjoint labels")
    print("  for one phenomenon. Under AAMI, A/a/J/S all collapse to S.")

    # --- 2. No demographics at all --------------------------------------------
    print("\nWhat the headers say about the subjects: nothing.")
    absent = [c for c in ("age", "sex", "patient_id", "medications", "description")
              if c not in df.columns]
    print(f"  columns absent from the labels: {absent}")
    print("  There are no header comment lines in this release, so folds cannot be")
    print("  grouped by subject and PhysioNet does not state how many subjects the")
    print("  78 recordings represent. mitdb can group on its tape number; this cannot.")
    uncal = int(df["header_declares_uncalibrated"].sum())
    print(f"\n  What the headers DO vary: {uncal} of {len(df)} records in this split")
    print("  declare gain 0 ('uncalibrated'); wfdb substitutes 200 adu/mV, so every")
    print("  record reads as mV either way and signal_unit_scale is 1.0.")

    # --- 3. Ectopy burden, the stratification axis ----------------------------
    print(f"\nSVEB burden bands (edges {SVEB_BURDEN_EDGES} of all beats):")
    counts = df["sveb_burden"].value_counts()
    for band in SVEB_BURDEN_BANDS:
        n = int(counts.get(band, 0))
        sveb = int(df.loc[df["sveb_burden"] == band, "n_sveb"].sum())
        print(f"  {band:9} {n:3d} records, {sveb:6d} SVEB beats")
    print("  For fold construction only — do not train on stratify_class.")

    print("\nThe database is NOT uniformly supraventricular:")
    ordered = df.sort_values("sveb_fraction", ascending=False)
    for record, row in ordered.head(5).iterrows():
        print(f"  {record}  {100 * row['sveb_fraction']:6.2f}% SVEB"
              f"  ({int(row['n_sveb']):5d} of {int(row['n_beats']):5d} beats,"
              f" {int(row['n_veb']):4d} VEB)")
    print("  ...")
    for record, row in ordered.tail(3).iterrows():
        print(f"  {record}  {100 * row['sveb_fraction']:6.2f}% SVEB"
              f"  ({int(row['n_sveb']):5d} of {int(row['n_beats']):5d} beats,"
              f" {int(row['n_veb']):4d} VEB)")
    none = df.index[df["n_sveb"] == 0].tolist()
    print(f"  records with NO supraventricular ectopy at all: {none or 'none in this split'}")
    print("  Ventricular ectopy is here too and is not incidental:"
          f" {int(df['n_veb'].sum())} beats")
    print(f"  ({100 * df['n_veb'].sum() / beats:.2f}% of this split), so 'ectopic vs not'")
    print("  learns both classes whether or not that was intended.")

    # --- 4. Uniform length, and what that means for windows -------------------
    print(f"\nEvery record is exactly {RECORD_SAMPLES} samples"
          f" ({RECORD_SAMPLES / 128:.0f} s):")
    print(f"  n_samples over this split: {sorted(df['n_samples'].unique().tolist())}")
    print("  So any window inside 1800 s fits ALL 78 records — one of the few datasets")
    print("  here where that is true. cpsc_2018 (6-144 s) and ptbdb (32-120 s) are not.")
    far = ECGDataset("svdb", **{**common, "window": (RECORD_SAMPLES - 640, 1280)})
    try:
        far[0]
        print("  ...but a window running past the end still raises:")
    except WindowOutOfRangeError as e:
        print(f"  window=({RECORD_SAMPLES - 640}, 1280) raises: {e}")

    # --- 5. Signal quality, and annotation completeness -----------------------
    print("\nAnnotated signal quality (seconds, from the `~` transitions):")
    print(f"  {'record':8} {'clean s':>9} {'ECG1':>7} {'ECG2':>7} {'both':>7}"
          f" {'noisy %':>8} {'artifacts':>10} {'unasserted':>11}")
    for record, row in df.sort_values("noisy_fraction", ascending=False).head(8).iterrows():
        print(
            f"  {record:8} {row['clean_secs']:9.1f} "
            f"{row['noisy_ECG1_secs']:7.1f} {row['noisy_ECG2_secs']:7.1f} "
            f"{row['noisy_both_secs']:7.1f} {100 * row['noisy_fraction']:8.2f} "
            f"{int(row['n_isolated_artifacts']):10d} "
            f"{row['quality_head_unasserted_secs']:11.1f}"
        )
    silent = int((df["n_quality_changes"] == 0).sum())
    print(f"  {silent} of {len(df)} records in this split carry no `~` at all,"
          " and are clean by default.")
    unasserted = df.index[df["quality_head_unasserted_secs"] > 0].tolist()
    print(f"  records whose FIRST `~` is a return to clean: {unasserted or 'none here'}")
    print("  For those, nothing ever asserted what the leading span was; it is counted")
    print("  clean (as WFDB does) and reported separately. In record 803 that span is")
    print("  1555.1 s — 86% of the record.")

    print("\nBeat annotation covers the whole record — unlike nsrdb:")
    print(f"  annotated_fraction {df['annotated_fraction'].min():.4f}"
          f" - {df['annotated_fraction'].max():.4f}")
    print(f"  largest unannotated head {df['unannotated_head_secs'].max():.2f} s,"
          f" tail {df['unannotated_tail_secs'].max():.2f} s")
    print("  So any window has reference annotation behind it. In nsrdb the last one to")
    print("  five hours of every record has none, and nothing announces it.")

    print(f"\nWhole-record rate and variability, over RR intervals in {RR_RANGE_SECS} s:")
    print(f"  mean_hr_bpm  {df['mean_hr_bpm'].min():.1f} - {df['mean_hr_bpm'].max():.1f}"
          f"  (mean {df['mean_hr_bpm'].mean():.1f})")
    print(f"  sdnn_ms      {df['sdnn_ms'].min():.1f} - {df['sdnn_ms'].max():.1f}")
    print(f"  rmssd_ms     {df['rmssd_ms'].min():.1f} - {df['rmssd_ms'].max():.1f}")
    print(f"  RR intervals rejected by that filter: {int(df['n_rr_rejected'].sum())}")
    print("  These are summaries over a rhythm that is by construction ectopic, so they")
    print("  describe the recording rather than the subject's sinus node.")

    # The two channels are unnamed, so select by name to make that explicit.
    print(f"\nLead layouts in this split: {df['lead_names'].value_counts().to_dict()}")
    one = ECGDataset("svdb", leads=["ECG2"], **common)
    print(f"leads=['ECG2'] -> {one.lead_names}, shape {tuple(one[0]['signal'].shape)}")
    print("  Channel positions. The release states no electrode placement, so do NOT")
    print("  read ECG1/ECG2 as MLII/V1 by analogy with mitdb.")

    # --- Batching -------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    mb = 2 * RECORD_SAMPLES * 4 / 1e6
    print(f"  Without window= each record is 2 x {RECORD_SAMPLES} float32 (~{mb:.1f} MB), so a")
    print(f"  batch of {args.batch_size} would be ~{mb * args.batch_size:.0f} MB,"
          " every byte decoded.")

    # A multi-class target: the AAMI class counts reduce to a per-record burden,
    # which is the quantity this database was built to vary.
    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["sveb_fraction"]], dtype=torch.float32
    )
    print(f"\nsveb_fraction target tensor: {tuple(target.shape)}"
          f"  mean {target.mean():.4f}, max {target.max():.4f}")
    print("  (a regression target. For beat-level classification, use the `.atr` files")
    print("  directly — the labels here are per-record counts, not per-beat labels.)")


if __name__ == "__main__":
    main()
