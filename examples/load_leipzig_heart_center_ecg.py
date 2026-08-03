#!/usr/bin/env python3
"""
Example: Leipzig Heart Center ECG Database — paediatric and CHD arrhythmias.

39 recordings from electrophysiological studies: 29 children with AV reentrant or
AV nodal reentrant tachycardia and 10 adults with repaired Tetralogy of Fallot.
18.5 hours of signal, 118,214 cardiologist-placed annotations. Four things this
script demonstrates, all of which will bite you otherwise:

1. **Records are not all 12-lead.** They carry 14, 18, 19 or 20 channels in six
   distinct layouts, and only channels 0-11 — the surface ECG — are the same
   channel in the same position in every record. Channel index 12 is `ABL12`,
   `RVA12` or `ART` depending on the record. So `leads=` is effectively mandatory
   if you want a homogeneous tensor, and `channel_index()` is how you reach an
   intracardiac channel.
2. **Length varies by two orders of magnitude**, 77.7 s to 2 h 30 m, so
   `window=(start, length)` is mandatory to batch at all.
3. **Three layers of label.** The subject-level `diagnosis` (7 classes), the
   `diagnosis_family` folds are built on (3 classes — a coarsening, not a clinical
   grouping), and the beat-level `tachy_*` counts, which are richer than either and
   are where AVRT/AVNRT/VT/AFIB episodes actually live.
4. **The test split is 3 records.** With 39 records the default 8/1/1 fold layout
   gives a test set too small to mean much, so this script also shows
   `split=None, fold_numbers=[...]` — the way to do cross-validation instead.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset (ODC-By, open):
    https://physionet.org/content/leipzig-heart-center-ecg/1.0.0/
  - Labels are never on the Hub, so they need that local copy. The fold CSVs are
    on the Hub and download automatically.

Usage:
  python examples/load_leipzig_heart_center_ecg.py \\
      --data-path /path/to/leipzig-heart-center-ecg/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.leipzig_heart_center_ecg import (
    BEAT_NAMES,
    BEAT_SYMBOLS,
    ECG_LEADS,
    IEGM_CHANNELS,
    TACHYCARDIA_AUX,
    channel_index,
)

#: 10 s at 977 Hz. Records run 77.7 s to 2 h 30 m, so a window is required to batch
#: at all. 60 s (58,620 samples) is the longest window that still fits the shortest
#: record, x0027 at 75,873 samples — anything longer raises WindowOutOfRangeError.
WINDOW = (0, 9770)


def main():
    parser = argparse.ArgumentParser(description="Load the Leipzig Heart Center database")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("leipzig_heart_center_ecg")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}")
    print("          ^ the 12 SURFACE ECG channels only. Records also carry 2-8")
    print("            intracardiac channels whose count and order vary per record.")
    print(f"Duration: {config.duration_seconds} s median — records are NOT uniform")
    print()

    try:
        dataset = ECGDataset(
            "leipzig_heart_center_ecg",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            window=WINDOW,
            leads=list(ECG_LEADS),
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:  {sorted(sample.keys())}")
    print(f"Signal shape: {tuple(sample['signal'].shape)}  (window {WINDOW}, leads=ECG_LEADS)")
    print(f"  record            {sample['record_id']}")
    print(f"  cohort            {labels['cohort']}")
    print(f"  diagnosis         {labels['diagnosis']}     <- train on this")
    print(f"  diagnosis_family  {labels['diagnosis_family']}     <- folds only")
    print(f"  gender / age      {labels['gender']} / {labels['age']} years")
    print(f"  ap_location       {labels['ap_location'] or '(none — AVNRT has no pathway)'}")
    print(
        f"  n_signals         {labels['n_signals']}  of which "
        f"{labels['n_iegm_channels']} intracardiac"
    )
    print(
        f"  n_samples         {labels['n_samples']} "
        f"({labels['n_samples'] / config.default_sampling_rate / 60:.1f} min)"
    )
    print(f"  n_beats           {labels['n_beats']}")

    # labels_df is aligned POSITIONALLY with metadata_df — row i of one is row i of
    # the other — and carries a RangeIndex, not record ids. Re-index it by record
    # name so the per-record lookups below read clearly.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].astype(str).to_numpy()

    print("\n--- 1. Six channel layouts, so index 12 is not one channel ---")
    print(
        f"{df['channel_names'].nunique()} distinct layouts in this split "
        f"({df['n_signals'].nunique()} distinct channel counts: "
        f"{sorted(int(n) for n in df['n_signals'].unique())})"
    )
    print("\n  record  n_sig  index 12   RVA12  CS12   ABL12")
    for record in list(df.index)[:6]:
        names = df.loc[record, "channel_names"]
        at_12 = names.split("|")[12]
        print(
            f"  {record:6s}  {int(df.loc[record, 'n_signals']):5d}  {at_12:9s}"
            f"  {str(channel_index(names, 'RVA12')):5s}"
            f"  {str(channel_index(names, 'CS12')):5s}"
            f"  {str(channel_index(names, 'ABL12')):5s}"
        )
    print("\n  -> None is a channel this record does not have. Never hardcode an")
    print("     index past 11; look the name up in that record's own header.")
    missing_abl = df.index[
        df["channel_names"].map(lambda n: channel_index(n, "ABL12") is None)
    ].tolist()
    print(f"  Records in this split with no ABL12: {missing_abl}")
    print("  (4 of the 39 records lack it; the README says every child's record has")
    print("   an ablation channel, and these do not)")

    print("\n--- 2. Reading an intracardiac channel by name ---")
    record = str(df.index[0])
    names = df.loc[record, "channel_names"].split("|")
    iegm = [n for n in names if n not in ECG_LEADS]
    print(f"  {record} intracardiac channels: {iegm}")
    for channel in iegm[:3]:
        described = IEGM_CHANNELS.get(channel, "NOT DOCUMENTED in the release")
        print(f"    {channel:8s} index {channel_index(names, channel):2d}  {described}")
    # leads= resolves against config.lead_names, which is the ECG only, so an
    # intracardiac channel has to be read from the record directly.
    print("  ECGDataset(leads=...) cannot select these — config.lead_names is the")
    print("  ECG only, deliberately, because there is no dataset-wide IEGM order.")

    print("\n--- 3. Three layers of label ---")
    print("  subject-level diagnosis (the ground truth):")
    for name, n in df["diagnosis"].value_counts().items():
        print(f"    {n:3d}  {name}")
    print("  diagnosis_family (what folds are stratified on — a coarsening):")
    for name, n in df["diagnosis_family"].value_counts().items():
        print(f"    {n:3d}  {name}")
    print("  beat-level annotations over this split:")
    total = int(df["n_beats"].sum())
    for symbol in BEAT_SYMBOLS:
        n = int(df[f"beat_{symbol}"].sum())
        if not n:
            continue
        holding = int((df[f"beat_{symbol}"] > 0).sum())
        print(
            f"    {symbol:2s} {n:7d} ({100 * n / total:5.2f}%) in {holding:2d} records"
            f"  {BEAT_NAMES[symbol]}"
        )
    print(f"    total {total} beats in the classes the README tabulates, plus")
    print(
        f"    {int(df['n_unclassifiable'].sum())} unclassifiable (Q), "
        f"{int(df['n_quality_marks'].sum())} signal-quality marks (~) and"
    )
    print(
        f"    {int(df['n_rhythm_changes'].sum())} rhythm markers (+) — "
        f"{int(df['n_annotations'].sum())} annotations in all."
    )

    print("\n  Which tachycardia each X beat was — richer than diagnosis:")
    for aux, column in TACHYCARDIA_AUX.items():
        n = int(df[column].sum())
        if n:
            print(f"    {aux:12s} {n:7d} beats in " f"{int((df[column] > 0).sum()):2d} records")
    print("  Note a child referred for AVNRT can still show AFIB or VT beats here,")
    print("  which is exactly why diagnosis is not the whole label.")

    print("\n--- 4. Signal quality: use the cardiologist's marks, not amplitudes ---")
    flagged = df.loc[df["n_quality_marks"] > 0, "n_quality_marks"]
    print(f"  {len(flagged)} of {len(df)} records carry '~' marks: {flagged.to_dict()}")
    print("  Signals are clipped at the amplifier rail (+/-10.24 mV on most")
    print("  channels), and amplitude_range_mv cannot flag that without excluding")
    print("  the intracardiac channels, which legitimately reach +/-51 mV.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print("  leads= makes this homogeneous; without it a batch mixes 14-, 18-, 19-")
    print("  and 20-channel records. Without window= the records in this split total")
    print(f"  {df['n_samples'].sum() / 977 / 3600:.1f} hours of signal.")

    print("\n--- 5. 39 records means cross-validation, not one 8/1/1 split ---")
    print("  The default layout leaves 3 records in test and 4 in val. For a dataset")
    print("  this small, select folds directly instead (split=None):")
    cv = ECGDataset(
        "leipzig_heart_center_ecg",
        split=None,
        fold_numbers=[1, 2, 3],
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=list(ECG_LEADS),
    )
    print(f"  folds 1-3 together: {len(cv)} records; each sample['split'] reports its")
    print(f"  own default split, e.g. {cv[0]['record_id']} -> {cv[0]['split']!r}")


if __name__ == "__main__":
    main()
