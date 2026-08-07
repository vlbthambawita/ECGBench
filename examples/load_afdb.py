#!/usr/bin/env python3
"""
Example: MIT-BIH Atrial Fibrillation Database with labels.

25 long-term two-lead Holter recordings of subjects with AF, and 623 manually
reviewed rhythm episodes over them. Five things to demonstrate:

1. **AF burden is the label**, not a diagnosis — every subject here has AF. It
   runs from 0.24% of the record (05091) to 100% (07162, 07859), and that spread
   is the whole point of the dataset.
2. **The two channels are not named leads.** The headers call them `ECG1` and
   `ECG2` and the release never says which anatomical leads they are. Unlike
   `mitdb`, there is no MLII to ask for; `leads=` selects a channel position.
3. **Records are 9,205,760 samples (~74 MB of float32 each).**
   `window=(start, length)` is needed to batch at all, and because it is pushed
   into the reader it avoids decoding the other 10 hours. Length is not uniform —
   06453 is shorter — so the window has to fit the *shortest* record.
4. **Two records ship no signals.** 00735 and 03665 have real rhythm labels and
   no `.dat`. They are in the `original` version and flagged invalid; `clean`
   excludes them. This script shows both.
5. **Record ids are zero-padded strings.** `00735` is not `735`, and reading a
   fold CSV without `dtype=str` silently breaks every path.

Labels come straight from the annotation files, so this works without running the
split pipeline first. The fold CSVs do come from the pipeline (or the Hub) — pass
--metadata-source local after copying output/afdb/{clean,original}/ into the
dataset root.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_afdb.py --data-path /path/to/afdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.afdb import AF_CODES, RHYTHM_NAMES

#: 10 s at 250 Hz. Records are 9,205,760 samples, so a window is required to batch
#: at all — and because window= pushes down into the reader, it also avoids
#: decoding the other 10 h 13.5 min.
WINDOW = (0, 2500)

#: The shortest record with signals: 06453 stops at 8,325,000 samples. Any window
#: has to end at or before this, or it raises on that one record.
SHORTEST_RECORD_SAMPLES = 8_325_000


def main():
    parser = argparse.ArgumentParser(description="Load MIT-BIH AFDB with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("afdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- channel positions; the release names no leads")
    print(f"Duration: {config.duration_seconds} s per record (06453 is shorter)")
    print(f"Patients: {config.patient_id_column}  <- no subject id ships, so folds are ungrouped")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("afdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    # In the `original` version the first record may be 00735 or 03665, whose ECG
    # was never released — dataset[i] raises for those. Show a loadable one, and
    # demonstrate the failure deliberately further down.
    index = next(
        (
            i
            for i in range(len(dataset))
            if bool(dataset.labels_df.iloc[i]["has_signals"])
        ),
        0,
    )
    if index:
        print(f"          (record 0 ships no signal; showing record {index} instead)")

    sample = dataset[index]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW} of 9205760)")
    print(f"  record_id         {sample['record_id']!r}   <- a STRING, zero-padded")
    print(f"  lead_names        {labels['lead_names']}")
    print(f"  af_burden         {labels['af_burden']:.4f}   ({labels['af_class']})")
    rhythm = labels["dominant_rhythm"]
    print(f"  dominant_rhythm   {rhythm} ({RHYTHM_NAMES.get(rhythm, '?')})")
    print(f"  rhythms           {labels['rhythms']}")
    print(f"  n_beats           {labels['n_beats']}  (unaudited .qrs detections)")
    print(f"  mean_heart_rate   {labels['mean_heart_rate_bpm']:.1f} bpm")
    print(f"  record_seconds    {labels['record_seconds']:.1f}")
    print(f"  unannotated tail  {labels['unannotated_tail_secs']:.0f} s at the end")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. AF burden is the label -------------------------------------------
    print("\nAF burden across this split (AFIB + AFL as a fraction of annotated time):")
    for record, row in df.sort_values("af_burden").iterrows():
        bar = "#" * max(1, round(40 * row["af_burden"]))
        print(
            f"  {record}  {100 * row['af_burden']:6.2f}%  {row['af_class']:10s}"
            f" {int(row['n_episodes_AFIB']):3d} AFIB episodes  {bar}"
        )
    print(f"  classes: {df['af_class'].value_counts().to_dict()}")
    print(
        "  NOTE: folds are stratified on a binary 20% cut (stratify_class), because "
        "StratifiedKFold\n        needs >= n_folds records per class and 'sustained' "
        "has only 3 in the whole release."
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
        marker = "  <- counted as AF" if code in AF_CODES else ""
        print(
            f"  {code:5s} {seconds[code] / 3600:7.2f} h ({100 * seconds[code] / grand:5.2f}%)"
            f" in {holding:3d} records  {RHYTHM_NAMES[code]}{marker}"
        )

    print("\nEpisode count and duration say different things:")
    fragmented = df["n_episodes_AFIB"].idxmax()
    print(
        f"  {fragmented}: {int(df.loc[fragmented, 'n_episodes_AFIB'])} AFIB episodes for "
        f"{100 * df.loc[fragmented, 'af_burden']:.1f}% of the record"
    )
    longest = df["longest_af_episode_secs"].idxmax()
    print(
        f"  {longest}: longest single AF episode "
        f"{df.loc[longest, 'longest_af_episode_secs'] / 3600:.2f} h"
    )

    # --- 2. The two channels are unnamed -------------------------------------
    print(f"\nLead layouts in this split: {df['lead_names'].value_counts().to_dict()}")
    one = ECGDataset("afdb", leads=["ECG2"], **common)
    print(f"leads=['ECG2'] -> {one.lead_names}, shape {tuple(one[index]['signal'].shape)}")
    print("  These are channel positions. The release states no electrode placement,")
    print("  so do NOT read ECG1/ECG2 as MLII/V1 by analogy with mitdb.")

    # --- 3. Corrected vs unaudited beat annotations ---------------------------
    corrected = df[df["has_corrected_beats"]]
    print(f"\nBeat annotations: {int(df['n_beats'].sum())} unaudited .qrs detections")
    if len(corrected):
        for record, row in corrected.iterrows():
            print(
                f"  {record} also ships manually corrected .qrsc: "
                f"{int(row['n_beats'])} -> {int(row['n_beats_corrected'])} beats"
            )
    else:
        print("  no record in this split ships corrected beats (only 05091 and 07859 do)")

    # --- 4. Records with no signals, and the short record --------------------
    print("\nRecords whose ECG was never released:")
    missing = df[~df["has_signals"].astype(bool)]
    if len(missing):
        print("  (these are why iterating the `original` version raises — use `clean`")
        print("   for training, and `original` only to see what was excluded and why)")
        for record in missing.index:
            try:
                dataset[list(df.index).index(record)]
                print(f"  {record}: loaded (unexpected!)")
            except WindowOutOfRangeError as e:
                print(f"  {record}: labels present, signal absent -> {type(e).__name__}")
                print(f"      af_burden {df.loc[record, 'af_burden']:.4f}, "
                      f"dominant_rhythm {df.loc[record, 'dominant_rhythm']}, "
                      f"{int(df.loc[record, 'n_beats'])} beat annotations")
                print(f"      {e}")
    else:
        print(f"  none in {args.version}/{args.split} — the clean version excludes both")
        print("      (00735 and 03665; run with --version original to see them)")

    print(f"\nThe shortest record holds {SHORTEST_RECORD_SAMPLES} samples (06453), so a")
    print("window must fit inside that rather than inside the nominal 9205760:")
    far = ECGDataset("afdb", **{**common, "window": (9_000_000, 100_000)})
    raised = 0
    for i in range(len(far)):
        if not bool(df.iloc[i]["has_signals"]):
            continue  # a signal-less record raises for any window; not the point here
        try:
            far[i]
        except WindowOutOfRangeError as e:
            raised += 1
            if raised == 1:
                print(f"  window=(9000000, 100000) raises: {e}")
    if not raised:
        print("  window=(9000000, 100000) loaded every record with signals in this split")

    # --- 5. Batching ----------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    try:
        batch = next(iter(loader))
        print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    except WindowOutOfRangeError as e:
        # Expected on version="original", which holds 00735 and 03665. There is no
        # signal to stack for them, so the version is not iterable at all — take
        # `clean` to train, and `original` only to inspect what was excluded.
        print(f"\nThis version cannot be batched: {e}")
        print("  Use version='clean' for anything that iterates.")
    mb = 2 * 9_205_760 * 4 / 1e6
    print(f"  Without window= each record is 2 x 9205760 float32 (~{mb:.0f} MB), so a batch")
    print(f"  of {args.batch_size} would be ~{mb * args.batch_size:.0f} MB, every byte decoded.")

    # AF burden as a regression target, which is what this dataset supports best.
    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["af_burden"].fillna(0.0)], dtype=torch.float32
    )
    print(f"\nAF-burden target tensor: {tuple(target.shape)}  mean {target.mean():.3f}")


if __name__ == "__main__":
    main()
