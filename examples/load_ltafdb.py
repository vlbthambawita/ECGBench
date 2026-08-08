#!/usr/bin/env python3
"""
Example: Long Term AF Database (LTAFDB) with labels.

84 two-lead Holter recordings of subjects with paroxysmal or sustained AF,
typically 24-25 hours each, and 8,995,973 manually verified beat annotations over
them. Five things to demonstrate:

1. **AF burden is the label**, not a diagnosis — 83 of the 84 subjects have
   annotated AF. It runs from 0% (record 30) to 100%, and the distribution is
   strongly bimodal, which is what makes the three-way `af_class` meaningful.
2. **The beat annotations are reference-grade and huge.** Unlike `afdb`, whose
   `.qrs` beats are unaudited detections, LTAFDB's `.atr` beats are typed
   (N/A/V/Q) and manually verified. The unaudited `.qrs` detector output ships
   too and is reported separately — never add the two together.
3. **The signal outlasts the annotation, by hours in some records.** 35 of the 84
   records stop annotating more than ten minutes before the signal ends, and
   record 117 stops 8.05 hours early. A `window=` into that tail returns waveform
   with no reference behind it.
4. **Records are a full day long** — the median is 11,059,200 samples, 88 MB of
   float32. `window=(start, length)` is needed to batch at all, and because it is
   pushed into the reader it avoids decoding the other 24 hours. Length is not
   uniform: 55 distinct lengths, and record 30 is only 2,826,240 samples.
5. **Record ids are zero-padded strings.** `00` is not `0`, and reading a fold CSV
   without `dtype=str` silently breaks every path.

Labels come straight from the annotation files, so this works without running the
split pipeline first. The fold CSVs do come from the pipeline (or the Hub) — pass
--metadata-source local after copying output/ltafdb/{clean,original}/ into the
dataset root.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_ltafdb.py --data-path /path/to/ltafdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.ltafdb import AF_CODES, BEAT_NAMES, BEAT_SYMBOLS, RHYTHM_NAMES

#: 10 s at 128 Hz. Records run to 12 million samples, so a window is required to
#: batch at all — and because window= pushes down into the reader, it also avoids
#: decoding the other 24 hours.
WINDOW = (0, 1280)

#: The shortest record: 30 stops at 2,826,240 samples (6.13 h) where the median
#: record holds 11,059,200. Any window has to end at or before this to work on
#: every record in the release.
SHORTEST_RECORD_SAMPLES = 2_826_240


def main():
    parser = argparse.ArgumentParser(description="Load Long Term AF Database with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("ltafdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- POSITIONS; the headers call both 'ECG'")
    print(f"Duration: {config.duration_seconds} s nominal (6.1 h to 26.4 h in fact)")
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
        dataset = ECGDataset("ltafdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    print(f"  record_id         {sample['record_id']!r}   <- a STRING, zero-padded")
    print(f"  lead_names        {labels['lead_names']}   <- both channels, same name")
    print(f"  adc_gains         {labels['adc_gains']} adu/mV  <- measured, per channel")
    print(f"  af_burden         {labels['af_burden']:.4f}   ({labels['af_class']})")
    rhythm = labels["dominant_rhythm"]
    print(f"  dominant_rhythm   {rhythm} ({RHYTHM_NAMES.get(rhythm, '?')})")
    print(f"  rhythms           {labels['rhythms']}")
    print(f"  n_beats           {int(labels['n_beats'])}  (typed reference beats from .atr)")
    print(f"  n_detections      {int(labels['n_detections'])}  (unaudited .qrs — do not add)")
    print(f"  mean_heart_rate   {labels['mean_heart_rate_bpm']:.1f} bpm")
    print(f"  record_hours      {labels['record_hours']:.2f}")
    print(f"  unannotated tail  {labels['unannotated_tail_secs']:.0f} s at the end")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. AF burden is the label -------------------------------------------
    print("\nAF burden across this split (AFIB as a fraction of annotated time):")
    for record, row in df.sort_values("af_burden").iterrows():
        bar = "#" * max(1, round(40 * row["af_burden"]))
        print(
            f"  {record:>3s}  {100 * row['af_burden']:6.2f}%  {row['af_class']:10s}"
            f" {int(row['n_episodes_AFIB']):4d} AFIB episodes  {bar}"
        )
    print(f"  classes: {df['af_class'].value_counts().to_dict()}")
    print(
        "  NOTE: af_class IS the fold label here. afdb had to coarsen the same "
        "quantity to a\n        binary cut because 25 records cannot fill three "
        "classes over ten folds; 84 can."
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
            f"  {code:5s} {seconds[code] / 3600:8.2f} h ({100 * seconds[code] / grand:5.2f}%)"
            f" in {holding:3d} records  {RHYTHM_NAMES[code]}{marker}"
        )
    print("  There is no AFL code in this release — afdb has one, so the two")
    print("  databases' 'af_burden' columns are computed over different code sets.")

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

    # --- 2. Reference beats vs unaudited detections ---------------------------
    print("\nTyped reference beats (.atr) across this split:")
    total = int(df["n_beats"].sum())
    for symbol in BEAT_SYMBOLS:
        count = int(df[f"beat_{symbol}"].sum())
        print(f"  {symbol}  {count:9d}  ({100 * count / total:5.2f}%)  {BEAT_NAMES[symbol]}")
    print(f"  total {total:9d} typed beats")
    print(
        f"  {int(df['n_detections'].sum()):9d} unaudited .qrs detections, all labelled N "
        "whatever they are"
    )
    print(
        f"  {int(df['n_missed_beats'].sum()):9d} missed-beat and "
        f"{int(df['n_pauses'].sum())} pause markers"
    )

    # --- 3. AF terminations, which exist in only half the database ------------
    terminations = df[df["n_af_terminations"] > 0]
    print(
        f"\nAF-termination markers in this split: {int(df['n_af_terminations'].sum())} "
        f"across {len(terminations)} records"
    )
    print("  These are hand-placed 'T' markers in the .qrs files, and they exist only in")
    print("  records 00-75 — the 100- and 200-series carry none. The AF Termination")
    print("  Challenge Database's 80 one-minute excerpts were cut around them.")
    print("  Note 'T' means something else entirely in an .atr rhythm code: trigeminy.")

    # --- 4. The unannotated tail ---------------------------------------------
    print("\nWhere the annotations stop, relative to where the signal does:")
    tail = df.sort_values("unannotated_tail_secs", ascending=False)
    for record, row in tail.head(5).iterrows():
        print(
            f"  {record:>3s}  record {row['record_hours']:5.2f} h, annotation stops "
            f"{row['unannotated_tail_secs'] / 3600:5.2f} h before the end"
        )
    print(f"  median across this split: {df['unannotated_tail_secs'].median():.1f} s")
    print("  A window= reaching into that tail returns waveform with no reference behind it.")

    # --- 5. Windowing and batching -------------------------------------------
    print(f"\nThe shortest record in the release holds {SHORTEST_RECORD_SAMPLES} samples (30),")
    print("against 11,059,200 in a median one, so a window must fit the shortest — and the")
    print("shortest is per split, not per release:")
    shortest = int(df["n_samples"].min())
    print(f"  shortest here: {df['n_samples'].idxmin()} at {shortest} samples")
    start = shortest - 50_000
    far = ECGDataset("ltafdb", **{**common, "window": (start, 100_000)})
    raised = 0
    for i in range(len(far)):
        try:
            far[i]
        except WindowOutOfRangeError as e:
            raised += 1
            if raised == 1:
                print(f"  window=({start}, 100000) raises: {e}")
    print(f"  ...and does so for {raised} of {len(far)} records in this split")

    print(f"\nLead layouts in this split: {df['lead_names'].value_counts().to_dict()}")
    one = ECGDataset("ltafdb", leads=["ECG2"], **common)
    print(f"leads=['ECG2'] -> {one.lead_names}, shape {tuple(one[0]['signal'].shape)}")
    print("  ECG1/ECG2 are POSITIONS that ECGBench assigns. Every header in this release")
    print("  names both channels 'ECG', and no electrode placement is stated anywhere,")
    print("  so do NOT read them as MLII/V1 by analogy with mitdb.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    mb = 2 * 11_059_200 * 4 / 1e6
    print(f"  Without window= a median record is 2 x 11059200 float32 (~{mb:.0f} MB), so a")
    print(f"  batch of {args.batch_size} would be ~{mb * args.batch_size:.0f} MB, all decoded.")

    # AF burden as a regression target, which is what this dataset supports best.
    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["af_burden"].fillna(0.0)], dtype=torch.float32
    )
    print(f"\nAF-burden target tensor: {tuple(target.shape)}  mean {target.mean():.3f}")


if __name__ == "__main__":
    main()
