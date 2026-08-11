#!/usr/bin/env python3
"""
Example: Apnea-ECG Database with labels.

70 overnight single-lead ECG recordings with an expert apnea annotation for every
one of their 34,313 minutes — the reference dataset for detecting sleep apnea from
the ECG alone, and the PhysioNet/CinC Challenge 2000 data. Five things to
demonstrate:

1. **The ground truth is per minute, not per record.** `apnea_sequence` is one
   character per minute, `A` or `N`, so minute *i* of a record is labelled by
   `apnea_sequence[i]`. Setting `window=(i * 6000, 6000)` makes the returned
   tensor exactly the labelled unit, which is what this script builds a target
   from. `apnea_class` is a whole-night summary the challenge used to *describe*
   records; it is not the task.
2. **The release's own learning/test split leaks subjects, and ECGBench does not
   use it.** These 70 records come from 30 subjects — 27 of them contributed more
   than one night — and 18 of those subjects have records on *both* sides of the
   challenge's a/b/c vs x division. Nothing in the release says so: there is no
   subject identifier anywhere. `subject_id` is reconstructed, and folds are
   grouped on it. `challenge_set` is kept only for reproducing 2000-era results.
3. **Two pairs of records are the same recording.** `x35` is `x22` shifted by
   40 s and `c06` is `c05` shifted by 80 s, bit for bit. Both are kept, and the
   grouping puts each pair in one fold.
4. **Records are whole nights (2.4M-3.5M samples), so `window=` is required to
   batch at all.** Length is *not* uniform — 6.75 h to 9.62 h — so a window sized
   for one record need not fit another.
5. **The single channel is not a named lead.** All 70 headers call it `ECG` and
   the release documents no electrode placement, so `leads=` selects a channel
   position.

Labels come straight from the annotation files and `additional-information.txt`,
so this works without running the split pipeline first. The fold CSVs come from
the Hub by default.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_apnea_ecg.py --data-path /path/to/apnea-ecg/1.0.0/
"""

import argparse

import numpy as np
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.apnea_ecg import (
    APNEA_CLASS_NAMES,
    DUPLICATE_RECORDS,
    RR_RANGE_SECS,
    SAMPLES_PER_MINUTE,
)

#: The first annotated minute: 6,000 samples at 100 Hz. Deliberately aligned to
#: the annotation grid — minute i of a record is window (i * 6000, 6000) and is
#: labelled by apnea_sequence[i]. Records hold 2.4M-3.5M samples, so a window is
#: required to batch at all, and because window= pushes down into the reader it
#: avoids decoding the other eight hours.
MINUTE = 0
WINDOW = (MINUTE * SAMPLES_PER_MINUTE, SAMPLES_PER_MINUTE)

#: The shortest record: x17 holds 2,430,000 samples (6.75 h). Any window has to
#: end at or before this, or it raises on that one record.
SHORTEST_RECORD_SAMPLES = 2_430_000

#: The longest: a12, 3,462,000 samples (9.62 h).
LONGEST_RECORD_SAMPLES = 3_462_000


def minute_labels(sequence: str) -> np.ndarray:
    """Turn an apnea_sequence into a boolean array, one entry per minute."""
    return np.frombuffer(sequence.encode("ascii"), dtype="S1") == b"A"


def main():
    parser = argparse.ArgumentParser(description="Load Apnea-ECG with labels")
    parser.add_argument("--data-path", default=None, help="Path to the version directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("apnea_ecg")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- a channel position; the release names no lead")
    print(f"Duration: mean {config.duration_seconds:.0f} s, but 24,300-34,620 s in fact")
    print(f"Patients: {config.patient_id_column}  <- RECONSTRUCTED; no subject id ships")
    print(
        f"Predefined splits: {config.has_predefined_splits}"
        "  <- the release has one, and it leaks subjects"
    )
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("apnea_ecg", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    sequence = labels["apnea_sequence"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(
        f"Signal shape:  {tuple(sample['signal'].shape)}"
        f"  (minute {MINUTE} of {labels['n_annotated_minutes']},"
        f" record holds {labels['n_samples']} samples)"
    )
    print(f"  record_id            {sample['record_id']!r}")
    print(f"  subject_id           {labels['subject_id']}")
    print(f"  challenge_set        {labels['challenge_set']}")
    print(
        f"  apnea_class          {labels['apnea_class']}"
        f"  ({APNEA_CLASS_NAMES[labels['apnea_class']]})"
    )
    print(
        f"  ahi / ai / hi        {labels['ahi']} / {labels['ai']} / {labels['hi']}"
        f"  -> {labels['ahi_severity']}"
    )
    print(f"  age / sex / bmi      {labels['age']:.0f} {labels['sex']}" f" {labels['bmi']:.1f}")
    print(f"  duration_hours       {labels['duration_hours']:.2f}")
    print(
        f"  apnea minutes        {labels['n_apnea_minutes']}"
        f" of {labels['n_annotated_minutes']}"
        f"  ({100 * labels['apnea_minute_fraction']:.1f}%)"
    )
    print(f"  mean_hr_bpm          {labels['mean_hr_bpm']:.1f}")
    print(f"  has_respiration      {labels['has_respiration']}")
    print(f"  apnea_sequence[:60]  {sequence[:60]}")
    print(f"  -> label for the returned window (minute {MINUTE}):" f" {sequence[MINUTE]!r}")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. The ground truth is per minute ------------------------------------
    print("\nThe task is per-minute binary classification, not per-record:")
    minutes = int(df["n_annotated_minutes"].sum())
    apnea = int(df["n_apnea_minutes"].sum())
    print(
        f"  this split carries {minutes} annotated minutes, {apnea} of them apnea"
        f" ({100 * apnea / minutes:.1f}%)"
    )
    print(
        f"  vs {len(df)} records — {minutes / len(df):.0f}x more supervision than the"
        " record-level"
    )
    print("  apnea_class suggests.")
    print("\n  An 'A' means apnea was in progress AT THE BEGINNING of that minute.")
    print("  (PhysioNet's first description said 'during the following minute'; it")
    print("  published a correction, which some papers still repeat.)")

    record = df.index[0]
    per_minute = minute_labels(df.loc[record, "apnea_sequence"])
    print(f"\n  {record}: {per_minute.size} minutes, {int(per_minute.sum())} apnea")
    print("  first two hours, one character per minute:")
    for hour in range(min(2, (per_minute.size + 59) // 60)):
        chunk = df.loc[record, "apnea_sequence"][hour * 60 : (hour + 1) * 60]
        print(f"    h{hour}  {chunk}")

    print("\n  Record-level class over this split (a whole-night SUMMARY, not the target):")
    for letter, count in df["apnea_class"].value_counts().sort_index().items():
        print(f"    {letter}  {count:3d}  {APNEA_CLASS_NAMES[letter]}")
    print(
        f"  AHI severity, for description only: " f"{df['ahi_severity'].value_counts().to_dict()}"
    )
    print("  Folds are balanced on apnea_class (40/10/20 across the release), not on")
    print("  AHI severity — its four bins split 23/5/11/31 and a class of 5 cannot be")
    print("  spread over 10 folds.")

    # --- 2. The challenge split leaks subjects ---------------------------------
    print(f"\n{len(df)} records in this split come from {df['subject_id'].nunique()}" " subjects:")
    repeated = df["subject_id"].value_counts()
    for subject, count in repeated[repeated > 1].items():
        records = sorted(df.index[df["subject_id"] == subject])
        sets = sorted(set(df.loc[records, "challenge_set"]))
        flag = "  <- SPANS the challenge learning/test split" if len(sets) > 1 else ""
        print(f"  {subject:12} {count} nights: {records}  {sets}{flag}")
    print(
        f"  {int((repeated > 1).sum())} of {df['subject_id'].nunique()} subjects in this"
        " split contributed more than one night."
    )
    print("\n  Across the whole release: 30 subjects for 70 records, and 18 of those")
    print("  subjects (49 records) appear in BOTH the challenge learning set and its")
    print("  test set. That is why has_predefined_splits is false — training on a/b/c")
    print("  and testing on x, the standard protocol here, shows the model 70% of its")
    print("  test records' subjects during training.")
    print(f"  challenge_set in this split: {df['challenge_set'].value_counts().to_dict()}")
    print("  ECGBench folds are grouped on subject_id, so no subject spans a fold.")

    # --- 3. Duplicate recordings ----------------------------------------------
    print("\nTwo pairs of records are the same recording, bit for bit:")
    for duplicate, (canonical, offset) in DUPLICATE_RECORDS.items():
        print(
            f"  {duplicate} == {canonical} shifted by {offset} samples" f" ({offset / 100:.0f} s)"
        )
    present = [r for r in DUPLICATE_RECORDS if r in df.index]
    both = [r for r in present if DUPLICATE_RECORDS[r][0] in df.index]
    print(
        f"  in this split: {present or 'neither'}"
        f"; with their counterpart too: {both or 'neither'}"
    )
    print("  Both records are kept — each is official, with its own official")
    print("  annotations — and the grouping puts each pair in one fold. An ungrouped")
    print("  split could otherwise place identical waveform in train and test.")

    # --- 4. Whole-night records need window= ----------------------------------
    print("\nRecords are whole nights, and NOT of uniform length:")
    print(
        f"  this split: {int(df['n_samples'].min())}"
        f" to {int(df['n_samples'].max())} samples"
        f" ({df['duration_hours'].min():.2f}-{df['duration_hours'].max():.2f} h)"
    )
    mb = SHORTEST_RECORD_SAMPLES * 4 / 1e6
    print(
        f"  the shortest record in the release (x17) is {SHORTEST_RECORD_SAMPLES}"
        f" samples, ~{mb:.0f} MB of float32;"
    )
    print(
        f"  the longest (a12) is {LONGEST_RECORD_SAMPLES}." " A window sized for a12 raises on x17:"
    )
    far = ECGDataset("apnea_ecg", **{**common, "window": (2_430_000, 6000)})
    raised = 0
    for i in range(len(far)):
        try:
            far[i]
        except WindowOutOfRangeError as e:
            raised += 1
            if raised == 1:
                print(f"    window=(2430000, 6000) raises: {e}")
    print(f"  {raised} of {len(far)} records in this split are too short for it")

    # --- 5. The channel is unnamed --------------------------------------------
    print(f"\nLead layouts in this split: {df['lead_names'].value_counts().to_dict()}")
    named = ECGDataset("apnea_ecg", leads=["ECG"], **common)
    print(f"leads=['ECG'] -> {named.lead_names}," f" shape {tuple(named[0]['signal'].shape)}")
    print("  A channel position. The release states no electrode placement, so this")
    print("  must not be stacked with a 12-lead dataset's lead I or II by name.")

    print(f"\nHeart rate from the .qrs detections, over RR in {RR_RANGE_SECS} s:")
    print(
        f"  mean_hr_bpm {df['mean_hr_bpm'].min():.1f} - {df['mean_hr_bpm'].max():.1f}"
        f"  sdnn_ms {df['sdnn_ms'].min():.1f} - {df['sdnn_ms'].max():.1f}"
    )
    print("  Those detections are machine-generated by sqrs125 and were never")
    print("  hand-edited, so they describe the record rather than reference it.")

    # --- Batching and the target tensor ---------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    longest_mb = LONGEST_RECORD_SAMPLES * 4 / 1e6
    print(
        f"  Without window= a batch of {args.batch_size} whole nights would be"
        f" ~{longest_mb * args.batch_size:.0f} MB, every byte decoded."
    )
    print(
        f"  With it, {args.batch_size} x {WINDOW[1]} samples is"
        f" {args.batch_size * WINDOW[1] * 4 / 1e6:.2f} MB — and the reader never"
        " touches the rest."
    )

    # The target for the returned windows: one label per record, for the ONE
    # minute each window covers. Iterate the minutes to train on more than that.
    import torch

    target = torch.tensor(
        [seq[MINUTE] == "A" for seq in dataset.labels_df["apnea_sequence"]],
        dtype=torch.float32,
    )
    print(
        f"\nPer-minute target for minute {MINUTE}: {tuple(target.shape)}"
        f"  positive rate {target.mean():.3f}"
    )
    print("  To train on the whole database, iterate minutes rather than records:")
    print("    for minute in range(n_annotated_minutes):")
    print("        ds = ECGDataset('apnea_ecg', window=(minute * 6000, 6000), ...)")
    print("        y  = [seq[minute] == 'A' for seq in ds.labels_df['apnea_sequence']]")
    print(f"  That is {minutes} labelled examples in this split alone.")


if __name__ == "__main__":
    main()
