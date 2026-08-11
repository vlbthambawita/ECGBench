#!/usr/bin/env python3
"""
Example: Post-Ictal Heart Rate Oscillations in Partial Epilepsy, with labels.

Seven single-lead recordings, 1.5 h to 3.8 h each, made during inpatient
EEG/ECG/video epilepsy monitoring of five women. The database exists to show one
thing: transient 0.01-0.10 Hz heart-rate oscillations appearing in the minutes
*after* a seizure. Six things to demonstrate:

1. **Seven records, five patients, and no subject id ships.** sz02, sz03 and
   sz04 are three recordings of the same woman. ECGBench reconstructs that
   grouping from beat morphology and folds respect it — this script prints the
   grouping and then checks the split against it, because reading the absent
   column as "one record per patient" would put her in train and in test.
2. **The seizure times are not annotations.** They live in `times.seize`, and
   the label columns are the only machine-readable form of them. The script uses
   them to cut a post-ictal window, which is the intended use of this database.
3. **`window=` is not optional here.** Records are 1.08M to 2.71M samples and
   vary by a factor of 2.5, so a batch of whole records is both large and
   ragged. Because `window=` is pushed into the reader, it decodes only what it
   returns.
4. **The beat annotations are unaudited machine output that opens with a
   warm-up.** Every record starts with exactly 50 `?` (WFDB "Learning")
   detections. They count as beats and fold into AAMI `Q`, so `aami_Q` is 50 per
   record before anything genuinely unclassifiable.
5. **`s` is ST change, not a beat**, and it is the only episode layer. Meanwhile
   rhythm annotation exists in *one* record, so a zero elsewhere means "never
   assessed".
6. **One channel, and it is a position, not a named lead.** The headers say
   "ECG" and the release states no electrode placement anywhere.

Labels come from the headers, the .ari files and times.seize, so this works
without running the split pipeline first. The fold CSVs come from the Hub by
default.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_szdb.py --data-path /path/to/szdb/1.0.0/
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.szdb import BEAT_NAMES, SUBJECT_IDS, SZDB_AAMI_CLASSES

#: 60 s at 200 Hz. Records run 1,079,998 to 2,711,998 samples (1.500 h to 3.767
#: h), so a window is needed to batch at all — and because window= pushes into
#: the reader, it avoids decoding the other hours.
WINDOW = (0, 12_000)

#: The shortest records: sz01 and sz04 hold 1,079,998 samples (5,399.99 s). Any
#: window has to end at or before this, or it raises on those two.
SHORTEST_RECORD_SAMPLES = 1_079_998

#: How much signal to take after a seizure ends. The paper reports the
#: oscillation lasting two to six minutes, so six covers it.
POST_ICTAL_SECS = 360


def main():
    parser = argparse.ArgumentParser(description="Load szdb with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("szdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- a channel position; the release names no lead")
    print(f"Duration: median {config.duration_seconds:.0f} s, 5,400-13,560 s in fact")
    print(f"Folds:    {config.n_folds}  <- not 10; 7 records from 5 subjects cannot make 10")
    print(f"Patients: {config.patient_id_column}  <- RECONSTRUCTED, see below")
    print()
    print("!! THE BEAT ANNOTATIONS IN THIS DATABASE ARE UNAUDITED. The shipped ANNOTATORS")
    print("!! file says: 'unaudited beat annotations from an automated detector'. The .ari")
    print("!! extension is the tell — every audited MIT-BIH database uses .atr. Every beat,")
    print("!! ectopy and HRV figure below is that detector's output.")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("szdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(
        f"Signal shape:  {tuple(sample['signal'].shape)}"
        f"  (window {WINDOW} of {int(labels['n_samples'])})"
    )
    print(f"  record_id             {sample['record_id']!r}")
    print(f"  subject_id            {labels['subject_id']}  (reconstructed)")
    print(f"  lead_names            {labels['lead_names']}")
    print(f"  cohort_label          {labels['cohort_label']}   <- the same for all 7 records")
    print(
        f"  duration_secs         {labels['duration_secs']:.0f}"
        f"  ({labels['duration_secs'] / 3600:.2f} h)"
    )
    print(f"  adc_gain              {labels['adc_gain']:.0f} adu/mV  <- 25 or 10, by record")
    print(
        f"  n_seizures            {int(labels['n_seizures'])}"
        f"   starts {labels['seizure_starts_secs']} s"
        f"   durations {labels['seizure_durations_secs']} s"
    )
    print(
        f"  n_beats               {int(labels['n_beats'])}"
        f"  ({int(labels['n_learning_beats'])} of them the detector's warm-up)"
    )
    print(f"  mean_hr_bpm           {labels['mean_hr_bpm']:.1f}")
    print(
        f"  annotated_fraction    {labels['annotated_fraction']:.5f}"
        "  <- covers the whole record"
    )

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. The subject grouping is reconstructed, and it matters -------------
    print("\n" + "=" * 74)
    print("1. SEVEN RECORDS, FIVE PATIENTS, AND NO SUBJECT ID IN THE RELEASE")
    print("=" * 74)
    print("The paper describes five women; the release ships seven records and no subject")
    print("column anywhere — not in the headers, not in RECORDS, not in the annotations.")
    print("ECGBench recovers the grouping from beat morphology: median beats over a")
    print("P-through-T window correlate at 0.9989 between sz03 and sz04, ABOVE either")
    print("record's own first-half-to-second-half self-control, where the best")
    print("cross-subject pair reaches 0.85. That gives exactly the five subjects AND the")
    print("two multi-seizure subjects the paper states.\n")
    for subject in sorted(set(SUBJECT_IDS.values())):
        records = [r for r, s in SUBJECT_IDS.items() if s == subject]
        here = [r for r in records if r in df.index]
        seizures = int(df.loc[here, "n_seizures"].sum()) if here else 0
        note = ""
        if here:
            note = f"  (in this split: {here}, {seizures} seizure{'s' if seizures != 1 else ''})"
        print(f"  {subject}  records {records}{note}")
    print("\n  Recheck it yourself — it reads all 16.8 h, so it is not run at load time:")
    print("    from ecgbench.labels.szdb import verify_subject_grouping")
    print("    verify_subject_grouping('/path/to/szdb/1.0.0').head()")

    in_split = df["subject_id"].nunique()
    print(f"\n  Subjects in this split: {in_split}; records: {len(df)}")
    print("  Folds are grouped on subject_id, so no woman appears in two splits. With")
    print("  patient_id_column left null — the natural reading of 'there is no column' —")
    print("  sz02, sz03 and sz04 would have been scattered across train, val and test.")

    # --- 2. The seizure times, and the post-ictal window ---------------------
    print("\n" + "=" * 74)
    print("2. THE SEIZURE TIMES — IN times.seize, NOT IN ANY ANNOTATION")
    print("=" * 74)
    print(f"  {'record':8} {'subject':14} {'dur h':>6} {'n':>2}  seizure intervals (s)")
    for record, row in df.sort_values("first_seizure_start_secs").iterrows():
        starts = [float(v) for v in str(row["seizure_starts_secs"]).split("|")]
        ends = [float(v) for v in str(row["seizure_ends_secs"]).split("|")]
        spans = ", ".join(f"{s:.0f}-{e:.0f} ({e - s:.0f}s)" for s, e in zip(starts, ends))
        print(
            f"  {record:8} {row['subject_id']:14} {row['duration_secs'] / 3600:6.2f} "
            f"{int(row['n_seizures']):2d}  {spans}"
        )
    print("\n  Onset and offset were read from the SIMULTANEOUS EEG by an")
    print("  electroencephalographer blinded to the heart-rate analysis. The EEG was never")
    print("  released, so they cannot be checked against their own source. And note the")
    print("  file holds 10 seizures where the paper describes 11 — the shortest listed is")
    print("  25 s against the paper's 15 s, so one has no released interval.")

    # Cut the post-ictal window the database exists for, on the first record.
    record = df.index[0]
    row = df.loc[record]
    end = float(str(row["seizure_ends_secs"]).split("|")[0])
    fs = config.default_sampling_rate
    post_ictal = (int(end * fs), POST_ICTAL_SECS * fs)
    print(f"\n  Post-ictal window on {record}: seizure ends at {end:.0f} s, so")
    print(f"  window={post_ictal} takes the {POST_ICTAL_SECS} s after it.")
    post = ECGDataset("szdb", labels=True, **{**common, "window": post_ictal})
    index = list(post.metadata_df[config.record_id_column]).index(record)
    signal = post[index]["signal"]
    print(
        f"  -> {tuple(signal.shape)}, {float(signal.min()):.2f} to {float(signal.max()):.2f} mV."
        "  Only these samples were decoded."
    )
    print("  This is the intended use: the reported oscillation is 0.01-0.10 Hz, two to six")
    print("  minutes long, and up to 41 bpm peak-to-trough — invisible in any whole-record")
    print("  HRV summary, which is why mean_hr_bpm and sdnn_ms below are descriptions of")
    print("  the recording rather than of the phenomenon.")

    # --- 3. Beats, the warm-up, and what `s` is ------------------------------
    print("\n" + "=" * 74)
    print("3. UNAUDITED BEATS, A 50-DETECTION WARM-UP, AND ST MARKERS THAT ARE NOT BEATS")
    print("=" * 74)
    print("Detected beats by raw symbol, with the AAMI EC57 class each reduces to:")
    for symbol, name in BEAT_NAMES.items():
        column = "n_learning_beats" if symbol == "?" else f"beat_{symbol}"
        total = int(df[column].sum())
        if total:
            print(f"  {symbol:2s} {total:8d}  -> AAMI {SZDB_AAMI_CLASSES[symbol]}   {name}")
    learning = int(df["n_learning_beats"].sum())
    print(
        f"\n  Every record contributes exactly 50 `?`: {learning} over {len(df)} records here."
    )
    print("  They are QRS complexes the detector LOCATED but had not yet learned to")
    print("  classify, not beats it found unclassifiable — so they count in n_beats and in")
    print("  the RR series, and fold into AAMI Q. Subtract n_learning_beats from aami_Q to")
    print(f"  recover genuinely unclassifiable beats: {int(df['aami_Q'].sum()) - learning}"
          " in this split.")

    print(f"\n  ST markers (`s`, WFDB symbol 18) in this split: {int(df['n_st_markers'].sum())}")
    print(f"  ST episodes they delimit:                       {int(df['n_st_episodes'].sum())}")
    print("  `s` is ST CHANGE, not a beat — counting it as one would inflate every beat")
    print("  total and put a non-beat into the AAMI reduction.")
    for record, row in df[df["n_st_episodes"] > 0].iterrows():
        print(
            f"    {record:8} {int(row['n_st_episodes']):2d} episodes "
            f"({int(row['n_st_depression_episodes'])} depression, "
            f"{int(row['n_st_elevation_episodes'])} elevation), "
            f"{row['st_secs']:.0f} s total, longest {row['longest_st_episode_secs']:.0f} s"
        )
    quiet = df.index[df["n_st_episodes"] == 0].tolist()
    print(f"    records with no ST episode: {quiet or 'none'}")

    # --- 4. Rhythm annotation exists in one record --------------------------
    annotated = df[df["has_rhythm_annotation"]]
    print(f"\n  Rhythm annotation: {len(annotated)} of {len(df)} records here carry any `+`.")
    for record, row in annotated.iterrows():
        print(
            f"    {record:8} {row['af_secs']:.1f} s of atrial fibrillation "
            f"in {int(row['n_af_episodes'])} episode(s)"
        )
    silent = df.index[~df["has_rhythm_annotation"]].tolist()
    print(f"    records with NO rhythm marker: {len(silent)} {silent}")
    print("    For those, af_secs == 0.0 means the rhythm was NEVER ASSESSED. Across the")
    print("    release only sz02 has any, and its 17.4 s of AF ends 25 s before that")
    print("    record's second seizure begins.")

    # --- 5. Records are long and ragged, so window= is required --------------
    print("\n" + "=" * 74)
    print("5. LENGTH VARIES BY A FACTOR OF 2.5, SO USE window= RATHER THAN A TRANSFORM")
    print("=" * 74)
    print(f"  {'record':8} {'samples':>10} {'hours':>7}")
    for record, row in df.sort_values("n_samples").iterrows():
        print(f"  {record:8} {int(row['n_samples']):10d} {row['duration_secs'] / 3600:7.3f}")
    print(f"\n  Shortest in the release: {SHORTEST_RECORD_SAMPLES} samples (sz01, sz04).")
    print("  A window ending past that raises WindowOutOfRangeError naming the record and")
    print("  its true length. window= is also pushed into wfdb as sampfrom/sampto, so it")
    print("  decodes only what it returns — and unlike a lambda transform it pickles, so")
    print("  DataLoader(num_workers>0) works under the spawn start method.")

    # --- 6. One channel, and a rail that depends on the record ---------------
    print("\n" + "=" * 74)
    print("6. ONE CHANNEL, TWO DIFFERENT AMPLITUDE RAILS")
    print("=" * 74)
    print(f"  lead_names: {config.lead_names} — the headers say 'ECG' and the release states")
    print("  no electrode placement anywhere, so this is a channel position. Selecting it")
    print("  by name still works and is still the right habit:")
    named = ECGDataset("szdb", leads=["ECG"], **common)
    print(f"    ECGDataset(leads=['ECG'])[0]['signal'].shape == {tuple(named[0]['signal'].shape)}")
    print("\n  The samples are 8-BIT — 256 levels, [-100, +155] adu — whatever the headers'")
    print("  declared resolution of 12 says, and the gain differs by record, so the")
    print("  clipping rail in millivolts does too:")
    for gain, group in df.groupby("adc_gain"):
        low, high = -100 / gain, 155 / gain
        print(
            f"    gain {gain:5.0f} adu/mV -> rail [{low:+.1f}, {high:+.1f}] mV"
            f"   records {list(group.index)}"
        )
    print("  amplitude_range_mv is the union, [-10.0, +15.5], so no record can fail")
    print("  validation for a rail its own header declares. Four of the seven records")
    print("  actually sit at both ends of theirs.")

    # --- Batch ---------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print("\n" + "=" * 74)
    print(f"One batch of {args.batch_size} through DataLoader + ecg_collate_fn")
    print("=" * 74)
    print(f"  signal      {tuple(batch['signal'].shape)}")
    print(f"  record_id   {batch['record_id']}")
    # batch["labels"] is a LIST of per-record dicts, not a dict of columns.
    print(f"  subject_id  {[row['subject_id'] for row in batch['labels']]}")
    print("  Stacking only works because window= makes every record the same length.")

    print("\nA target tensor, if you wanted one — but read this first:")
    print("  This is FIVE PATIENTS. There is no negative class, no control cohort and no")
    print("  held-out person to generalise to: the whole release is 7 recordings, 3 of")
    print("  them the same woman. Use it to study post-ictal dynamics on data with")
    print("  blinded EEG-derived seizure times, not to train a classifier.")
    targets = torch.tensor([float(row["seizure_secs"]) for row in batch["labels"]])
    print(f"  e.g. seconds of seizure per record in this batch: {targets.tolist()}")


if __name__ == "__main__":
    main()
