#!/usr/bin/env python3
"""
Example: Preterm Infant Cardio-Respiratory Signals Database (PICS), with labels.

Ten NICU recordings of ten preterm infants, 20.3 h to 70.3 h each, 439.8 h in
total. It is the only neonatal dataset in ECGBench and the youngest cohort in the
catalogue: these hearts run at 130-167 bpm, so a normal RR interval is 0.36-0.46 s
and adult HRV parameter choices are simply mis-specified here.

Six things to demonstrate:

1. **The label is an event time, not a diagnosis.** Every infant is a preterm
   infant in the same unit, so the record-level class is a constant. The ground
   truth is 622 manually validated bradycardia onsets and 3,797,503 verified R
   peaks — both time series, both reachable aligned to `window=`.
2. **Two records sample at 250 Hz**, so a window in samples is a different length
   of time depending on the record. This is the trap most likely to produce
   quietly wrong results.
3. **The bradycardia onset sits one sample after the R peak that opens the first
   RR > 0.6 s**, which is why `np.isin(onsets, rpeaks)` finds almost nothing.
4. **R-peak annotation does not cover the whole recording** — 94.0% to 99.9%, and
   infant10's last 2.13 h carry none at all. An empty return means "not annotated
   here", not "no beats".
5. **Every record clips at the converter rail and holds minutes of perfectly
   constant signal**, and ECGBench's whole-record `flat_line` check cannot see
   either. Read the label columns, not the validation report.
6. **The single channel is named three different ways** across the ten headers, so
   `leads=["II"]` is right for seven records and raises for three.

Labels come from the headers and the annotation files, so this works without
running the split pipeline first. The fold CSVs come from the Hub by default.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_picsdb.py --data-path /path/to/picsdb/1.0.0/
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.picsdb import BRADYCARDIA_RR_SECS, bradycardia_onsets, rpeaks

#: 15,000 samples. THIS IS 30 s OF EIGHT RECORDS AND 60 s OF THE OTHER TWO —
#: window= counts samples, and infant1 and infant5 run at 250 Hz. Every record is
#: at least 36,604,500 samples long (infant7, the shortest), so any window that
#: fits there fits everywhere.
WINDOW = (0, 15_000)

#: infant7's length, and the ceiling on `start + length` for any window.
SHORTEST_RECORD_SAMPLES = 36_604_500

#: Seconds of signal to take around a bradycardia onset. The release's definition
#: is RR > 0.6 s sustained over at least two beats, so an event is seconds long.
EVENT_CONTEXT_SECS = 30


def main():
    parser = argparse.ArgumentParser(description="Load picsdb with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("picsdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Leads:    {config.lead_names}  <- the PREDOMINANT name; see section 6")
    print(f"Rates:    {config.sampling_rates} Hz  <- per-record, not a choice; see section 2")
    print(f"Duration: median {config.duration_seconds / 3600:.1f} h, 20.3-70.3 h in fact")
    print(f"Folds:    {config.n_folds}  <- one infant per fold, leave-one-infant-out")
    print()
    print("!! THIS IS TEN INFANTS FROM ONE NICU. There is no negative class, no control")
    print("!! cohort and no diagnosis: cohort_label is a constant. What is learnable here")
    print("!! is an EVENT WITHIN a recording, and the split has one infant in val and one")
    print("!! in test — use split=None with fold_numbers=[...] for real cross-validation.")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
    )

    try:
        dataset = ECGDataset("picsdb", labels=True, window=WINDOW, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")
    data_path = dataset.data_path

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    print(f"  record_id             {sample['record_id']!r}")
    print(f"  subject_id            {labels['subject_id']}")
    print(f"  lead_name             {labels['lead_name']!r}")
    print(f"  cohort_label          {labels['cohort_label']}  <- the same for all 10 records")
    print(f"  sampling_rate         {int(labels['sampling_rate'])} Hz")
    print(
        f"  duration_secs         {labels['duration_secs']:.0f}"
        f"  ({labels['duration_secs'] / 3600:.2f} h)"
    )
    print(
        f"  n_bradycardias        {int(labels['n_bradycardias'])}"
        f"  ({labels['bradycardias_per_hour']:.2f} per hour)"
    )
    print(f"  n_rpeaks              {int(labels['n_rpeaks'])}")
    print(f"  mean_hr_bpm           {labels['mean_hr_bpm']:.1f}  <- an infant, not an adult")
    print(f"  annotated_fraction    {labels['annotated_fraction']:.4f}")
    print(f"  rail_secs             {labels['rail_secs']:.1f} s clipped at the converter rail")
    print(f"  flat_secs             {labels['flat_secs']:.1f} s of perfectly constant signal")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. The dataset is an event-detection dataset -------------------------
    print("\n" + "=" * 74)
    print("1. THE GROUND TRUTH IS AN EVENT TIME, NOT A CLASS")
    print("=" * 74)
    print(f"  {'record':13} {'infant':10} {'hours':>6} {'Hz':>4} {'brady':>6} {'/h':>5}"
          f" {'R peaks':>9} {'HR':>6}")
    for record, row in df.iterrows():
        print(
            f"  {record:13} {row['subject_id']:10} {row['duration_secs'] / 3600:6.2f}"
            f" {int(row['sampling_rate']):4d} {int(row['n_bradycardias']):6d}"
            f" {row['bradycardias_per_hour']:5.2f} {int(row['n_rpeaks']):9d}"
            f" {row['mean_hr_bpm']:6.1f}"
        )
    print(
        f"\n  {int(df['n_bradycardias'].sum())} bradycardia onsets and "
        f"{int(df['n_rpeaks'].sum())} R peaks in this split."
    )
    print("  cohort_label is 'preterm_infant' for every row, and so is the stratification")
    print("  label — ten records over ten folds admits no other class. Folds are grouped on")
    print("  subject_id, which is 1:1 here, so each fold is exactly one infant.")

    # --- 2. Two sampling rates ------------------------------------------------
    print("\n" + "=" * 74)
    print("2. A WINDOW IN SAMPLES IS NOT A WINDOW IN TIME")
    print("=" * 74)
    for rate, group in df.groupby("sampling_rate"):
        secs = WINDOW[1] / int(rate)
        print(
            f"  {int(rate)} Hz: {list(group.index)}"
            f"\n          window={WINDOW} is {secs:.0f} s of these records"
        )
    print("\n  Sampling rate is a per-record PROPERTY here, not a choice of representation:")
    print("  infant1 and infant5 are the 250 Hz 'compound' recordings the release describes.")
    print("  ECGDataset(sampling_rate=250) raises rather than handing back a mixed-rate")
    print("  subset — filter on the sampling_rate label column instead. And convert samples")
    print("  to seconds per record, never with one global rate.")

    # --- 3. Bradycardia onsets, aligned to a window ---------------------------
    print("\n" + "=" * 74)
    print("3. THE ONSET MARKS THE R PEAK OPENING THE FIRST LONG RR, ONE SAMPLE LATE")
    print("=" * 74)
    record = df.index[0]
    row = df.loc[record]
    fs = int(row["sampling_rate"])
    onset_secs = [float(v) for v in str(row["bradycardia_onsets_secs"]).split("|")]
    event = onset_secs[0]
    start = max(0, int((event - EVENT_CONTEXT_SECS / 2) * fs))
    length = EVENT_CONTEXT_SECS * fs
    print(f"  {record}: first onset at {event:.3f} s, so window=({start}, {length})")

    event_ds = ECGDataset("picsdb", window=(start, length), **common)
    index = list(event_ds.metadata_df[config.record_id_column]).index(record)
    signal = event_ds[index]["signal"]
    onsets = bradycardia_onsets(data_path, record, start, length)
    peaks = rpeaks(data_path, record, start, length)
    print(
        f"  -> signal {tuple(signal.shape)}, {float(signal.min()):.2f} to "
        f"{float(signal.max()):.2f} mV.  Only these samples were decoded."
    )
    print(f"  -> {len(onsets)} onset(s) and {len(peaks)} R peak(s) inside the window,")
    print(f"     at sample {onsets.tolist()} of the returned tensor.")

    if len(onsets) and len(peaks) > 2:
        onset = int(onsets[0])
        after = peaks[peaks >= onset]
        before = peaks[peaks < onset]
        if len(after) and len(before):
            nearest = min(int(before[-1]), int(after[0]), key=lambda p: abs(p - onset))
            opener = nearest
            rr = np.diff(peaks) / fs
            j = int(np.searchsorted(peaks, onset, side="right"))
            enclosing = rr[j - 1] if 0 < j < len(peaks) else float("nan")
            print(
                f"\n  Nearest R peak to the onset: sample {opener}, i.e. {opener - onset:+d}"
                f" sample(s) away."
            )
            print(f"  RR interval containing the onset: {enclosing:.3f} s"
                  f"  (the definition is > {BRADYCARDIA_RR_SECS} s)")
    print("\n  Over the whole release, 526 of the 622 onsets sit within two samples of the")
    print("  beat opening the first RR > 0.6 s, 493 of them exactly one sample after it,")
    print("  and all 622 within 10 s. Only 32 land exactly ON a .qrsc sample, so")
    print("  np.isin(onsets, rpeaks) finds almost nothing and looks like a bug. Re-measure:")
    print("    from ecgbench.labels.picsdb import verify_bradycardia_onsets")
    print("    verify_bradycardia_onsets('/path/to/picsdb/1.0.0/')")

    # --- 4. R-peak coverage ---------------------------------------------------
    print("\n" + "=" * 74)
    print("4. R-PEAK ANNOTATION DOES NOT COVER THE WHOLE RECORDING")
    print("=" * 74)
    print(f"  {'record':13} {'covered':>8} {'head s':>8} {'tail s':>9} {'gaps':>5} {'gap s':>9}")
    for record_name, row in df.sort_values("annotated_fraction").iterrows():
        print(
            f"  {record_name:13} {row['annotated_fraction']:8.4f} "
            f"{row['annotated_head_secs']:8.1f} {row['annotated_tail_secs']:9.1f} "
            f"{int(row['n_annotation_gaps']):5d} {row['annotation_gap_secs']:9.1f}"
        )
    print("\n  infant10's last 2.13 h carry no R peaks at all, and infant5 opens with 1,631 s")
    print("  of unannotated signal. rpeaks() returning an empty array there means 'not")
    print("  annotated here', not 'no beats' — a supervised window has to stay inside the")
    print("  annotated span, and nothing in the WFDB headers says where that ends.")

    # --- 5. Rails and flat runs, which no check catches ------------------------
    print("\n" + "=" * 74)
    print("5. CLIPPING AND DEAD SIGNAL THAT VALIDATION CANNOT SEE")
    print("=" * 74)
    print(f"  {'record':13} {'rail s':>8} {'rail %':>7} {'flat s':>8} {'longest flat':>13}"
          f" {'min mV':>8} {'max mV':>8}")
    for record_name, row in df.sort_values("rail_secs", ascending=False).iterrows():
        print(
            f"  {record_name:13} {row['rail_secs']:8.1f} {100 * row['rail_fraction']:7.3f}"
            f" {row['flat_secs']:8.1f} {row['longest_flat_secs']:11.1f} s"
            f" {row['min_mv']:8.2f} {row['max_mv']:8.2f}"
        )
    print("\n  Every record touches both 16-bit rails, so amplitude_range_mv is the union of")
    print("  ten per-record rails ([-40.96, +40.92] mV) rather than a physiologic bound —")
    print("  gain and baseline differ per record. All 10 records PASS every check and")
    print("  `clean` equals `original`, because flat_line tests variance over the WHOLE")
    print("  record and a 20-70 hour recording passes that trivially. The 24-minute constant")
    print("  run in infant5 is invisible to it. Read these columns before choosing a window.")

    # --- 6. One channel, three names ------------------------------------------
    print("\n" + "=" * 74)
    print("6. ONE CHANNEL, NAMED THREE DIFFERENT WAYS")
    print("=" * 74)
    for name, group in df.groupby("lead_name"):
        print(f"  {name:4} {len(group)} record(s): {list(group.index)}")
    print("\n  The release says only 'a single channel of a 3-lead ECG', and nothing states")
    print("  that the 'ECG' channel is lead II. record_lead_layouts therefore makes")
    print("  ECGDataset resolve leads= against each record's OWN header:")
    named = ECGDataset("picsdb", leads=["II"], window=WINDOW, **common)
    ok, refused = [], []
    for i, record_name in enumerate(named.metadata_df[config.record_id_column]):
        try:
            named[i]
            ok.append(record_name)
        except ValueError:
            refused.append(record_name)
    print(f"    leads=['II'] returns a signal for {len(ok)}: {ok}")
    print(f"    and RAISES for {len(refused)}: {refused}")
    print("  That is the honest answer — the alternative is handing back the one channel")
    print("  there is under a name the header never used. Omit leads= to take whatever")
    print("  channel each record holds.")

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
    print(f"  rates       {[int(row['sampling_rate']) for row in batch['labels']]} Hz"
          "  <- same tensor width, different durations")
    print("  Stacking only works because window= makes every record the same number of")
    print("  SAMPLES. It does not make them the same number of seconds.")

    print("\nA target tensor, if you wanted one — but read this first:")
    print("  There is no record-level class to predict. A per-record count is a property of")
    print("  a 20-70 hour recording of ONE infant, and there are ten infants in the whole")
    print("  release. The task this database supports is predicting an onset from the")
    print("  signal preceding it, within a recording.")
    targets = torch.tensor([float(row["n_bradycardias"]) for row in batch["labels"]])
    print(f"  e.g. bradycardia count per record in this batch: {targets.tolist()}")


if __name__ == "__main__":
    main()
