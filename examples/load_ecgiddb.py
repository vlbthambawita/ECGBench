#!/usr/bin/env python3
"""
Example: ECG-ID Database with labels.

310 twenty-second Lead I recordings from 90 volunteers, collected in 2004-2005 to
test whether an ECG identifies the person who produced it. Five things to
demonstrate, and the first two are the ones that will surprise you:

1. **Every record stores the same lead twice.** The channels are `ECG I` (raw,
   with every cardiograph filter deliberately switched off) and `ECG I filtered`
   (the thesis's own offline preprocessing of those same samples). So the tensor is
   (2, 10000) and both rows are Lead I. `leads=["ECG I"]` is how you avoid handing
   a model the same lead twice.
2. **The label is the subject, and ECGBench's folds cannot be used for it.** The
   ground truth here is identity — which of 90 people — so `subject_id` is both the
   label and `patient_id_column`. Folds group by subject, which means no fold's
   model has ever seen the person it would be asked to recognise. That is right for
   any other use of these recordings and wrong for this one; this script shows the
   within-subject session split you need instead.
3. **The .atr annotations stop about 40% of the way in.** 10 R-peaks and 10 T-peaks
   per record, from an unaudited automatic detector, all inside the first 5.1-11.7 s
   of a 20.000 s record. `annotated_fraction` says where they end.
4. **Length is uniform, unusually.** All 310 records hold exactly 10,000 samples,
   so any `window=` inside (0, 10000) fits every record and nothing can raise
   `WindowOutOfRangeError`.
5. **The raw channel is genuinely noisy, on purpose.** 3 of the 310 records drift
   past ±10 mV and are excluded from `clean`.

Labels come from the record headers and the .atr files, so this works without
running the split pipeline first. The fold CSVs come from the Hub by default.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_ecgiddb.py --data-path /path/to/ecgiddb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.ecgiddb import AGE_CUT_YEARS, ANNOTATION_SOURCE, RR_RANGE_SECS

#: The first five seconds. Safe on every record — all 310 hold exactly 10,000
#: samples — and inside the annotated span of every one of them, whose last
#: annotation lands at sample 2,542 at the earliest. window= is pushed into wfdb's
#: sampfrom/sampto, so the other 15 s are never decoded.
WINDOW = (0, 2500)

#: Every record: 10,000 samples at 500 Hz.
RECORD_SAMPLES = 10_000

#: The earliest and latest last-annotation samples in the release.
EARLIEST_LAST_ANNOTATION = 2542
LATEST_LAST_ANNOTATION = 5869

#: Excluded from `clean` by amplitude_outlier — baseline drift the thesis chose not
#: to filter. Person_47_rec_2 is a genuine 10.885 mV excursion; the other two drift
#: to -154 mV.
EXCLUDED_FROM_CLEAN = ("Person_47_rec_2", "Person_76_rec_2", "Person_88_rec_1")


def main():
    parser = argparse.ArgumentParser(description="Load ECG-ID with labels")
    parser.add_argument("--data-path", default=None, help="Path to the version directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("ecgiddb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- ONE physical lead, stored twice")
    print(f"Duration: {config.duration_seconds} s ({RECORD_SAMPLES} samples), uniform")
    print(f"Patients: {config.patient_id_column}  <- also the LABEL; see below")
    print(f"Label:    {config.label_column}  <- 90-class identity, no diagnosis exists")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
    )

    try:
        dataset = ECGDataset("ecgiddb", labels=True, window=WINDOW, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(
        f"Signal shape:  {tuple(sample['signal'].shape)}"
        f"  (window {WINDOW}, record holds {int(labels['n_samples'])} samples)"
    )
    print(f"  record_id                {sample['record_id']!r}")
    print(f"  subject_id               {labels['subject_id']}  <- the label")
    print(f"  record_name              {labels['record_name']}  <- NOT unique across subjects")
    print(f"  age / sex                {labels['age']:.0f} / {labels['sex']}")
    print(f"  ecg_date                 {labels['ecg_date']}")
    print(
        f"  session {int(labels['session_index'])} of "
        f"{int(labels['n_sessions_for_subject'])}"
        f"           day {int(labels['days_since_first_session'])} of "
        f"{int(labels['session_span_days'])}"
    )
    print(f"  n_records_for_subject    {int(labels['n_records_for_subject'])}")
    print(
        f"  annotated span           samples 0-{int(labels['last_annotation_sample'])}"
        f" of {int(labels['n_samples'])}"
        f"  ({100 * labels['annotated_fraction']:.1f}%)"
    )
    print(
        f"  mean_hr_bpm              {labels['mean_hr_bpm']:.1f}"
        f"  (over {int(labels['n_rr_used'])} RR intervals)"
    )
    print(f"  mean_rt_interval_ms      {labels['mean_rt_interval_ms']:.1f}  <- R peak to T PEAK")
    print(f"  annotation_source        {labels['annotation_source']}")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. One lead, stored twice --------------------------------------------
    print("\nBoth channels are Lead I — raw and the thesis's filtered version:")
    print(f"  lead_names in this split: {df['signal_descriptions'].value_counts().to_dict()}")
    raw_only = ECGDataset("ecgiddb", leads=["ECG I"], window=WINDOW, **common)
    print(
        f"  leads=['ECG I']          -> {raw_only.lead_names},"
        f" shape {tuple(raw_only[0]['signal'].shape)}"
    )
    filtered = ECGDataset("ecgiddb", leads=["ECG I filtered"], window=WINDOW, **common)
    print(
        f"  leads=['ECG I filtered'] -> {filtered.lead_names},"
        f" shape {tuple(filtered[0]['signal'].shape)}"
    )
    import torch

    correlation = float(torch.corrcoef(sample["signal"])[0, 1])
    print(f"  correlation between the two channels of {sample['record_id']}: {correlation:.3f}")
    print("  The filter chain was level-9 db8 wavelet baseline removal, an adaptive")
    print("  50 Hz bandstop and a 5th-order Butterworth lowpass at 40 Hz. It is")
    print("  zero-phase, so the two channels are sample-aligned and the annotation")
    print("  indices apply to both. Feeding a model both rows feeds it one lead twice.")
    print("  ecgbench.labels.ecgiddb.scan_noise_levels(data_path) quantifies what the")
    print("  filter removed: median 0.187 mV RMS, and 41.9 mV for Person_76/rec_2.")

    # --- 2. The label is the subject, and the folds group by it ----------------
    print(f"\n{len(df)} records in this split come from {df['subject_id'].nunique()} subjects:")
    repeated = df["subject_id"].value_counts()
    for subject, count in repeated.head(5).items():
        sessions = int(df.loc[df["subject_id"] == subject, "n_sessions_for_subject"].iloc[0])
        span = int(df.loc[df["subject_id"] == subject, "session_span_days"].iloc[0])
        print(f"  {subject:10} {count:2d} records over {sessions} session(s), {span:3d} days")
    print(
        f"  {int((repeated > 1).sum())} of {df['subject_id'].nunique()} subjects here"
        " contributed more than one record."
    )
    print("\n  THE FOLDS CANNOT BE USED FOR IDENTIFICATION. subject_id is the label AND")
    print("  the grouping column, so every subject's records sit in one fold and no")
    print("  fold's model has seen the person it would have to recognise. Across the")
    print("  release, folds hold 7-11 subjects each and no subject spans a fold.")
    print("\n  For identification, split WITHIN subject instead — hold out each")
    print("  subject's later sessions, using the multi-session subjects:")
    multi = df[df["is_multi_session"]]
    print(
        f"    is_multi_session in this split: {len(multi)} records from"
        f" {multi['subject_id'].nunique()} subjects"
        f" (20 of the 90 across the release)"
    )
    if len(multi):
        enrol = multi[multi["session_index"] == 1]
        verify = multi[multi["session_index"] > 1]
        print(
            f"    session_index == 1 -> {len(enrol)} enrolment records;"
            f" > 1 -> {len(verify)} verification records"
        )
        print("      ids = dataset.metadata_df['record_id']")
        print("      enrol  = ids[dataset.labels_df['session_index'] == 1]")
        print("      verify = ids[dataset.labels_df['session_index'] > 1]")
        print(
            f"    days between sessions here:"
            f" {int(verify['days_since_first_session'].min())}"
            f"-{int(verify['days_since_first_session'].max())}"
            if len(verify)
            else "    (no later sessions in this split)"
        )
    print("  The thesis's own 195/115 train/test division is NOT recoverable: it is")
    print("  described in prose and recorded in no file.")

    # --- 3. Where the annotations stop ----------------------------------------
    print(f"\nThe {int(df['n_annotations'].sum())} annotations in this split are all")
    print(f"{ANNOTATION_SOURCE} output, and they stop early:")
    print(
        f"  last_annotation_sample {int(df['last_annotation_sample'].min())}"
        f"-{int(df['last_annotation_sample'].max())} of {RECORD_SAMPLES}"
        f"  ({100 * df['annotated_fraction'].min():.1f}-"
        f"{100 * df['annotated_fraction'].max():.1f}%)"
    )
    print(
        f"  unannotated tail {df['unannotated_tail_secs'].min():.1f}"
        f"-{df['unannotated_tail_secs'].max():.1f} s of every 20.0 s record"
    )
    print(f"  R-peaks per record {df['n_r_peaks'].value_counts().to_dict()}")
    print(f"  T-peaks per record {df['n_t_peaks'].value_counts().to_dict()}")
    print(
        f"  Release-wide the last annotation is never earlier than sample"
        f" {EARLIEST_LAST_ANNOTATION} nor later than {LATEST_LAST_ANNOTATION},"
    )
    print(f"  so window={WINDOW} is inside the annotated span of every record.")
    print(f"  heart rate {df['mean_hr_bpm'].min():.1f}-{df['mean_hr_bpm'].max():.1f} bpm over")
    print(f"  RR in {RR_RANGE_SECS} s; {int(df['n_rr_rejected'].sum())} intervals rejected.")
    print("  Ten machine-detected beats is not an HRV measurement and not a")
    print("  beat-detection reference — use qtdb or ludb for that.")

    # --- 4. Uniform length ----------------------------------------------------
    print("\nLength is uniform, which is rare for a PhysioNet release:")
    print(
        f"  n_samples in this split: {df['n_samples'].value_counts().to_dict()}"
        f"  ({df['duration_secs'].min():.3f}-{df['duration_secs'].max():.3f} s)"
    )
    whole = ECGDataset("ecgiddb", **common)
    print(
        f"  no window= -> {tuple(whole[0]['signal'].shape)};"
        f" window={WINDOW} -> {tuple(sample['signal'].shape)}"
    )
    print(f"  Any window inside (0, {RECORD_SAMPLES}) fits every record, so")
    print("  WindowOutOfRangeError cannot fire here. Contrast cpsc_2018 (6-144 s).")

    # --- 5. Demographics, folds and the excluded records ----------------------
    print("\nDemographics — the entire shipped metadata, from three header comments:")
    print(f"  sex   {df['sex'].value_counts().to_dict()}")
    print(
        f"  age   {df['age'].min():.0f}-{df['age'].max():.0f}," f" median {df['age'].median():.0f}"
    )
    print(f"  dates {sorted(df['ecg_date'].unique())}")
    print(f"  Folds are stratified on sex x an age cut at {AGE_CUT_YEARS}:")
    print(f"    {df['stratify_class'].value_counts().to_dict()}")
    print("  That cross is the finest cut that puts every class in all 10 folds — its")
    print("  smallest cell holds exactly 10 subjects. It is a fold-construction")
    print("  device, not a clinical grouping.")
    print(f"\n  {len(EXCLUDED_FROM_CLEAN)} of the 310 records fail amplitude_outlier and are")
    print(f"  absent from clean/: {', '.join(EXCLUDED_FROM_CLEAN)}.")
    print("  The raw channel is unfiltered by design, so two of them drift to -154 mV;")
    print("  use version='original' to get them back with quality_issues attached.")

    # --- Batching and the target tensor ---------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}")
    print(f"  ids {batch['record_id']}")

    # The identity target. Fit the encoder on the split's own subjects — there is
    # no global 90-class ordering, and a fold does not contain all 90.
    subjects = sorted(df["subject_id"].unique())
    index = {subject: i for i, subject in enumerate(subjects)}
    target = torch.tensor([index[s] for s in dataset.labels_df["subject_id"]], dtype=torch.long)
    print(
        f"\nIdentity target: {tuple(target.shape)} over"
        f" {len(subjects)} classes present in this split"
    )
    print("  There is no 90-class target for a fold: the other subjects are in other")
    print("  folds by construction. Build the class list from the records you have,")
    print("  and for the real task re-split within subject as shown above.")


if __name__ == "__main__":
    main()
