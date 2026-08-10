#!/usr/bin/env python3
"""
Example: SHDB-AF (Saitama Heart Database — Atrial Fibrillation) with labels.

128 twenty-four-hour two-lead Holter recordings from 122 Japanese subjects, with a
45-column clinical table on all of them and beat-level rhythm annotations on 98.
Six things worth demonstrating, because each one is a way to get this dataset wrong:

1. **There are two label layers, and they are different kinds of thing.**
   `AF_Type` is a clinical diagnosis from the medical report — one value per
   recording, present for all 128. `af_burden` is measured from the annotation
   marks and exists for 98. They agree closely and not perfectly.
2. **30 recordings have no rhythm annotation at all**, and they are not a random
   sample: the annotated 98 were drawn from a subset stratified on age, sex and
   diagnosis. `has_rhythm_annotation` is the column to filter on, and it is one
   axis of the fold label for exactly that reason.
3. **`(N` means "not annotated", not sinus rhythm.** The protocol marked
   supraventricular arrhythmia only, so `N` pools sinus rhythm with ventricular
   ectopy, pauses and noise. `rhythm_secs_N` is not sinus time.
4. **`(AB` is in the files and in no documentation**, and the release's own
   published beat table omits its 5,021 beats — and does not reproduce for `N` or
   `AFIB` either, because it was computed for v1.0.0's 100 annotated recordings.
5. **Records 005 and 020 are the same recording** — identical SHA-256 for `.dat`
   and `.qrs` in the release's own manifest — filed as two Holters three years
   apart. `duplicate_of` names the pair.
6. **A record is 24 hours long**, 138 MB of float32, so `window=` is required to
   batch at all — and length is not uniform, so the window has to fit the 9-hour
   outlier.

Labels come from `AdditionalData.csv` plus the annotation files, so this works
without running the split pipeline first. The fold CSVs come from the Hub by
default; pass --metadata-source local after copying
output/shdb_af/{clean,original}/ into the dataset root.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_shdb_af.py --data-path /path/to/shdb-af/1.0.1/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.shdb_af import (
    AF_CODES,
    RHYTHM_NAMES,
    SVT_CODES,
    UNLABELLED_CODE,
)

#: 10 s at 200 Hz. A record is up to 17,340,000 samples, so a window is required to
#: batch at all — and because window= pushes down into the reader (sampfrom/sampto),
#: it also avoids decoding the other 24 hours.
WINDOW = (0, 2000)

#: The shortest record in the release: 107 holds 6,480,000 samples (9.00 h) where
#: 87 of the 128 hold exactly 17,280,000 (24 h), and 022 is the other outlier at
#: 14,357,400 (19.94 h). Any window must end at or before this to work everywhere.
SHORTEST_RECORD_SAMPLES = 6_480_000


def main():
    parser = argparse.ArgumentParser(description="Load SHDB-AF with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("shdb_af")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- ECG1 = modified CC5, ECG2 = NASA")
    print(f"Duration: {config.duration_seconds} s nominal (9.0 h to 24.1 h in fact)")
    print(f"Patients: {config.patient_id_column}  <- 122 subjects, 128 recordings")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("shdb_af", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    print(f"  record_id            {sample['record_id']!r}   <- a STRING, zero-padded")
    print(f"  Subject_ID           {labels['Subject_ID']!r}")
    print(f"  lead_names           {labels['lead_names']}")
    print(f"  adc_gains            {labels['adc_gains']} adu/mV  <- per-channel scaling,")
    print("                        not an amplifier setting; see the config")
    print(f"  AF_Type              {labels['AF_Type']}   <- clinical diagnosis")
    print(f"  Age_at_Holter / Sex  {labels['Age_at_Holter']} / {labels['Sex']}")
    print(f"  has_rhythm_annot.    {labels['has_rhythm_annotation']}")
    if labels["has_rhythm_annotation"]:
        print(f"  af_burden            {labels['af_burden']:.4f}   ({labels['af_class']})")
        print(f"  af_beat_fraction     {labels['af_beat_fraction']:.4f}"
              "   <- larger: AF beats are faster")
        print(f"  dominant_rhythm      {labels['dominant_rhythm']} "
              f"({RHYTHM_NAMES.get(labels['dominant_rhythm'], '?')})")
        print(f"  rhythms              {labels['rhythms']}")
    print(f"  n_beats              {int(labels['n_beats'])}  (unaudited epltd detections)")
    print(f"  mean_heart_rate      {labels['mean_heart_rate_bpm']:.1f} bpm")
    print(f"  record_hours         {labels['record_hours']:.2f}")
    print(f"  duplicate_of         {labels['duplicate_of']!r}")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()
    annotated = df[df["has_rhythm_annotation"].astype(bool)]

    # --- 1. Two label layers -------------------------------------------------
    print("\nThe two label layers, side by side over this split:")
    print(f"  AF_Type (all {len(df)} records):     {df['AF_Type'].value_counts().to_dict()}")
    print(f"  af_class ({len(annotated)} annotated):  {df['af_class'].value_counts().to_dict()}")
    print("\n  AF_Type against measured AF burden:")
    for af_type, group in annotated.groupby("AF_Type"):
        with_af = int((group["af_burden"] > 0).sum())
        print(
            f"    {af_type:7s} n={len(group):3d}  {with_af:3d} with any annotated AF, "
            f"burden median {group['af_burden'].median():.4f} "
            f"max {group['af_burden'].max():.4f}"
        )
    print("  A clinical diagnosis is not a measurement of the recording; do not treat")
    print("  AF_Type as a label for the waveform, or af_burden as a diagnosis.")

    # --- 2. The unannotated 30 -----------------------------------------------
    print("\nAnnotation coverage is not uniform across diagnoses:")
    import pandas as pd

    print(pd.crosstab(df["AF_Type"], df["has_rhythm_annotation"].astype(bool)).to_string())
    print("  This is why the fold label crosses AF_Type with annotation availability:")
    print(f"  {df['stratify_class'].value_counts().to_dict()}")
    print("  Filtering to has_rhythm_annotation changes the demographic mix — the 98")
    print("  annotated recordings were drawn from a subset stratified on age, sex and")
    print("  diagnosis, so they are not a random sample of the 128.")

    # --- 3 & 4. The rhythm codes --------------------------------------------
    print("\nTime spent in each annotated rhythm, over this split:")
    seconds = {
        code: float(annotated[f"rhythm_secs_{code}"].sum())
        for code in RHYTHM_NAMES
        if float(annotated[f"rhythm_secs_{code}"].sum()) > 0
    }
    grand = sum(seconds.values())
    for code in sorted(seconds, key=lambda c: -seconds[c]):
        holding = int((annotated[f"rhythm_secs_{code}"] > 0).sum())
        note = ""
        if code in AF_CODES:
            note = "  <- counted as AF"
        elif code == UNLABELLED_CODE:
            note = "  <- NOT sinus rhythm: everything the protocol did not mark"
        elif code == "AB":
            note = "  <- undocumented upstream"
        print(
            f"  {code:5s} {seconds[code] / 3600:8.2f} h ({100 * seconds[code] / grand:5.2f}%)"
            f" in {holding:3d} records  {RHYTHM_NAMES[code]}{note}"
        )
    svt = sum(seconds.get(c, 0.0) for c in SVT_CODES)
    print(f"  Any supraventricular arrhythmia: {svt / 3600:.2f} h ({100 * svt / grand:.2f}%)")

    print("\nBeats per rhythm — the quantity the release's own table publishes:")
    total_beats = int(annotated["n_beats"].sum())
    for code in sorted(RHYTHM_NAMES, key=lambda c: -int(annotated[f"beats_{c}"].fillna(0).sum())):
        beats = int(annotated[f"beats_{code}"].fillna(0).sum())
        if beats:
            print(f"  {code:5s} {beats:9d}  ({100 * beats / total_beats:5.2f}%)")
    print(f"  total {total_beats:9d}")
    print("  The published table has five rows and omits AB entirely. Over the whole")
    print("  release AFL (195,659), AT (48,800) and PAT+NOD (4,416) reproduce exactly,")
    print("  while N is 170,276 beats short of the published figure and AFIB 59,154 —")
    print("  224,409 beats, or 2.12 records' worth. v1.0.1 withdrew two annotated")
    print("  records (016, 030) as duplicates and the table was never regenerated.")

    # --- 5. The duplicate ----------------------------------------------------
    duplicated = df[df["duplicate_of"].fillna("").astype(str) != ""]
    print(f"\nRecords flagged as duplicates in this split: {list(duplicated.index)}")
    print("  005 and 020 have the SAME SHA-256 for .dat and .qrs in the release's own")
    print("  SHA256SUMS.txt, and the clinical table files them as two Holters three years")
    print("  apart (ages 47 and 50). Both carry Subject_ID 4899921, so patient grouping")
    print("  keeps them in one fold — but a per-record metric double-counts one recording.")

    # --- 6. Windowing and batching ------------------------------------------
    print(f"\nThe shortest record in the release holds {SHORTEST_RECORD_SAMPLES} samples (107),")
    print("against 17,280,000 in 87 of the 128, so a window must fit the shortest — and the")
    print("shortest is per split, not per release:")
    shortest = int(df["n_samples"].min())
    print(f"  shortest here: {df['n_samples'].idxmin()} at {shortest} samples")
    # Sized to fall just past the shortest record and inside the longer ones, so the
    # error is selective rather than universal — which is the behaviour worth seeing.
    start = shortest - 100_000
    far = ECGDataset("shdb_af", **{**common, "window": (start, 200_000)})
    raised = 0
    for i in range(len(far)):
        try:
            far[i]
        except WindowOutOfRangeError as e:
            raised += 1
            if raised == 1:
                print(f"  window=({start}, 200000) raises: {e}")
    print(f"  ...and does so for {raised} of {len(far)} records in this split")

    print(f"\nLead layouts in this split: {df['lead_names'].value_counts().to_dict()}")
    one = ECGDataset("shdb_af", leads=["ECG2"], **common)
    print(f"leads=['ECG2'] -> {one.lead_names}, shape {tuple(one[0]['signal'].shape)}")
    print("  ECG2 is a NASA lead and ECG1 a modified CC5 — documented placements, which")
    print("  makes this the only two-lead Holter in the catalogue whose channel names mean")
    print("  something. They are still not any of the standard twelve.")

    print("\nAmplitude scale is per channel, not per release:")
    gains = annotated["adc_gains"].str.split("|", expand=True).astype(float)
    print(f"  gains in this split run {gains.min().min():.1f} to {gains.max().max():.1f} adu/mV")
    print("  Each channel was independently scaled to fill the 16-bit range, so absolute")
    print("  millivolt amplitudes are NOT comparable between records. units='uV' rescales")
    uv = ECGDataset("shdb_af", units="uV", **common)
    print(f"  units='uV' -> peak {float(uv[0]['signal'].abs().max()):.1f} uV vs "
          f"{float(sample['signal'].abs().max()):.4f} mV")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    mb = 2 * 17_280_000 * 4 / 1e6
    print(f"  Without window= a 24 h record is 2 x 17280000 float32 (~{mb:.0f} MB), so a")
    print(f"  batch of {args.batch_size} would be ~{mb * args.batch_size:.0f} MB, all decoded.")

    # AF burden as a regression target is what this dataset supports best; the
    # diagnosis is the classification target, and it exists for every record.
    import torch

    classes = sorted(df["AF_Type"].dropna().unique())
    target = torch.tensor([classes.index(x) for x in df["AF_Type"]], dtype=torch.long)
    print(f"\nAF_Type target tensor: {tuple(target.shape)} over {classes}")
    burden = torch.tensor(
        [float(x) for x in df["af_burden"].fillna(-1.0)], dtype=torch.float32
    )
    print(f"AF-burden target tensor: {tuple(burden.shape)}, "
          f"{int((burden < 0).sum())} records masked as unannotated (-1)")
    print("  Do NOT fill the unannotated 30 with 0.0 — 11 of them have a PAF diagnosis,")
    print("  so a zero there is a wrong label rather than a missing one.")


if __name__ == "__main__":
    main()
