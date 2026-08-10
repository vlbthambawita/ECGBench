#!/usr/bin/env python3
"""
Example: BIDMC Congestive Heart Failure Database with labels.

15 twenty-hour two-lead Holter recordings of subjects with **severe** congestive
heart failure (NYHA class III-IV) — the severe-CHF counterpart to `nsrdb`'s normal
cohort. Six things to demonstrate:

1. **The beat annotations are unaudited machine output.** PhysioNet says so
   plainly, and the `.ecg` extension is the tell — every audited MIT-BIH database
   in this catalogue uses `.atr`. So every ectopy count and HRV figure below
   describes what one 1980s detector reported, not ground truth. This is the first
   thing the script prints, because it conditions everything else.
2. **There is no clinical label, and that is the point.** `cohort_label` is
   `severe_chf` and `nyha_class` is `III-IV` for all 15 records. Folds are
   stratified on the subject's sex (11 M, 4 F) — the one axis PhysioNet documents
   about this cohort, and the only one that survives 15 records over 10 folds.
3. **`r` is a ventricular beat, and it outnumbers `V` in 9 of the 15 records.**
   Counting `beat_V` alone undercounts ventricular ectopy across most of the
   database. The script shows the raw symbols beside the AAMI reduction so the gap
   is visible rather than described.
4. **Rhythm annotation exists in only 4 of 15 records, and its absence is not a
   negative.** chf06 is 80% atrial fibrillation; 11 records carry no rhythm marker
   at all, so `af_secs == 0` there means "never assessed".
5. **Records are ~20 h (17.8M-18.0M samples, ~142 MB of float32 each).**
   `window=(start, length)` is needed to batch at all, and because it is pushed
   into the reader it avoids decoding the other 20 hours.
6. **The two channels are not named leads.** The headers call them `ECG1` and
   `ECG2` and the release never says which anatomical leads they are, so `leads=`
   selects a channel position — as in `afdb` and `nsrdb`, and unlike `mitdb`.

Labels come straight from the headers and annotation files, so this works without
running the split pipeline first. The fold CSVs come from the Hub by default (or
from the pipeline — pass --metadata-source local after copying
output/chfdb/{clean,original}/ into the dataset root).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_chfdb.py --data-path /path/to/chfdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.chfdb import BEAT_NAMES, COHORT_LABEL, RR_RANGE_SECS
from ecgbench.labels.svdb import AAMI_CLASSES

#: 10 s at 250 Hz. Records are 17.8M-18.0M samples, so a window is required to
#: batch at all — and because window= pushes down into the reader, it also avoids
#: decoding the other ~20 h.
WINDOW = (0, 2500)

#: The shortest record: chf06 holds 17,789,952 samples (19.767 h). Any window has
#: to end at or before this, or it raises on that one record.
SHORTEST_RECORD_SAMPLES = 17_789_952


def main():
    parser = argparse.ArgumentParser(description="Load BIDMC CHFDB with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("chfdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- channel positions; the release names no leads")
    print(f"Duration: nominal {config.duration_seconds:.0f} s, 71,160-71,995 s in fact")
    print(f"Patients: {config.patient_id_column}  <- no subject id ships, so folds are ungrouped")
    print()
    print("!! THE BEAT ANNOTATIONS IN THIS DATABASE ARE UNAUDITED. PhysioNet: 'Annotation")
    print("!! files (with the suffix .ecg) were prepared using an automated detector and")
    print("!! have not been corrected manually.' Every beat, ectopy and HRV figure below")
    print("!! is that detector's output. Do not train a beat classifier on it.")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("chfdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(
        f"Signal shape:  {tuple(sample['signal'].shape)}"
        f"  (window {WINDOW} of {labels['n_samples']})"
    )
    print(f"  record_id            {sample['record_id']!r}")
    print(f"  lead_names           {labels['lead_names']}")
    print(f"  cohort_label         {labels['cohort_label']}   <- the same for all 15 records")
    print(f"  nyha_class           {labels['nyha_class']}      <- likewise constant")
    print(f"  age / sex            {labels['age']:.0f} {labels['sex']}")
    print(
        f"  duration_secs        {labels['duration_secs']:.0f}"
        f"  ({labels['duration_secs'] / 3600:.2f} h)"
    )
    print(
        f"  n_beats              {labels['n_beats']}" f"  ({labels['n_ectopic_beats']} not normal)"
    )
    print(f"  n_veb / veb_fraction {int(labels['n_veb'])} / {labels['veb_fraction']:.4f}")
    print(f"  mean_hr_bpm          {labels['mean_hr_bpm']:.1f}")
    print(f"  sdnn_ms              {labels['sdnn_ms']:.1f}")
    print(
        f"  af_secs              {labels['af_secs']:.1f}"
        f"   has_rhythm_annotation {bool(labels['has_rhythm_annotation'])}"
    )
    print(
        f"  annotated_fraction   {labels['annotated_fraction']:.5f}"
        "  <- covers the whole record, unlike nsrdb"
    )

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. There is no clinical label ---------------------------------------
    print(f"\ncohort_label over this split: {df['cohort_label'].value_counts().to_dict()}")
    print(f"nyha_class   over this split: {df['nyha_class'].value_counts().to_dict()}")
    print(f"  One class ({COHORT_LABEL}) for every record, asserted by the release rather")
    print("  than derived from the files. This is a positive class or a severity-matched")
    print("  comparison group, not a classification task in itself.")
    print(
        f"  Folds are stratified on sex instead: {df['sex'].value_counts().to_dict()}"
        " in this split"
    )
    print("  (11 M / 4 F across the whole release). That is for fold construction only;")
    print("  do not train on stratify_class.")

    # --- 2. `r` is ventricular, and the raw symbols hide it -------------------
    print("\nDetected beats by raw symbol, with the AAMI EC57 class each reduces to:")
    for symbol, name in BEAT_NAMES.items():
        total = int(df[f"beat_{symbol}"].sum())
        if total:
            print(f"  beat_{symbol:2s} {total:9d}  -> AAMI {AAMI_CLASSES[symbol]}   {name}")
    v_only = int(df["beat_V"].sum())
    ront = int(df["beat_r"].sum())
    print(
        f"\n  beat_V alone is {v_only}, but AAMI ventricular is {int(df['n_veb'].sum())}"
        f" — the {ront} `r` beats"
    )
    print("  (R-on-T PVCs) are ventricular too. `r` OUTNUMBERS `V` in 9 of the 15 records,")
    print("  so a pipeline counting only beat_V undercounts ventricular ectopy across most")
    print("  of this database. Use n_veb / veb_fraction / aami_V.")
    more_ront = df.index[df["beat_r"] > df["beat_V"]].tolist()
    print(f"  records in this split where r > V: {more_ront or 'none'}")

    print("\nVentricular ectopy burden — the real per-record signal here:")
    for record, row in df.sort_values("veb_fraction", ascending=False).iterrows():
        bar = "#" * round(60 * row["veb_fraction"])
        print(
            f"  {record:8} {100 * row['veb_fraction']:6.2f}% ventricular "
            f"({int(row['n_veb']):6d} of {int(row['n_beats']):6d} beats)"
            f"  {int(row['veb_per_hour']):5d}/h  {bar}"
        )
    print("  Across the release this spans 0.017% (chf12) to 20.52% (chf02) — three orders")
    print("  of magnitude, so a per-record metric is not comparable without controlling")
    print("  for it. This is also why folds cannot be stratified on it: with 15 records")
    print("  over 10 folds, StratifiedKFold needs one class of 10+, and no meaningful")
    print("  banding of this quantity provides one.")

    # --- 3. Rhythm annotation is sparse, and absence is not a negative --------
    annotated = df[df["has_rhythm_annotation"]]
    print(
        f"\nRhythm annotation: {len(annotated)} of {len(df)} records in this split carry"
        " any `+` marker."
    )
    if len(annotated):
        print(f"  {'record':8} {'AF %':>7} {'AF h':>7} {'episodes':>9} {'unasserted head s':>18}")
        for record, row in annotated.iterrows():
            print(
                f"  {record:8} {100 * row['af_fraction']:7.2f} {row['af_secs'] / 3600:7.2f} "
                f"{int(row['n_af_episodes']):9d} {row['rhythm_head_unasserted_secs']:18.1f}"
            )
    silent = df.index[~df["has_rhythm_annotation"]].tolist()
    print(f"  records with NO rhythm marker at all: {len(silent)} {silent}")
    print("  For those, af_secs == 0.0 means the rhythm was NEVER ASSESSED, not that there")
    print("  was no AF. Check has_rhythm_annotation before treating a zero as a negative.")
    print("  (chf06 also opens with 1,757 s before its first marker, and that marker is")
    print("  `(N` — implying AF that was never marked. rhythm_head_unasserted_secs has it.)")

    # --- 4. HRV, with two compounding caveats --------------------------------
    print(f"\nWhole-record HRV, over RR intervals in {RR_RANGE_SECS} s:")
    print(
        f"  mean_hr_bpm  {df['mean_hr_bpm'].min():.1f} - {df['mean_hr_bpm'].max():.1f}"
        f"  (mean {df['mean_hr_bpm'].mean():.1f})"
    )
    print(f"  sdnn_ms      {df['sdnn_ms'].min():.1f} - {df['sdnn_ms'].max():.1f}")
    print(f"  rmssd_ms     {df['rmssd_ms'].min():.1f} - {df['rmssd_ms'].max():.1f}")
    print(f"  RR intervals rejected by that filter: {int(df['n_rr_rejected'].sum())}")
    print("  Two caveats compound: the beats are unaudited, and these are whole-record")
    print("  summaries over ~20 h of activity and sleep in subjects with severe heart")
    print("  failure and heavy ectopy. A real HRV analysis would segment, exclude ectopic")
    print("  couplings, and not use uncorrected machine labels at all.")

    # --- 5. No signal-quality layer ------------------------------------------
    print(
        f"\nSignal-quality annotations: n_quality_changes"
        f" {int(df['n_quality_changes'].sum())},"
        f" n_isolated_artifacts {int(df['n_isolated_artifacts'].sum())}"
    )
    print("  Zero, in every record — this release ships no quality layer at all, where")
    print("  nsrdb, svdb and mitdb all carry per-channel bitmasks. There are deliberately")
    print("  no clean_secs/noisy_secs columns: they would assert 298.9 h of clean signal")
    print("  that nobody assessed. Judge quality from the waveform.")

    # --- 6. The two channels are unnamed -------------------------------------
    print(f"\nLead layouts in this split: {df['lead_names'].value_counts().to_dict()}")
    one = ECGDataset("chfdb", leads=["ECG2"], **common)
    print(f"leads=['ECG2'] -> {one.lead_names}, shape {tuple(one[0]['signal'].shape)}")
    print("  These are channel positions. The release states no electrode placement, so")
    print("  do NOT read ECG1/ECG2 as MLII/V1 by analogy with mitdb. Note also that only")
    print("  the current .hea files name them: the superseded .hea- copies shipped beside")
    print("  them carry no signal descriptions at all.")

    # A window that fits the longest record but not the shortest.
    print(f"\nThe shortest record holds {SHORTEST_RECORD_SAMPLES} samples (chf06), so a")
    print("window must fit inside that rather than inside the longest record's 17998848:")
    far = ECGDataset("chfdb", **{**common, "window": (17_900_000, 90_000)})
    raised = 0
    for i in range(len(far)):
        try:
            far[i]
        except WindowOutOfRangeError as e:
            raised += 1
            if raised == 1:
                print(f"  window=(17900000, 90000) raises: {e}")
    print(f"  {raised} of {len(far)} records in this split are too short for it")

    # --- Batching -------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    mb = 2 * 17_998_848 * 4 / 1e6
    print(f"  Without window= the longest record is 2 x 17998848 float32 (~{mb:.0f} MB), so a")
    print(
        f"  batch of {args.batch_size} would be ~{mb * args.batch_size:.0f} MB,"
        " every byte decoded."
    )

    # There is no class to predict here, so the useful target is a per-record
    # continuous quantity. Ventricular ectopy burden is the obvious one.
    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["veb_fraction"]], dtype=torch.float32
    )
    print(
        f"\nveb_fraction target tensor: {tuple(target.shape)}"
        f"  mean {target.mean():.4f}  max {target.max():.4f}"
    )
    print("  (a regression target; there is no classification target in this database,")
    print("  and it is derived from unaudited detector output, so treat it as weak")
    print("  supervision a human would have to confirm)")


if __name__ == "__main__":
    main()
