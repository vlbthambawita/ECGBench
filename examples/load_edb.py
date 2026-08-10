#!/usr/bin/env python3
"""
Example: European ST-T Database with labels.

90 two-hour two-lead ambulatory recordings from 79 subjects with diagnosed or
suspected myocardial ischaemia, and the reference benchmark for ST-segment and
T-wave change analysis. Five things to demonstrate:

1. **Lead layout varies more than in any other dataset here, and no lead is in
   every record.** All 90 records store two leads, but they use fifteen different
   orderings of eleven different lead pairs. MLIII/V4 and V4/MLIII are both
   present, 15 records each, so `signal[0]` is a limb lead in one half and a chest
   lead in the other. `leads=["MLIII"]` resolves against each record's own header —
   and *raises* for the 43 records that have no MLIII, because V5 (51 records) is
   the widest any lead reaches. This script shows both halves of that.
2. **The signals carry a large uncorrected DC offset.** Gain was calibrated against
   the original analog calibration signals; offset was not. 116 of the 180 signals
   sit more than 1 mV off zero, and 21 records never cross 0 mV at all — e0114
   lives entirely between +5.6 and +9.8 mV. Baseline removal is a prerequisite, and
   this script measures the offset rather than asserting it.
3. **The ground truth is episodes, and the aux text is the only thing that says
   what.** 368 ST and 401 T episodes, each an onset/extremum/end triple, marked
   independently in each signal. Three sub-cases are *not* episodes and are counted
   separately: 166 extreme-T threshold crossings, 21 axis-shift spans the annotators
   flagged as artefact mimicking ischaemia, and 12 episodes with no end annotation.
4. **The deviations are relative to each subject's own reference waveform** from
   their record's first 30 seconds — not to an absolute isoelectric line. A fixed ST
   threshold cannot reproduce these labels, and the reference waveforms themselves
   were printed on plastic rulers that no longer exist.
5. **Folds are subject-grouped on a RECONSTRUCTED patient id.** The release ships no
   subject identifier; `edb.txt` says 90 records come from 79 subjects and nothing
   says which. ECGBench reconstructs it from the header and the reconstruction
   reproduces both the published subject count and the published "70 men aged 30 to
   84". This script prints the multi-record subjects it recovers.

Labels come straight from the headers and annotation files, so this works without
running the split pipeline first. The fold CSVs come from the Hub by default (or from
the pipeline — pass --metadata-source local after copying
output/edb/{clean,original}/ into the dataset root).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_edb.py --data-path /path/to/edb/1.0.0/
"""

import argparse

import numpy as np
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.edb import (
    AAMI_CLASSES,
    AAMI_ORDER,
    BEAT_NAMES,
    RECORD_SAMPLES,
    RECORD_SECONDS,
    RHYTHM_NAMES,
    ST_BURDEN_BANDS,
    ST_BURDEN_EDGES,
)

#: 10 s at 250 Hz. Records are 1,800,000 samples each — 14.4 MB of float32 per
#: record — so batching without a window decodes two hours to use ten seconds.
WINDOW = (0, 2500)

#: The two widest-reaching leads: V5 is in 51 of 90 records and MLIII in 47.
#: Neither is universal, which is the point.
WIDEST_LEAD = "V5"


def main():
    parser = argparse.ArgumentParser(description="Load the European ST-T Database with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("edb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- the MODAL layout, and only 19 of 90 records")
    print(f"          {len(config.record_lead_layouts)} layouts in the release;"
          " select by NAME, never by index")
    print(f"Duration: {config.duration_seconds:.0f} s, uniform — all 90 records are"
          f" {RECORD_SAMPLES} samples")
    print(f"Patients: {config.patient_id_column}  <- RECONSTRUCTED; the release ships"
          " no subject id")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("edb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(k for k in sample if k != 'labels')}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}"
          f"  (window {WINDOW} of {RECORD_SAMPLES})")
    print(f"  record_id             {sample['record_id']!r}")
    print(f"  lead_names            {labels['lead_names']}"
          "   <- THIS record's layout, not config.lead_names")
    print(f"  patient_id            {labels['patient_id']!r}  (reconstructed)")
    print(f"  age / sex             {labels['age']} / {labels['sex']!r}")
    print(f"  angina_type           {labels['angina_type']!r}")
    print(f"  n_st_episodes         {int(labels['n_st_episodes'])}"
          f" ({int(labels['n_st_up'])} elevation,"
          f" {int(labels['n_st_down'])} depression)")
    print(f"  n_t_episodes          {int(labels['n_t_episodes'])}")
    print(f"  peak_st_deviation_uv  {int(labels['peak_st_deviation_uv'])}"
          "   <- vs THIS subject's reference waveform")
    print(f"  ischaemic_fraction    {labels['ischaemic_fraction']:.4f}")
    print(f"  st_t_class            {labels['st_t_class']!r}")
    print(f"  st_burden_band        {labels['st_burden_band']!r}"
          "   <- fold label, not a target")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. Fifteen lead layouts, and no universal lead ----------------------
    print("\nLead layouts in this split:")
    for layout, n in df["lead_names"].value_counts().items():
        print(f"  {layout:16} {n:3d} records")
    print("  Note MLIII|V4 and V4|MLIII are BOTH present: the same pair in either")
    print("  order, so an index-based selection crosses a limb and a chest lead with")
    print("  no error and nothing in the fold CSVs to warn you.")

    present = {}
    for layout in df["lead_names"]:
        for name in layout.split("|"):
            present[name] = present.get(name, 0) + 1
    print("\n  Records containing each lead, in this split:")
    for name, n in sorted(present.items(), key=lambda kv: -kv[1]):
        universal = "  <- in every record" if n == len(df) else ""
        print(f"    {name:6} {n:3d} / {len(df)}{universal}")

    named = ECGDataset("edb", leads=[WIDEST_LEAD], **common)
    ok = failed = 0
    example_error = ""
    first_shape = None
    for i in range(len(named)):
        try:
            row = named[i]
            ok += 1
            if first_shape is None:
                first_shape = (str(row["record_id"]), tuple(row["signal"].shape))
        except ValueError as e:
            failed += 1
            example_error = str(e)
    print(f"\n  leads=['{WIDEST_LEAD}'] -> {named.lead_names}")
    # NOT named[0] — record 0 of this split stores MLIII/V4 and has no V5 at all,
    # which is exactly the trap being demonstrated.
    if first_shape:
        print(f"  first record that resolves: {first_shape[0]}, shape {first_shape[1]}")
    print(f"  loaded {ok} records, raised for {failed} that store no {WIDEST_LEAD}")
    if example_error:
        print(f"    {example_error}")
    print("  This is the documented behaviour, not a bug: there is no lead common to")
    print("  all 90 records, so every name-based selection here excludes some of them.")
    print("  To batch edb you need leads= (for a consistent channel) AND a record")
    print("  filter — ecg_collate_fn stacks with default_collate, so a batch mixing")
    print("  layouts is fine by shape but wrong by physiology.")

    # --- 2. The uncorrected DC offset ----------------------------------------
    print("\nDC offset, measured from the samples (not asserted):")
    print(f"  {'record':8} {'leads':16} {'lead 0 median':>14} {'lead 1 median':>14}"
          f" {'crosses 0':>10}")
    offsets = []
    for i in range(min(8, len(dataset))):
        row = dataset[i]
        signal = row["signal"].numpy()
        medians = np.median(signal, axis=1)
        crosses = bool((signal.min() < 0) and (signal.max() > 0))
        offsets.append(medians)
        print(f"  {str(row['record_id']):8} {row['labels']['lead_names']:16}"
              f" {medians[0]:14.3f} {medians[1]:14.3f} {str(crosses):>10}")
    print("  Values are millivolts. Across the whole release 116 of the 180 signals")
    print("  sit more than 1 mV off zero and 58 more than 3 mV, up to +9.05 mV, while")
    print("  peak-to-peak is a normal 4.02 mV median. Subtract a baseline before")
    print("  training, and do not read absolute amplitude as ST level.")
    print(f"  amplitude_range_mv is {config.validation.amplitude_range_mv} — the 12-bit")
    print("  ADC rail, the only threshold that means anything when the offset is free.")

    # --- 3. The episode inventory, and the three things that are not episodes -
    print("\nST and T episodes over this split:")
    print(f"  ST episodes          {int(df['n_st_episodes'].sum()):5d}"
          f"  (signal 0: {int(df['n_st_episodes_sig0'].sum())},"
          f" signal 1: {int(df['n_st_episodes_sig1'].sum())})")
    print(f"    elevation          {int(df['n_st_up'].sum()):5d}")
    print(f"    depression         {int(df['n_st_down'].sum()):5d}")
    print(f"  T episodes           {int(df['n_t_episodes'].sum()):5d}"
          f"  (signal 0: {int(df['n_t_episodes_sig0'].sum())},"
          f" signal 1: {int(df['n_t_episodes_sig1'].sum())})")
    print(f"    T amplitude up     {int(df['n_t_up'].sum()):5d}")
    print(f"    T amplitude down   {int(df['n_t_down'].sum()):5d}")
    print("\n  NOT episodes, and counted apart from them:")
    print(f"    extreme-T markers  {int(df['n_extreme_t_markers'].sum()):5d}"
          "  400 uV crossings INSIDE a T episode")
    print(f"    axis-shift spans   {int(df['n_axis_shift_episodes'].sum()):5d}"
          "  positional artefact the annotators flagged")
    print(f"    unterminated       {int(df['n_unterminated_episodes'].sum()):5d}"
          "  onset + extremum, no end annotation")
    print("  The axis shifts are spelled in LOWER CASE ((st0+ not (ST0+) precisely so")
    print("  they can be told apart, so case-folding the aux text merges recognised")
    print("  artefact into the findings. The unterminated ones are closed at the record")
    print("  end: 10 of the 12 plainly run past it, but e0409's two ST depressions open")
    print("  at 8.4 and 17.2 min and simply stop being annotated — dropping them would")
    print("  zero that record's ischaemia.")

    print("\n  Time in episode (the two signals are annotated INDEPENDENTLY):")
    summed = df["st_episode_secs"].sum() / 3600
    union = df["st_secs_any_signal"].sum() / 3600
    print(f"    ST summed over signals  {summed:6.1f} h   <- can exceed the recording")
    print(f"    ST union, any signal    {union:6.1f} h   <- bounded by it")
    over = df.index[df["st_episode_secs"] > RECORD_SECONDS].tolist()
    print(f"    records whose summed ST exceeds 7200 s: {over or 'none in this split'}")
    print("    That is not an error — it is concurrent ST depression in both channels.")
    print("    Use st_secs_any_signal (or ischaemic_fraction) for a fraction.")

    # --- 4. Deviations are relative to the subject's own reference ------------
    print("\nPeak deviations, in microvolts against each subject's own reference:")
    busiest = df.sort_values("n_st_episodes", ascending=False)
    print(f"  {'record':8} {'ST':>3} {'T':>3} {'peak ST':>8} {'peak T':>7}"
          f" {'ischaemic':>10} {'class':>10}")
    for record, row in busiest.head(6).iterrows():
        print(f"  {record:8} {int(row['n_st_episodes']):3d} {int(row['n_t_episodes']):3d}"
              f" {int(row['peak_st_deviation_uv']):8d} {int(row['peak_t_deviation_uv']):7d}"
              f" {row['ischaemic_fraction']:10.3f} {row['st_t_class']:>10}")
    quiet = df.index[df["n_st_episodes"] == 0].tolist()
    print(f"  records with NO ST episode: {quiet or 'none in this split'}")
    print("  These are the negative controls an ST detector is scored against, which is")
    print("  why the 'none' band is kept separate even though 4 records cannot fill 10")
    print("  folds. Two of the four have no T episode either.")
    print("\n  A deviation of +600 uV does NOT mean the ST segment sits 600 uV above")
    print("  isoelectric. It means 600 uV above where it sat in that record's first")
    print("  30 s. Subjects with prior infarction carry fixed elevation or depression")
    print("  underneath, and these annotations mark the TRANSIENT change on top of it.")

    # --- 5. Reconstructed subject grouping -----------------------------------
    print("\nSubjects (reconstructed from the header, not released):")
    per_subject = df["patient_id"].value_counts()
    multi = per_subject[per_subject > 1]
    print(f"  {len(per_subject)} subjects over {len(df)} records in this split")
    print(f"  subjects with more than one record: {multi.to_dict() or 'none here'}")
    for subject in multi.index:
        records = df.index[df["patient_id"] == subject].tolist()
        row = df.loc[records[0]]
        print(f"    {subject}: {records}  age {row['age']} {row['sex']!r},"
              f" {row['recorder_type']!r}")
    print("  Ungrouped, those records would straddle train and test. Across the whole")
    print("  release the reconstruction gives 79 subjects — the count edb.txt states —")
    print("  and 70 men aged 30-84, which is edb.txt's phrasing exactly. That agreement")
    print("  is the check; an attempt to confirm it from the signals was inconclusive.")

    # --- Clinical background, and what it is not ------------------------------
    print("\nHeader clinical text (subject background, NOT per-record annotation):")
    for column in ("angina_type", "mi_location", "st_t_class"):
        print(f"  {column:22} {df[column].value_counts().to_dict()}")
    print(f"  {'n_diseased_vessels':22} "
          f"{df['n_diseased_vessels'].value_counts(dropna=False).to_dict()}")
    print(f"  myocardial_infarction  {int(df['myocardial_infarction'].sum())} records,"
          f" hypertension {int(df['hypertension'].sum())},"
          f" bypass_graft {int(df['bypass_graft'].sum())}")
    print("  Every subject was selected for suspected ischaemia, so this is not a")
    print("  case/control axis. Train on the episode columns.")

    print("\nBeats and rhythm:")
    beats = int(df["n_beats"].sum())
    for symbol, name in BEAT_NAMES.items():
        total = int(df[f"beat_{symbol}"].sum())
        if total:
            print(f"  beat_{symbol:2s} {total:8d}  {100 * total / beats:6.3f}%"
                  f"  -> AAMI {AAMI_CLASSES[symbol]}   {name}")
    print(f"  {beats} beats in this split, under AAMI EC57:")
    print("   ", {cls: int(df[f"aami_{cls}"].sum()) for cls in AAMI_ORDER})
    print("  aami_* is the column to combine with mitdb, svdb and incartdb on.")

    print(f"\n  dominant_rhythm: {df['dominant_rhythm'].value_counts().to_dict()}")
    print("  Sinus in every record, which is why it is not the label column. The")
    print("  non-sinus rhythm spans are short but present:")
    for code, name in RHYTHM_NAMES.items():
        if code == "N":
            continue
        minutes = df[f"rhythm_secs_{code}"].sum() / 60
        if minutes > 0:
            n = int((df[f"rhythm_secs_{code}"] > 0).sum())
            print(f"    {code:5} {minutes:7.1f} min in {n:2d} records   {name}")

    print("\nAnnotated signal quality (per channel, from the `~` bitmask):")
    print(f"  signal 0: noisy {df['sig0_noisy_secs'].sum() / 3600:5.1f} h,"
          f" unreadable {df['sig0_unreadable_secs'].sum() / 3600:4.1f} h")
    print(f"  signal 1: noisy {df['sig1_noisy_secs'].sum() / 3600:5.1f} h,"
          f" unreadable {df['sig1_unreadable_secs'].sum() / 3600:4.1f} h")
    print(f"  usable_fraction min {df['usable_fraction'].min():.4f}")
    silent = int((df["n_quality_changes"] == 0).sum())
    print(f"  {silent} record(s) in this split carry no `~` at all, and are clean throughout.")
    print(f"  median unasserted leading span: "
          f"{df['quality_head_unasserted_secs'].median() / 60:.1f} min — no record has a")
    print("  `~` at sample 0, so the span before the first one is clean by implication.")
    print("  NB the shipped annotations.shtml subtype table disagrees with the files for")
    print("  three of its nine values; ecgbench.labels.edb.decode_quality reads the")
    print("  bitmask instead, which fits all 8,918 annotations.")

    # --- Windows and batching -------------------------------------------------
    print(f"\nEvery record is exactly {RECORD_SAMPLES} samples"
          f" ({RECORD_SECONDS:.0f} s), so any window inside 7200 s fits all 90:")
    print(f"  window={WINDOW} -> {tuple(sample['signal'].shape)}")
    far = ECGDataset("edb", **{**common, "window": (RECORD_SAMPLES - 1250, 2500)})
    try:
        far[0]
        print("  ...but a window running past the end still raises:")
    except WindowOutOfRangeError as e:
        print(f"  window=({RECORD_SAMPLES - 1250}, 2500) raises: {e}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    mb = 2 * RECORD_SAMPLES * 4 / 1e6
    print(f"  Without window= each record is 2 x {RECORD_SAMPLES} float32 (~{mb:.1f} MB), so a")
    print(f"  batch of {args.batch_size} would be ~{mb * args.batch_size:.0f} MB,"
          " every byte decoded to use 10 s.")

    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["ischaemic_fraction"]], dtype=torch.float32
    )
    print(f"\nischaemic_fraction target tensor: {tuple(target.shape)}"
          f"  mean {target.mean():.4f}, max {target.max():.4f}")
    print("  (a per-record regression target. For episode DETECTION — what this database")
    print("  was built to evaluate — read the `.atr` files directly: the labels here are")
    print("  per-record summaries, not per-sample episode masks.)")
    print(f"\n  ST burden bands (edges {ST_BURDEN_EDGES} episodes): "
          f"{ {b: int((df['st_burden_band'] == b).sum()) for b in ST_BURDEN_BANDS} }")
    print("  For fold construction only — do not train on stratify_class.")


if __name__ == "__main__":
    main()
