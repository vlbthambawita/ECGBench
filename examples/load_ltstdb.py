#!/usr/bin/env python3
"""
Example: Long-Term ST Database with labels.

86 continuous ambulatory recordings of 17-48 hours from 80 subjects, annotated
beat by beat and episode by episode by three expert groups working independently
and meeting to agree a consensus. 1,992 hours, 8,897,780 reference beats, and the
largest annotated ST-episode inventory in this catalogue. Six things to
demonstrate:

1. **There are THREE sets of ST episode annotations and they disagree by a factor
   of two.** `.sta` (75 uV, 30 s) marks 1,795 ischaemic and 516 rate-related
   episodes, `.stb` (100 uV, 30 s) 1,130 and 234, `.stc` (100 uV, 60 s) 857 and
   116. None is more correct than the others. Any figure from this database is
   meaningless without its criterion, so this script prints all three side by side.
2. **Ischaemic and rate-related ST change are different findings**, and so are the
   1,493 axis shifts and 895 conduction-change shifts, which are artefacts of body
   position rather than findings about the heart. Telling them apart is the task
   this database exists for; summing them is throwing it away.
3. **This dataset cannot be batched whole.** 68 records store two signals and 18
   store three, in twelve layouts, and no lead is present in all 86 — MLIII, the
   widest, reaches 29. `ecg_collate_fn` stacks with `default_collate`, so a mixed
   batch raises. This script shows the filter that makes a batch possible.
4. **22 records name no leads at all.** Their headers describe both signals as
   `ECG` and say "Electrode locations were not recorded", which makes that the
   single largest layout in the release at 26%. For those, `leads=["ECG"]` returns
   signal 0 and no name reaches signal 1.
5. **`window=` is not optional here.** A record is up to 43,050,000 samples per
   channel; without a window every `__getitem__` decodes a full day of ECG. And the
   window must fit the *shortest* record, 15,200,000 samples.
6. **Subject grouping is published, in the record name.** `sXYYYZ`: X signals,
   subject YYY, record Z. s20271-s20274 are one person and hold 416 of the
   release's 1,795 ischaemic episodes between them.

It also prints the ten records whose tapes also produced European ST-T Database
records. Do not train on one and evaluate on the other.

Labels come straight from the headers and annotation files, so this works without
running the split pipeline first. The fold CSVs come from the Hub by default (or
from the pipeline — pass --metadata-source local after copying
output/ltstdb/{clean,original}/ into the dataset root).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_ltstdb.py --data-path /path/to/ltstdb/1.0.0/
"""

import argparse

from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.ltstdb import (
    AAMI_CLASSES,
    AAMI_ORDER,
    BEAT_NAMES,
    ISCHEMIC_BURDEN_EDGES,
    ISCHEMIC_BURDEN_NAMES,
    ST_ANNOTATORS,
)

#: 10 s at 250 Hz. The shortest record is 15,200,000 samples (16.9 h) and the
#: longest 43,050,000 (47.8 h), so a window must fit the former, and reading a
#: whole record to use ten seconds of it decodes up to 172 MB of float32.
WINDOW = (0, 2500)

#: The widest-reaching named lead: MLIII is in 29 of the 86 records. Nothing
#: reaches all of them, which is the point.
WIDEST_LEAD = "MLIII"


def main():
    parser = argparse.ArgumentParser(description="Load the Long-Term ST Database with labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("ltstdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- the MODAL layout, and it names NOTHING")
    print(f"          {len(config.record_lead_layouts)} layouts in the release,"
          " of 2 AND 3 signals; select by NAME, never by index")
    print(f"Duration: {config.duration_seconds:.0f} s is the MEDIAN — records run"
          " 15,200,000 to 43,050,000 samples")
    print(f"Patients: {config.patient_id_column}  <- published, inside the record name")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
    )

    try:
        dataset = ECGDataset("ltstdb", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(k for k in sample if k != 'labels')}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}"
          f"  (window {WINDOW} of {int(labels['sig_len'])})")
    print(f"  record_id                 {sample['record_id']!r}")
    print(f"  patient_id                {labels['patient_id']!r}"
          "   <- the subject field of the record name")
    print(f"  lead_names                {labels['lead_names']}"
          "   <- THIS record's layout, not config.lead_names")
    print(f"  duration_hours            {labels['duration_hours']:.2f}")
    print(f"  age / sex                 {labels['age']} / {labels['sex']!r}")
    print(f"  diagnoses                 {labels['diagnoses']!r}")
    print(f"  n_ischemic_episodes       {int(labels['n_ischemic_episodes'])}"
          "   <- criterion A (.sta)")
    print(f"  n_rate_related_episodes   {int(labels['n_rate_related_episodes'])}")
    print(f"  n_axis_shifts             {int(labels['n_axis_shifts'])}"
          "   <- artefact, not a finding")
    print(f"  peak_st_deviation_uv      {int(labels['peak_st_deviation_uv'])}"
          "   <- vs the annotator-placed baseline")
    print(f"  ischemic_fraction         {labels['ischemic_fraction']:.4f}")
    print(f"  st_class                  {labels['st_class']!r}")
    print(f"  ischemic_burden_band      {labels['ischemic_burden_band']!r}"
          "   <- fold label, not a target")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. Three criteria, three answers --------------------------------------
    print("\nST episodes over this split, under each of the three shipped criteria:")
    print(f"  {'file':6} {'criterion':16} {'ischaemic':>10} {'rate-related':>13}"
          f" {'total':>7} {'records w/ isch.':>17}")
    for ext, suffix, criterion in ST_ANNOTATORS:
        isch = int(df[f"n_ischemic_episodes{suffix}"].sum())
        rate = int(df[f"n_rate_related_episodes{suffix}"].sum())
        n_rec = int((df[f"n_ischemic_episodes{suffix}"] > 0).sum())
        print(f"  .{ext:5} {criterion:16} {isch:10d} {rate:13d} {isch + rate:7d}"
              f" {n_rec:14d}/{len(df)}")
    print("  Same recordings, same annotators, different detection thresholds. The")
    print("  unsuffixed columns are .sta; the others are _b and _c. NEVER quote an")
    print("  episode count from this database without saying which criterion it is.")

    print("\n  The four quantities that are IDENTICAL in all three files, because they")
    print("  are marks rather than threshold crossings:")
    for column, what in (
        ("n_axis_shifts", "axis shifts — body position, mimics ischaemia"),
        ("n_conduction_change_shifts", "conduction-change shifts — likewise"),
        ("n_noise_events", "noise events"),
        ("n_unreadable_intervals", "unreadable intervals"),
    ):
        total = int(df[column].sum())
        print(f"    {column:28} {total:5d}  in {int((df[column] > 0).sum()):2d} records"
              f"   {what}")

    # --- 2. Ischaemic is not rate-related --------------------------------------
    print("\nIschaemic vs rate-related, per record — the distinction the database is for:")
    print(f"  {'record':8} {'isch':>5} {'rate':>5} {'axis':>5} {'cc':>4}"
          f" {'peak uV':>8} {'frac':>7} {'class':>26}")
    busiest = df.sort_values("n_ischemic_episodes", ascending=False)
    for record, row in busiest.head(5).iterrows():
        print(f"  {record:8} {int(row['n_ischemic_episodes']):5d}"
              f" {int(row['n_rate_related_episodes']):5d}"
              f" {int(row['n_axis_shifts']):5d} {int(row['n_conduction_change_shifts']):4d}"
              f" {int(row['peak_st_deviation_uv']):8d} {row['ischemic_fraction']:7.3f}"
              f" {row['st_class']:>26}")
    rate_only = df.index[df["st_class"] == "rate_related_only"].tolist()
    quiet = df.index[df["n_ischemic_episodes"] == 0].tolist()
    print(f"\n  st_class over this split: {df['st_class'].value_counts().to_dict()}")
    print(f"  records with NO ischaemic episode: {len(quiet)} of {len(df)}")
    print(f"  ...of which rate-related only:     {len(rate_only)}")
    print("  Record s20011 is the case to understand: 20 criterion-A episodes, every")
    print("  one rate-related, and its header says why — \"all episodes in lead 1 are")
    print("  compatible with heart-rate induced non-ischemic changes and are so")
    print("  labeled. It is recognized that this is an arbitrary decision.\"")

    print("\n  Episodes are counted at their EXTREMUM, not their onset:")
    print(f"    already running at sample 0     {int(df['n_episodes_open_at_start'].sum()):4d}"
          "   no onset annotation; measured from 0")
    print(f"    still running at the last sample {int(df['n_unterminated_episodes'].sum()):3d}"
          "   no end annotation; closed at the end")
    print("    Counting onsets would miss the first group entirely and disagree with")
    print("    the release's own .cnt summaries for 14 of the 86 records. Counting")
    print("    extrema reproduces them in all 258 blocks (86 records x 3 criteria x 6).")

    print("\n  Time in episode (the leads are annotated INDEPENDENTLY):")
    summed = df["ischemic_secs"].sum() / 3600
    union = df["ischemic_secs_any_lead"].sum() / 3600
    hours = df["duration_hours"].sum()
    print(f"    ischaemic, summed over leads {summed:7.1f} h   <- can exceed the recording")
    print(f"    ischaemic, union over leads  {union:7.1f} h   <- bounded by it")
    print(f"    recorded                     {hours:7.1f} h")
    print("    Use ischemic_secs_any_lead, or ischemic_fraction, for any proportion.")

    # --- 3 & 4. Twelve layouts, two lead counts, 22 unnamed --------------------
    print("\nLead layouts in this split:")
    for layout, n in df["lead_names"].value_counts().items():
        unnamed = "   <- electrode locations not recorded" if layout == "ECG|ECG" else ""
        print(f"  {layout:16} {n:3d} records ({len(layout.split('|'))} signals){unnamed}")
    print(f"  signals per record: {df['n_leads'].value_counts().sort_index().to_dict()}")
    print(f"  records naming their leads: {int(df['leads_named'].sum())} of {len(df)}")

    present = {}
    for layout in df["lead_names"]:
        for name in set(layout.split("|")):
            present[name] = present.get(name, 0) + 1
    print("\n  Records containing each lead, in this split:")
    for name, n in sorted(present.items(), key=lambda kv: -kv[1]):
        universal = "  <- in every record" if n == len(df) else ""
        print(f"    {name:6} {n:3d} / {len(df)}{universal}")

    named = ECGDataset("ltstdb", leads=[WIDEST_LEAD], **common)
    ok = failed = 0
    example_error = ""
    first = None
    for i in range(len(named)):
        try:
            row = named[i]
            ok += 1
            if first is None:
                first = (str(row["record_id"]), tuple(row["signal"].shape))
        except ValueError as e:
            failed += 1
            example_error = str(e)
    print(f"\n  leads=['{WIDEST_LEAD}'] -> {named.lead_names}")
    if first:
        print(f"  first record that resolves: {first[0]}, shape {first[1]}")
    print(f"  loaded {ok} records, raised for {failed} that store no {WIDEST_LEAD}")
    if example_error:
        print(f"    {example_error}")
    print("  Across the release V4/MLIII and MLIII/V4 are BOTH present — 20 records")
    print("  and 6 — so an index-based selection crosses a chest and a limb lead with")
    print("  no error. And for the 22 ECG/ECG records leads=['ECG'] returns SIGNAL 0;")
    print("  no name that reaches signal 1, because the placement was never recorded.")

    # --- 5. Windows and batching ------------------------------------------------
    shortest = int(df["sig_len"].min())
    longest = int(df["sig_len"].max())
    print(f"\nRecord length in this split: {shortest:,} to {longest:,} samples"
          f" ({shortest / 250 / 3600:.1f} to {longest / 250 / 3600:.1f} h).")
    print(f"  window={WINDOW} -> {tuple(sample['signal'].shape)}, and it fits all of them.")
    far = ECGDataset("ltstdb", **{**common, "window": (shortest - 1250, 2500)})
    for i in range(len(far)):
        try:
            far[i]
        except WindowOutOfRangeError as e:
            print(f"  window=({shortest - 1250:,}, 2500) raises: {e}")
            break
    else:
        print(f"  window=({shortest - 1250:,}, 2500) fits every record in this split.")
    mb = 2 * longest * 4 / 1e6
    print(f"  Without window=, the longest record alone is 2 x {longest:,} float32"
          f" (~{mb:.0f} MB).")

    print("\nBatching needs a RECORD FILTER, not just leads=:")
    print("  ecg_collate_fn stacks with torch's default_collate, so a batch mixing 2-")
    print("  and 3-signal records raises RuntimeError — and no leads= value fixes it,")
    print("  because no lead is in every record. Filter first:")
    keep = [
        i for i, layout in enumerate(df["lead_names"])
        if WIDEST_LEAD in layout.split("|")
    ]
    print(f"  {len(keep)} of {len(dataset)} records in this split hold {WIDEST_LEAD}")
    subset = torch_subset(named, keep)
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"  One batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")

    # --- 6. Subject grouping ----------------------------------------------------
    print("\nSubjects (published, in the record name — sXYYYZ):")
    per_subject = df["patient_id"].value_counts()
    multi = per_subject[per_subject > 1]
    print(f"  {len(per_subject)} subjects over {len(df)} records in this split")
    print(f"  subjects with more than one record: {multi.to_dict() or 'none here'}")
    for subject in multi.index:
        records = df.index[df["patient_id"] == subject].tolist()
        episodes = int(df.loc[records, "n_ischemic_episodes"].sum())
        print(f"    {subject}: {records}  {episodes} ischaemic episodes between them")
    print("  Across the whole release subject 027 contributed four records holding 416")
    print("  of the 1,795 ischaemic episodes — 23% of the database in one person.")
    print("  Ungrouped folds would put the same day of the same heart on both sides.")

    # --- The European ST-T overlap ----------------------------------------------
    shared = df[df["edb_record"] != ""]
    print(f"\nRecords whose tapes ALSO produced a European ST-T record: {len(shared)}")
    for record, row in shared.iterrows():
        print(f"  {record} -> edb {row['edb_record']}"
              f"   ({int(row['n_ischemic_episodes'])} ischaemic episodes here)")
    print("  DO NOT train on this database and evaluate ST detection on edb, or the")
    print("  reverse. The tapes were redigitised and rescaled, so the samples differ")
    print("  and no correlation check finds the overlap — but the hours are the same.")
    pilot = df[df["pilot_record"] != ""]
    print(f"  ({len(pilot)} records also carry a pilot_record name from the unpublished")
    print("   1995-98 collection. Do NOT read those as record ids here: s20071's is")
    print("   's20511', which is a different subject's record in this release.)")

    # --- Beats ------------------------------------------------------------------
    print("\nBeats and rhythm:")
    beats = int(df["n_beats"].sum())
    for symbol, name in BEAT_NAMES.items():
        total = int(df[f"beat_{symbol}"].sum())
        if total:
            print(f"  beat_{symbol:2s} {total:9d}  {100 * total / beats:6.3f}%"
                  f"  -> AAMI {AAMI_CLASSES[symbol]}   {name}")
    print(f"  {beats:,} beats in this split, under AAMI EC57:")
    print("   ", {cls: int(df[f"aami_{cls}"].sum()) for cls in AAMI_ORDER})
    print("  aami_* is the column to combine with mitdb, svdb, incartdb and edb on.")
    print(f"  annotated_fraction: min {df['annotated_fraction'].min():.5f} — every record")
    print("  is annotated end to end, unlike nsrdb's 12.1% unannotated tail.")
    print(f"  mean heart rate {df['mean_hr_bpm'].min():.1f}-{df['mean_hr_bpm'].max():.1f} bpm")
    print("\n  There is NO rhythm annotation and NO signal-quality annotation in the")
    print("  .atr files — they hold beats and nothing else, which is unusual for a")
    print("  MIT-BIH-family release. Noise and unreadable spans live in the ST files:")
    print(f"    unreadable {df['unreadable_secs'].sum() / 3600:.2f} h in"
          f" {int((df['unreadable_secs'] > 0).sum())} records")

    # --- The header's clinical record --------------------------------------------
    print("\nHeader clinical fields — nullable boolean plus the verbatim text:")
    for column in ("hypertension", "previous_mi", "lv_hypertrophy",
                   "intraventricular_conduction_block", "bypass_grafting"):
        v = df[column]
        yes, no, na = int((v == True).sum()), int((v == False).sum()), int(v.isna().sum())  # noqa: E712
        examples = sorted({t for t in df.loc[v == True, f"{column}_text"] if t})[:2]  # noqa: E712
        print(f"  {column:34} yes {yes:3d}  no {no:3d}  NA {na:3d}   e.g. {examples}")
    print("  NA means the header said \"No data\". The answers are free text even where")
    print("  they look boolean, so read <field>_text whenever the detail matters.")
    print(f"\n  recorder: {df['recorder'].value_counts().to_dict()}")
    print(f"  age {df['age'].min():.0f}-{df['age'].max():.0f},"
          f" mean {df['age'].mean():.1f},"
          f" not recorded in {int(df['age'].isna().sum())} records;"
          f" sex {df['sex'].value_counts().to_dict()}")

    # --- Targets -------------------------------------------------------------------
    import torch

    target = torch.tensor(
        [float(x) for x in dataset.labels_df["ischemic_fraction"]], dtype=torch.float32
    )
    print(f"\nischemic_fraction target tensor: {tuple(target.shape)}"
          f"  mean {target.mean():.4f}, max {target.max():.4f}")
    print("  (a per-record regression target. For episode DETECTION — what this")
    print("  database was built to evaluate — read the .sta/.stb/.stc files directly:")
    print("  the labels here are per-record summaries, not per-sample episode masks.)")
    print(f"\n  Ischaemic burden bands (edges {ISCHEMIC_BURDEN_EDGES} episodes): "
          f"{ {b: int((df['ischemic_burden_band'] == b).sum()) for b in ISCHEMIC_BURDEN_NAMES} }")
    print("  For fold construction only — do not train on stratify_class.")


def torch_subset(dataset, indices):
    """``torch.utils.data.Subset``, imported here to keep the header imports light."""
    from torch.utils.data import Subset

    return Subset(dataset, indices)


if __name__ == "__main__":
    main()
