#!/usr/bin/env python3
"""
Example: QT Database with labels and per-beat wave boundaries.

105 fifteen-minute two-lead excerpts in which cardiologists marked the onset, peak
and end of the P, QRS, T and U waves of 3,623 selected beats. It is the reference
benchmark for ECG *delineation*, and almost nothing about it behaves like the other
two-lead Holter databases in this catalogue. Seven things to demonstrate:

1. **The ground truth is per beat, not per record.** `labels=True` gives a summary;
   the boundaries themselves come from
   `ecgbench.labels.qtdb.load_beat_annotations()`, which returns 3,623 rows with up
   to eleven fiducial points each. This script loads both and shows how to line the
   samples up with the tensor.
2. **The annotation is in the last five minutes and nowhere else.** The earliest
   manual annotation in the release sits at 600.464 s and the latest at 896.916 s,
   deliberately, to leave an algorithm ten minutes of learning data. So
   `window=(150000, 74993)` is the whole annotated region — a window from sample 0
   contains no ground truth at all, which the script demonstrates.
3. **Every record is an excerpt of another database's recording.** 100 of the 105
   share signal samples with `edb`, `sddb`, `mitdb`, `svdb`, `nsrdb` or `stdb` —
   verified from the waveforms, not inferred from the names. The script prints the
   leakage partner of every record.
4. **There is no diagnostic label.** `source_database` fills `label_column` because
   the release has no record-level class of any kind. It is provenance, not
   pathology, and the script says so where a reader would otherwise train on it.
5. **20 lead layouts, and the modal pair is a placeholder.** 57 records describe
   both channels only as `ECG1`/`ECG2`; the other 48 name them, and the 33 European
   ST-T records use the ESC's own electrode nomenclature, which does not match
   `edb`'s names for the same channels.
6. **Amplitude is unreliable for 34 records, and four sit 5.12 mV high.** Intervals
   are unaffected; millivolt comparisons across source databases are not.
7. **Two annotators, 11 records, and one of them has three beats.** `sel102` lost
   82 of annotator 2's 85 beats in the audit, so inter-observer figures have to be
   weighted.

Labels come from the headers and the nine annotation layers, so this works without
running the split pipeline first. The fold CSVs come from the Hub by default (or
pass --metadata-source local after copying output/qtdb/{clean,original}/ into the
dataset root).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_qtdb.py --data-path /path/to/qtdb/1.0.0/
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.qtdb import (
    ANNOTATED_WINDOW,
    SOURCE_CATALOGUE_SLUG,
    SOURCE_DATABASE_NAMES,
    load_beat_annotations,
)

#: Exactly the annotated region: samples 150,000 to 224,993, i.e. 600.0 s to the end
#: of the shortest record. Every record is 224,993 samples or longer, so this fits
#: all 105 — and because window= is pushed into the reader it also avoids decoding
#: the first ten minutes, which carry no ground truth.
WINDOW = ANNOTATED_WINDOW

#: The shortest record. All 23 sudden-death excerpts hold 224,993 samples rather
#: than the nominal 225,000, because their headers record a 7-sample delay applied
#: to signal 0. Any window must end at or before this.
SHORTEST_RECORD_SAMPLES = 224_993


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Local qtdb/1.0.0 root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("qtdb")
    print(f"=== {config.name} ({config.slug}) v{config.version} ===")
    print(f"  {config.leads} leads at {config.default_sampling_rate} Hz, "
          f"{config.duration_seconds:.0f} s nominal, {config.signal_format}")
    print(f"  licence: {config.license}")
    print(f"  lead_names (the MODAL layout): {config.lead_names}")
    print(f"  record_lead_layouts: {len(config.record_lead_layouts)} distinct layouts")
    print("  All 105 records pass validation, so clean/ == original/.")

    common = dict(
        data_path=args.data_path,
        version=args.version,
        metadata_source=args.metadata_source,
    )

    try:
        dataset = ECGDataset(
            "qtdb", split=args.split, window=WINDOW, labels=True, **common
        )
    except LabelSourceMissingError as e:
        print(f"\nLabels unavailable: {e}")
        return

    print(f"\n{len(dataset)} records in split '{args.split}' (version={args.version})")

    # labels_df is reindexed positionally so __getitem__ needs no per-item join, so
    # its index is 0..n-1 and the record names live in metadata_df. Anything that
    # wants to name a record has to go through this.
    names = list(dataset.metadata_df[config.record_id_column])

    sample = dataset[0]
    print(f"\nds[0] keys: {sorted(sample.keys())}")
    print(f"  record_id  {sample['record_id']}")
    print(f"  signal     {tuple(sample['signal'].shape)}  "
          f"(2 x {WINDOW[1]} = the annotated region only)")

    labels = sample["labels"]
    print("\n--- Provenance: where this excerpt came from ---")
    source = labels["source_database"]
    print(f"  source_database        {source}  ({SOURCE_DATABASE_NAMES[source]})")
    print(f"  source_record          {labels['source_record']}")
    print(f"  source_offset_secs     {labels['source_offset_secs']}")
    print(f"  source_sampling_rate   {labels['source_sampling_rate']} Hz")
    print(f"  resampled_from_source  {labels['resampled_from_source']}")
    partner = labels["source_catalogue_slug"]
    print(f"  source_catalogue_slug  {partner or '(not in the ECGBench catalogue)'}")
    print(f"  source_record_verified {labels['source_record_verified']}")

    print("\n--- The manual boundary annotations, summarised ---")
    for key in (
        "n_annotated_beats", "n_annotated_beats_published",
        "annotated_beats_matches_published", "waveform_pattern",
        "n_p_waves", "n_t_onsets", "n_t_ends", "n_u_waves",
        "median_qt_ms", "median_qtc_bazett_ms", "median_rr_ms",
        "median_heart_rate_bpm", "median_qrs_ms", "median_pr_ms",
        "annotation_start_secs", "annotation_end_secs",
        "has_second_annotator", "n_annotated_beats_annotator2",
    ):
        print(f"  {key:38s} {labels[key]}")

    print("\n--- Signal properties of THIS record ---")
    for key in (
        "lead_names", "positional_lead_names", "declared_gain_0", "declared_gain_1",
        "amplitude_calibrated", "dc_pedestal_mv", "n_samples",
        "signal_0_delay_samples",
    ):
        print(f"  {key:38s} {labels[key]}")

    # --- The provenance table, which is the leakage warning ------------------
    print("\n=== 100 of the 105 records are in another ECGBench dataset ===")
    counts = dataset.labels_df["source_database"].value_counts()
    for db, n in counts.items():
        partner = SOURCE_CATALOGUE_SLUG[db] or "-- not in the catalogue --"
        print(f"  {db:12s} {n:3d} records in this split   -> {partner}")
    shared = sum(n for db, n in counts.items() if SOURCE_CATALOGUE_SLUG[db])
    print(f"\n  {shared} of {len(dataset)} records in split '{args.split}' share signal")
    print("  samples with a dataset ECGBench also partitions. These folds are disjoint")
    print("  WITHIN qtdb only. Filter on source_database before combining:")
    print("    delineation_only = ds.labels_df.query(\"source_database == 'edb'\")")
    print("  30 of the 33 European ST-T excerpts are BIT-IDENTICAL to edb 1.0.0, and")
    print("  22 of the 23 sudden-death ones reproduce sddb exactly as trunc(sddb/4).")

    # --- The actual ground truth --------------------------------------------
    print("\n=== The ground truth is per beat, so it is a separate call ===")
    beats = load_beat_annotations(args.data_path)
    print(f"  load_beat_annotations() -> {beats.shape[0]} beats x {beats.shape[1]} cols")
    print(f"  columns: {list(beats.columns)}")

    record = sample["record_id"]
    mine = beats[beats["record_name"] == record]
    print(f"\n  {record}: {len(mine)} annotated beats")
    cols = ["p_onset", "p_peak", "qrs_onset", "qrs_peak", "qrs_offset",
            "t_peak", "t_offset", "qt_ms", "rr_ms"]
    print(mine[cols].head(4).to_string(index=False))
    print("  NaN means the annotator did not mark that point — information, not")
    print("  missing data. sel35 and sel37 mark QRS boundaries and no T wave at all.")

    # Boundaries are absolute samples in the record's own 250 Hz frame, so they have
    # to be shifted by the window start before they index the tensor.
    first = mine.iloc[0]
    onset_in_window = int(first["qrs_onset"]) - WINDOW[0]
    offset_in_window = int(first["t_offset"]) - WINDOW[0]
    print(f"\n  First beat: QRS onset at sample {int(first['qrs_onset'])} absolute")
    print(f"              = index {onset_in_window} in the windowed tensor")
    print(f"              T end at {int(first['t_offset'])} = index {offset_in_window}")
    qrs_to_t = sample["signal"][:, onset_in_window : offset_in_window + 1]
    print(f"  signal[:, onset:t_end+1] -> {tuple(qrs_to_t.shape)}  "
          f"= {first['qt_ms']:.0f} ms of QT interval")

    print("\n  Release-wide interval statistics, over the beats rather than the records:")
    for column in ("qt_ms", "qtc_bazett_ms", "rr_ms", "qrs_ms", "pr_ms"):
        values = beats[column].dropna()
        print(f"    {column:14s} n={len(values):5d}  median {values.median():7.1f}  "
              f"p1 {values.quantile(0.01):7.1f}  p99 {values.quantile(0.99):7.1f}")
    print("  The heart rates are low because the excerpts were chosen to avoid noise,")
    print("  which the paper itself warns about: a delineator validated only here has")
    print("  not been tested at tachycardia, on baseline wander, or on ectopic beats.")

    # --- Why the window matters ----------------------------------------------
    print("\n=== A window from sample 0 contains no ground truth ===")
    unwindowed = ECGDataset("qtdb", split=args.split, window=(0, 74993), **common)
    mark_columns = [c for c in beats.columns if c.endswith(("_onset", "_peak", "_offset"))]
    earliest = beats[mark_columns].min().min()
    latest = beats[mark_columns].max().max()
    print("  window=(0, 74993) covers samples 0-74992 = 0.0-300.0 s")
    print(f"  the earliest mark anywhere in the release is at sample "
          f"{int(earliest)} = {earliest / 250:.3f} s, the latest at "
          f"{int(latest)} = {latest / 250:.3f} s")
    print(f"  so all {len(unwindowed)} records load fine and every one of them is")
    print(f"  unlabelled signal. Use window={WINDOW} instead.")
    print(f"  (It fits every record because the shortest holds "
          f"{SHORTEST_RECORD_SAMPLES} samples.)")

    # --- Lead selection ------------------------------------------------------
    print("\n=== 20 lead layouts, and ECG1/ECG2 is a placeholder ===")
    layouts = dataset.labels_df["lead_names"].value_counts()
    for layout, n in layouts.head(6).items():
        print(f"  {layout:16s} {n:3d} records")
    positional = int(dataset.labels_df["positional_lead_names"].sum())
    print(f"  {positional} of {len(dataset)} records in this split declare no electrode")
    print("  placement at all, so ECG1/ECG2 are channel positions. Do NOT read them")
    print("  as MLII/V1 by analogy with mitdb.")

    named = dataset.labels_df.query("not positional_lead_names")
    if len(named):
        wanted = named["lead_names"].iloc[0].split(";")[0]
        has_it = [
            r for r in dataset.labels_df.index
            if wanted in dataset.labels_df.loc[r, "lead_names"].split(";")
        ]
        # With record_lead_layouts declared, the requested NAMES are resolved against
        # each record's own header rather than against a declared index. So this
        # returns the right physical lead where the record has it and refuses where it
        # does not — it never hands back whichever channel sits at that position.
        selected = ECGDataset(
            "qtdb", split=args.split, window=WINDOW, leads=[wanted], **common
        )
        print(f"\n  leads=['{wanted}'] over the whole split:")
        resolved = refused = 0
        first_error = ""
        for position, record_id in enumerate(names):
            try:
                selected[position]
                resolved += 1
            except ValueError as e:
                refused += 1
                if not first_error:
                    first_error = f"{record_id}: {e}"
        print(f"    {resolved} record(s) resolved, {refused} refused")
        print(f"    expected {len(has_it)} to resolve, from their own lead_names")
        if first_error:
            print("    the refusal names the record and its actual channels:")
            print(f"      {first_error}")
        print("  No lead is common to all 105 records, so leads= cannot make qtdb")
        print("  batchable without also restricting which records are loaded — exactly")
        print("  as for edb. That is the documented behaviour, not a bug.")

    print("\n  And the names disagree with edb for the SAME channels: edb's MLIII is")
    print("  qtdb's D3 or ML5, its V5 is CM5, its V2 is CM2/V1-V2/V2-V3. Of the 33")
    print("  shared records, leads=['V5'] selects 14 under edb's names and 2 under")
    print("  qtdb's, over signals that are bit-identical.")

    # --- Units and calibration ----------------------------------------------
    print("\n=== Amplitude: unreliable for 34 records, +5.12 mV on four ===")
    uncal = [n for n, ok in zip(names, dataset.labels_df["amplitude_calibrated"])
             if not ok]
    print(f"  {len(uncal)} of {len(dataset)} records in this split: "
          f"{sorted(uncal)[:6]}{'...' if len(uncal) > 6 else ''}")
    print("  24 are sudden-death Holters whose gains the paper calls estimates; 10 more")
    print("  declare a gain of 0, which wfdb silently replaces with 200 adu/mV.")
    pedestal = [(n, float(mv)) for n, mv
                in zip(names, dataset.labels_df["dc_pedestal_mv"]) if mv > 0]
    print(f"  dc_pedestal_mv > 0 for {len(pedestal)} record(s): "
          f"{[n for n, _ in pedestal]}")
    if pedestal:
        rec, offset = pedestal[0]
        position = names.index(rec)
        raw = dataset[position]["signal"]
        print(f"  {rec}: signal min {raw.min():.3f} mV, max {raw.max():.3f} mV — never")
        print("    negative, because wfdb honours its explicit baseline of 0 against an")
        print(f"    adc_zero of 1024. Subtract {offset} to compare with mitdb's copy:")
        print(f"    corrected min {(raw - offset).min():.3f} mV, "
              f"max {(raw - offset).max():.3f} mV")

    # --- Inter-observer ------------------------------------------------------
    print("\n=== Two annotators, 11 records, and one of them has three beats ===")
    second = load_beat_annotations(args.data_path, annotator="q2c")
    print(f"  annotator 2: {len(second)} beats over {second.record_name.nunique()} records")
    both = beats[beats["record_name"].isin(second["record_name"].unique())]
    print(f"  annotator 1 marked {len(both)} beats in those same records")
    per_record = (
        both.groupby("record_name").size().rename("annotator_1").to_frame()
        .join(second.groupby("record_name").size().rename("annotator_2"))
    )
    print(per_record.to_string())
    print("  Weight any inter-observer figure by annotator_2 — sel102's 3 beats against")
    print("  85 is the largest disagreement in the release and would otherwise vanish.")

    # --- Batching ------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    full_mb = 2 * 225_000 * 4 / 1e6
    win_mb = 2 * WINDOW[1] * 4 / 1e6
    print(f"  window= keeps this at {win_mb:.1f} MB per record instead of "
          f"{full_mb:.1f} MB, and")
    print("  every sample it drops is unannotated. It also survives")
    print("  DataLoader(num_workers>0) under spawn, where a lambda transform raises.")

    # --- Targets -------------------------------------------------------------
    print("\n=== There is no class to predict ===")
    print("  label_column is source_database, which is PROVENANCE, not pathology:")
    print(f"    {dict(dataset.labels_df['source_database'].value_counts())}")
    print("  It exists so folds are balanced across the seven sources. Do not train on")
    print("  it — a model that learns it has learned which database a signal came from.")
    print("\n  The real targets are the fiducial points. A per-sample mask for one beat:")
    mask = torch.zeros(sample["signal"].shape[1], dtype=torch.long)
    n_marked = 0
    for _, beat in mine.iterrows():
        for wave, value in ((1, "qrs"), (2, "t"), (3, "p")):
            start, end = beat.get(f"{value}_onset"), beat.get(f"{value}_offset")
            if np.isnan(start) or np.isnan(end):
                continue
            lo = int(start) - WINDOW[0]
            hi = int(end) - WINDOW[0]
            if 0 <= lo < hi < mask.shape[0]:
                mask[lo : hi + 1] = wave
                n_marked += 1
    covered = int((mask > 0).sum())
    print(f"  mask {tuple(mask.shape)} from {n_marked} annotated wave segments: "
          f"{covered} samples labelled ({100 * covered / mask.shape[0]:.1f}% of the window)")
    print("  0 = unlabelled, 1 = QRS, 2 = T, 3 = P. The rest is unlabelled because only")
    print("  ~1.6% of the beats in this database were ever annotated — which is the")
    print("  point of the 10-minute learning period, not a defect.")


if __name__ == "__main__":
    main()
