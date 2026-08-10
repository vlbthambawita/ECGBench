#!/usr/bin/env python3
"""
Example: Sudden Cardiac Death Holter Database with labels.

23 complete two-lead Holter recordings of patients who died — or nearly died —
*during* the recording. Seven things to demonstrate, because almost nothing about
this database behaves like its siblings:

1. **Use `version="original"`.** 20 of the 23 records contain NaN samples, so they
   fail `nan_values` and `clean/` holds 3 records with **empty val and test**. The
   NaN is WFDB's invalid-sample marker (digital −2048) for brief analog-tape
   dropouts, not corruption — but it is real, it lands in your tensor, and it will
   make your loss NaN. This script defaults to `original` and shows how much NaN
   is in the window it just loaded.
2. **The terminal event is in a header comment, not an annotation.** `vf_onset_secs`
   comes from `#vfon: HH:MM:SS` and is elapsed from the record start. There is not
   one `[` (VFON) annotation in the release. Onset lands from 6.1% to 98.9% of the
   way through, so no single `window=` captures it across records — which the
   script demonstrates rather than asserts.
3. **There are two annotators covering different records.** `.ari` is unaudited and
   covers all 23; `.atr` is the audited reference and covers only 12. Every beat
   column is prefixed `ari_` or `atr_` for that reason.
4. **The `(AFIB` markers are not an AF label.** They disagree with the published
   clinical rhythm in both directions. The script prints them side by side so the
   disagreement is visible.
5. **The audited annotation stops early — at exactly 24 h in four records.** A late
   window has no reference behind it and nothing errors.
6. **Nothing clinical is in the files.** Age, sex, history, medication and rhythm
   come from a table on the landing page, transcribed into `ecgbench.labels.sddb`.
7. **The two channels are not named leads.** The headers call both `ECG`; ECGBench
   spells them `ECG1`/`ECG2`, which are channel positions.

Labels come from the headers, both annotation files and that transcribed table, so
this works without running the split pipeline first. The fold CSVs come from the
Hub by default (or pass --metadata-source local after copying
output/sddb/{clean,original}/ into the dataset root).

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub.

Usage:
  python examples/load_sddb.py --data-path /path/to/sddb/1.0.0/
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.dataset import WindowOutOfRangeError
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.sddb import BEAT_NAMES, COHORT_LABEL, RR_RANGE_SECS
from ecgbench.labels.svdb import AAMI_CLASSES

#: 10 s at 250 Hz. Records run 3.54M to 22.63M samples, so a window is required to
#: batch at all — and because window= pushes down into the reader, it also avoids
#: decoding the other 3 to 25 hours.
WINDOW = (0, 2500)

#: The shortest record: 41 holds 3,540,000 samples (14,160 s, 3.93 h). Any window
#: has to end at or before this, or it raises on that one record. This is a much
#: tighter bound than the sibling long-term databases impose — sddb's longest
#: record is 6.4x its shortest, where chfdb's spread is 1.2%.
SHORTEST_RECORD_SAMPLES = 3_540_000


def main():
    parser = argparse.ArgumentParser(description="Load the Sudden Cardiac Death Holter DB")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    # original, NOT clean — see point 1 in the module docstring.
    parser.add_argument("--version", default="original", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("sddb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:    {config.lead_names}  <- channel positions; both headers say 'ECG'")
    print(f"Duration: nominal {config.duration_seconds:.0f} s, 14,160-90,510 s in fact")
    print(f"Patients: {config.patient_id_column}  <- the subject id IS the record name")
    print()
    print("!! USE version='original'. 20 of the 23 records contain NaN samples (WFDB's")
    print("!! invalid-sample marker, -2048, for brief tape dropouts), so they fail")
    print("!! nan_values and clean/ holds 3 records with EMPTY val and test.")
    print("!! ECGDataset('sddb', split='val') defaults to clean and then either raises a")
    print("!! misleading \"No record in split 'val' matched a label row\" (with labels=True)")
    print("!! or builds a dataset of length 0 (without). The split is simply empty.")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
    )

    try:
        dataset = ECGDataset("sddb", labels=True, window=WINDOW, **common)
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
    # record_name has no leading zeros, so the config leaves zero_padded_identifiers
    # false and pandas reads it as an integer — hence str() rather than repr() here.
    print(f"  record_id             {sample['record_id']}  ({type(sample['record_id']).__name__})")
    print(f"  lead_names            {labels['lead_names']}   adc_gain {labels['adc_gain']}")
    print(f"  cohort_label          {labels['cohort_label']}  <- the same for all 23 records")
    print(f"  sex / age             {labels['sex']!r} {labels['age']}")
    print(f"  underlying_rhythm     {labels['underlying_rhythm']!r}")
    print(f"  rhythm_class          {labels['rhythm_class']}   has_pacing {labels['has_pacing']}")
    print(f"  history               {labels['history']!r}")
    print(f"  medication            {labels['medication']!r}")
    print(
        f"  duration_secs         {labels['duration_secs']:.0f}"
        f"  ({labels['duration_secs'] / 3600:.2f} h)"
    )
    print(
        f"  vf_onset_secs         {labels['vf_onset_secs']}"
        f"  ({labels['vf_onset_fraction']:.3f} of the way through)"
        if labels["has_vf_onset"]
        else "  vf_onset_secs         nan  <- this record has no terminal-event comment"
    )
    print(f"  has_audited_annotation {labels['has_audited_annotation']}")
    print(f"  ari_n_beats           {int(labels['ari_n_beats'])}  (unaudited, all 23 records)")
    print(f"  atr_n_beats           {int(labels['atr_n_beats'])}  (audited, 12 records only)")

    # --- 1. The NaN is real, and it is in the tensor you just loaded ----------
    print("\nNaN in the loaded window, and in the whole record:")
    n_nan_window = int(torch.isnan(sample["signal"]).sum())
    print(f"  this 10 s window:  {n_nan_window} NaN samples of {sample['signal'].numel()}")
    print("  Across the release: 201,708 invalid samples in 20 of 23 records — at most")
    print("  1.79 s in a run, median 4-84 ms, worst 0.93% of a channel (record 39).")
    print("  They are short scattered dropouts, so a given 10 s window usually has none,")
    print("  but over a whole record you will hit them. Handle before computing a loss:")
    print("    signal = torch.nan_to_num(signal)          # or mask the loss instead")
    print("  ecgbench.labels.sddb.scan_invalid_samples(data_path) gives per-record counts;")
    print("  original/ carries them in quality_issues.")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].astype(str).to_numpy()

    # --- 2. The terminal event, and why one window cannot capture it ----------
    print(f"\ncohort_label over this split: {df['cohort_label'].value_counts().to_dict()}")
    print(f"  One class ({COHORT_LABEL}) for every record: every subject sustained a")
    print("  ventricular tachyarrhythmia. There is no negative class here, so this is a")
    print("  positive cohort to pair with a control database, not a task in itself.")

    onset = df[df["has_vf_onset"]]
    print(f"\nTerminal-event onset, from the #vfon: header comment ({len(onset)} of {len(df)}):")
    print(f"  {'record':>7} {'hours':>7} {'onset h':>8} {'onset %':>8} {'after onset s':>14}")
    for record, row in onset.sort_values("vf_onset_fraction").iterrows():
        print(
            f"  {record:>7} {row['duration_secs'] / 3600:7.2f} {row['vf_onset_secs'] / 3600:8.2f} "
            f"{100 * row['vf_onset_fraction']:8.1f} {row['secs_after_vf_onset']:14.0f}"
        )
    none = df.index[~df["has_vf_onset"]].tolist()
    print(f"  records with NO onset comment: {none or 'none in this split'}")
    print("  (release-wide those are 40, 42 and 49 — the landing page marks them")
    print("  '(paced, no VF)', '(no VF)' and '(paced, no VF)')")
    print("\n  Onset spans 6.1% to 98.9% of the record across the release, so ONE window=")
    print("  cannot capture the event for more than a couple of records. Window per record")
    print("  instead — one ECGDataset per record, with its own onset:")
    if len(onset):
        record = onset.index[len(onset) // 2]
        row = onset.loc[record]
        start = int(row["vf_onset_secs"] * config.default_sampling_rate) - 1250
        per_record = ECGDataset(
            "sddb", window=(start, 2500), fold_numbers=None, labels=True, **common
        )
        idx = list(per_record.metadata_df[config.record_id_column].astype(str)).index(record)
        centred = per_record[idx]
        print(
            f"    record {record}: window=({start}, 2500) is 5 s either side of onset"
            f" -> {tuple(centred['signal'].shape)}"
        )
        print(
            f"    that window holds {int(torch.isnan(centred['signal']).sum())} NaN"
            f" and spans {np.nanmin(centred['signal'].numpy()):.3f}"
            f" to {np.nanmax(centred['signal'].numpy()):.3f} mV"
        )

    # --- 3. Two annotators, different records, different vocabularies --------
    audited = df[df["has_audited_annotation"]]
    print(
        f"\nAnnotator coverage: {len(audited)} of {len(df)} records in this split have the"
        " audited .atr"
    )
    print("  (12 of 23 release-wide — PhysioNet calls the audited set 'incomplete')")
    print("\nBeats by raw symbol in each annotator, with the AAMI EC57 class each reduces to:")
    for prefix, label in (("atr", "audited  "), ("ari", "unaudited")):
        print(f"  {label} (.{prefix}):")
        for symbol, name in BEAT_NAMES.items():
            column = f"{prefix}_beat_{symbol}"
            if column in df.columns and int(df[column].sum()):
                print(
                    f"    beat_{symbol:2s} {int(df[column].sum()):9d}"
                    f"  -> AAMI {AAMI_CLASSES[symbol]}   {name}"
                )
    print("\n  The two vocabularies are DISJOINT where it matters: release-wide .atr carries")
    print("  54,725 B (all of them in record 36, so absent from most splits), 23,123 / and")
    print("  412 f and no r; .ari carries 58,820 r and none of those three. beat_V alone is")
    print("  therefore not")
    print("  ventricular ectopy in either file — use the aami_* columns, which are the")
    print("  only cross-annotator-comparable counts here.")
    print(
        f"    atr: beat_V {int(df['atr_beat_V'].sum())} vs aami_V {int(df['atr_aami_V'].sum())}"
        f"   |   ari: beat_V {int(df['ari_beat_V'].sum())}"
        f" vs aami_V {int(df['ari_aami_V'].sum())}"
    )

    # --- 4. The (AFIB markers are not an AF label ----------------------------
    print("\nThe .ari detector's atrial-fibrillation time against the PUBLISHED rhythm:")
    print(f"  {'record':>7} {'clinical rhythm':>16} {'ari_afib %':>11}  verdict")
    for record, row in df.sort_values("ari_afib_fraction", ascending=False).iterrows():
        pct = 100 * row["ari_afib_fraction"]
        is_af = row["rhythm_class"] == "afib"
        if is_af and pct > 50:
            verdict = "agrees"
        elif is_af:
            verdict = "MISSES a published AF subject"
        elif pct > 20:
            verdict = "FALSE POSITIVE on a non-AF subject"
        else:
            verdict = "-"
        print(f"  {record:>7} {row['rhythm_class']:>16} {pct:11.2f}  {verdict}")
    print("  Wrong in both directions, so DO NOT use ari_afib_* as an AF label. Release-")
    print("  wide it finds 72.9-99.4% AF in three of the four published AF subjects but")
    print("  0.95% in record 37, which is also AF, and 22-36% in six published SINUS")
    print("  records. Use underlying_rhythm / rhythm_class from the clinical table.")

    # --- 5. The audited annotation stops early -------------------------------
    if len(audited):
        print("\nWhere the audited annotation actually stops:")
        print(f"  {'record':>7} {'hours':>7} {'last beat s':>12} {'tail gap s':>11}")
        for record, row in audited.sort_values(
            "atr_unannotated_tail_secs", ascending=False
        ).iterrows():
            last = row["duration_secs"] - row["atr_unannotated_tail_secs"]
            print(
                f"  {record:>7} {row['duration_secs'] / 3600:7.2f} {last:12.1f} "
                f"{row['atr_unannotated_tail_secs']:11.1f}"
            )
        print("  In records 30, 32, 35 and 51 the last audited beat sits at 86,398.6-86,399.4 s")
        print("  — a hard 24-HOUR CUTOFF on recordings that run to 25.1 h. Record 49 loses")
        print("  4,993.7 s and record 51 is also unannotated for its first 1,078.7 s. Nothing")
        print("  errors if you window past the cutoff; check atr_unannotated_tail_secs.")
        print("  The .ari files are the mirror image: they start 29.7-65.2 s in, because")
        print("  every record opens with exactly 50 '?' LEARN annotations —")
        print(f"  ari_n_learning is {sorted(set(df['ari_n_learning'].astype(int)))} here.")

    # --- 6. Quality and ST layers -------------------------------------------
    print(
        f"\nAnnotation layers beyond beats: atr_n_isolated_artifacts"
        f" {int(df['atr_n_isolated_artifacts'].sum())},"
        f" atr_n_quality_changes {int(df['atr_n_quality_changes'].sum())},"
        f" ari_n_st_episodes {int(df['ari_n_st_episodes'].sum())}"
    )
    subtypes = sorted({s for s in df["atr_quality_subtypes"] if s})
    print(f"  '~' quality subtypes present: {subtypes}")
    print("  WFDB defines that subtype as a per-channel bitmask (0/1/2/3 for two")
    print("  channels), but this release uses 0 and 51, strictly alternating. So 51 means")
    print("  'noisy' and 0 'clean', 51 is NOT a valid mask, and ECGBench reports")
    print("  atr_noisy_secs with NO per-channel split rather than guessing.")
    print(f"  atr_noisy_secs over this split: {df['atr_noisy_secs'].sum():.1f} s")
    print("  ari_n_st_episodes counts unaudited ST-segment episodes ('s' markers with")
    print("  aux notes like '(ST0+'), present in 22 of 23 records release-wide.")

    # --- 7. HRV, with the caveat that makes it unusual ----------------------
    print(f"\nWhole-record HRV from the .ari intervals, over RR in {RR_RANGE_SECS} s:")
    print(
        f"  mean_hr_bpm  {df['mean_hr_bpm'].min():.1f} - {df['mean_hr_bpm'].max():.1f}"
        f"  (mean {df['mean_hr_bpm'].mean():.1f})"
    )
    print(f"  sdnn_ms      {df['sdnn_ms'].min():.1f} - {df['sdnn_ms'].max():.1f}")
    print(f"  rmssd_ms     {df['rmssd_ms'].min():.1f} - {df['rmssd_ms'].max():.1f}")
    print(f"  RR intervals rejected by that filter: {int(df['n_rr_rejected'].sum())}")
    print("  These span the moment the subject's heart stopped working. Averaging heart")
    print("  rate across a terminal ventricular arrhythmia is not an HRV measurement of")
    print("  anything — segment around vf_onset_secs if you want physiology.")

    # --- The two channels are unnamed ---------------------------------------
    print(f"\nLead layouts in this split: {df['lead_names'].value_counts().to_dict()}")
    one = ECGDataset("sddb", leads=["ECG2"], window=WINDOW, **common)
    print(f"leads=['ECG2'] -> {one.lead_names}, shape {tuple(one[0]['signal'].shape)}")
    print("  Channel positions, not anatomy. Every current .hea describes BOTH channels")
    print("  as bare 'ECG'; the release states no electrode placement anywhere. Do NOT")
    print("  read ECG1/ECG2 as MLII/V1 by analogy with mitdb. (The superseded .hea- copies")
    print("  say 'record 30, signal 0' instead — the 2008 revision introduced 'ECG'.)")

    # A window that fits the longest record but not the shortest.
    print(f"\nThe shortest record holds {SHORTEST_RECORD_SAMPLES} samples (41, 3.93 h),")
    print("so a window must fit inside that, not inside the longest record's 22627500:")
    far = ECGDataset("sddb", window=(3_500_000, 90_000), **common)
    raised = 0
    for i in range(len(far)):
        try:
            far[i]
        except WindowOutOfRangeError as e:
            raised += 1
            if raised == 1:
                print(f"  window=(3500000, 90000) raises: {e}")
    print(f"  {raised} of {len(far)} records in this split are too short for it")

    # --- Batching ------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nOne batch: signal {tuple(batch['signal'].shape)}  ids {batch['record_id']}")
    mb = 2 * 22_627_500 * 4 / 1e6
    print(f"  Without window= the longest record is 2 x 22627500 float32 (~{mb:.0f} MB), so a")
    print(
        f"  batch of {args.batch_size} would be ~{mb * args.batch_size:.0f} MB,"
        " every byte decoded."
    )

    # There is no class to predict, so the useful targets are per-record continuous
    # quantities. The onset time is the one this database exists for.
    target = torch.tensor(
        [float(x) for x in dataset.labels_df["vf_onset_fraction"]], dtype=torch.float32
    )
    finite = target[~torch.isnan(target)]
    print(
        f"\nvf_onset_fraction target tensor: {tuple(target.shape)}"
        f"  {int(torch.isnan(target).sum())} NaN (records with no onset),"
        f"  finite range {finite.min():.3f}-{finite.max():.3f}"
    )
    print("  (a regression target, and the only one derived from the files rather than")
    print("  from a detector or from the landing page)")


if __name__ == "__main__":
    main()
