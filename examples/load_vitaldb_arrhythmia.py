#!/usr/bin/env python3
"""
Example: VitalDB Arrhythmia Database — annotations, with the waveforms elsewhere.

This release ships **no ECG waveforms**. It is beat and rhythm labels for
intraoperative Lead II recordings that live in the public VitalDB project, keyed
by VitalDB's own `case_id`. So there is deliberately no `vitaldb_arrhythmia`
config, no `ecgbench splits --dataset vitaldb_arrhythmia` and no `ECGDataset`:
nothing here is a signal file, so there is nothing to validate and nothing to
partition. Samples come from `vitaldb.load_case(...)` over the network.

Six things this script surfaces, all of which bite a naive read of the CSVs:

1. **Split on `subjectid`, not `case_id`.** 482 cases come from 473 patients —
   eight gave two cases and one gave three, so case-level folds leak.
2. **`total_beats` counts rows, not beats.** 676,250 rows, but only 658,874
   classify a heartbeat. Use `is_beat`.
3. **Three cases ship different columns**, and case 2453 writes its segment
   boundaries into `beat_type` as `Start`/`End`. `load_annotations` normalises
   all three; a raw `value_counts` reports two fake beat classes.
4. **`beat_type` has an undocumented fifth value `P`** (7 beats, 4 cases).
5. **`time_second` is an offset into the whole surgery**, starting as late as
   33,628 s, and the annotated window is ~20 min of it. Slicing from 0 gets you
   signal with no labels.
6. **`(case_id, time_second)` is not unique** — 111 cases repeat a timestamp.

Prerequisites:
  - pip install ecgbench
  - A local copy of the annotations (they are not on the HuggingFace Hub — there
    are no fold CSVs to publish).
  - Optional, for the waveform section only: pip install vitaldb

Usage:
  python examples/load_vitaldb_arrhythmia.py \\
      --data-path /path/to/vitaldb-arrhythmia/1.0.0/
"""

import argparse

from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.vitaldb_arrhythmia import (
    BEAT_TYPES,
    RHYTHM_LABELS,
    SAMPLING_RATE_HZ,
    WAVEFORM_TRACK,
    bad_signal_intervals,
    case_window,
    load_annotations,
    load_cases,
    load_vitaldb_arrhythmia,
    rhythm_segments,
)

#: A case with ventricular tachycardia, an undocumented `P` beat and several
#: bad-signal intervals — enough quirks to be worth printing in full.
DEMO_CASE = 1018

#: The one case whose annotation file uses a different schema.
ODD_CASE = 2453


def main():
    parser = argparse.ArgumentParser(description="Explore VitalDB arrhythmia annotations")
    parser.add_argument(
        "--data-path", required=True, help="Dataset root (holds metadata.csv, Annotation_Files/)"
    )
    parser.add_argument(
        "--fetch-waveform",
        action="store_true",
        help="Download one case's ECG from VitalDB (needs `pip install vitaldb` + network)",
    )
    args = parser.parse_args()

    print("This dataset ships no waveforms, so it has no config and no ECGDataset.")
    print(f"Samples come from VitalDB: {WAVEFORM_TRACK} at {SAMPLING_RATE_HZ} Hz.\n")

    try:
        cases = load_cases(args.data_path)
    except LabelSourceMissingError as exc:
        raise SystemExit(f"{exc}\n") from exc

    # ------------------------------------------------------------------- cases
    print("=" * 72)
    print("CASES")
    print("=" * 72)
    print(f"{len(cases)} cases from {cases['subjectid'].nunique()} patients")
    repeats = cases["subjectid"].value_counts()
    repeats = repeats[repeats > 1]
    print(
        f"  {len(repeats)} patients contributed more than one case "
        f"({repeats.to_dict()}) -> GROUP ON subjectid, NOT case_id\n"
    )

    span = cases["analysis_end_time_sec"] - cases["analysis_start_time_sec"]
    print(
        f"annotated window per case: mean {span.mean():.0f} s, median {span.median():.0f} s, "
        f"range {span.min():.0f}-{span.max():.0f} s"
    )
    print(f"  total annotated: {span.sum():,.0f} s")
    print(
        f"  window starts {cases['analysis_start_time_sec'].min():.0f}-"
        f"{cases['analysis_start_time_sec'].max():.0f} s into the surgery — "
        "time_second is an offset into the whole recording\n"
    )

    print(
        f"age: mean {cases['age_years'].mean():.1f} y, "
        f"{int(cases['age_censored'].sum())} published as '>89' (de-identification)"
    )
    print(f"sex: {cases['sex'].value_counts().to_dict()}")
    print(f"department: {cases['department'].value_counts().to_dict()}\n")

    # ----------------------------------------------------------------- rhythms
    print("=" * 72)
    print("RHYTHM AND BEAT LABELS (recomputed from the annotation files)")
    print("=" * 72)
    print("Reading all 482 annotation files, a few seconds...\n")
    summary = load_vitaldb_arrhythmia(args.data_path)

    print(f"{'rhythm':<40} {'cases':>6}")
    counts = {}
    for label in RHYTHM_LABELS:
        n = int(cases["rhythm_class_list"].map(lambda labels, k=label: k in labels).sum())
        if n:
            counts[label] = n
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {RHYTHM_LABELS[label]:<38} {n:>6}")

    print(
        f"\nannotation rows: {int(summary['n_rows'].sum()):,}  "
        f"(== metadata total_beats, which counts ROWS)"
    )
    print(
        f"classified beats: {int(summary['n_beats'].sum()):,}  "
        f"({int(summary['n_rows'].sum() - summary['n_beats'].sum()):,} rows annotate no beat)"
    )
    for letter, name in BEAT_TYPES.items():
        column = f"beats_{letter}"
        if column in summary.columns:
            print(f"  {letter} {name:<26} {int(summary[column].sum()):>9,}")
    print("  ^ P is undocumented: the paper names four beat classes, not five.\n")

    # -------------------------------------------------------------- one case
    print("=" * 72)
    print(f"ONE CASE IN DETAIL — case {DEMO_CASE}")
    print("=" * 72)
    start, end = case_window(args.data_path, DEMO_CASE)
    print(f"annotated window: {start:.1f}-{end:.1f} s ({end - start:.1f} s of surgery)")

    beats = load_annotations(args.data_path, DEMO_CASE)
    print(f"{len(beats)} rows, {int(beats['is_beat'].sum())} of them beats")
    print(f"beat types: {beats['beat_type'].value_counts(dropna=False).to_dict()}")
    print(f"rhythms:    {beats['rhythm_label'].value_counts(dropna=False).to_dict()}\n")

    segments = rhythm_segments(args.data_path, DEMO_CASE)
    arrhythmic = segments[segments["rhythm_label"].notna() & segments["rhythm_label"].ne("N")]
    print(f"{len(segments)} rhythm segments, {len(arrhythmic)} of them not sinus:")
    print(
        arrhythmic.head(6)[
            ["rhythm_label", "start_second", "end_second", "duration_second", "n_beats"]
        ].to_string(index=False)
    )
    print(
        "\nsegment durations partition the window "
        f"({segments['duration_second'].sum():.0f} s of {end - start:.0f} s); the README's "
        "per-rhythm seconds overlap and sum to more than the data spans."
    )

    intervals = bad_signal_intervals(args.data_path, DEMO_CASE)
    print(
        f"\n{len(intervals)} bad-signal intervals, "
        f"{int((~intervals['closed']).sum())} of them unterminated in the file"
    )

    # ------------------------------------------------------- the odd schema
    print("\n" + "=" * 72)
    print(f"THE ODD SCHEMA — case {ODD_CASE}")
    print("=" * 72)
    odd = load_annotations(args.data_path, ODD_CASE)
    print("Shipped with a `caseid` column, no `bad_signal_quality_label`, and its")
    print("boundary markers written into `beat_type` as 'Start'/'End'. After")
    print("load_annotations() it is indistinguishable from any other case:")
    print(f"  columns: {list(odd.columns)}")
    print(f"  Start/End left in beat_type: {int(odd['beat_type'].isin(['Start', 'End']).sum())}")
    print(f"  recovered markers: {odd['bad_signal_quality_label'].dropna().tolist()[:6]} ...")

    # ---------------------------------------------------------- the waveform
    print("\n" + "=" * 72)
    print("GETTING THE WAVEFORM")
    print("=" * 72)
    print("The signal is not in this package. Fetch it from VitalDB by case_id:\n")
    print("    import vitaldb")
    print(f"    vals = vitaldb.load_case({DEMO_CASE}, ['{WAVEFORM_TRACK}'], 1/{SAMPLING_RATE_HZ})")
    print(f"    ecg = vals['{WAVEFORM_TRACK}']")
    print(f"    # annotations cover {start:.0f}-{end:.0f} s, so index the labelled stretch:")
    print(
        f"    window = ecg[int({start:.0f} * {SAMPLING_RATE_HZ}):"
        f"int({end:.0f} * {SAMPLING_RATE_HZ})]"
    )
    print("    # and an R-peak at t seconds sits at sample int(t * 500) of `ecg`, not of `window`")

    if args.fetch_waveform:
        try:
            import vitaldb
        except ImportError:
            print("\n`pip install vitaldb` to run this section.")
            return
        print(f"\nDownloading case {DEMO_CASE} from VitalDB...")
        vals = vitaldb.load_case(DEMO_CASE, [WAVEFORM_TRACK], 1 / SAMPLING_RATE_HZ)
        ecg = vals[WAVEFORM_TRACK]
        print(
            f"  got {ecg.shape[0]:,} samples "
            f"({ecg.shape[0] / SAMPLING_RATE_HZ / 60:.1f} min at {SAMPLING_RATE_HZ} Hz)"
        )
        first = beats[beats["is_beat"]].iloc[0]
        index = int(first["time_second"] * SAMPLING_RATE_HZ)
        print(
            f"  first annotated beat: {first['beat_type']} at "
            f"{first['time_second']:.3f} s -> sample {index:,}"
        )


if __name__ == "__main__":
    main()
