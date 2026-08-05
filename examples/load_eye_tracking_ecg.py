#!/usr/bin/env python3
"""
Example: Eye Tracking Dataset for 12-Lead ECG Interpretation — gaze, not signals.

This release ships **no ECG waveforms**. It holds ten *printed* 12-lead ECGs and
aggregate eye-tracking metrics for 63 clinicians reading them — 630 sessions in
all. So there is deliberately no `eye_tracking_ecg` config, no
`ecgbench splits --dataset eye_tracking_ecg` and no `ECGDataset`: there is no
sampled signal to decode, no sampling rate, and no patient behind a record. The
unit of observation is a reader session, so ten folds over "records" would be
partitioning ten pictures.

Five things this script surfaces, all of which bite a naive read of the CSVs:

1. `1`, `2`, `3` are **leads I, II and III** — the AOI grid numbers the limb-lead
   boxes instead of naming them, so the limb leads look absent. `aoi_lead` fixes it.
2. AOI labels are **scoped to their image** (`V1 NSR` vs `V1 AFib`), so grouping
   by `Label` compares nothing across images. Group by `aoi_area` / `aoi_lead`.
3. "Never happened" is encoded as **`-1`, not a blank** — `notna()` reports 100%
   populated and every mean is silently wrong.
4. **`Age` is `0` for 54 of 63 readers**, an anonymisation artefact.
5. **`II-3 VTach` names two different regions**, so `(reader, image, label)` is
   not a unique key. `aoi_occurrence` disambiguates.

Prerequisites:
  - pip install ecgbench
  - A local copy of the dataset (it is not on the HuggingFace Hub — there are no
    fold CSVs to publish).

Usage:
  python examples/load_eye_tracking_ecg.py \\
      --data-path /path/to/eye-tracking-ecg/1.0.0/
"""

import argparse

from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.eye_tracking_ecg import (
    STIMULUS_IMAGES,
    classify_area,
    load_aoi_metrics,
    load_eye_tracking_ecg,
    load_respondents,
    load_sessions,
    stimulus_image_path,
)


def main():
    parser = argparse.ArgumentParser(description="Explore ECG eye-tracking gaze data")
    parser.add_argument("--data-path", required=True, help="Dataset root (holds Datasets/)")
    args = parser.parse_args()

    print("This dataset ships no waveforms, so it has no config and no ECGDataset.")
    print("It is consumed as tables, and any split is the task's to define.\n")

    try:
        readers = load_respondents(args.data_path)
        sessions = load_sessions(args.data_path)
        grid = load_eye_tracking_ecg(args.data_path)
    except LabelSourceMissingError as exc:
        raise SystemExit(f"{exc}\n") from exc

    # ------------------------------------------------------------------ readers
    print("=" * 72)
    print("READERS")
    print("=" * 72)
    print(
        f"{len(readers)} readers, {len(sessions)} sessions "
        f"({len(sessions) // len(readers)} images each)\n"
    )
    print(readers["Group"].value_counts().to_string())
    print(f"\nGender: {readers['Gender'].value_counts().to_dict()}")
    # Age is 0 for most readers in the source; the loader has already NaN'd it, so
    # this count is the honest one rather than a mean age of about four.
    print(
        f"Age recorded for {int(readers['Age'].notna().sum())} of {len(readers)} "
        f"readers -- the rest carry the 0 sentinel, converted to NaN"
    )

    # ----------------------------------------------------------------- stimuli
    print("\n" + "=" * 72)
    print("STIMULI -- images, never decoded signals")
    print("=" * 72)
    for stimulus in sorted(STIMULUS_IMAGES):
        rows = grid[grid.ParentStimulus == stimulus]
        n_leads = rows.loc[rows.aoi_kind == "lead", "aoi_lead"].nunique()
        flag = "  <- 16-lead trace" if n_leads > 12 else ""
        print(
            f"  {stimulus:32} {stimulus_image_path(args.data_path, stimulus).name:36}"
            f" {len(rows) // len(readers):2} AOIs, {n_leads} leads{flag}"
        )

    # Sessions are nominally 30 s but really are not, so prefer the percentage
    # columns over absolute milliseconds when comparing readers.
    short = (sessions.Duration < 29_000).sum()
    print(
        f"\nSession duration: {sessions.Duration.min():,}-{sessions.Duration.max():,} ms; "
        f"{short} of {len(sessions)} ran under 29 s despite the nominal 30 s."
    )

    # -------------------------------------------------------------- AOI naming
    print("\n" + "=" * 72)
    print("AOI LABELS -- scoped per image, and numbered for the limb leads")
    print("=" * 72)
    print(
        f"{grid.Label.nunique()} distinct labels for ~25 real regions, "
        "because each label carries its image's suffix:"
    )
    for label in sorted(grid.loc[grid.aoi_lead == "V1", "Label"].unique())[:4]:
        print(f"  {label}")
    print("\nSo group by aoi_area / aoi_lead, never by Label. Decoding:")
    for area in ("1", "2", "3", "V5-3", "V1 short", "Information"):
        kind, lead = classify_area(area)
        print(f"  {area:14} -> kind={kind:13} lead={lead}")

    reused = grid.loc[grid.aoi_occurrence > 0, "Label"].unique()
    print(f"\n{len(reused)} label(s) name two different regions: {list(reused)}.")
    print("  (reader, image, label) is therefore NOT unique -- use aoi_occurrence.")

    # ------------------------------------------------------------- gaze by lead
    print("\n" + "=" * 72)
    print("WHERE READERS LOOK")
    print("=" * 72)
    leads = grid[grid.aoi_kind == "lead"]
    print("Mean dwell time per lead box, % of session (top 6):")
    print(
        leads.groupby("aoi_lead")["Time_spent_G_Percentage"].mean().nlargest(6).round(2).to_string()
    )

    print("\nMean ms to first gaze on a lead box, by expertise:")
    print(leads.groupby("Group")["Hit_time_G"].mean().sort_values().round(0).to_string())
    print(
        "  Hit_time_G was -1 (never gazed) on "
        f"{int(grid.Hit_time_G.isna().sum()):,} of {len(grid):,} AOI rows; "
        "as NaN those are excluded rather than counted as -1 ms."
    )

    # TTFF_F is the one inconsistency the loader does not touch: it holds a
    # plausible value even where no fixation happened, so mask it yourself.
    no_fixation = (grid.Fixations_Count == 0).sum()
    print(
        f"\n{no_fixation:,} rows have Fixations_Count == 0 yet still carry a "
        "TTFF_F value; mask on Fixations_Count > 0 before using it."
    )

    # ----------------------------------------------------- coarse long/short view
    print("\n" + "=" * 72)
    print("THE SAME SESSIONS, SCORED LONG vs SHORT")
    print("=" * 72)
    long_short = load_aoi_metrics(args.data_path, table="long_short")
    print(
        long_short.groupby("aoi_area")["Time_spent_G_Percentage"]
        .agg(["mean", "std"])
        .round(1)
        .to_string()
    )
    print("\n  Long  = the rhythm strips along the bottom")
    print("  Short = the twelve short lead traces")

    print(
        "\nBuilding a task? Split by reader (generalise to a new clinician) or by "
        "image (generalise to a new abnormality).\nThose are different "
        "experiments, which is why ECGBench ships neither as the default."
    )


if __name__ == "__main__":
    main()
