#!/usr/bin/env python3
"""
Example: ECGDMMLD — a drug-exposure dataset where the label lies about the drug.

4,211 ten-second 12-lead ECGs at 1 kHz from 22 healthy volunteers in a randomised,
double-blind, 5-period crossover Phase I trial (NCT02308748) built to test whether
blocking **late sodium current** shortens the QT prolongation caused by a pure hERG
blocker. Every subject passed through all five arms: dofetilide alone (the hERG
positive control), dofetilide with mexiletine, dofetilide with lidocaine,
moxifloxacin with diltiazem, and placebo.

Six things this script goes out of its way to show, because each is different here
from the rest of ECGBench — and the first is different from its own sibling
`ecgcipa`:

1. **`treatment` names the period's regimen, not the drug in the blood.** The
   agents were staged hours apart within each period, so a record labelled
   "Mexiletine + Dofetilide" at the 2-hour timepoint contains mexiletine and no
   dofetilide at all. This is the single most important thing to know before
   training on this dataset, and section 6 makes it visible.
2. **There is no class to predict.** Nobody in this cohort has cardiac disease.
   `treatment` is the stratification label and the closest thing to a class.
3. **The samples really are millivolts.** `signal_unit_scale: 1.0`, unlike
   `ecgcipa`'s 0.001 — the headers here declare per-lead gains against `/mV`.
4. **Records come in near-duplicate triplicates.** Three segments per subject per
   timepoint, seconds apart at the same concentration. 4,211 records are closer to
   1,404 independent observations — and closer still to 22 for anything
   subject-level.
5. **The study's endpoint IS computable here**, which is the happy difference from
   `ecgcipa`. `is_baseline` flags each period's pre-dose triplicate, so change from
   baseline is a per-record quantity — `load_baseline_deltas()` returns it.
6. **Every record ships twice.** `signal_path` is the raw 10 s ECG;
   `median_beat_path` is a derived 16-channel median beat whose `.atr` fiducials are
   what the published intervals were measured from. Three of the 4,211 median
   headers are corrupt upstream; `median_beat_readable` says which.

Usage:
  python examples/load_ecgdmmld.py --data-path /path/to/ecgdmmld/1.0.0/
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config

#: Interval columns and their clinical names, in the order a clinician reads them.
#: HR and QTcF are derived by the label loader — the release ships neither.
INTERVALS = [
    ("hr_bpm", "HR"), ("rr_ms", "RR"), ("pr_ms", "PR"), ("qrs_ms", "QRS"),
    ("qt_ms", "QT"), ("qtcf_ms", "QTcF"), ("jtpeak_ms", "J-Tpeak"),
    ("tpeak_tend_ms", "Tpeak-Tend"),
]

#: Plasma concentration columns per treatment arm, in the order they come on board.
#: Dofetilide is pg/mL; the others are ng/mL — the unit is in the column name
#: precisely because pooling them numerically is a 1000x error.
ANALYTES_OF_ARM = {
    "Dofetilide": ["plasma_dofetilide_pg_ml"],
    "Mexiletine + Dofetilide": [
        "plasma_mexiletine_ng_ml", "plasma_dofetilide_pg_ml",
    ],
    "Lidocaine + Dofetilide": [
        "plasma_lidocaine_ng_ml", "plasma_dofetilide_pg_ml",
    ],
    "Moxifloxacin + Diltiazem": [
        "plasma_moxifloxacin_ng_ml", "plasma_diltiazem_ng_ml",
    ],
    "Placebo": [],
}


def main():
    parser = argparse.ArgumentParser(description="Load ECGDMMLD via ECGBench")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("ecgdmmld")
    print(f"Dataset: {config.name} v{config.version}")
    print(f"Split:   {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:   {config.leads} — {', '.join(config.lead_names)}")
    print("         ^ uppercase 'A', like ptbxl. The sibling ecgcipa writes the")
    print("           same three leads aVR/aVL/aVF, so do not hardcode either.")
    print(f"Samples: {int(config.duration_seconds * config.default_sampling_rate)} "
          "per record, uniform across all 4,211")
    print("Rate:    acquired at 500 Hz, up-sampled to 1 kHz by the depositors —")
    print("         the extra samples are interpolated, not measured.")
    print()

    dataset = ECGDataset(
        "ecgdmmld", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source, labels=True,
    )
    print(f"Records: {len(dataset)}  (of 4,211, from 22 healthy volunteers)")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nLabel fields ({len(labels)}): {list(labels)}")
    print("\nFirst record:")
    print(f"  ID:          {sample['record_id']}")
    print(f"  signal:      {tuple(sample['signal'].shape)}  (leads, samples)")
    print(f"  subject:     {labels['patient_id']}  "
          f"({labels['age_years']}y {labels['sex']}, {labels['race']})")
    print(f"  treatment:   {labels['treatment']!r}  "
          f"(sequence {labels['treatment_sequence']!r}, period {labels['period']})")
    print(f"  timepoint:   {labels['timepoint_hours']:+g} h from the period's first "
          f"dose (baseline={labels['is_baseline']})")
    print("  intervals:   " + "  ".join(
        f"{shown}={labels[name]:.0f}"
        for name, shown in INTERVALS if labels[name] == labels[name]  # skip NaN
    ))
    print("               HR in bpm, the rest in ms. HR and QTcF are DERIVED —")
    print("               the release ships neither.")
    print(f"  median beat: {labels['median_beat_path']}  (+ .atr fiducials)")

    # --- 1. The samples are already millivolts --------------------------------
    micro = ECGDataset(
        "ecgdmmld", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, units="uV",
    )
    lead_ii_mv = sample["signal"][1]
    lead_ii_uv = micro[0]["signal"][1]
    print("\nUnits — every channel declares its own gain against '/mV', so wfdb")
    print("already returns MILLIVOLTS and signal_unit_scale is 1.0 (ecgcipa needs")
    print("0.001; copying that value here would divide every sample by 1000):")
    print(f"  units='mV' (default): lead II min {lead_ii_mv.min():+9.3f} "
          f"max {lead_ii_mv.max():+9.3f}")
    print(f"  units='uV':           lead II min {lead_ii_uv.min():+9.1f} "
          f"max {lead_ii_uv.max():+9.1f}")

    # --- 2. Windowing: 10,000 samples per record is a lot to decode ------------
    # Length is uniform (all 4,211 records are exactly 10,000 samples), so unlike
    # cpsc_2018 or ptbdb any window inside [0, 10000) fits every record. window= is
    # pushed into wfdb's sampfrom/sampto, so it decodes only what it returns — and
    # unlike a cropping lambda it survives DataLoader(num_workers>0) under spawn.
    windowed = ECGDataset(
        "ecgdmmld", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, window=(2000, 5000),
    )
    print(f"\nwindow=(2000, 5000) -> {tuple(windowed[0]['signal'].shape)}  "
          "(5 s starting at 2 s)")
    print("  Every record is exactly 10,000 samples, so any window inside")
    print("  [0, 10000) fits all of them — no WindowOutOfRangeError to plan around.")

    # --- 3. Selecting leads by name -------------------------------------------
    limb = ECGDataset(
        "ecgdmmld", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, leads=["I", "II", "V5"],
    )
    print(f"\nleads=['I', 'II', 'V5'] -> {tuple(limb[0]['signal'].shape)}")
    print("  Matching is case-insensitive, so 'avr' and 'AVR' both resolve.")

    # --- 4. Triplicates, and why 4,211 overstates the sample size -------------
    # labels_df is aligned POSITIONALLY with metadata_df and carries a RangeIndex,
    # not record IDs — attach them explicitly before reporting anything per record.
    frame = dataset.labels_df.copy()
    frame["record_id"] = dataset.metadata_df[config.record_id_column].to_numpy()

    groups = frame.groupby(["patient_id", "period", "timepoint_hours"]).size()
    print(f"\nThis split: {len(frame)} records from {frame['patient_id'].nunique()} "
          f"subjects over {len(groups)} (subject, period, timepoint) groups")
    print(f"  {(groups == 3).sum()} groups hold 3 near-duplicate records each — same")
    print("  person, same posture, same concentration, seconds apart. Folds are")
    print("  grouped on patient_id, so all three always land on the same side.")

    unreadable = int((~frame["median_beat_readable"]).sum())
    print(f"  {unreadable} record(s) here have a corrupt median-beat header (3 in the")
    print("  full release): one channel's .dat filename has digits from the gain")
    print("  field spliced in, so wfdb.rdrecord raises. The raw/ ECGs are fine.")

    # --- 5. Treatment: the stratification label -------------------------------
    print("\nTreatment, counted both ways. This is a COMPLETE crossover — every")
    print("subject passed through every arm — so the subject counts are all but")
    print("equal and no split of this dataset can separate the arms:")
    by_record = frame["treatment"].value_counts()
    by_subject = frame.groupby("treatment")["patient_id"].nunique()
    print(f"  {'treatment':26s} {'records':>8s} {'subjects':>9s}")
    for arm in by_record.index:
        print(f"  {arm:26s} {by_record[arm]:8d} {by_subject.get(arm, 0):9d}")

    # --- 6. THE TRAP: the label names the period, not the drug on board -------
    print("\nWhy `treatment` must not be used as an exposure label. Within each")
    print("period the drugs were staged hours apart, so the arm's second agent is")
    print("simply absent early on. Fraction of each arm's records with a measured")
    print("concentration of each analyte:")
    for arm in sorted(by_record.index):
        analytes = ANALYTES_OF_ARM.get(arm, [])
        arm_rows = frame[frame["treatment"] == arm]
        if not analytes:
            print(f"  {arm:26s} no analyte — placebo sets the diurnal drift")
            continue
        shown = "  ".join(
            f"{a.removeprefix('plasma_').removesuffix('_ng_ml').removesuffix('_pg_ml')}"
            f"={arm_rows[a].notna().mean():.0%}"
            for a in analytes
        )
        print(f"  {arm:26s} {shown}")
    print("  A 'Mexiletine + Dofetilide' record at TPT 1.5-3 h has mexiletine and")
    print("  NO dofetilide. Train on the plasma columns, or on treatment crossed")
    print("  with timepoint_hours — never on treatment alone.")

    # --- 7. The endpoint, which here IS attachable to a waveform --------------
    print("\nChange from baseline — the study's actual endpoint. Unlike ecgcipa,")
    print("where it exists only on unjoinable triplicate-average rows, here every")
    print("period has its own pre-dose triplicate and the delta is per record:")
    try:
        from ecgbench.labels.ecgdmmld import load_baseline_deltas

        deltas = load_baseline_deltas(dataset.data_path, config)
        deltas = deltas.loc[deltas.index.intersection(frame["record_id"])]
        post = deltas[~deltas["is_baseline"]]
        print(f"  {'arm':26s} {'delta QTcF':>11s} {'delta J-Tpeak':>14s}")
        for arm in sorted(post["treatment"].unique()):
            rows = post[post["treatment"] == arm]
            print(f"  {arm:26s} {rows['delta_qtcf_ms'].mean():+11.1f} "
                  f"{rows['delta_jtpeak_ms'].mean():+14.1f}")
        print("  ms, averaged over all post-dose timepoints. Read every arm against")
        print("  placebo. The hypothesis under test: dofetilide alone (pure hERG")
        print("  block) prolongs QTcF and J-Tpeak, while adding mexiletine or")
        print("  lidocaine (late-sodium block) shortens J-Tpeak back toward")
        print("  baseline. Placebo-correction needs a cross-subject aggregation and")
        print("  is deliberately left to you.")
    except FileNotFoundError:
        print("  (needs a local copy of the release; pass --data-path)")

    missing = int(frame["qrs_ms"].isna().sum())
    print(f"\n{missing} record(s) here have no QRS duration and no J-Tpeak: no QRS")
    print("offset could be annotated (9 in the full release). Both are NaN, not 0.")
    print("tpeak_tpeakp_ms is NA in ALL 4,211 rows — the column is documented but")
    print("the release never populated it, and no .atr marks a secondary T peak.")

    # --- 8. A batch, and a target tensor --------------------------------------
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=ecg_collate_fn)
    batch = next(iter(loader))
    print(f"\nFirst batch: signal {tuple(batch['signal'].shape)} "
          f"{batch['signal'].dtype}, {len(batch['labels'])} label dicts")

    arms = sorted(frame["treatment"].unique())
    targets = torch.tensor([arms.index(row["treatment"]) for row in batch["labels"]])
    print(f"Targets:     {targets.tolist()}")
    print(f"  over {arms}")
    print("  A regression target reads the same way — e.g. QTcF in ms:")
    qtcf_targets = torch.tensor(
        [float(row["qtcf_ms"]) for row in batch["labels"]], dtype=torch.float32
    )
    print(f"  qtcf_ms:   {[round(v, 1) for v in qtcf_targets.tolist()]}")
    print("\n  treatment is the fold-balancing label, not a waveform class: a")
    print("  pre-dose record in the dofetilide arm is a drug-free ECG that this")
    print("  label calls 'Dofetilide'. Filter on is_baseline, on timepoint_hours,")
    print("  or on the plasma concentration before treating it as exposure.")


if __name__ == "__main__":
    main()
