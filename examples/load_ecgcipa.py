#!/usr/bin/env python3
"""
Example: CiPA ECG Validation Study — a drug-exposure dataset, not a diagnosis one.

5,749 ten-second 12-lead ECGs at 1 kHz from 60 healthy volunteers in a Phase I
trial (NCT03070470) built to test whether QTc *and* J-Tpeakc together can separate
balanced ion-channel block from predominant hERG block. Fifty subjects were
randomised in parallel to ranolazine, verapamil, lopinavir+ritonavir, chloroquine
or placebo; ten more crossed over between dofetilide and diltiazem+dofetilide.

Five things this script goes out of its way to show, because every one of them is
different here from the rest of ECGBench:

1. **There is no class to predict.** Nobody in this cohort has cardiac disease. The
   labels are which drug, how long after dosing, at what plasma concentration, and
   what the intervals did. `treatment` is the stratification label and the closest
   thing to a class.
2. **The samples are microvolts.** `signal_unit_scale: 0.001` converts them, so
   `units="mV"` (the default) and `units="uV"` differ by 1000x on the same record.
3. **Records come in near-duplicate triplicates.** Three segments per subject per
   timepoint, seconds apart at the same concentration. 5,749 records are closer to
   1,917 independent observations — and closer still to 60 for anything
   subject-level.
4. **The study's own endpoints cannot be joined to a waveform.** Change from
   baseline and placebo-corrected change exist only on `adeg.csv`'s
   triplicate-average rows, which carry no record ID. This script shows both what
   you can attach and what you cannot.
5. **Every record ships twice.** `signal_path` is the raw 10 s ECG;
   `median_beat_path` is a derived 16-channel median beat whose `.atr` fiducials are
   what the published intervals were measured from.

Usage:
  python examples/load_ecgcipa.py --data-path /path/to/ecgcipa/1.0.0/
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config

#: Interval columns and their clinical names, in the order a clinician reads them.
INTERVALS = [
    ("hr_bpm", "HR"), ("rr_ms", "RR"), ("pr_ms", "PR"), ("qrs_ms", "QRS"),
    ("qt_ms", "QT"), ("qtcf_ms", "QTcF"), ("jtpeak_ms", "J-Tpeak"),
    ("jtpeakc_ms", "J-Tpeakc"), ("tpeak_tend_ms", "Tpeak-Tend"),
]

#: Plasma concentration column per treatment arm, for the exposure-response demo.
#: Dofetilide is pg/mL; the others are ng/mL — the unit is in the column name
#: precisely because pooling them numerically is a 1000x error.
ANALYTE_OF_ARM = {
    "Ranolazine": "plasma_ranolazine_ng_ml",
    "Verapamil": "plasma_verapamil_ng_ml",
    "Chloroquine": "plasma_chloroquine_ng_ml",
    "Lopinavir+Ritonavir": "plasma_lopinavir_ng_ml",
    "Dofetilide": "plasma_dofetilide_pg_ml",
    "Diltiazem+Dofetilide": "plasma_dofetilide_pg_ml",
}


def main():
    parser = argparse.ArgumentParser(
        description="Load the CiPA ECG Validation Study via ECGBench"
    )
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("ecgcipa")
    print(f"Dataset: {config.name} v{config.version}")
    print(f"Split:   {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:   {config.leads} — {', '.join(config.lead_names)}")
    print("         ^ lowercase 'a'; the derived medians/ headers of the SAME")
    print("           records spell them AVR/AVL/AVF and add VCGMAG/X/Y/Z.")
    print(f"Samples: {int(config.duration_seconds * config.default_sampling_rate)} "
          "per record, uniform across all 5,749")
    print()

    dataset = ECGDataset(
        "ecgcipa", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source, labels=True,
    )
    print(f"Records: {len(dataset)}  (of 5,749, from 60 healthy volunteers)")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nLabel fields ({len(labels)}): {list(labels)}")
    print("\nFirst record:")
    print(f"  ID:          {sample['record_id']}")
    print(f"  signal:      {tuple(sample['signal'].shape)}  (leads, samples)")
    print(f"  subject:     {labels['patient_id']}  "
          f"({labels['age_years']}y {labels['sex']}, {labels['race']})")
    print(f"  treatment:   {labels['treatment']!r}  (arm {labels['planned_arm']!r})")
    print(f"  timepoint:   {labels['timepoint']} into period {labels['period']}, "
          f"replicate {labels['replicate_number']} of 3")
    print("  intervals:   " + "  ".join(
        f"{shown}={labels[name]:.0f}" for name, shown in INTERVALS
    ))
    print("               HR in bpm, the rest in ms")
    print(f"  median beat: {labels['median_beat_path']}  (+ .atr fiducials)")

    # --- 1. The samples are microvolts, and signal_unit_scale converts them ----
    micro = ECGDataset(
        "ecgcipa", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, units="uV",
    )
    lead_ii_mv = sample["signal"][1]
    lead_ii_uv = micro[0]["signal"][1]
    print("\nUnits — the headers declare gain 0.26595744680851063(0)/uV, so the")
    print("stored samples are MICROVOLTS and signal_unit_scale is 0.001:")
    print(f"  units='mV' (default): lead II min {lead_ii_mv.min():+9.3f} "
          f"max {lead_ii_mv.max():+9.3f}")
    print(f"  units='uV':           lead II min {lead_ii_uv.min():+9.1f} "
          f"max {lead_ii_uv.max():+9.1f}")
    print("  Leave it at 'mV' unless you specifically want the source scale.")

    # --- 2. Windowing: 10,000 samples per record is a lot to decode ------------
    # Length is uniform (all 5,749 records are exactly 10,000 samples), so unlike
    # cpsc_2018 or ptbdb any window inside [0, 10000) fits every record. window= is
    # pushed into wfdb's sampfrom/sampto, so it decodes only what it returns — and
    # unlike a cropping lambda it survives DataLoader(num_workers>0) under spawn.
    windowed = ECGDataset(
        "ecgcipa", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, window=(2000, 5000),
    )
    print(f"\nwindow=(2000, 5000) -> {tuple(windowed[0]['signal'].shape)}  "
          "(5 s starting at 2 s)")
    print("  Every record is exactly 10,000 samples, so any window inside")
    print("  [0, 10000) fits all of them — no WindowOutOfRangeError to plan around.")

    # --- 3. Selecting leads by name -------------------------------------------
    limb = ECGDataset(
        "ecgcipa", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, leads=["I", "II", "V5"],
    )
    print(f"\nleads=['I', 'II', 'V5'] -> {tuple(limb[0]['signal'].shape)}")
    print("  Matching is case-insensitive, so 'avr' and 'aVR' both resolve.")

    # --- 4. Triplicates, and why 5,749 overstates the sample size -------------
    # labels_df is aligned POSITIONALLY with metadata_df and carries a RangeIndex,
    # not record IDs — attach them explicitly before reporting anything per record.
    frame = dataset.labels_df.copy()
    frame["record_id"] = dataset.metadata_df[config.record_id_column].to_numpy()

    groups = frame.groupby(["patient_id", "period", "timepoint_n"]).size()
    print(f"\nThis split: {len(frame)} records from {frame['patient_id'].nunique()} "
          f"subjects over {len(groups)} (subject, period, timepoint) groups")
    print(f"  {(groups == 3).sum()} groups hold 3 near-duplicate records each — same")
    print("  person, same posture, same concentration, seconds apart. Folds are")
    print("  grouped on patient_id, so all three always land on the same side.")

    # --- 5. Treatment: the stratification label -------------------------------
    print("\nTreatment, counted both ways — records are weighted by how many")
    print("timepoints each subject completed:")
    by_record = frame["treatment"].value_counts()
    by_subject = frame.groupby("treatment")["patient_id"].nunique()
    print(f"  {'treatment':24s} {'records':>8s} {'subjects':>9s}")
    for arm in by_record.index:
        print(f"  {arm:24s} {by_record[arm]:8d} {by_subject.get(arm, 0):9d}")

    # --- 6. What the dataset is actually for: exposure vs repolarisation ------
    print("\nQTcF against plasma concentration, per arm (the study's question).")
    print("Baseline is each arm's pre-dose records; Cmax rows are its top decile")
    print("of concentration:")
    print(f"  {'arm':24s} {'pre-dose':>9s} {'top decile':>11s} {'delta':>7s}  "
          f"{'concentration':>14s}")
    for arm in sorted(set(by_record.index)):
        arm_rows = frame[frame["treatment"] == arm]
        pre = arm_rows[arm_rows["nominal_hours_from_period_start"] <= 0]
        analyte = ANALYTE_OF_ARM.get(arm)
        if analyte is None:
            # Placebo has no analyte to measure, so its comparison group is every
            # post-dose record. It is the drift the drug arms must be read against.
            peak = arm_rows[arm_rows["nominal_hours_from_period_start"] > 0]
            shown = "     n/a      "
        else:
            exposed = arm_rows[arm_rows[analyte].notna()]
            if exposed.empty:
                continue
            peak = exposed[exposed[analyte] >= exposed[analyte].quantile(0.9)]
            unit = "pg/mL" if analyte.endswith("_pg_ml") else "ng/mL"
            shown = f"{peak[analyte].mean():9.0f} {unit}"
        if pre.empty or peak.empty:
            continue
        print(f"  {arm:24s} {pre['qtcf_ms'].mean():9.1f} "
              f"{peak['qtcf_ms'].mean():11.1f} "
              f"{peak['qtcf_ms'].mean() - pre['qtcf_ms'].mean():+7.1f}  {shown}")
    print("  Read every arm against placebo, which sets the diurnal drift. The")
    print("  hypothesis under test: a predominant-hERG blocker (dofetilide)")
    print("  prolongs QTcF, while a balanced blocker (verapamil, ranolazine) does")
    print("  not despite comparable hERG affinity.")

    # Filter on the boolean, not on plasma_below_lloq != "": the list column is
    # empty for uncensored records, and pandas reads that empty string back from a
    # CSV as NaN, so the string comparison matches everything on a re-read frame.
    below = int(frame["plasma_any_below_lloq"].sum())
    print(f"\n{below} record(s) in this split have an analyte reported as 0 because")
    print("it was below the limit of quantification — that is censoring, not")
    print("absence. plasma_below_lloq names which analyte, so the two are")
    print("distinguishable; nothing else in adpc.csv is 0.")

    missing = frame["qt_ms"].isna().sum()
    print(f"{missing} record(s) have no QT: no T annotation could be placed. "
          f"{frame['pr_ms'].isna().sum()} have no PR (no P onset). Both are NaN,")
    print("not 0 — 19 of the 5,749 records are affected in the full release.")

    # --- 7. The endpoint you cannot attach to a waveform ----------------------
    print("\nWhat is NOT here: change from baseline. adeg.csv carries it only on")
    print("its 17,870 DTYPE=AVERAGE rows, which have a blank EGREFID — they are")
    print("triplicate means, one level of aggregation above the signals:")
    try:
        from ecgbench.labels.ecgcipa import load_triplicate_averages

        averages = load_triplicate_averages(dataset.data_path, config)
        qtcf = averages[(averages["parameter"] == "QTCF") & averages["CHG"].notna()]
        print(f"  {len(averages)} average rows, {len(qtcf)} with a QTcF change")
        print(f"  largest QTcF increase: {qtcf['CHG'].max():+.1f} ms "
              f"(subject {qtcf.loc[qtcf['CHG'].idxmax(), 'patient_id']}, "
              f"{qtcf.loc[qtcf['CHG'].idxmax(), 'TRTA']})")
        print("  Keyed by (patient_id, period, timepoint_n) — no record to join to.")
    except FileNotFoundError:
        print("  (needs a local copy of the release; pass --data-path)")

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
    print("  label calls 'Dofetilide'. Filter on nominal_hours_from_period_start")
    print("  or on the plasma concentration before treating it as exposure.")


if __name__ == "__main__":
    main()
