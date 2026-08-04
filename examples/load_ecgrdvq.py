#!/usr/bin/env python3
"""
Example: ECGRDVQ — the dataset where J-Tpeak separates what QTc cannot.

5,232 ten-second 12-lead ECGs at 1 kHz from 22 healthy volunteers in a randomised,
double-blind, 5-period crossover Phase I trial (study SCR-002, NCT01873950) built to
test whether measuring **J-Tpeak alongside QTc** can tell a drug that blocks only
hERG apart from one that also blocks inward currents — and so prolongs QTc without
the same proarrhythmic risk. Every subject passed through all five arms: dofetilide
(predominant hERG, the positive control), quinidine (hERG + sodium), ranolazine
(hERG + late sodium), verapamil (hERG + L-type calcium), and placebo.

This is **SCR-002, the first of the three FDA studies** in ECGBench — `ecgdmmld` is
SCR-003 and `ecgcipa` is SCR-004. Seven things this script goes out of its way to
show, because each differs either from the rest of ECGBench or from its own siblings:

1. **`treatment` really is the drug here** — and that is the difference from
   `ecgdmmld`, where the arms are staged combinations and a record labelled
   "Mexiletine + Dofetilide" at 2 h contains no dofetilide. Each period of this
   study dosed a single agent. One caveat survives: 327 pre-dose records carry
   their period's drug name while containing no drug. Section 5 makes that visible.
2. **There is no class to predict.** Nobody in this cohort has cardiac disease.
   `treatment` is the stratification label and the closest thing to a class.
3. **The samples really are millivolts.** `signal_unit_scale: 1.0`, like `ecgdmmld`
   and unlike `ecgcipa`'s 0.001 — the headers declare per-lead gains against `/mV`.
4. **Records come in exact triplicates.** Three segments per subject per timepoint,
   seconds apart at the same concentration, and here *all* 1,744 groups hold
   precisely 3. So 5,232 records are closer to 1,744 independent observations — and
   closer still to 22 for anything subject-level.
5. **The pharmacokinetic table is long, not wide**, because one agent per period
   means one measurement per record. And **dofetilide is reported in pg/mL while the
   other three analytes are ng/mL**, so section 4 shows the normalised column that
   exists to stop you averaging across a 1000x scale change.
6. **The study's endpoint is computable per record.** `is_baseline` flags each
   period's pre-dose triplicate, so change from baseline is a per-record quantity —
   and section 6 reconstructs the study's actual finding from it.
7. **Every record ships twice.** `signal_path` is the raw 10 s ECG;
   `median_beat_path` is a derived 16-channel median beat whose `.atr` fiducials are
   what the published intervals were measured from. Nine of the 5,232 median beats
   were never published, which is why nine records have no PR, QRS, QT or J-Tpeak;
   `median_beat_available` says which.

Usage:
  python examples/load_ecgrdvq.py --data-path /path/to/ecgrdvq/1.0.0/
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

#: What each arm blocks — the reason the study exists. "Predominant hERG" drugs
#: prolong J-Tpeak as well as QTc; multichannel blockers prolong QTc alone.
ION_CHANNEL_PROFILE = {
    "Dofetilide": "predominant hERG (positive control)",
    "Quinidine Sulph": "hERG + peak/late sodium",
    "Ranolazine": "hERG + late sodium",
    "Verapamil HCL": "hERG + L-type calcium",
    "Placebo": "control — sets the diurnal drift",
}


def main():
    parser = argparse.ArgumentParser(description="Load ECGRDVQ via ECGBench")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("ecgrdvq")
    print(f"Dataset: {config.name} v{config.version}")
    print(f"Split:   {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Leads:   {config.leads} — {', '.join(config.lead_names)}")
    print("         ^ uppercase 'A', like ptbxl and ecgdmmld. The sibling ecgcipa")
    print("           writes the same three aVR/aVL/aVF, so hardcode neither.")
    print(f"Samples: {int(config.duration_seconds * config.default_sampling_rate)} "
          "per record, uniform across all 5,232")
    print("Rate:    acquired at 500 Hz, up-sampled to 1 kHz by the depositors —")
    print("         the extra samples are interpolated, not measured.")
    print()

    dataset = ECGDataset(
        "ecgrdvq", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source, labels=True,
    )
    print(f"Records: {len(dataset)}  (of 5,232, from 22 healthy volunteers)")

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
    dose = labels["dose"]
    if dose == dose:  # not NaN
        print(f"  dose:        {dose:g} {labels['dose_unit']}  "
              "<- note the unit: dofetilide is ug, the rest mg")
    else:
        print("  dose:        none (placebo)")
    print(f"  timepoint:   {labels['timepoint_hours']:+g} h from the period's dose "
          f"(baseline={labels['is_baseline']})")
    print("  intervals:   " + "  ".join(
        f"{shown}={labels[name]:.0f}"
        for name, shown in INTERVALS if labels[name] == labels[name]  # skip NaN
    ))
    print("               HR in bpm, the rest in ms. HR and QTcF are DERIVED —")
    print("               the release ships neither.")
    print(f"  median beat: {labels['median_beat_path']}  (+ .atr fiducials)")
    print("               16 channels (12 leads + VCGMAG + vx/vy/vz), and its")
    print("               LENGTH VARIES: 968-1,876 samples. ecgdmmld's is a fixed")
    print("               1,200, so never assume a shape.")

    # --- 1. The samples are already millivolts --------------------------------
    micro = ECGDataset(
        "ecgrdvq", split=args.split, version=args.version, data_path=args.data_path,
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
    # Length is uniform (all 5,232 records are exactly 10,000 samples), so unlike
    # cpsc_2018 or ptbdb any window inside [0, 10000) fits every record. window= is
    # pushed into wfdb's sampfrom/sampto, so it decodes only what it returns — and
    # unlike a cropping lambda it survives DataLoader(num_workers>0) under spawn.
    windowed = ECGDataset(
        "ecgrdvq", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, window=(2000, 5000),
    )
    print(f"\nwindow=(2000, 5000) -> {tuple(windowed[0]['signal'].shape)}  "
          "(5 s starting at 2 s)")
    print("  Every record is exactly 10,000 samples, so any window inside")
    print("  [0, 10000) fits all of them — no WindowOutOfRangeError to plan around.")

    # --- 3. Selecting leads by name -------------------------------------------
    limb = ECGDataset(
        "ecgrdvq", split=args.split, version=args.version, data_path=args.data_path,
        metadata_source=args.metadata_source, leads=["I", "II", "V5"],
    )
    print(f"\nleads=['I', 'II', 'V5'] -> {tuple(limb[0]['signal'].shape)}")
    print("  Matching is case-insensitive, so 'avr' and 'AVR' both resolve.")

    # --- 4. Triplicates, and why 5,232 overstates the sample size -------------
    # labels_df is aligned POSITIONALLY with metadata_df and carries a RangeIndex,
    # not record IDs — attach them explicitly before reporting anything per record.
    frame = dataset.labels_df.copy()
    frame["record_id"] = dataset.metadata_df[config.record_id_column].to_numpy()

    groups = frame.groupby(["patient_id", "period", "timepoint_hours"]).size()
    print(f"\nThis split: {len(frame)} records from {frame['patient_id'].nunique()} "
          f"subjects over {len(groups)} (subject, period, timepoint) groups")
    print(f"  {(groups == 3).sum()} of {len(groups)} groups hold exactly 3 "
          "near-duplicate records —")
    print("  same person, same posture, same concentration, seconds apart. Folds")
    print("  are grouped on patient_id, so all three land on the same side. In the")
    print("  full release this is exact: 1,744 of 1,744 groups hold 3.")

    absent = int((~frame["median_beat_available"]).sum())
    print(f"  {absent} record(s) here have no published median beat (9 in the full")
    print("  release). Every interval was measured FROM the median beat, so those")
    print("  rows have no PR, QRS, QT, J-Tpeak or Tpeak-Tend — only RR. Their raw/")
    print("  ECGs are intact.")
    repaired = int(frame["pr_ms_repaired"].sum())
    print(f"  {repaired} record(s) here had a 32-bit wrap in PR repaired (2 in the")
    print("  full release): the P onset fell before the median beat's start, so an")
    print("  unsigned subtraction wrapped. pr_ms_repaired flags them.")

    # --- 5. Treatment: the stratification label, and what it does mean ---------
    print("\nTreatment, counted both ways. This is a near-complete crossover — 21 of")
    print("22 subjects passed through every arm — so the subject counts are all but")
    print("equal and no split of this dataset can separate the arms:")
    by_record = frame["treatment"].value_counts()
    by_subject = frame.groupby("treatment")["patient_id"].nunique()
    print(f"  {'treatment':18s} {'records':>8s} {'subjects':>9s}  ion-channel profile")
    for arm in by_record.index:
        print(f"  {arm:18s} {by_record[arm]:8d} {by_subject.get(arm, 0):9d}  "
              f"{ION_CHANNEL_PROFILE.get(arm, '')}")

    print("\nUnlike ecgdmmld, `treatment` here IS the drug that was administered —")
    print("each period dosed a single agent, so there is no staged-combination trap.")
    print("The one caveat: pre-dose records carry a drug name with no drug in them.")
    predose = frame[frame["is_baseline"]]
    print(f"  {len(predose)} of {len(frame)} records here are pre-dose "
          f"({len(predose) / len(frame):.1%}); 327 of 5,232 in the release.")
    print("  Measured concentration, by arm — 'has PK' is the share of records with")
    print("  a plasma value, which is 0% for placebo and 0% pre-dose:")
    print(f"  {'treatment':18s} {'has PK':>7s} {'unit':>6s} {'mean ng/mL':>11s}")
    for arm in sorted(by_record.index):
        rows = frame[frame["treatment"] == arm]
        unit = rows["plasma_concentration_unit"].dropna()
        mean_ng = rows["plasma_concentration_ng_ml"].mean()
        print(f"  {arm:18s} {rows['plasma_concentration'].notna().mean():7.0%} "
              f"{(unit.iloc[0] if len(unit) else '—'):>6s} "
              f"{(f'{mean_ng:11.1f}' if mean_ng == mean_ng else '          —')}")
    print("  DOFETILIDE IS pg/mL AND THE OTHER THREE ARE ng/mL, so the raw")
    print("  plasma_concentration column mixes two scales 1000x apart. The")
    print("  plasma_concentration_ng_ml column above is the same quantity in one")
    print("  unit — use it for anything that crosses arms.")

    # --- 6. The endpoint, and the study's actual finding -----------------------
    print("\nChange from baseline — the study's actual endpoint. Every one of the 109")
    print("(subject, period) pairs has its own pre-dose triplicate, so the delta is")
    print("a per-record quantity:")
    try:
        from ecgbench.labels.ecgrdvq import load_baseline_deltas

        deltas = load_baseline_deltas(dataset.data_path, config)
        deltas = deltas.loc[deltas.index.intersection(frame["record_id"])]
        post = deltas[~deltas["is_baseline"]]
        print(f"  {'arm':18s} {'dQTcF':>7s} {'dJ-Tpeak':>9s} {'dTpeak-Tend':>12s}")
        for arm in sorted(post["treatment"].unique()):
            rows = post[post["treatment"] == arm]
            print(f"  {arm:18s} {rows['delta_qtcf_ms'].mean():+7.1f} "
                  f"{rows['delta_jtpeak_ms'].mean():+9.1f} "
                  f"{rows['delta_tpeak_tend_ms'].mean():+12.1f}")
        print("  ms, averaged over all post-dose timepoints.")

        # Placebo-correct, which is what makes the result legible. The placebo arm
        # drifts substantially over the same 24 hours, so the raw column above
        # understates every drug's effect on J-Tpeak.
        placebo = post[post["treatment"] == "Placebo"]
        if len(placebo):
            print("\n  PLACEBO-CORRECTED (arm minus placebo) — this is the study's")
            print("  finding, and it is why the dataset exists:")
            base_qtcf = placebo["delta_qtcf_ms"].mean()
            base_jt = placebo["delta_jtpeak_ms"].mean()
            print(f"  {'arm':18s} {'dQTcF':>7s} {'dJ-Tpeak':>9s}")
            for arm in sorted(a for a in post["treatment"].unique() if a != "Placebo"):
                rows = post[post["treatment"] == arm]
                print(f"  {arm:18s} "
                      f"{rows['delta_qtcf_ms'].mean() - base_qtcf:+7.1f} "
                      f"{rows['delta_jtpeak_ms'].mean() - base_jt:+9.1f}")
            print("  Every arm prolongs QTcF. Only the predominant-hERG blockers")
            print("  (dofetilide, quinidine) also prolong J-Tpeak; the multichannel")
            print("  blockers (ranolazine, verapamil) leave it flat or shorten it.")
            print("  That separation is what QTc alone cannot give you.")
            print("  This is a descriptive reconstruction over all post-dose records,")
            print("  NOT the published concentration-response model.")
    except FileNotFoundError:
        print("  (needs a local copy of the release; pass --data-path)")

    missing_qt = int(frame["qt_ms"].isna().sum())
    secondary = int(frame["tpeak_tpeakp_ms"].notna().sum())
    print(f"\n{missing_qt} record(s) here have no QT and no Tpeak-Tend (13 in the full")
    print("release): 9 have no median beat at all, and 4 more had no T offset that")
    print("could be annotated — all four are quinidine records whose T waves are the")
    print("flattest in the release. NaN, not 0.")
    print(f"{secondary} record(s) here carry a secondary T peak in tpeak_tpeakp_ms "
          "(42 in the")
    print("full release, mostly quinidine). This is the opposite of ecgdmmld, where")
    print("the column is empty in every row and no .atr marks one — a pipeline")
    print("written against that release will silently drop a real measurement here.")

    # --- 7. A batch, and a target tensor --------------------------------------
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
    print("\n  treatment is the fold-balancing label. It names the drug the subject")
    print("  was given, which is more than ecgdmmld's label does — but a pre-dose")
    print("  record in the dofetilide arm is still a drug-free ECG that this label")
    print("  calls 'Dofetilide'. Filter on is_baseline, on timepoint_hours, or on")
    print("  plasma_concentration_ng_ml before treating it as exposure.")


if __name__ == "__main__":
    main()
