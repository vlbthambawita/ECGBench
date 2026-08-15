#!/usr/bin/env python3
"""
Example: EDGAR — Experimental Data and Geometric Analysis Repository, with labels.

**EDGAR is not one dataset**, and this script's main job is to stop you treating
it like one. It is 24 electrocardiographic-imaging experiments from ten
institutions — 2,943 recordings, 20 subjects, 29 distinct electrode counts, six
sampling rates, five measurement surfaces and two unit conventions — pooled into
one record table because ECGBench needs one. Filter it before you use it.

Seven things to demonstrate:

1. **The first `mat` dataset in ECGBench**, and its signal paths are not paths:
   `<file>.mat:<variable>:<orientation>:<unit>`, because a MATLAB container
   declares none of those reliably.
2. **The recordings have to be unpacked first.** EDGAR publishes only zips, so
   the split pipeline extracts into `ecgbench_extracted/` on first run.
3. **`recording_surface` is the label, and one of its values is not a
   potential** — the 16 KIT `transmembrane` runs are simulated membrane
   voltages.
4. **The real ground truth is the pacing site**, and it comes from the CARTO
   tables: 2,157 records carry x/y/z coordinates.
5. **Two experiments declare the wrong unit**, so `declared_unit` and
   `unit_applied` disagree for four records and `unit_source` says why.
6. **You cannot batch this dataset as it stands** — records differ in both
   electrode count and length, so `default_collate` raises on the first batch.
   Filter to one experiment, then `window=`.
7. **Four subjects hold 92% of the recordings**, so the default fold layout is
   not a 8/1/1 split of anything.

Labels come from the curated table in `ecgbench.labels.edgar` plus each
recording's own MATLAB struct, so this works after one pipeline run. The fold
CSVs come from the Hub by default.

Prerequisites:
  - pip install ecgbench[torch,mat]
  - A local copy of the repository (free registration at edgar.sci.utah.edu)
  - One prior run of: ecgbench splits --dataset edgar --data-path <path>
    (that is what unpacks the archives the signal paths point into)

Usage:
  python examples/load_edgar.py --data-path /path/to/EDGAR/
"""

import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels.edgar import EXPERIMENTS, RECORDING_SURFACES, SIGNAL_FREE_POSTS

#: The experiment to batch in section 6. Its 944 recordings are one subject's
#: 120-electrode BSPM at 2 kHz — uniform in leads, which is what batching needs.
BATCHABLE_EXPERIMENT = "charles_pat1"

#: Samples to take from each of those. Their lengths run 246 to 364, so a window
#: of 246 fits every one of them and no window fits the dataset as a whole.
BATCH_WINDOW = (0, 246)


def main():
    parser = argparse.ArgumentParser(description="Load edgar with labels")
    parser.add_argument("--data-path", default=None, help="Path to the EDGAR root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("edgar")
    print(f"Dataset:  {config.name}")
    print(f"Mirror:   {config.version}   (a rolling portal, not a versioned release)")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Format:   {config.signal_format}  <- the first MATLAB dataset in ECGBench")
    print(f"Leads:    {config.leads} is the MODE, not the dataset — 29 counts, 54 to 2223")
    print(f"Names:    {config.lead_names}  <- deliberately empty; see section 1")
    print(f"Rates:    {config.sampling_rates} Hz, nominal {config.default_sampling_rate}")
    print(f"Folds:    {config.n_folds}, grouped on {config.patient_id_column!r}")
    print()
    print("!! EDGAR IS 24 EXPERIMENTS FROM TEN INSTITUTIONS POOLED INTO ONE TABLE.")
    print("!! Electrode counts, rates, surfaces, species and units all differ between")
    print("!! them, and four subjects hold 92% of the recordings. Filter on")
    print("!! `experiment`, `recording_surface` and `n_leads` before training on it.")
    print()

    dataset = ECGDataset(
        "edgar",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        labels=True,
    )
    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (leads, samples)")
    print(f"  record_id            {sample['record_id']}")
    print(f"  experiment           {labels['experiment']}")
    print(f"  subject_id           {labels['subject_id']}   ({labels['species']})")
    print(f"  recording_surface    {labels['recording_surface']}")
    print(f"  electrode_array      {labels['electrode_array']}")
    print(f"  intervention         {labels['intervention']}")
    print(f"  n_leads / n_samples  {int(labels['n_leads'])} / {int(labels['n_samples'])}")
    print(f"  sampling_rate_hz     {labels['sampling_rate_hz']}")
    print(f"  unit_applied         {labels['unit_applied']}  ({labels['unit_source']})")

    # labels_df is aligned positionally to metadata_df, so give it the record ids
    # back — otherwise every "record N" printed below is a row number.
    df = dataset.labels_df.copy()
    df.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. The signal reference ----------------------------------------------
    print("\n" + "=" * 78)
    print("1. THE SIGNAL PATH IS NOT A PATH")
    print("=" * 78)
    print("  <file>.mat:<variable>:<orientation>:<unit>, e.g.")
    for record_id in (df.index[0], *df.index[df["orientation"] == "sl"][:1]):
        print(f"    {df.loc[record_id, 'signal_path']}")
    print()
    print("  <variable>    22 distinct names across the release — ts, bspm, ECG, EGM,")
    print("                ens, eps, pots, lichaampots, heartleadpots, and one per")
    print("                simulated pacing site (Simulation_04_LVLAT and friends).")
    print("  <orientation> ls = (leads, samples), sl = its transpose. NOT inferable:")
    print("                KIT stores 2223 leads x 225 samples and Dalhousie stores")
    print("                1142 samples x 120 leads, so 'leads are the shorter axis'")
    print("                is wrong in both directions.")
    print("  <unit>        mV or uV — EDGAR mixes both, so one signal_unit_scale")
    print("                could only ever be right for part of the release.")
    print(f"  Transposed experiments in this split: "
          f"{sorted(set(df.loc[df['orientation'] == 'sl', 'experiment']))}")

    # --- 2. What is actually in the table --------------------------------------
    print("\n" + "=" * 78)
    print("2. WHAT IS IN THIS SPLIT (the release has 24 experiments and 20 subjects)")
    print("=" * 78)
    print(f"  {'experiment':22s} {'subject':26s} {'n':>5s} {'leads':>10s} {'Hz':>7s}  surface")
    for experiment, group in df.groupby("experiment"):
        leads = sorted(set(int(v) for v in group["n_leads"]))
        rates = sorted(set(group["sampling_rate_hz"].dropna()))
        rate_text = f"{rates[0]:g}" if len(rates) == 1 else "-"
        surfaces = "+".join(sorted(set(group["recording_surface"])))
        print(
            f"  {experiment:22s} {group['subject_id'].iloc[0]:26s} {len(group):5d} "
            f"{str(leads if len(leads) < 3 else f'{min(leads)}..{max(leads)}'):>10s} "
            f"{rate_text:>7s}  {surfaces}"
        )
    print("\n  Two of EDGAR's 26 experiments ship no signals at all:")
    for post, reason in SIGNAL_FREE_POSTS.items():
        print(f"    {post}\n      {reason.split('.')[0]}.")

    # --- 3. The label, and the value that is not a potential --------------------
    print("\n" + "=" * 78)
    print("3. recording_surface — AND ONE VALUE THAT IS NOT A POTENTIAL")
    print("=" * 78)
    counts = Counter(df["recording_surface"])
    for surface in RECORDING_SURFACES:
        n = counts.get(surface, 0)
        arrays = sorted(set(df.loc[df["recording_surface"] == surface, "electrode_array"]))
        print(f"  {surface:16s} {n:5d}  {', '.join(arrays) if arrays else '-'}")
    print()
    print("  !! `transmembrane` is the simulated MEMBRANE VOLTAGE of KIT's TMV source")
    print("  !! models — a different physical quantity from every other record here")
    print("  !! (their resting level is exactly -84 mV). Do not train on them as")
    print("  !! though they were electrograms.")
    print()
    print("  `electrode_array` is the distinction `recording_surface` cannot make: a")
    print("  599-electrode rigid cage and a 247-electrode sock are both `epicardium`,")
    print("  and they are not the same measurement.")

    # --- 4. The pacing site ------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. THE GROUND TRUTH IS THE PACING SITE, AND IT HAS COORDINATES")
    print("=" * 78)
    paced = df[df["pacing_site_x"].notna()]
    print(f"  {len(paced)} of {len(df)} records carry CARTO x/y/z for the paced site.")
    for experiment, group in paced.groupby("experiment"):
        # dropna=False: KIT numbers its sites without naming a chamber, so that
        # column is empty for it and a plain groupby would report zero sites.
        sites = group.groupby(["pacing_chamber", "pacing_site"], dropna=False).ngroups
        print(f"    {experiment:22s} {len(group):5d} records over {sites:3d} sites")
    if len(paced):
        row = paced.iloc[0]
        print(
            f"\n  e.g. {paced.index[0]}\n"
            f"       chamber {row['pacing_chamber']!r} site {int(row['pacing_site'])} at "
            f"({row['pacing_site_x']:.3f}, {row['pacing_site_y']:.3f},"
            f" {row['pacing_site_z']:.3f})"
        )
    print("\n  Localising that from the body-surface map is what EDGAR exists for.")
    print("  Fold membership is grouped by subject, so a model cannot memorise one")
    print("  patient's torso and be scored on the same patient's other pacing sites.")

    # --- 5. The unit the file declares is not always the unit -------------------
    print("\n" + "=" * 78)
    print("5. TWO EXPERIMENTS DECLARE THE WRONG UNIT")
    print("=" * 78)
    disagree = df[
        df["declared_unit"].notna()
        & (df["declared_unit"].str.lower() != df["unit_applied"].str.lower())
    ]
    print(f"  {len(disagree)} of {len(df)} records in this split disagree with their")
    print("  file (4 of 2,943 across the release — Valencia pat1 and pat2, ECG and EGM):")
    for record_id, row in disagree.iterrows():
        print(
            f"    {record_id[:56]:58s} declared {row['declared_unit']:>4s} "
            f"-> applied {row['unit_applied']}"
        )
    print("\n  Valencia's files say 'mV' for samples reaching 5350. Their own")
    print("  Docs/Readme.txt says microV, which is what ECGBench applies — believing")
    print("  the file would put body-surface potentials at five volts.")

    # --- 6. Batching -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. YOU CANNOT BATCH THIS DATASET AS IT STANDS")
    print("=" * 78)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    try:
        next(iter(loader))
        print("  (this split happens to be homogeneous — unusual)")
    except RuntimeError as e:
        print(f"  DataLoader over the whole split raises, as it must:\n    {e}"[:300])

    indices = [
        i
        for i, experiment in enumerate(dataset.metadata_df[config.record_id_column])
        if df.loc[experiment, "experiment"] == BATCHABLE_EXPERIMENT
    ]
    n_leads = int(df.loc[df["experiment"] == BATCHABLE_EXPERIMENT, "n_leads"].iloc[0])
    print(
        f"\n  Fix: pick one experiment — {BATCHABLE_EXPERIMENT} has {len(indices)} records"
        f" in this\n  split, all {n_leads} electrodes — and window to its shortest"
        f" record, {BATCH_WINDOW}."
    )

    windowed = ECGDataset(
        "edgar",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        labels=True,
        window=BATCH_WINDOW,
    )
    loader = DataLoader(
        Subset(windowed, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
    )
    batch = next(iter(loader))
    print(f"\n  signals   {tuple(batch['signal'].shape)}  (batch, leads, samples)")
    print(f"  record_id {[r.split('__')[-1] for r in batch['record_id']]}")

    # ecg_collate_fn keeps label dicts as a LIST of dicts, not a dict of lists.
    target = torch.tensor(
        [
            [float(row["pacing_site_x"]), float(row["pacing_site_y"]),
             float(row["pacing_site_z"])]
            for row in batch["labels"]
        ],
        dtype=torch.float32,
    )
    print(f"  target    {tuple(target.shape)}  (batch, xyz) — the paced site in CARTO space")
    print("\n  window= is plain data, so it survives DataLoader(num_workers>0) under the")
    print("  'spawn' start method, where a lambda transform would raise PicklingError.")

    # --- 7. The fold layout ------------------------------------------------------
    print("\n" + "=" * 78)
    print("7. THE FOLDS ARE PATIENT-SAFE, NOT EQUAL")
    print("=" * 78)
    subjects = Counter(df["subject_id"])
    biggest = subjects.most_common(4)
    share = sum(n for _, n in biggest) / max(len(df), 1)
    print(f"  Four subjects hold {share:.0%} of this split:")
    for subject, n in biggest:
        print(f"    {subject:26s} {n:5d}")
    print(f"  ...and the smallest contribute {min(subjects.values())} record(s) each.")
    print("\n  So the default fold-10 test split is a couple of dozen records, not a")
    print("  tenth of the release. That is what 20 experiments look like; splitting")
    print("  one subject's 944 recordings across train and test would instead make")
    print("  pacing-site localisation look solved. For a different question, use")
    print("  ECGDataset(split=None, fold_numbers=[...]) and group them yourself.")
    print("\n  Curated per-experiment facts: ecgbench.labels.edgar.EXPERIMENTS")
    print(f"  ({len(EXPERIMENTS)} entries — authoritative archive, surface, orientation,")
    print("  unit and its provenance, and the citations EDGAR requires per dataset.)")


if __name__ == "__main__":
    main()
