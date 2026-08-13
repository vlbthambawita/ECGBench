#!/usr/bin/env python3
"""
Example: ECG-capable smartwatches — four watches and a reference electrocardiograph
reading the same patient simulator.

915 recordings of a METRON PS-440 patient simulator, taken *simultaneously* by a
Philips TC30 hospital electrocardiograph and by an Apple Watch Series 9, a Samsung
Galaxy Watch 6, a Fitbit Sense 2 and a Withings ScanWatch, under IEC
60601-2-25:2011. No human was recorded. Six things to demonstrate, and the first
three decide whether you can use the dataset correctly at all:

1. **The label is the simulator setting, not a diagnosis.** 36 settings in four
   families — heart rate 30-300 bpm, R-wave amplitude 500-2000 µV, ST offset -800
   to +800 µV, and one 2 Hz square wave. `nominal_rate_bpm`,
   `nominal_r_amplitude_uv` and `nominal_st_offset_uv` carry them, each NaN outside
   its own family. Measuring how far a device departs from them is the question
   this release exists to answer, so nothing estimates them for you.
2. **The smartwatch records are lead I and their headers say `II`.** The simulator's
   right-arm output went to the crown and its left-arm output to the caseback, which
   is lead I; all 720 smartwatch headers name the channel `II` anyway. ECGBench
   follows the files, so `leads=["II"]` silently mixes a genuine lead II (the
   12-lead reference) with an arm-to-arm lead I (the watches). This script prints
   `derivation` so the trap is visible rather than described.
3. **The reference is 12-lead and the watches are single-lead**, at four different
   sampling rates and three different lengths — so a batch needs BOTH `leads=` and
   `window=`. Without them `default_collate` raises. `leads=` is re-resolved
   against each record's own layout, so a chest lead works for the reference and
   *refuses* for a watch instead of returning its only channel.
4. **`clean` contains no Samsung record at all.** Every Samsung record is 15,001
   samples where 15,000 is 30.000 s, and that extra final sample is WFDB's
   invalid-sample marker, which reads back as NaN. All 179 fail `nan_values`. The
   signal before it is intact: `window=(0, 15000)` reads one with no NaN.
5. **Folds group on the simulator setting, because there is no patient.** The five
   repetitions of a setting correlate at ~0.95, and the same setting on a *different*
   device at 0.803 — the five instruments recorded the same output at the same
   instant. So a setting lies wholly inside one fold, and `setting_id` cannot be
   trained as a classification target across folds; the numeric nominal columns can.
6. **The default test fold holds two of the four families.** `amp_test` has 4
   settings and `sqr-2hz` has 1, so they reach 4 folds and 1 fold respectively at
   any fold count above four. Use `split=None` with `fold_numbers=[...]` for those.

The splits are NOT on the HuggingFace Hub — the release is restricted-access under
the PhysioNet Restricted Health Data License 1.5.0 — so this script uses
`metadata_source="local"` and needs the fold tree copied next to the signals:

    ecgbench splits --dataset ecg_capable_smartwatches --data-path <DATA>
    cp -r output/ecg_capable_smartwatches/{clean,original} <DATA>

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset (credentialed download from PhysioNet).

Usage:
  python examples/load_ecg_capable_smartwatches.py \
      --data-path /path/to/ecg-capable-smartwatches/1.0.0/
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError

#: The first 5,500 samples. The shortest record in the release is a Philips one of
#: exactly 5,500 samples, so this is the largest fixed window that fits every
#: record — and, because the five devices sample at four different rates, it is
#: 11.0 s of Philips and 22.0 s of Fitbit. A window in samples is not a window in
#: time here.
WINDOW = (0, 5_500)

#: Samsung's 15,001st sample is WFDB's invalid marker; 15,000 is 30.000 s at 500 Hz.
SAMSUNG_CLEAN_WINDOW = (0, 15_000)


def main():
    parser = argparse.ArgumentParser(description="Load the ECG-capable smartwatches dataset")
    parser.add_argument("--data-path", default=None, help="Path to the 1.0.0/ directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config("ecg_capable_smartwatches")
    print(f"Dataset:  {config.name}")
    print(f"Version:  {config.version}   licence: {config.license}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Leads:    {config.lead_names}  <- what the FILES say; the watches record lead I")
    print(f"          alternate layouts: {sorted(config.alternate_lead_names or {})} "
          "(the 12-lead reference)")
    print(f"Rates:    {config.sampling_rates}  <- one per device, a per-record property")
    print(f"Grouping: {config.patient_id_column}  <- a simulator setting, NOT a patient")
    print()

    common = dict(
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source="local",  # the splits are not published to the Hub
    )

    try:
        dataset = ECGDataset("ecg_capable_smartwatches", labels=True, **common)
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    labels = sample["labels"]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (whole record, no window)")
    print(f"  record_id            {sample['record_id']!r}")
    print(f"  device_model         {labels['device_model']!r}  ({labels['device_role']})")
    print(f"  derivation           {labels['derivation']!r}  <- NOT what the header says")
    print(f"  setting_id / family  {labels['setting_id']} / {labels['family']}")
    print(f"  nominal rate         {labels['nominal_rate_bpm']} bpm")
    print(f"  nominal amplitude    {labels['nominal_r_amplitude_uv']} uV")
    print(f"  nominal ST offset    {labels['nominal_st_offset_uv']} uV")
    print(f"  rate / samples       {labels['sampling_rate']} Hz / {labels['n_samples']}"
          f"  ({labels['duration_secs']:.2f} s)")
    print(f"  amplitude            {labels['min_mv']:.3f} to {labels['max_mv']:.3f} mV")

    # labels_df is aligned POSITIONALLY with metadata_df, not indexed by record id.
    frame = dataset.labels_df.copy()
    frame.index = dataset.metadata_df[config.record_id_column].to_numpy()

    # --- 1. Which device, and which lead it really recorded ------------------
    print(f"\nDevices in this split ({args.version}, no signals decoded):")
    for model, rows in frame.groupby("device_model"):
        print(f"  {model:<24} {len(rows):>3} records"
              f"   {int(rows['sampling_rate'].max()):>3} Hz"
              f"   {int(rows['n_leads'].max()):>2} lead(s)"
              f"   {rows['derivation'].iloc[0]}")
    print("  -> the headers name every one of these channels 'II'. Filter on")
    print("     `derivation` before comparing morphology across devices.")

    # --- 2. The labels are the simulator settings ----------------------------
    print("\nSimulator settings in this split:")
    for family, rows in frame.groupby("family"):
        print(f"  {family:<12} {len(rows):>3} records"
              f"   settings: {sorted(rows['setting_id'].unique())}")
    print("  -> a setting lies wholly inside ONE fold (they are the grouping unit),")
    print("     so train on the numeric nominal_* columns, not on setting_id.")

    # --- 3. Batching needs leads= AND window= --------------------------------
    batched = ECGDataset(
        "ecg_capable_smartwatches", labels=True, leads=["II"], window=WINDOW, **common
    )
    loader = DataLoader(
        batched, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
    )
    batch = next(iter(loader))
    print(f"\nBatch signal:  {tuple(batch['signal'].shape)}"
          f"  (leads=['II'] + window={WINDOW} make this stackable)")
    print(f"Batch records: {batch['record_id']}")

    # A regression target: the nominal heart rate, where the family defines one.
    targets = torch.tensor(
        [row["nominal_rate_bpm"] if row["family"] == "freq_test" else float("nan")
         for row in batch["labels"]],
        dtype=torch.float32,
    )
    print(f"Targets (bpm): {targets.tolist()}  (NaN where the family is not freq_test)")

    # --- 4. leads= refuses rather than guessing ------------------------------
    chest = ECGDataset(
        "ecg_capable_smartwatches", leads=["V4"], window=WINDOW, **common
    )
    reference = frame.index[frame["n_leads"] == 12]
    watch = frame.index[frame["n_leads"] == 1]
    print("\nleads=['V4'] against the two layouts:")
    if len(reference):
        position = list(frame.index).index(reference[0])
        print(f"  {reference[0]:<32} {tuple(chest[position]['signal'].shape)}"
              "   <- the reference's true V4")
    if len(watch):
        position = list(frame.index).index(watch[0])
        try:
            chest[position]
        except ValueError as e:
            print(f"  {watch[0]:<32} refused: {str(e).split('.')[0]}.")

    # --- 5. Samsung's trailing invalid sample --------------------------------
    samsung = frame.index[frame["trailing_invalid_sample"]]
    print(f"\n{len(samsung)} of the {len(frame)} records in this split end in WFDB's "
          "invalid-sample marker.")
    if len(samsung):
        whole = ECGDataset("ecg_capable_smartwatches", **common)
        windowed = ECGDataset(
            "ecg_capable_smartwatches", window=SAMSUNG_CLEAN_WINDOW, **common
        )
        position = list(frame.index).index(samsung[0])
        n_whole = int(torch.isnan(whole[position]["signal"]).sum())
        n_windowed = int(torch.isnan(windowed[position]["signal"]).sum())
        print(f"  {samsung[0]}: {n_whole} NaN over the whole record, "
              f"{n_windowed} under window={SAMSUNG_CLEAN_WINDOW}")
    else:
        print("  (none here — they are all Samsung records, and `clean` excludes")
        print("   every one of them. Run with --version original to see them.)")

    # --- 6. What the reference shows that the watches do not ------------------
    amplitude = frame[frame["family"] == "amp_test"]
    if not amplitude.empty:
        print("\nMean peak-to-peak span (mV) by nominal R-wave amplitude:")
        table = amplitude.pivot_table(
            index="device_model", columns="nominal_r_amplitude_uv",
            values="span_mv", aggfunc="mean",
        ).round(3)
        print(table.to_string())
        print("  -> span_mv is the record's own range, NOT an R-wave measurement, and")
        print("     every record was rescaled to fill int16 — so read the ordering,")
        print("     not the absolute millivolts.")
    else:
        print("\nNo amp_test records in this split: amp_test reaches only 4 of the 10")
        print("folds, because the release has 4 amplitude settings. Use split=None with")
        print("fold_numbers=[2, 6, 8, 9] to get them.")


if __name__ == "__main__":
    main()
