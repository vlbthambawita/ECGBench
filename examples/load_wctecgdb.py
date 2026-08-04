#!/usr/bin/env python3
"""
Example: Wilson Central Terminal ECG Database — 37 channels, raw and filtered.

540 ten-second segments from 92 hospital-admitted cardiac patients at 800 Hz. The
dataset exists to measure the Wilson Central Terminal — the reference V1-V6 are
measured against, which conventional ECG assumes is 0 V and which this release
shows reaching a substantial fraction of lead II. So the WCT channel is the point,
not an artefact.

Four things this script goes out of its way to show, because all four are easy to
get wrong here and none of them applies to the other 12-lead datasets in ECGBench:

1. **This is not a 12-lead record.** 37 channels: I, II, III and V1-V6, the three
   limb electrode potentials LA/RA/LL, the six true unipolar chest leads UV1-UV6 —
   each present both raw and filtered — plus WCT. aVR, aVL and aVF do not exist in
   the release; derive them from I and II if you need them.
2. **Every channel ships twice.** Channels 0-17 are the raw acquisition, 18-35 the
   same signals after DC removal and a 0.05-150 Hz band-pass. Mixing the two
   families in one tensor mixes two preprocessing states, and the raw
   limb/unipolar channels carry several mV of DC offset that the filtered ones do
   not. Select by name.
3. **Segments cluster hard by patient.** 540 records come from 92 patients, 1-31
   each; five patients account for a quarter of the dataset. Folds are grouped on
   patient_id for exactly this reason, and any per-record statistic you compute is
   weighted by segment count rather than by person.
4. **The only label is a patient-level admission diagnosis**, free text, 43
   distinct strings, 10 patients with none recorded. It says why the patient was
   admitted, not what these ten seconds show.

Usage:
  python examples/load_wctecgdb.py --data-path /path/to/wctecgdb/1.0.1/
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config

#: The filtered diagnostic-lead family, by name — never by index.
FILTERED_LEADS = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]


def as_channels(value):
    """Normalise a reconstructed-precordials cell to a list, whether it came from
    the label loader (a real list) or the generated metadata CSV (';'-joined — NOT
    comma-joined, because the channel list contains commas itself)."""
    if isinstance(value, list):
        return value
    text = "" if value is None else str(value)
    if text in ("", "nan"):
        return []
    return [part for part in text.split(";") if part]


def main():
    parser = argparse.ArgumentParser(
        description="Load the Wilson Central Terminal ECG database via ECGBench"
    )
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("wctecgdb")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version}) @ {config.default_sampling_rate} Hz")
    print(f"Channels: {config.leads} — 18 raw, 18 filtered, then WCT")
    print(f"  raw:      {', '.join(config.lead_names[:18])}")
    print(f"  filtered: {', '.join(config.lead_names[18:36])}, {config.lead_names[36]}")
    print("          ^ no aVR/aVL/aVF anywhere in this release")
    print()

    dataset = ECGDataset(
        "wctecgdb", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source, labels=True,
    )
    print(f"Records:  {len(dataset)}  (of 540, from 92 patients)")

    sample = dataset[0]
    labels = sample["labels"]
    signal = sample["signal"]
    print(f"\nLabel fields: {list(labels)}")
    print("\nFirst record:")
    print(f"  ID:                {sample['record_id']}")
    print(f"  signal:            {tuple(signal.shape)}  (channels, samples)")
    print(f"  patient:           {labels['patient_id']} / {labels['segment']}")
    print(f"  age / sex:         {labels['age']} / {labels['sex']}")
    print(f"  diagnosis:         {labels['diagnosis']!r}")
    print(f"  diagnosis group:   {labels['diagnosis_group']!r}  (stratification label)")
    print(f"  synthesised chans: {as_channels(labels['reconstructed_precordials'])}")

    # --- 1. Raw and filtered are two preprocessing states, not two leads --------
    print("\nSame physical signal, raw vs filtered (this record, in mV):")
    names = list(config.lead_names)
    for lead in ("II", "LA", "UV3"):
        raw = signal[names.index(f"{lead}-Raw")]
        clean = signal[names.index(lead)]
        print(f"  {lead:4s} raw  min {raw.min():+8.3f}  max {raw.max():+8.3f}   "
              f"filtered  min {clean.min():+8.3f}  max {clean.max():+8.3f}")
    print("  The raw limb-electrode and unipolar channels are unreferenced")
    print("  potentials: their offset is the DC the 0.05-150 Hz filter removes,")
    print("  not a calibration error. Pick one family and stay in it.")

    # --- 2. The WCT is not zero — that is the whole point ----------------------
    wct = signal[names.index("WCT")]
    lead_ii = signal[names.index("II")]
    print(f"\nWCT peak-to-peak {(wct.max() - wct.min()):.3f} mV against lead II's "
          f"{(lead_ii.max() - lead_ii.min()):.3f} mV")
    print(f"  = {100 * (wct.max() - wct.min()) / (lead_ii.max() - lead_ii.min()):.0f}% "
          "of lead II on this record. Conventional ECG assumes this is 0 V.")
    print("  V1 = UV1 - WCT holds by construction, which is what makes the true")
    print("  unipolar leads UV1-UV6 recoverable at all.")

    # --- 3. Selecting a lead family by name -----------------------------------
    filtered = ECGDataset(
        "wctecgdb", split=args.split, version=args.version,
        data_path=args.data_path, metadata_source=args.metadata_source,
        leads=FILTERED_LEADS,
    )
    print(f"\nleads={FILTERED_LEADS} -> {tuple(filtered[0]['signal'].shape)}")
    print("  ^ the conventional 9 leads, filtered. Add aVR/aVL/aVF yourself:")
    print("    aVR = -(I + II)/2,  aVL = I - II/2,  aVF = II - I/2")

    # --- 4. Patient clustering, and why a per-record rate misleads -------------
    # labels_df is aligned POSITIONALLY with metadata_df and carries a RangeIndex,
    # not record IDs — attach them explicitly before reporting anything per record.
    frame = dataset.labels_df.copy()
    frame["record_id"] = dataset.metadata_df[config.record_id_column].to_numpy()
    per_patient = frame.groupby("patient_id").size()

    print(f"\nThis split: {len(frame)} records from {len(per_patient)} patients "
          f"({per_patient.min()}-{per_patient.max()} segments each)")
    top = per_patient.sort_values(ascending=False).head(5)
    print(f"  top 5 patients contribute {top.sum()} records "
          f"({100 * top.sum() / len(frame):.1f}%): {top.to_dict()}")

    print("\nDiagnosis groups, counted both ways — the columns disagree because")
    print("segment counts vary 30x between patients:")
    by_record = frame["diagnosis_group"].value_counts()
    by_patient = frame.drop_duplicates("patient_id")["diagnosis_group"].value_counts()
    print(f"  {'group':38s} {'records':>8s} {'patients':>9s}")
    for group in by_record.index:
        print(f"  {group:38s} {by_record[group]:8d} {by_patient.get(group, 0):9d}")
    print("  Weight by patient, or group by patient_id, before quoting any rate.")

    reported = frame["diagnosis_reported"].astype(bool)
    print(f"\n{(~reported).sum()} of {len(frame)} records have no diagnosis recorded "
          f"('not reported' is a value here, not a blank).")

    synthesised = frame[frame["has_reconstructed_precordials"].astype(bool)]
    if len(synthesised):
        print(f"\n{len(synthesised)} record(s) carry SYNTHESISED precordial channels "
              "(V = UV - WCT), not measurements:")
        for _, row in synthesised.iterrows():
            print(f"  {row['record_id']}: {as_channels(row['reconstructed_precordials'])}")
        print("  Exclude these when evaluating precordial reconstruction — otherwise")
        print("  the method under test is scored against its own output.")

    # --- 5. A batch, and a target tensor --------------------------------------
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=ecg_collate_fn)
    batch = next(iter(loader))
    print(f"\nFirst batch: signal {tuple(batch['signal'].shape)} "
          f"{batch['signal'].dtype}, {len(batch['labels'])} label dicts")

    classes = sorted(frame["diagnosis_group"].unique())
    targets = torch.tensor([
        classes.index(row["diagnosis_group"]) for row in batch["labels"]
    ])
    print(f"Targets:     {targets.tolist()}")
    print(f"  over {classes}")
    print("  diagnosis_group is a coarse reduction of free-text admission")
    print("  diagnoses, used to balance the folds. It is a PATIENT-level label")
    print("  about an admission, so it is not a waveform target — a segment in")
    print("  the 'Other tachyarrhythmia' group need not contain a tachycardia.")


if __name__ == "__main__":
    main()
