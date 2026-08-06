#!/usr/bin/env python3
"""
Example: ZZU-pECG — 14,190 paediatric ECGs, and two different lead layouts.

The only genuinely paediatric 12-lead resource in the catalogue: 11,643
hospitalised children aged 1 day to 15 years. Four things worth seeing rather
than reading:

1. **Not every record has 12 leads.** 1,856 of 14,190 store only 9, dropping V2,
   V4 and V6 — and the 9-lead layout is not a prefix of the 12-lead one, so
   stored position 7 is V2 in one and V3 in the other. `leads=["V2"]` therefore
   *raises* on a reduced record rather than handing back V3, which is what the
   config's `alternate_lead_names` is for. Demonstrated below.
2. **Record length varies by a factor of 24**, from 5 s to 120 s. Any fixed
   `window=` has to fit the shortest record.
3. **The code columns mix codes with prose.** `AHA_code` gives an AHA code where
   one exists and a plain-English description where it does not, because 14 of
   the 105 findings have no AHA equivalent. Reading it as a code vocabulary
   invents phantom codes.
4. **Age is in days, and that matters here.** Paediatric ECG norms change fast:
   right-axis dominance and right precordial T-wave inversion are normal in an
   infant and abnormal in an adolescent. 546 records are under a year old.

Labels come from the shipped `AttributesDictionary.csv` plus its two dictionary
files, so this works without running the split pipeline first.

Prerequisites:
  - pip install ecgbench[torch]
  - A local copy of the dataset; labels are not on the HuggingFace Hub. The
    waveforms ship as a two-part split zip (Child_ecg.zip + Child_ecg.z01) which
    must be joined before extracting.

Usage:
  python examples/load_zzu_pecg.py --data-path /path/to/ZZU_pECG/
"""

import argparse
from collections import Counter

import pandas as pd
from torch.utils.data import DataLoader

from ecgbench import ECGDataset, ecg_collate_fn, load_config
from ecgbench.labels import LabelSourceMissingError

#: Sized to the SHORTEST record in the release, 2,500 samples (5 s at 500 Hz).
#: Anything wider raises WindowOutOfRangeError on those records. It is also what
#: makes a DataLoader batch possible at all here — without it the collate has to
#: cope with lengths from 2,500 to 60,000.
WINDOW = (0, 2500)

#: Present in both layouts, so this selection works for every record.
SAFE_LEADS = ["I", "II", "V1", "V5"]

#: Absent from the 9-lead layout, so this selection raises on those records.
REDUCED_ONLY_LEAD = "V2"


def main():
    parser = argparse.ArgumentParser(description="Load ZZU-pECG with its labels")
    parser.add_argument("--data-path", default=None, help="Path to the dataset root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--version", default="clean", choices=["clean", "original"])
    parser.add_argument("--metadata-source", default="hf", choices=["hf", "local"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config("zzu_pecg")
    print(f"Dataset:  {config.name} v{config.version}")
    print(f"Split:    {args.split} ({args.version})")
    print(f"Leads:    {config.lead_names}")
    print(f"          alternate: {config.alternate_lead_names}")
    print(f"Rate:     {config.default_sampling_rate} Hz, records 5-120 s")
    print(
        f"Validation expected_samples: {config.validation.expected_samples} "
        f"<- empty on purpose, lengths vary"
    )
    print()

    try:
        dataset = ECGDataset(
            "zzu_pecg",
            split=args.split,
            version=args.version,
            data_path=args.data_path,
            metadata_source=args.metadata_source,
            labels=True,
            # Read at load time, so a 120 s record decodes only these 5 s. On this
            # dataset that is also what makes batching possible.
            window=WINDOW,
        )
    except LabelSourceMissingError as e:
        print(f"Labels unavailable: {e}")
        return

    print(f"Records:  {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys:   {sorted(sample.keys())}")
    print(f"Signal shape:  {tuple(sample['signal'].shape)}  (window {WINDOW})")
    labels = sample["labels"]
    print(f"  record        {sample['record_id']}")
    print(f"  patient       {labels['patient_id']}")
    print(f"  n_leads       {labels['n_leads']}   <- 9 or 12, per record")
    print(f"  age           {labels['age_days']:.0f} days ({labels['age_years']:.2f} years)")
    print(f"  sex           {labels['sex']}")
    print(f"  duration      {labels['duration_seconds']:.1f} s ({labels['n_samples']} samples)")
    print(f"  aha_codes     {labels['aha_codes']}")
    print(f"  ecg_findings  {labels['ecg_findings']}")
    print(f"  icd10_codes   {labels['icd10_codes']}")
    print(f"  disease_grps  {labels['disease_groups'] or '(none of the 19 studied)'}")
    print(f"  primary_grp   {labels['primary_disease_group']}   <- folds only, never train on it")
    print(f"  quality       pSQI {labels['psqi_mean']:.3f}  basSQI {labels['bassqi_mean']:.3f}")

    # Use labels_df for split statistics — iterating the Dataset decodes signals.
    df = dataset.labels_df

    print("\nTHE LEAD TRAP — two layouts in one dataset:")
    n_nine = int((df["n_leads"] == 9).sum())
    print(
        f"  9-lead records in this split: {n_nine} of {len(df)} " f"({100 * n_nine / len(df):.1f}%)"
    )
    print(f"  12-lead layout position 7 is {config.lead_names[7]}")
    print(f"  9-lead  layout position 7 is {config.alternate_lead_names[9][7]}")
    print("  So an index-based selection silently crosses those two leads.")

    safe = ECGDataset(
        "zzu_pecg",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=SAFE_LEADS,
    )
    print(
        f"\n  leads={SAFE_LEADS} (in both layouts) -> "
        f"{tuple(safe[0]['signal'].shape)} for every record"
    )

    risky = ECGDataset(
        "zzu_pecg",
        split=args.split,
        version=args.version,
        data_path=args.data_path,
        metadata_source=args.metadata_source,
        window=WINDOW,
        leads=[REDUCED_ONLY_LEAD],
    )
    # labels_df is aligned row-for-row with the dataset, so a position in one is
    # the same record in the other.
    n_leads = list(df["n_leads"])
    twelve_at = n_leads.index(12)
    nine_at = n_leads.index(9)

    print(f"\n  leads=['{REDUCED_ONLY_LEAD}'] on a 12-lead record: ", end="")
    print(f"{tuple(risky[twelve_at]['signal'].shape)}  (fine)")
    print(f"  leads=['{REDUCED_ONLY_LEAD}'] on a 9-lead record:  ", end="")
    try:
        risky[nine_at]
        print("returned a signal — THIS WOULD BE THE BUG")
    except ValueError as e:
        print(f"raises: {str(e).split('.')[0]}.")
    print("  Filter on n_leads if you need all 12; do not silently accept 9.")

    print("\nRecord length varies, which is why window= is not optional:")
    dur = pd.to_numeric(df["duration_seconds"])
    print(
        f"  {dur.min():.0f}-{dur.max():.0f} s, median {dur.median():.0f} s, "
        f"{dur.nunique()} distinct lengths"
    )
    print(f"  WINDOW={WINDOW} fits the shortest ({dur.min():.0f} s); anything wider raises")
    print("  WindowOutOfRangeError naming the record and its true length.")

    print("\nECG findings are multi-label (AHA vocabulary, normalised):")
    counts = Counter(c for s in df["aha_codes"].fillna("") for c in str(s).split(",") if c)
    for code, n in counts.most_common(8):
        print(f"  {code:34s} {n:5d}")
    print(
        f"  {len(counts)} distinct codes; {int(df['n_findings'].sum())} finding-record "
        f"pairs over {len(df)} records (max {int(df['n_findings'].max())} on one record)"
    )
    prose = [c for c in counts if not c[:1].isupper() or " " in c]
    print(f"  {len(prose)} of those 'codes' are prose, because the AHA vocabulary has no")
    print(f"  code for them, e.g. {prose[:2]}")

    print("\nICD-10 disease groups (the diagnosis axis, from the discharge record):")
    diagnosed = df["n_disease_groups"] > 0
    print(
        f"  records with one of the 19 studied codes: {int(diagnosed.sum())} "
        f"({100 * diagnosed.mean():.1f}%)"
    )
    for name, n in df["primary_disease_group"].value_counts().items():
        print(f"  {name:28s} {n:5d}")
    print("  These are ADMISSION diagnoses, so a record can carry one whose ECG")
    print("  signature is absent from that particular tracing.")

    print("\nReduced-lead records are NOT a neutral subset to drop:")
    rate_9 = df.loc[df["n_leads"] == 9, "n_disease_groups"].gt(0).mean()
    rate_12 = df.loc[df["n_leads"] == 12, "n_disease_groups"].gt(0).mean()
    print(f"  diagnosed among 9-lead:  {100 * rate_9:.1f}%")
    print(f"  diagnosed among 12-lead: {100 * rate_12:.1f}%")
    print("  Dropping the 9-lead records preferentially discards sick children.")

    age = pd.to_numeric(df["age_days"])
    print(
        f"\nAge in DAYS: {age.min():.0f}-{age.max():.0f} "
        f"(median {age.median():.0f} = {age.median() / 365.25:.1f} years)"
    )
    print(f"  under 1 year: {int((age < 365).sum())} records")
    print(f"  Sex: {df['sex'].value_counts().to_dict()}")
    print("  Use age_days, not age_years: rounding collapses the whole infant range.")

    # Multi-hot target over the commonest findings — the release's ECG task.
    top = [c for c, _ in counts.most_common(10)]
    code_lists = df["aha_codes"].fillna("").astype(str).str.split(",")
    targets = pd.DataFrame(
        {c: code_lists.apply(lambda lst, c=c: int(c in lst)) for c in top}, index=df.index
    )
    print(f"\nMulti-hot target over the 10 commonest findings: {targets.shape}")
    print(f"  positives per record: mean {targets.sum(axis=1).mean():.2f}")

    # BATCHING NEEDS BOTH ADAPTERS HERE. window= makes the sample count uniform
    # and leads= makes the lead count uniform; ecg_collate_fn stacks signals with
    # torch's default_collate, which requires identical shapes. Without leads= a
    # batch that happens to mix a 9-lead and a 12-lead record raises RuntimeError.
    print("\nBatching this dataset needs leads= as well as window=:")
    try:
        DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn
        ).__iter__().__next__()
        print("  a mixed-lead batch stacked — only because this batch happened to be uniform")
    except RuntimeError as e:
        print(f"  without leads=: RuntimeError ({str(e).splitlines()[0][:70]}...)")
        print("  because the batch mixes 9-lead and 12-lead records.")

    loader = DataLoader(safe, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn)
    batch = next(iter(loader))
    print(f"  with leads={SAFE_LEADS}: signal {tuple(batch['signal'].shape)}")
    print("  Selecting leads by name is what makes the lead dimension uniform, and it")
    print("  is safe across both layouts. window= does the same for the sample axis.")

    print("\nNote on the records the clean version drops: 2,000 fail amplitude_outlier.")
    print("There is a hard rail at about 26.6 mV that 11.8% of records touch, with")
    print("sustained rather than spike excursions, and the release's own basSQI agrees")
    print("they are the poor-quality ones (median 0.961 against 0.983). Use")
    print("version='original' to get them back, with is_valid and quality_issues.")


if __name__ == "__main__":
    main()
