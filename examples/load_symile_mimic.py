#!/usr/bin/env python3
"""
Example: Symile-MIMIC — a multimodal cohort over MIMIC-IV-ECG's records.

Symile-MIMIC pairs 11,622 hospital admissions with a chest X-ray, an ECG and up to
50 blood labs, all drawn from MIMIC-IV. **The ECGs are MIMIC-IV-ECG's own
recordings** — all 11,610 distinct `ecg_study_id`s are MIMIC-IV-ECG `study_id`s,
and `ecg_path` matches `record_list.csv` in 100% of rows. So there is deliberately
no `symile_mimic` config and no `ecgbench splits --dataset symile_mimic`: a second
ten-fold partition over records `mimic_iv_ecg` already partitions would let someone
train on one split and evaluate on the other without noticing.

Instead you load MIMIC-IV-ECG on ECGBench's folds and join this cohort onto it,
which is what this script does.

Five things worth seeing in the output:

1. **The row unit is the admission, not the ECG.** 12 ECG studies serve two
   admissions each, so `by_study_id()` makes the de-duplication policy explicit.
2. **The column named `study_id` is the CXR's, not the ECG's** — a duplicate of
   `cxr_study_id`. Joining MIMIC-IV-ECG on it matches nothing, so `load_cohort()`
   drops it and leaves the real key under `ecg_study_id`.
3. **CheXpert labels have four states, not two**: 1.0/0.0/-1.0 uncertain/NaN not
   mentioned. `chexpert_targets()` makes you resolve the last two yourself.
4. **The release's train/val/test split is not ECGBench's**, and the two are
   independent — 75.6% of the test studies sit in ECGBench's train split.
5. **The shipped ECG tensors are not millivolts** and cannot be converted back:
   each is min-max normalised to [-1, 1] with the min and max discarded.

Prerequisites:
  - pip install ecgbench[torch]
  - Local copies of BOTH datasets, each credentialed on PhysioNet. Symile-MIMIC has
    no ECGBench folds; MIMIC-IV-ECG has none of these columns.
  - ECGBench's fold CSVs for `mimic_iv_ecg` are never published to the Hub, so
    generate them and copy them into the MIMIC-IV-ECG root first:

      ecgbench splits --dataset mimic_iv_ecg --data-path /path/to/mimic-iv-ecg/1.0/
      cp -r output/mimic_iv_ecg/clean output/mimic_iv_ecg/original \\
            /path/to/mimic-iv-ecg/1.0/

Usage:
  python examples/load_symile_mimic.py \\
      --mimic-path  /path/to/mimic-iv-ecg/1.0/ \\
      --symile-path /path/to/symile-mimic/1.0.0/
"""

import argparse

from ecgbench import ECGDataset
from ecgbench.labels import LabelSourceMissingError
from ecgbench.labels.symile_mimic import (
    CHEXPERT_LABELS,
    ECG_LEAD_NAMES,
    RETRIEVAL_CANDIDATES,
    SPLIT_ADMISSIONS,
    SPLIT_CHEXPERT_LABELS,
    SPLIT_CSVS,
    as_leads_first,
    by_study_id,
    chexpert_targets,
    labs_frame,
    load_cohort,
    load_split,
    load_split_tensors,
    retrieval_queries,
)

STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    parser = argparse.ArgumentParser(description="Join Symile-MIMIC onto MIMIC-IV-ECG's folds")
    parser.add_argument("--mimic-path", required=True, help="MIMIC-IV-ECG root (waveforms)")
    parser.add_argument("--symile-path", required=True, help="Symile-MIMIC release root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--fold", type=int, default=1, help="one ECGBench fold; 800k records total")
    parser.add_argument(
        "--tensors", action="store_true",
        help="also read a shipped ECG tensor (needs data_npy/, 33 GB in full)",
    )
    args = parser.parse_args()

    print("Symile-MIMIC is a cohort over MIMIC-IV-ECG's records, so it is consumed")
    print("through MIMIC-IV-ECG's splits. There is no symile_mimic config.\n")

    # metadata_source="local": MIMIC-IV-ECG's fold CSVs are credentialed and never
    # published to the Hub, so they are the ones you generated yourself.
    ds = ECGDataset(
        "mimic_iv_ecg",
        split=args.split,
        data_path=args.mimic_path,
        metadata_source="local",
        fold_numbers=[args.fold],
        leads=STANDARD_12,
    )
    print(f"MIMIC-IV-ECG {args.split} fold {args.fold}: {len(ds):,} records")

    try:
        # prefix= because MIMIC-IV-ECG's own label frame also carries subject_id
        # and ecg_time; an unprefixed join would silently keep one of each.
        cohort = load_cohort(args.symile_path, prefix="sym_")
    except LabelSourceMissingError as e:
        print(f"Symile-MIMIC unavailable: {e}")
        return

    print(f"Symile-MIMIC cohort:         {cohort.shape[0]:,} admissions x "
          f"{cohort.shape[1]} columns")
    print(f"  indexed by {cohort.index.name!r} — the admission, the release's row unit")

    print("\nThe join key is not the column called 'study_id':")
    print("  'sym_study_id' is absent — it was the CXR's id, a duplicate of cxr_study_id")
    print(f"  present instead: 'sym_ecg_study_id' -> {'sym_ecg_study_id' in cohort.columns}")

    keyed = by_study_id(cohort, prefix="sym_")
    print(f"\nby_study_id():               {len(keyed):,} rows, index {keyed.index.name!r}")
    print(f"  {len(cohort) - len(keyed)} row(s) dropped: 12 ECG studies each served two")
    print("  admissions of the same patient, hours apart. Default policy keeps the")
    print("  earliest admittime; pass on_duplicate='raise' to refuse instead.")

    study_ids = ds.metadata_df["study_id"]
    joined = keyed.reindex(study_ids.values)
    matched = int(joined.notna().any(axis=1).sum())
    print(
        f"\njoined onto the fold:        {matched:,} of {len(study_ids):,} records "
        f"({100 * matched / len(study_ids):.2f}%)"
    )
    print("  A low match rate is expected and correct: Symile-MIMIC is a 11,610-record")
    print("  cohort carved out of MIMIC-IV-ECG's 800,035, not a label layer over all")
    print("  of it. Only admissions with a CXR, an ECG and a lab qualified.")

    matched_rows = joined[joined.notna().any(axis=1)]

    print("\nThe 50 blood labs, for the matched records:")
    values = labs_frame(matched_rows, "value", names=True, prefix="sym_")
    per_record = values.notna().sum(axis=1)
    print(f"  {values.shape[1]} labs; mean {per_record.mean():.1f} measured per admission "
          f"(min {per_record.min()}, max {per_record.max()})")
    coverage = values.notna().mean().sort_values(ascending=False)
    def pct(items):
        return ", ".join(f"{n} {100 * v:.0f}%" for n, v in items)

    print(f"  best covered:  {pct(coverage.head(3).items())}")
    print(f"  worst covered: {pct(coverage.tail(3).items())}")
    print("  Missing labs are genuine NaNs — no sentinel rails, unlike MIMIC-IV-ECG's")
    print("  machine_measurements.csv — so notna() is the right test.")

    print("\nCheXpert findings — four states, and you resolve two of them:")
    print(f"  the cohort table carries all {len(CHEXPERT_LABELS)}; the split CSVs only "
          f"{len(SPLIT_CHEXPERT_LABELS)}")
    default = chexpert_targets(matched_rows, prefix="sym_")
    keep = chexpert_targets(matched_rows, uncertain="keep", not_mentioned="nan", prefix="sym_")
    print(f"  {'finding':28s} {'1.0':>6} {'0.0':>6} {'-1.0':>6} {'NaN':>6}"
          "   -> default pos/neg/nan")
    for name in ("Atelectasis", "Cardiomegaly", "No Finding"):
        raw, res = keep[name], default[name]
        print(f"  {name:28s} {int(raw.eq(1).sum()):>6} {int(raw.eq(0).sum()):>6} "
              f"{int(raw.eq(-1).sum()):>6} {int(raw.isna().sum()):>6}   -> "
              f"{int(res.eq(1).sum())}/{int(res.eq(0).sum())}/{int(res.isna().sum())}")
    print("  Defaults are uncertain='nan' (ignore the label) and")
    print("  not_mentioned='negative'. -1.0 stays out of the negatives either way.")

    row = matched_rows.iloc[0]
    print(f"\nOne record — study_id {matched_rows.index[0]}:")
    # reindex over 800k study_ids leaves NaNs behind, so the int columns of the
    # joined frame are float64. Cast on the way out.
    print(f"  admission (hadm_id)   {int(row['sym_hadm_id'])}")
    print(f"  patient / age / sex   {int(row['sym_subject_id'])} / "
          f"{int(row['sym_age'])} / {row['sym_gender']}")
    print(f"  ecg_time              {row['sym_ecg_time']}   (date-shifted: MIMIC spans 2110-2208)")
    print(f"  CXR                   {row['sym_cxr_ViewPosition']}, {row['sym_cxr_path']}")
    print(f"  labs measured         {int(values.iloc[0].notna().sum())} of 50")
    print(f"  signal from MIMIC     {tuple(ds[0]['signal'].shape)} in mV, leads reordered by name")

    print("\nThe release's own splits — a different partition from ECGBench's:")
    for split in SPLIT_CSVS:
        frame = load_split(args.symile_path, split)
        note = ""
        if len(frame) != SPLIT_ADMISSIONS[split]:
            note = (f"  = {SPLIT_ADMISSIONS[split]} queries x {RETRIEVAL_CANDIDATES} "
                    "retrieval candidates")
        print(f"  {split:14s} {len(frame):>6,} rows{note}")
    print("  10,000 + 750 + 464 = 11,214 of 11,622 admissions; the other 408 were")
    print("  discarded by the release's patient-disjointness filter.")

    test = load_split(args.symile_path, "test")
    queries = retrieval_queries(test)
    print(f"\nThe CXR-retrieval task: {len(queries)} queries, "
          f"{RETRIEVAL_CANDIDATES} candidates each")
    print("  Candidates are drawn from the split itself, so every test admission is")
    print("  both a query and a negative for others — the negatives add pairings, not records.")

    # by_study_id consumes ecg_study_id into the index, so the study ids are .index.
    test_studies = set(by_study_id(test, on_duplicate="keep_all").index)
    in_fold = sum(1 for sid in study_ids if sid in test_studies)
    print(f"\n{in_fold} of this ECGBench {args.split} fold's {len(study_ids):,} records are")
    print("  Symile-MIMIC *test* records. Across the whole partition, 75.6% of the 464")
    print("  test studies sit in ECGBench's train split and 349 of 461 test subjects")
    print("  appear there. Pick one partition and stay inside it.")

    if args.tensors:
        ecg, hadm = load_split_tensors(args.symile_path, "test", "ecg")
        leads_first = as_leads_first(ecg[:1])
        print(f"\nShipped ECG tensor:          {ecg.shape} float32, aligned to "
              f"{len(hadm)} hadm_ids")
        print(f"  as_leads_first ->           {leads_first.shape}, ECGBench's orientation")
        print(f"  value range                 [{leads_first.min():.3f}, {leads_first.max():.3f}] "
              "— unitless, NOT millivolts")
        print("  Each record was min-max normalised over all 12 leads at once and the")
        print("  min and max were not shipped, so mV cannot be recovered. For real")
        print("  millivolts use ECGDataset('mimic_iv_ecg', ...) as above.")
        print(f"  lead order is MIMIC-IV-ECG's: {list(ECG_LEAD_NAMES)}")
        print("  note aVF at index 4, before aVL — not the conventional order.")


if __name__ == "__main__":
    main()
