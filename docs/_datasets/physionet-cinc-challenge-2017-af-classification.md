---
slug: "physionet-cinc-challenge-2017-af-classification"
name: "PhysioNet/CinC Challenge 2017 (AF Classification)"
category: "one-lead"
order: 2
status: "completed"
source_url: "https://physionet.org/content/challenge-2017/1.0.0/"
url_label: "physionet.org"
format: "1-lead (ECG) · 9.05–60.95 s · 300 Hz · WFDB"
patients: "—"
records: "8,528"
access: "open"
license: "ODC-By 1.0"
origin_institution: "AliveCor Inc.; Emory University; MIT"
origin_country: "USA"
leads: 1
paper_title: "Clifford et al., CinC 2017"
paper_doi: "https://doi.org/10.22489/CinC.2017.065-469"
search_keywords: "cinc challenge 2017 af atrial fibrillation alivecor usa mit harvard emory single lead handheld consumer noisy signal quality relabelling four class"
patients_class: "count-na"

sections:
  - type: description
    title: "Overview"
    body: |
      **8,528 single-lead recordings from a consumer handheld device, each labelled
      normal / AF / other rhythm / too noisy to classify.** This is the reference
      benchmark for atrial fibrillation detection from *consumer* ECG rather than
      clinical 12-lead, and the only dataset in this catalogue whose fourth class
      is "unusable signal" — signal quality is part of the task, not a
      preprocessing step to get past.

      Every recording was taken by a member of the public who had bought one of
      three generations of AliveCor's single-channel device and held one electrode
      in each hand, giving a nominal **lead I (LA-RA) equivalent**. The device
      recorded for an average of 30 s, transmitted the trace acoustically to a
      phone over a 19 kHz carrier, and stored it at 300 Hz / 16 bits with a
      0.5–40 Hz passband.

      **The channel is called `ECG`, not `I`, and that is deliberate.** The paper
      states that "many of the ECGs were inverted (RA-LA) since the device did not
      require the user to rotate it in any particular orientation", and no record
      says which. ECGBench declares the source's own channel name so nobody
      stacks this with 12-lead lead I by name while an unknown fraction of the
      traces carry the opposite sign.

      **Record length varies and is not a nuisance variable.** 2,714 to 18,286
      samples — 9.05 s to 60.95 s, in **1,487 distinct lengths** — so records
      cannot be batched without `window=`, and the window has to fit the shortest.
      Length also correlates with the label: noisy recordings average 24.4 s
      against 32–34 s for the other three classes. A model fed whole records can
      learn duration instead of rhythm.

      **This is the public training set only.** The challenge used 12,186
      recordings; the other 3,658 are the hidden test set, which was never
      released. Nothing here reproduces the challenge's own scoring split — see
      "The shipped `validation/` directory is not a split" below, which is the trap
      that follows from that.

  - type: table
    title: "The four classes, and what relabelling did to them"
    headers: ["Class", "Code", "Shipped v0", "Shipped v3 (the label)", "Diff", "Share of v3"]
    rows:
      - ["normal", "`N`", "5,154", "**5,076**", "−78", "59.52%"]
      - ["other rhythm", "`O`", "2,557", "**2,415**", "−142", "28.32%"]
      - ["atrial fibrillation", "`A`", "771", "**758**", "−13", "8.89%"]
      - ["too noisy to classify", "`~`", "46", "**279**", "**+233**", "3.27%"]
      - ["**total**", "", "**8,528**", "**8,528**", "0", "100%"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the shipped files, after
      verifying **all 17,066 files under `training/` against the release's own
      `MD5SUMS`** and all 610 under `validation/` against its `SHA256SUMS` — every
      one matches.

      **The catalogue previously said 12,186 records.** That is the figure the
      paper's abstract gives, and it counts the hidden test set: 8,528 public +
      3,658 private. Only the 8,528 were ever released, so that is what ECGBench
      partitions and what this page now states.

      **`patients` is `—` because no patient identifier exists**, and unlike most
      one-record-per-patient datasets this one cannot even assert that. The
      recordings came from members of the public who had bought a handheld device,
      and nothing in the release says whether anyone contributed more than one.
      See "Folds are stratified but ungrouped" below.

      The class table is **single-label** — every record carries exactly one of the
      four codes, none carries two, and none carries none — so the column sums to
      the record total, and the stratification label *is* the training target.
      That is unusual here; most ECGBench datasets need a reduction for folds that
      you must not train on.

  - type: description
    title: "Four label versions ship, and the file numbers are one behind the paper's"
    body: |
      The challenge relabelled its data twice mid-competition, using the entrants'
      own disagreement to find the records it trusted least: the whole dataset was
      ranked by how much the top-performing algorithms disagreed, and eight ECG
      experts independently relabelled the 1,129 most contentious recordings. Over
      those, Fleiss' κ was **0.245** — "fair agreement" — so label noise here is
      substantial and acknowledged by the organisers rather than hypothetical.

      Four `REFERENCE-v<N>.csv` files ship, and **their numbers do not match the
      paper's V1/V2/V3**. Matching the shipped training counts against the paper's
      Table 2 gives:

      | Shipped file | N / A / O / ~ | The paper calls it |
      |---|---|---|
      | `REFERENCE-v0.csv` | 5154 / 771 / 2557 / 46 | **V1** — unofficial phase, Feb–Apr 2017 |
      | `REFERENCE-v1.csv` | 5040 / 736 / 2469 / 283 | *not tabulated in the paper* |
      | `REFERENCE-v2.csv` | 5050 / 738 / 2456 / 284 | **V2** — official phase, Apr–Sep 2017 |
      | `REFERENCE-v3.csv` | 5076 / 758 / 2415 / 279 | **V3** — final scoring |

      So "v1" names a different file depending on whether you read the paper or
      the directory, and one shipped version has no name in the paper at all.
      `REFERENCE.csv` is byte-identical to `REFERENCE-v3.csv`, so **v3 is the
      label**. ECGBench names its columns after the *shipped* files —
      `class_code_v0` … `class_code_v3` — because those are what a reader can
      verify on disk.

      **412 of the 8,528 labels changed between v0 and v3**, and the movement is
      almost entirely into the noisy class:

      | v0 ↓ → v3 → | normal | AF | other | noisy |
      |---|---|---|---|---|
      | **normal** | 5,019 | 9 | 12 | **114** |
      | **AF** | 19 | 684 | 36 | **32** |
      | **other** | 38 | 65 | 2,367 | **87** |
      | **noisy** | 0 | 0 | 0 | 46 |

      233 records moved *into* `noisy` and **none moved out** — the second pass
      was a visual recheck for recordings too noisy to read, and it only ever
      added. `n_distinct_labels` counts how many labels a record was ever given:
      **8,104** records were never relabelled, **418** carry two labels across the
      four versions and **6** carry three. Filtering on it is the closest this
      release comes to an annotation-confidence measure.

  - type: description
    title: "The shipped validation/ directory is not a split"
    body: |
      The release contains a `validation/` directory of 300 records, and it is
      **not held-out data**. All 300 of its `.mat` files are **byte-identical** to
      their `training/` counterparts — checked, all 300 — and the paper says what
      it is for: "Validation was 300 records (3.5%) of training set just to ensure
      the algorithm produced the expected results."

      Using it as a test set means evaluating on training data. ECGBench therefore
      sets `has_predefined_splits: false`, lets those records take part in the
      ten folds like any other, and flags them as
      `in_challenge_validation_subset` so they can be **excluded** from a
      comparison against published challenge numbers:

      ```python
      ds.labels_df[~ds.labels_df.in_challenge_validation_subset]
      ```

      The real 3,658-record test set was never published, so the challenge's own
      scoring split cannot be reproduced from this release at all. Numbers
      computed on ECGBench folds are not comparable to the challenge leaderboard.

  - type: description
    title: "Folds are stratified but ungrouped, and that cannot be fixed here"
    body: |
      Every one of the 8,528 headers is exactly **two lines with no `#` comment of
      any kind** — no age, no sex, no subject id, no clinical text, nothing. So
      `patient_id_column` is null and the engine uses plain `StratifiedKFold`.

      This is worse than the usual "one record per patient, so grouping is moot".
      The contributors were members of the public using their own devices, and
      nothing in the release says whether one person recorded once or fifty times.
      **A repeat contributor would straddle folds undetectably**, and no
      information in the files could detect it. Treat the resulting generalisation
      estimate as an upper bound.

      The header timestamp is **de-identified rather than removed**, and is not a
      wall clock: the date is `1/<month>/2000` in every record — day always 1, year
      always 2000, only the month varying — and the hour field runs 0–12 for 8,378
      records and 21–23 for the other 150, which is not a 24-hour distribution.
      `header_timestamp` is exposed verbatim so nobody has to guess; do not read it
      as a recording time.

  - type: table
    title: "Validation summary (300 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "8,528", "all records, with is_valid + quality_issues"]
      - ["clean", "8,527", "99.99% pass rate — one record excluded"]

  - type: description
    title: "The one record that fails, and the two checks that cannot fire"
    body: |
      **`A04894` is the only exclusion**, for `amplitude_outlier`: its trace runs
      to **−10.636 mV**, just past the ±10 mV range. The device's stated dynamic
      range is ±5 mV and the extremes over all 8,528 records are −10.636 mV to
      +8.318 mV, so this is a genuine rail-hitting excursion on a handheld
      recording, not a units error. The range was left at the standard ±10 mV
      rather than widened to make that record disappear — which is what
      `amplitude_range_mv` is for. Nothing exceeds ±20 mV, so a mis-scaled copy
      (microvolts, or the gain dropped) would still be caught on its first record.

      Two other checks are worth explaining because they *cannot* fire here:

      - **`truncated_signal` is disabled**, by leaving `expected_samples` empty.
        There are 1,487 distinct record lengths between 2,714 and 18,286 samples,
        so no single threshold separates a truncated record from a short one, and
        any threshold would drop thousands of sound records.
      - **`nan_values` and `flat_line` never fire.** No record contains a NaN
        sample, and the lowest per-record standard deviation anywhere in the
        release is 0.032 mV — noisy, but never flat.

      All 8,528 records read, so `corrupt_header` is clean too. Fold membership is
      identical between `original/` and `clean/`; `clean/` is the same partition
      minus one row.

  - type: description
    title: "Overlap with other datasets: none"
    body: |
      No `related:` edge is declared, and the reason is structural rather than
      merely unverified. This is the only AliveCor single-lead cohort in the
      catalogue: the PhysioNet/CinC Challenge 2020 and 2021 releases that bundle
      other datasets are **12-lead** meta-datasets built from eight clinical
      cohorts (CPSC, PTB, PTB-XL, Georgia, Chapman-Shaoxing, Ningbo, St
      Petersburg INCART), none of which is this one, and no other release in the
      catalogue redistributes these recordings.

      The one genuine relative is the **3,658-record hidden test set**, which is
      not public and so cannot be an entry here.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset challenge2017 --data-path /path/to/challenge-2017/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Variable-length records: a window is needed to batch at all, and because
      # window= is pushed into the reader it also avoids decoding the rest.
      # 2,700 samples is 9 s at 300 Hz — it fits inside the SHORTEST record in the
      # release (A05493, 2,714 samples), so it fits all 8,528. Anything longer
      # raises WindowOutOfRangeError on the short tail.
      ds = ECGDataset(
          "challenge2017",
          split="train",
          data_path="/path/to/challenge-2017/1.0.0/",
          window=(0, 2700),
          labels=True,
      )

      len(ds)                                # 6823
      ds[0]["signal"].shape                  # torch.Size([1, 2700])
      ds[0]["record_id"]                     # 'A00001'
      ds.lead_names                           # ('ECG',) — the header's own name.
                                              # NOT 'I': the device gives a lead I
                                              # equivalent but does not enforce
                                              # orientation, so many traces are inverted.

      ds[0]["labels"]["class_code"]           # 'N'
      ds[0]["labels"]["class_name"]           # 'normal'
      ds[0]["labels"]["is_af"]                # False
      ds[0]["labels"]["duration_seconds"]     # 30.0 — the FULL record, not the window
      ds[0]["labels"]["n_distinct_labels"]    # 1 — never relabelled
      ds[0]["labels"]["header_timestamp"]     # '05:05:15 1/05/2000' — de-identified
      ds[0]["labels"]["in_challenge_validation_subset"]
                                              # True — record 0 happens to be one of the
                                              # 300 the release duplicates under validation/

      # The label IS the training target here — single-label, four classes, no
      # reduction involved:
      ds.labels_df["class_name"].value_counts()
      # normal 4061, other_rhythm 1932, atrial_fibrillation 606, noisy 224

      # Records the organisers' relabelling touched, i.e. the contentious ones:
      ds.labels_df["label_revised"].sum()     # 337 in this split (412 overall)

      # Drop the 239 records in this split that ship a duplicate copy under
      # validation/ before comparing with published challenge results:
      ds.labels_df[~ds.labels_df.in_challenge_validation_subset]   # 6584 records
---
