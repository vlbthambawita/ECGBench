---
slug: "brno-university-of-technology-ecg-quality-database-but-qdb"
name: "Brno University of Technology ECG Quality Database (BUT QDB)"
category: "one-lead"
order: 7
status: "completed"
source_url: "https://physionet.org/content/butqdb/1.0.0/"
url_label: "physionet.org"
format: "1-lead (Bittium Faros 180, chest) + 3-axis accel. · 24.0–38.7 h · 1,000 Hz · WFDB · sample-by-sample quality labels"
patients: "15"
records: "18"
access: "open"
license: "CC BY 4.0"
origin_institution: "Brno University of Technology"
origin_country: "Czech Republic"
leads: 1
paper_title: "Nemcova et al., PhysioNet 2020 (method: Smital et al., IEEE TBME 2020)"
paper_doi: "https://doi.org/10.1109/tbme.2020.2969719"
search_keywords: "but qdb brno ecg quality signal quality assessment annotation noise artifact wearable czech republic bittium faros ambulatory free-living inter-annotator agreement consensus accelerometer one-lead single-lead physionet"

sections:
  - type: description
    title: "Overview"
    body: |
      **The only dataset in this catalogue whose ground truth is a label per
      sample.** Every other release answers "what is wrong with this patient"; BUT
      QDB answers "can this stretch of signal be analysed at all", and it answers it
      at 1 kHz over 99.4 hours of free-living wearable ECG.

      18 single-lead recordings, each longer than 24 hours, from 15 people wearing a
      Bittium Faros 180 on the chest between August 2018 and October 2019 while
      carrying out ordinary daily activities. 478.7 hours of ECG at 1,000 Hz, with
      3-axis accelerometry at 100 Hz alongside every recording. There is **no cardiac
      diagnosis anywhere in the release** — this is a quality-assessment benchmark.

      **Three ECG experts graded the signal independently, and the release ships all
      three opinions plus their consensus.** Three classes: class 1, every waveform
      (P, QRS, T) measurable; class 2, QRS reliably detectable but nothing finer;
      class 3, QRS not reliably detectable and the signal unusable for anything.

      **Only 20.8% of the recorded time is graded, and 88.6% of the graded time is
      three of the 18 records.** That is the first thing to plan around, and the
      second is where the graded segments *are* — see the next section.

      **`clean` and `original` are the same 18 records, deliberately.** Excluding
      recordings for being noisy would destroy exactly what this database exists to
      measure: the noisiest recording, `105001`, is the single most valuable one.

  - type: description
    title: "The trap: `window=(0, n)` has no labels behind it for 15 of the 18 records"
    body: |
      Three recordings — `100001`, `105001`, `111001` — are graded end to end. The
      other 15 get **two 20-minute segments each**, plus five extra segments the
      authors picked for being noisy (four of 20 minutes, one of 2 minutes).

      **Those two standard segments sit at exactly the same offsets in all 15
      records**, which the release does not mention and which was recomputed from the
      annotation files:

      | Segment | Samples | Wall-clock into the recording |
      |---|---|---|
      | 1 | `28,800,000` – `30,000,000` | 8 h 00 m – 8 h 20 m |
      | 2 | `57,600,000` – `58,800,000` | 16 h 00 m – 16 h 20 m |

      So the obvious first thing to try returns unlabelled signal:

      ```python
      from ecgbench.labels.butqdb import quality_vector
      quality_vector(path, "100002", start=0,         length=10_000)  # all 0 = ungraded
      quality_vector(path, "100002", start=28_800_000, length=10_000)  # classes 1, 2
      ```

      `annotated_blocks(path, record_id)` returns the bounds per record, and the five
      extra segments are elsewhere: `113001` at 36,119,999; `114001` at 11,214,750
      (the 2-minute one, which is 120,001 samples, not 120,000) and 11,674,750;
      `124001` at 33,699,999 and 65,099,999.

  - type: code
    title: "The actual ground truth: one class per sample, aligned to `window=`"
    language: python
    body: |
      from ecgbench import ECGDataset
      from ecgbench.labels.butqdb import (
          annotated_blocks, load_quality_intervals, quality_vector,
      )

      PATH = "/path/to/butqdb/1.0.0/"

      # quality_vector takes the SAME (start, length) as ECGDataset(window=...),
      # so the label array lines up with the signal tensor sample for sample.
      ds = ECGDataset("butqdb", split="train", data_path=PATH,
                      window=(0, 10_000), labels=True)
      sample = ds[0]
      sample["signal"].shape                      # torch.Size([1, 10000])
      sample["record_id"]                         # 100001   <- int64, so str() it

      y = quality_vector(PATH, "100001", start=0, length=10_000)
      y.shape, y.dtype                            # (10000,), int8
      {int(k): int((y == k).sum()) for k in (0, 1, 2, 3)}
      # {0: 0, 1: 2953, 2: 7047, 3: 0}             <- 0 means NEVER ANNOTATED

      # Each expert individually, and they disagree — on this window expert 1 calls
      # all of it class 2 and expert 2 calls all of it class 1:
      quality_vector(PATH, "100001", annotator="expert_1", start=0, length=10_000)
      quality_vector(PATH, "100001", annotator="expert_2", start=0, length=10_000)

      # The interval table, 0-based and half-open (the file is 1-based inclusive):
      load_quality_intervals(PATH, "100001")      # start, end, quality_class, ...
      load_quality_intervals(PATH, "100001", annotator=None)   # all four at once

      # Where a window may legitimately go:
      annotated_blocks(PATH, "100002")            # 28800000-30000000, 57600000-58800000

  - type: table
    title: "The 18 recordings, recomputed"
    headers: ["Record", "Subject", "Session", "Sex", "Age", "Hours", "Graded (min)", "Blocks", "Class 1", "Class 2", "Class 3", "3 experts unanimous", "Fold"]
    rows:
      - ["`100001`", "100", "1", "F", "28", "24.19", "1451", "1", "68.6%", "31.1%", "**0.3%**", "56.4%", "2"]
      - ["`100002`", "100", "2", "F", "28", "24.10", "40", "2", "65.2%", "34.8%", "**0.0%**", "71.2%", "2"]
      - ["`103001`", "103", "1", "M", "21", "24.20", "40", "2", "72.4%", "27.2%", "**0.3%**", "77.4%", "1"]
      - ["`103002`", "103", "2", "M", "21", "24.02", "40", "2", "90.5%", "9.5%", "**0.0%**", "88.4%", "1"]
      - ["`103003`", "103", "3", "M", "21", "24.01", "40", "2", "84.9%", "15.1%", "**0.0%**", "82.2%", "1"]
      - ["`104001`", "104", "1", "F", "21", "24.22", "40", "2", "66.3%", "33.7%", "**0.0%**", "74.1%", "8"]
      - ["`105001`", "105", "1", "M", "22", "38.65", "2319", "1", "41.9%", "24.1%", "**34.0%**", "81.5%", "5"]
      - ["`111001`", "111", "1", "F", "28", "25.18", "1511", "1", "41.9%", "53.6%", "**4.4%**", "59.9%", "10"]
      - ["`113001`", "113", "1", "F", "23", "25.32", "60", "3", "66.0%", "31.2%", "**2.8%**", "72.2%", "6"]
      - ["`114001`", "114", "1", "F", "24", "25.46", "62", "4", "62.5%", "15.0%", "**22.6%**", "89.8%", "7"]
      - ["`115001`", "115", "1", "F", "83", "24.43", "40", "2", "68.3%", "31.7%", "**0.0%**", "71.1%", "6"]
      - ["`118001`", "118", "1", "F", "48", "24.75", "40", "2", "63.5%", "36.5%", "**0.0%**", "77.6%", "7"]
      - ["`121001`", "121", "1", "M", "37", "25.34", "40", "2", "84.0%", "16.0%", "**0.0%**", "59.0%", "3"]
      - ["`122001`", "122", "1", "F", "59", "34.11", "40", "2", "33.9%", "16.1%", "**50.0%**", "95.2%", "3"]
      - ["`123001`", "123", "1", "F", "43", "37.07", "40", "2", "90.3%", "9.4%", "**0.4%**", "68.6%", "4"]
      - ["`124001`", "124", "1", "M", "44", "24.02", "80", "4", "28.8%", "56.9%", "**14.3%**", "53.1%", "4"]
      - ["`125001`", "125", "1", "M", "70", "24.01", "40", "2", "97.9%", "1.9%", "**0.1%**", "97.6%", "5"]
      - ["`126001`", "126", "1", "M", "58", "25.66", "40", "2", "98.6%", "1.4%", "**0.0%**", "96.2%", "9"]

  - type: description
    title: "The class mix is deliberately worse than free-living ECG actually is"
    body: |
      Over the 99.4 graded hours, by the expert consensus:

      | Class | Share of graded time | Hours |
      |---|---|---|
      | 1 — all waveforms measurable | 51.5% | 51.15 |
      | 2 — QRS detectable, nothing finer | 33.3% | 33.12 |
      | 3 — **unusable** | 15.2% | 15.12 |

      Two reasons not to read 15.2% as a property of 24-hour wearable ECG. **Five of
      the graded segments were "subjectively selected"** to raise the proportion of
      poor signal. And **87.0% of all class-3 time is one record**: `105001` alone
      contributes 13.15 of the 15.12 unusable hours.

      Only six records carry an appreciable class-3 burden at all — `122001` 50.0%,
      `105001` 34.0%, `114001` 22.6%, `124001` 14.3%, `111001` 4.4%, `113001` 2.8% —
      and the other twelve sit between 0.00% and 0.38%.

  - type: description
    title: "The experts disagree a great deal, and that is the ceiling on any result"
    body: |
      All three agree on **248,410,246 of the 357,799,001 graded samples — 69.4%**.
      Pairwise agreement averaged over the 18 records is 0.82 (experts 1–2), 0.81
      (1–3) and 0.89 (2–3). **Expert 1 is systematically stricter**, calling class 2
      where the other two call class 1: on `121001` expert 1 gives 51.8% class 1
      against expert 2's 90.4%.

      Anything reported as accuracy against "the" label is reporting agreement with a
      consensus that three humans reached this loosely. Per-record and per-expert
      figures are in the labels (`expert_1_class1_fraction` …
      `expert_unanimous_fraction`, `mean_expert_agreement`), so the ceiling is
      visible rather than implicit. Records range from 53.1% unanimous (`124001`) to
      97.6% (`125001`).

      **The fourth column triple is the consensus, and it behaves as a majority
      vote.** The release states the annotation file holds "3 columns × 3 annotators +
      consensus" but not how the consensus was formed. Measured over every graded
      sample: a majority of the three exists at 99.863% of them, the consensus equals
      that majority at all but **3,103** of those samples (99.99913%), and it differs
      from every expert at only **378** samples in the whole release. The residuals
      sit at interval boundaries — what one expects from a consensus drawn segment by
      segment. `consensus_matches_majority` records it per record, so a re-release
      formed by a different rule would show up rather than being assumed away.

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the shipped files, after
      verifying the local copy against the release's own `SHA256SUMS.txt` — **all 95
      listed files match**.

      | Figure | Published | Recomputed | Note |
      |---|---|---|---|
      | recordings | 18 | **18** | — |
      | subjects | 15 (9 F, 6 M) | **15 (9 F, 6 M)** | — |
      | records by sex | 10 F / 8 M | **10 F / 8 M** | — |
      | sessions per subject | 13×1, one woman ×2, one man ×3 | **13×1, `100`×2, `103`×3** | — |
      | minimum length | "24 hours" | **24.006 h** (`103003`) | longest is 38.65 h (`105001`) |
      | age range | 21 – 83 | **21 – 83** | — |
      | mean / median age | 41 / 37 | **40.6 / 37 per subject** | 37.7 / 28 per *record* — see below |
      | fully annotated | 3 signals | **3** (`100001`, `105001`, `111001`) | — |
      | extra noisy segments | 5 (four ×20 min, one ×2 min) | **5** (`113001`, `114001`×2, `124001`×2) | the 2-min one is 120,001 samples |

      **The release's own summary mixes two denominators in one sentence.** "Broadly
      balanced in terms of gender (10 female records and 8 male records) and age (21
      to 83 years, mean 41 years, median 37 years)" — the gender figures are per
      *record* and the age figures are per *subject*. Per record the cohort is much
      younger: mean 37.7, **median 28**, because the two repeat subjects are 28 and
      21 and contribute five of the 18 records. Neither figure is wrong; quoting them
      together is.

      **The age distribution is bimodal, not broad.** Eight subjects are 21–28 and
      seven are 37–83, with nobody between 28 and 37, so any age effect estimated
      here is estimated across a gap. Three subjects smoke; height 153–184 cm, weight
      43–85 kg, BMI 17.4–29.4.

  - type: description
    title: "Every record saturates the converter, so `amplitude_outlier` is a no-op"
    body: |
      Gain and baseline differ **per record** — 0.99998 to 1.996 ADC units per µV,
      baseline −18,289 to +11,462 — so each record has a different physical span:
      `100001` covers [−10.30, +22.53] mV, `124001` [−32.77, +15.78], `104001`
      [−32.77, +32.77]. **All 18 records attain both of their 16-bit rails**, so the
      configured `amplitude_range_mv` has to be their union, `[-32.769, 32.769]`, and
      the check cannot exclude anything.

      The bounds are *attained*, so the float32 rounding trap applies: what the
      loader actually produces is −32.76765823364258 mV and +32.768115997314453 mV, so
      a bound of 32.768 would have excluded the very record it was computed from.

      Saturation is much milder than in `toliet-thigh-based-ecg-toilet-seat`: 66,166
      samples across the release sit at a rail, the worst record being `114001` at
      0.015% of its samples and `100001` at two samples out of 87 million. No check in
      `CHECK_REGISTRY` measures it, so `clipped_fraction`, `min_mv` and `max_mv` are in
      the labels.

      **WFDB's invalid-sample marker does not occur.** Format 16 reserves −32768,
      which `wfdb.rdrecord` turns into NaN — and `nan_values` fails a record on a
      single NaN. There is not one anywhere in the release, so `nan_values` passes on
      all 18 records rather than passing by luck; `n_invalid_samples` reports it so a
      re-release that introduces the marker is caught.

  - type: table
    title: "Validation summary (1000 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "18", "all records, with is_valid + quality_issues"]
      - ["clean", "18", "100% pass rate — nothing excluded, by design"]

  - type: description
    title: "Nothing fails a check, and that is the correct outcome here"
    body: |
      No record has a NaN sample, an unreadable header, a flat lead or a truncated
      signal, and `amplitude_outlier` cannot fire (above). So `clean` is all 18
      records and `original` is the same 18 with `is_valid` and `quality_issues`
      attached.

      That is not a validation gap — it is the only defensible setting for this
      dataset. ECGBench's checks exist to exclude recordings that cannot be analysed;
      here, *which stretches cannot be analysed* is the ground truth three experts
      were paid to produce. Use the annotations, not `is_valid`:

      ```python
      usable = ds.labels_df[ds.labels_df["consensus_class3_fraction"] < 0.05]
      ```

      `expected_samples` is deliberately empty in the config: every record is a
      different length, 86,420,000 to 139,147,000 samples, and each is a complete
      recording that ran until the battery or the subject stopped.

  - type: description
    title: "Ten folds, grouped on subject and balanced on unusable signal"
    body: |
      Folds are built with `StratifiedGroupKFold`, grouped on `subject_id` and
      stratified on `stratify_class` — whether more than 1% of the record's graded
      time is class 3.

      | Class | Records | Subjects |
      |---|---|---|
      | `class3_low` | 12 | 9 |
      | `class3_high` | 6 | 6 |

      **Why class-3 burden and not sex.** What the folds have to balance is unusable
      signal, because rejecting unusable signal is the task. Only six records carry an
      appreciable class-3 burden, so **at most six of the ten folds can hold one — and
      this axis achieves exactly six**, the arithmetic maximum. Measured over the
      shipped files at `random_state=42`:

      | Stratified on | Classes | Folds holding a class-3 record |
      |---|---|---|
      | **class-3 burden** | 12/6 records, 9/6 subjects | **6 of 10** |
      | sex | 10 F / 8 M records, 9/6 subjects | 5 of 10 |
      | age < 45 | 13/5 records, 10/5 subjects | 6 of 10 |
      | dominant consensus class | 15/2/1 records | 5 of 10 |
      | annotated fully or not | 3/15 records | 5 of 10 |
      | sex × class-3 burden | every class < 10 records | raises |

      Sex is what `mit-bih-normal-sinus-rhythm-database` and
      `bidmc-congestive-heart-failure-database` use, and it is not better here: 15
      subjects over 10 folds put one or two records in a fold, so the female fraction
      per fold is 0, 0.5 or 1 whatever it is stratified on. **The threshold is not
      tuned** — twelve records sit at 0.00%–0.38% class 3 and six at 2.8%–50.0%, so
      any cut in between gives the same partition. Every cross of two axes raises,
      because `StratifiedGroupKFold` needs at least one class holding `n_folds`
      records and no cross of 18 records has one.

      | Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | records | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
      | subjects | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
      | graded minutes | 120 | 1491 | 80 | 120 | 2359 | 100 | 102 | 40 | 40 | 1511 |
      | class-3 minutes | 0.1 | 3.7 | 20.0 | 11.6 | 789.2 | 1.7 | 14.0 | 0.0 | 0.0 | 67.0 |
      | `class3_high` records | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
      | fully graded records | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |

      No subject spans a fold: `103`'s three recordings are all in fold 1 and `100`'s
      two are both in fold 2. The three fully-graded records land in three different
      folds (2, 5, 10) without annotation depth being in the stratification at all.

      **Use the folds, not the default split.** With folds 1–8 → train, 9 → val,
      10 → test:

      | Split | Records | Subjects | Hours | Graded hours | F / M | `class3_high` |
      |---|---|---|---|---|---|---|
      | train | 16 | 13 | 427.9 | 73.5 | 9 / 7 | 5 |
      | val | 1 | 1 | 25.7 | 0.7 | 0 / 1 | 0 |
      | test | 1 | 1 | 25.2 | 25.2 | 1 / 0 | 1 |

      One record is not an evaluation set. That is the arithmetic of 18 recordings,
      not a defect in the partition — for a real evaluation, cross-validate:
      `split=None` with `fold_numbers=[...]` selects by fold from `folds.csv` and
      ignores the default layout.

  - type: description
    title: "The accelerometer ships but gets no records"
    body: |
      Every recording has a companion WFDB record `<id>_ACC`: three channels
      (`ACCx`, `ACCy`, `ACCz`) at **100 Hz in milli-g**. The release's own usage notes
      propose motion as an input to quality assessment, either alongside the ECG or on
      its own.

      ECGBench gives it no records and does not declare 100 Hz as a sampling rate — it
      is not an ECG, and declaring it would make ECGBench offer to load accelerometry
      as though it were an ECG lead. `acc_path` in the labels points at each one, and
      `wfdb.rdrecord` reads them directly.

  - type: description
    title: "Overlap with other datasets in this catalogue: none"
    body: |
      No `related:` edge is declared. This is an original 2018–2019 Brno recording
      campaign on a Bittium Faros 180 under free-living conditions, and no other
      release in the catalogue shares a subject, a recording or an institution with
      it — Brno University of Technology appears nowhere else here.

      The nearest neighbours are by *purpose* rather than content.
      `mit-bih-normal-sinus-rhythm-database` and the other MIT-BIH Holters carry `~`
      signal-quality annotations, but those mark noise transitions in a two-lead
      clinical Holter recording, not a graded three-class scale from three
      independent experts on a consumer wearable. Nothing else in the catalogue
      offers per-sample quality labels at all.

      The overlap that matters is **inside** this release: two subjects contributed
      more than one 24-hour recording (`100` twice, `103` three times), which is why
      folds group on `subject_id`.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset butqdb --data-path /path/to/butqdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      ds = ECGDataset(
          "butqdb",
          split="train",
          data_path="/path/to/butqdb/1.0.0/",
          labels=True,
          # MANDATORY in practice: records are 86.4M-139.1M samples, so one record is
          # 346-556 MB of float32. window= is pushed into wfdb's sampfrom/sampto, so
          # only these 10 s are decoded.
          window=(0, 10_000),
      )

      len(ds)                                     # 16
      ds[0]["signal"].shape                       # torch.Size([1, 10000])
      ds[0]["record_id"]                          # 100001   <- int64, so str() it
      ds.lead_names                               # ('ECG',)  <- one channel, no anatomy
                                                  # stated anywhere in the release
      ds[0]["labels"]["subject_id"]               # '100'
      ds[0]["labels"]["session_index"]            # 1
      ds[0]["labels"]["sex"]                      # 'F'
      ds[0]["labels"]["age"]                      # 28
      ds[0]["labels"]["duration_secs"]            # 87087.0     (24.19 h)
      ds[0]["labels"]["annotated_secs"]           # 87087.0     <- graded end to end
      ds[0]["labels"]["fully_annotated"]          # True        <- true for 3 of 18
      ds[0]["labels"]["n_annotated_blocks"]       # 1
      ds[0]["labels"]["consensus_class1_fraction"]  # 0.6864
      ds[0]["labels"]["consensus_class2_fraction"]  # 0.3111
      ds[0]["labels"]["consensus_class3_fraction"]  # 0.0025
      ds[0]["labels"]["expert_unanimous_fraction"]  # 0.5639  <- the annotators agreed
                                                    # on 56% of this record
      ds[0]["labels"]["clipped_fraction"]         # 0.0       <- CHECK THIS: every record
                                                  # reaches a rail, so amplitude_outlier
                                                  # cannot fire
      ds[0]["labels"]["min_mv"]                   # -10.304108   <- this record's rails;
      ds[0]["labels"]["max_mv"]                   #  22.528557   they differ per record
      ds[0]["labels"]["acc_path"]                 # '100001/100001_ACC'
      ds[0]["labels"]["dominant_consensus_class"] # 1   <- A REDUCTION. Do not train on
                                                  # it: the label is per sample.

      # The source is microvolts with a per-record gain; signal_unit_scale = 0.001.
      uv = ECGDataset("butqdb", split="train", window=(0, 10_000), units="uV",
                      data_path="/path/to/butqdb/1.0.0/")
      uv[0]["signal"].max()                       # tensor(22528.6)  vs 22.529 mV

      # All 18 records with the quality columns attached. Identical row set to
      # `clean` here — nothing is excluded, by design.
      everything = ECGDataset("butqdb", split="train", version="original",
                              window=(0, 10_000),
                              data_path="/path/to/butqdb/1.0.0/")
      len(everything)                             # 16

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/butqdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/kah4-0w24" }
      - { label: "Method paper (IEEE TBME, 2020)", url: "https://doi.org/10.1109/tbme.2020.2969719" }
---
