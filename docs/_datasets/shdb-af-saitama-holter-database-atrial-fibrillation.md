---
slug: "shdb-af-saitama-holter-database-atrial-fibrillation"
name: "SHDB-AF (Saitama Holter Database — Atrial Fibrillation)"
category: "two-lead"
order: 10
status: "completed"
source_url: "https://physionet.org/content/shdb-af/1.0.1/"
url_label: "physionet.org"
format: "2-lead (mod. CC5 + NASA) · 9-24 h · 200 Hz · WFDB"
patients: "122"
records: "128"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Saitama Medical University International Medical Center; Technion"
origin_country: "Japan"
leads: 2
paper_title: "Tsutsui et al., Scientific Data 2025"
paper_doi: "https://doi.org/10.13026/n6yq-fq90"
search_keywords: "shdb-af saitama heart database holter atrial fibrillation japan japanese cc5 nasa lead paroxysmal persistent af burden rhythm annotation physiozoo epltd pan-tompkins arnet2 rawecgnet generalization distribution shift clinical metadata echocardiography ablation comorbidity two-lead 200 hz"

sections:
  - type: description
    title: "Overview"
    body: |
      **The first AF database in this catalogue with a Japanese cohort, and it exists
      for that reason alone.** 128 twenty-four-hour two-lead Holter recordings from 122
      adults monitored at Saitama Medical University International Medical Center
      between May 2019 and May 2023 — **3,050.4 hours** of ECG. The authors built it to
      test whether AF detectors trained on MITDB, AFDB, LTAFDB, IRIDIA-AF, Icentia11k
      and CPSC2021 generalise to a population none of them contains, so it is
      constructed as a **held-out test set**, and a model tuned on it stops measuring
      the thing it was built to measure.

      **It is the only Holter release here where clinical context and time-resolved
      rhythm labels ship together.** `AdditionalData.csv` carries **45 columns per
      recording** — diagnosis, comorbidity, named medications, echocardiographic LAD
      and LVEF, up to two ablation procedures, pacemaker status — and 98 of the 128
      recordings carry rhythm marks placed beat by beat in PhysioZoo by a cardiology
      fellow. Most databases give you one or the other.

      **The annotation files use a layout nothing else in this catalogue uses, and
      ordinary WFDB code reads them as empty.** There is not one `+` rhythm change and
      not one typed beat symbol in the release. Every annotation is a `"` comment —
      10,349,733 of them — and the rhythm code sits in the `aux_note` of the one on the
      first beat of each interval. A reader filtering `symbol == "+"` finds zero
      episodes; a reader filtering `symbol in "NAVQ"` finds zero beats.

      **`(N` means "not annotated", not sinus rhythm.** The protocol covered
      supraventricular arrhythmia only — the release says so — so `N` pools sinus
      rhythm with ventricular ectopy, pauses and noise, 1,829.9 h of it. Do not train a
      sinus-rhythm detector on it.

      **Records 005 and 020 are the same recording**, with the same SHA-256 for `.dat`
      and `.qrs` in the release's own manifest, filed in the clinical table as two
      Holters three years apart. v1.0.1 withdrew 016 and 030 as duplicates and missed
      this pair, so the release holds **127 distinct recordings**. Its own section
      below.

      **A record is a full day**, 17,280,000 samples × 2 leads, 138 MB of float32, so
      `window=` is required to batch at all. Length is not uniform: record 107 is 9.00 h
      and record 022 is 19.94 h, so a window must fit 6,480,000 samples.

      **All 128 records pass every validation check**, so `clean/` equals `original/`.

  - type: table
    title: "Recomputed from the files"
    headers: ["Quantity", "Value", "Published", "Note"]
    rows:
      - ["Recordings", "**128**", "128", "but only **127 distinct** — 005 and 020 are byte-identical"]
      - ["Subjects", "**122**", "122", "6 contributed 2 recordings each; folds grouped on `Subject_ID`"]
      - ["Annotated recordings", "**98**", "98", "the other 30 have `.qrs` only"]
      - ["Total signal", "**3,050.4 h**", "—", "2,330.7 h of it annotated"]
      - ["Sampling rate", "**200 Hz**", "200 Hz", "recorder was **125 Hz**; the release upsampled"]
      - ["Record length", "**6,480,000 – 17,340,000**", "~24 h", "10 distinct values; 87 records exactly 17,280,000"]
      - ["Beat detections (all 128)", "**13,532,154**", "—", "unaudited `epltd` Pan-Tompkins"]
      - ["`.atr` annotations (98)", "**10,349,733**", "10,574,142", "see \"About those counts\""]
      - ["`N` beats", "7,642,032", "7,812,308", "**−170,276**"]
      - ["`AFIB` beats", "2,453,805", "2,512,959", "**−59,154**"]
      - ["`AFL` beats", "**195,659**", "195,659", "exact"]
      - ["`AT` beats", "**48,800**", "48,800", "exact"]
      - ["`PAT`+`NOD` beats", "**4,416**", "4,416", "exact"]
      - ["`AB` beats", "**5,021**", "*absent*", "an undocumented sixth code"]
      - ["`AFIB` intervals", "794", "809", "**−15**"]
      - ["`AFL` / `AT` / `PAT`+`NOD` intervals", "**45 / 57 / 9**", "45 / 57 / 9", "exact"]
      - ["Interval duration median", "58.91 s", "47.5 s", "min 0.630 s against a published 2.5 s floor"]
      - ["Age at Holter", "**65.8 ± 12.1**", "68.0 ± 11.3", "per recording; per subject 65.7 ± 12.0"]
      - ["Sex", "**81 M / 47 F**", "75 M / 47 F", "per recording; per subject **78 M / 44 F**"]
      - ["`AF_Type`", "**PAF 80 · PerAF 15 · non-AF 33**", "same", "per recording"]
      - ["Stroke history", "**19 (14.8%)**", "19 (11.7%)", "the percentage is the release's arithmetic"]
      - ["Files verified", "**488 / 488**", "—", "against the release's own `SHA256SUMS.txt`"]

  - type: description
    title: "About those counts"
    body: |
      Every figure above was recomputed from the 128 headers, the 98 `.atr` files, the
      128 `.qrs` files and `AdditionalData.csv`, after verifying the local copy against
      the release's own `SHA256SUMS.txt` — **all 488 files match**. The record list
      comes from the shipped `RECORDS.txt`, so a partial download cannot enter the
      partition. (Note the `.txt`: every other WFDB release in this catalogue ships an
      extensionless `RECORDS`, and code that globs for that name here silently falls
      back to whatever `.hea` files happen to be on disk. For a 7.7 GB release that is
      how a half-finished download passes for a smaller database — which is exactly
      what the copy this config was written against turned out to be.)

      **The published beat table does not reproduce, and the reason is the version.**
      Three of its five rows match exactly: `AFL` at 195,659 beats in 45 intervals,
      `AT` at 48,800 in 57, `PAT`+`NOD` at 4,416 in 9. `N` comes out 170,276 beats
      short of the published 7,812,308 and `AFIB` 59,154 short of 2,512,959, with 794
      `AFIB` intervals against 809. The shortfall is **224,409 beats, or 2.12 times the
      mean 105,610 beats per recording** — two recordings' worth. v1.0.1 withdrew
      exactly two annotated recordings, 016 and 030, as duplicates, and the table was
      not regenerated: it describes v1.0.0's 100 annotated recordings. The same
      explanation covers the interval-duration summary, which states a 2.5 s minimum
      and a 47.5 s median where the shipped files give 0.630 s and 58.91 s and 50
      intervals fall under the stated floor. **A recomputation that differs in `N`,
      `AFIB` and the duration quartiles — and only those — is right, not broken.**

      **There is a sixth rhythm code in the files and in no documentation: `(AB`,**
      atrial bigeminy. 3 intervals across 2 recordings (047 once, 051 twice), 5,021
      beats, 3,674.2 s. Neither the landing page nor the shipped README lists it and it
      appears in no row of the published table. ECGBench counts it, because dropping an
      annotation for want of documentation would silently reassign those beats to
      whatever came before.

      **The demographic summary mixes recording-level and subject-level counts.** The
      landing page gives "Female: 47 (38.5%), Male: 75 (61.4%)", which sums to the 122
      subjects — but 47 is the number of *recordings* from female patients, and per
      subject it is 44 F / 78 M. Age is given as 68.0 ± 11.3 where the shipped column
      yields 65.8 ± 12.1 per recording and 65.7 ± 12.0 per subject, and nothing in the
      release accounts for the 2.2-year difference. Stroke is given as 11.7% where
      19/128 is 14.8%. The figures above are per recording throughout, and stated as
      such.

      **The shipped `README.md` is v1.0.0's and its column names are wrong.** It
      documents `<Study ID>`, `<UID>` and `<Height (m)>`, claims 127 unique patients,
      and says the `.dat` files carry `base_year` and `base_time` fields. The file that
      actually ships has `Subject_ID`, `Data_ID` and `Height`, 122 subjects, and every
      header is exactly three lines with no timestamp — v1.0.1 moved the start time
      into `Holter_start_time`. Follow the landing page.

      Two smaller cross-checks, both harmless but worth knowing before you rely on a
      column: `Holter_recording_length` is the header duration **minus exactly one
      second** for 126 of the 127 recordings that have both (022 is the exception), so
      use the header for duration. And `AF_Duration_Months` does **not** equal
      `Date_Holter − Date_of_First_Diagnosis_of_AF_AFL` — the median discrepancy is 59
      months — because the de-identification shifted each subject's dates
      independently. `AF_Duration_Months` is the release's own pre-shift interval and
      is the column to use.

  - type: description
    title: "005 and 020 are the same recording"
    body: |
      This is the one thing about SHDB-AF that can cost you a result, and nothing in
      the release announces it.

      `005.dat` and `020.dat` have the **same SHA-256 in the release's own
      `SHA256SUMS.txt`**, and so do `005.qrs` and `020.qrs`. Only the `.atr` differs.
      The clinical table presents them as two separate Holters three years apart —
      `Age_at_Holter` 47 and 50, `Date_Holter` 2021-05-21 and 2024-02-21 — so no
      metadata comparison finds it. Checked exhaustively: across all 488 files these
      are the **only** duplicated checksums, and an independent scan of every `.qrs`
      beat-position vector finds exactly one identical pair. So the release holds **127
      distinct recordings, not 128**.

      **ECGBench's folds are not leaky because of it — and that is luck, not design.**
      Both rows carry `Subject_ID` 4899921 and folds are grouped on `Subject_ID`, so
      the two copies always land in the same fold. `duplicate_of` in the labels names
      the partner (`"020"` on record 005, `"005"` on 020) so anyone computing a
      per-record metric can drop one rather than double-count a recording.

      **The two `.atr` files are an accidental annotation-repeatability sample**, and
      the only one this database offers — every other rhythm mark in the release was
      placed by the same single cardiology fellow, so there is no inter-observer
      estimate at all. The same recording annotated twice gives the same 17 marks and
      the same 9 `N` / 8 `AFIB` interval structure, with three beats moved across
      boundaries (94,795/7,207 against 94,798/7,204) and an AF burden of **0.03640
      against 0.03637**. Encouraging, and a sample of one.

      As for overlap with *other* datasets: there is none to declare. This is a
      single-institution Japanese cohort recorded 2019-2023 on a Fukuda monitor, and
      every other two-lead database in this catalogue comes from Boston, Europe or
      China, decades earlier and on different equipment. The databases the landing page
      names alongside it — MITDB, AFDB, LTAFDB, IRIDIA-AF, Icentia11k, CPSC2021 — are
      the *comparators* in the generalisation study that produced it, not sources it
      draws from. That is an argument from provenance rather than a sample-wise check,
      and it is stated that way on purpose.

  - type: description
    title: "Two label layers, and they answer different questions"
    body: |
      `load_labels` returns both, joined on `Data_ID`.

      **`AF_Type` is a clinical diagnosis** from the medical report written after the
      Holter: `PAF` (80 recordings), `PerAF` (15), `non-AF` (33). It exists for all 128
      and is the only label the 30 unannotated recordings have.

      **`af_burden` is a measurement** of the recording: AFIB seconds over annotated
      seconds. It exists for the 98 with a `.atr`. `af_beat_fraction` is the
      beat-counted version and runs higher — 0.222 against 0.196 on average — because
      AF beats are faster than the rest of the record.

      The two agree better than one might fear and not perfectly:

      | | annotated | with any annotated AFIB | with annotated AFL | median `af_burden` |
      |---|---|---|---|---|
      | `PAF` | 69 | **68** | 10 | 0.116 |
      | `PerAF` | 9 | 9 | 2 | **0.023** |
      | `non-AF` | 20 | **0** | 0 | 0.000 |

      **Not one of the 20 annotated `non-AF` recordings contains a single second of
      annotated AF**, which is a stronger consistency than most releases manage. The
      one `PAF` recording without any is **107** — also the shortest in the release at
      9.00 h. And note that `PerAF`'s median burden (0.023) is *lower* than `PAF`'s
      (0.116): the diagnosis describes the subject's history, not this particular day,
      so it is not a burden proxy in either direction.

      Time in each annotated rhythm, over the 98:

      | Code | Hours | Recordings | Meaning |
      |---|---|---|---|
      | `N` | **1,829.9** | 98 | **not annotated** — sinus rhythm, ectopy, pauses, noise |
      | `AFIB` | 455.9 | 77 | atrial fibrillation |
      | `AFL` | 35.8 | 12 | atrial flutter |
      | `AT` | 7.2 | 4 | atrial tachycardia |
      | `AB` | 1.0 | 2 | atrial bigeminy — **undocumented upstream** |
      | `PAT` | 0.3 | 1 | other SVT, e.g. Wolff-Parkinson-White |
      | `NOD` | 0.3 | 1 | intranodal tachycardia |

      `AT`, `AB`, `PAT` and `NOD` are far too rare to train on as classes. Flutter
      **is** annotated here, unlike in `ltafdb`, and is deliberately *not* folded into
      `af_burden` — it gets `afl_burden` instead, so the `af_burden` column means the
      same thing across `afdb`, `ltafdb` and this dataset.

      `af_class` bins `af_burden` on the same cuts those two use: **minimal 46,
      paroxysmal 49, sustained 3, unannotated 30**. The `unannotated` level is a
      deliberate fourth value rather than a NaN or a zero, because 11 of those 30
      recordings carry a `PAF` diagnosis and calling their burden zero would be a wrong
      label rather than a missing one.

  - type: description
    title: "How the folds are stratified"
    body: |
      Ten folds, `StratifiedGroupKFold`, **grouped on `Subject_ID`** because six
      subjects contributed two recordings each (005/020, 015/047, 035/036, 052/128,
      066/118, 129/133) — and one of those pairs is the byte-identical duplicate above.

      Stratified on `AF_Type` **crossed with whether the recording is annotated**,
      which is not any shipped column: those are the two axes a user slices on, and a
      fold holding no annotated persistent-AF recording would be useless for either
      purpose. The cross is taken only where the counts can afford it:

      | Fold class | n |
      |---|---|
      | `PAF+annotated` | 69 |
      | `non-AF+annotated` | 20 |
      | `PerAF` | 15 |
      | `non-AF+unannotated` | 13 |
      | `PAF+unannotated` | 11 |

      `PerAF` stays whole because its own cross is 9 annotated and 6 unannotated, and
      `StratifiedGroupKFold` does not raise for a class it cannot spread over ten folds
      — it quietly leaves folds without one.

  - type: description
    title: "Amplitude is per channel, not per release"
    body: |
      Every channel was independently rescaled to fill the 16-bit range, so the gains
      in the headers are **normalisation constants, not amplifier settings**. The 256
      signal lines carry **254 distinct gains** from 2,880.0 to 33,488.6 adu/mV, and
      255 of them a nonzero baseline (only 050's ECG1 is zero). The consequence is that
      the physical span differs **11.6-fold** across the release: record 141's ECG2
      covers 1.96 mV peak-to-peak and record 113's ECG1 covers 22.75 mV.
      **Do not compare absolute amplitudes between recordings.** `adc_gains` and
      `adc_baselines` in the labels give the pair per record.

      The scaling is tight enough to be worth one more note. 240 of the 256 channels
      reach digital −32767 and 225 reach 32765 or more, so the observed millivolt
      extremes over the whole release ([−11.9113, 10.8432], both in record 113's ECG1)
      sit within one quantisation step of the computed 16-bit rail. −32768 is WFDB's
      invalid-sample marker for format 16 and `wfdb` converts it to NaN on read — and
      **not one sample of the 780 million in this release is −32768**. One step of
      headroom is the difference between every record passing `nan_values` and most of
      the release failing it.

      `amplitude_range_mv` is therefore the union of all 256 rails plus a thousandth of
      a millivolt of slack, `[-11.92, 10.85]`. The slack is not cosmetic: signals are
      loaded as float32 and the bound is compared as float64, so a bound set to the
      attained rail excludes the very record it was computed from — which is how
      `chfdb` lost a record.

  - type: description
    title: "200 Hz files from a 125 Hz recorder"
    body: |
      The Fukuda Holter monitors digitised at **125 Hz**. The release band-passed with a
      zero-phase second-order IIR filter over [0.67–100] Hz and then **upsampled to
      200 Hz** with an anti-aliasing filter, which is what the headers declare and what
      `wfdb` returns.

      So `sampling_rates: [200]` is honest about the files and misleading about the
      information: there is no real content above 62.5 Hz, and 1.6 of every 2 samples
      is interpolation. Anything measuring high-frequency content, QRS slope, or
      fiducial timing to better than 8 ms should know that before comparing this
      dataset against a natively-200 Hz one. The beat positions were detected *after*
      resampling, by `epltd`, and are unaudited for all 128 recordings — `.atr` and
      `.qrs` hold the identical positions, so `n_beats` is not a verified beat count in
      the sense `ltafdb`'s is.

  - type: code
    title: "Loading with ECGBench"
    language: "python"
    body: |
      from ecgbench import ECGDataset

      # A record is 24 hours long, so window= is not optional for batching.
      # 6,480,000 samples is the shortest record (107), so any window must fit inside it.
      ds = ECGDataset(
          "shdb_af",
          split="train",
          version="clean",
          data_path="/path/to/shdb-af/1.0.1/",   # signals; fold CSVs come from the Hub
          window=(0, 2000),                       # 10 s at 200 Hz
          labels=True,
      )

      len(ds)                       # 100
      sample = ds[0]
      sample["record_id"]           # '001'   <- a STRING; zero-padded ids
      sample["signal"].shape        # torch.Size([2, 2000])
      ds.lead_names                 # ('ECG1', 'ECG2')   ECG1 = mod. CC5, ECG2 = NASA

      labels = sample["labels"]
      labels["AF_Type"]                    # 'PAF'      <- clinical diagnosis
      labels["Subject_ID"]                 # '2043771'
      labels["has_rhythm_annotation"]      # True
      labels["af_burden"]                  # 0.5639     <- measured, AFIB seconds / annotated
      labels["af_beat_fraction"]           # 0.7109     <- beat-counted, always higher
      labels["af_class"]                   # 'paroxysmal'
      labels["dominant_rhythm"]            # 'AFIB'     <- 'N' would mean 'not annotated'
      labels["record_hours"]               # 23.9167
      labels["duplicate_of"]               # ''         <- '020' on record 005

      # Select a channel by name; both are documented Holter placements here, which is
      # not true of any other two-lead database in this catalogue.
      nasa = ECGDataset("shdb_af", split="train", data_path="/path/to/shdb-af/1.0.1/",
                        window=(0, 2000), leads=["ECG2"])
      nasa[0]["signal"].shape       # torch.Size([1, 2000])

  - type: links
    title: "Links"
    items:
      - label: "PhysioNet — SHDB-AF v1.0.1"
        url: "https://physionet.org/content/shdb-af/1.0.1/"
      - label: "Paper (arXiv:2406.16974)"
        url: "https://arxiv.org/abs/2406.16974"
      - label: "ECGBench config"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/ecgbench/data/configs/shdb_af.yaml"
      - label: "ECGBench label loader"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/ecgbench/labels/shdb_af.py"
      - label: "Example script"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_shdb_af.py"
---
