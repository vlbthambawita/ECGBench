---
slug: "st-vincent-s-ucd-sleep-apnea-database-ucddb"
name: "St. Vincent's / UCD Sleep Apnea Database (UCDDB)"
category: "three-lead"
order: 1
status: "completed"
source_url: "https://physionet.org/content/ucddb/1.0.0/"
url_label: "physionet.org"
format: "3-lead Holter (V5, CC5, V5R) · 7.52–8.68 h · 128 Hz · EDF · AHI + sleep stages + respiratory events"
patients: "25"
records: "25"
access: "open"
license: "ODC Attribution"
origin_institution: "St. Vincent's University Hospital / University College Dublin"
origin_country: "Ireland — Dublin"
leads: 3
paper_title: "Dataset DOI"
paper_doi: "https://doi.org/10.13026/C26C7D"
search_keywords: "ucddb ucd sleep apnea dublin ireland holter v5 cc5 v5r overnight polysomnogram psg 3-lead edf ahi apnea hypopnea index obstructive central mixed sleep staging rechtschaffen kales reynolds lifecard jaeger toennies"

sections:
  - type: description
    title: "Overview"
    body: |
      25 full overnight sleep studies from the Sleep Disorders Clinic at St
      Vincent's University Hospital, Dublin. Each ships **two simultaneous
      recordings of the same night**: a 14-channel Jaeger-Toennies polysomnogram
      and a three-channel Reynolds Lifecard CF Holter ECG. ECGBench splits and
      validates the **Holter** — **203.4 hours** of V5/CC5/V5R at 128 Hz, 7.52 h to
      8.68 h per record.

      **This is the first EDF dataset in ECGBench.** `signal_format: "edf"` was
      added for it, with a reader that seeks to the data record a window starts in
      rather than decoding an eight-hour night to return thirty seconds of it. The
      polysomnograms are mixed-rate files (8, 64 and 128 Hz side by side) and the
      reader refuses them by name; their paths are still carried in the metadata
      for anyone who wants to read them separately.

      The subjects were selected at random over six months (September 2002 to
      February 2003) from patients referred for possible obstructive sleep apnea,
      central sleep apnea or primary snoring, and had to be over 18, free of known
      cardiac disease and autonomic dysfunction, and not on medication affecting
      heart rate: **21 men and 4 women**, aged 50 ± 10 years (28–68), BMI 31.6 ± 4.0
      kg/m² (25.1–42.5), apnea-hypopnea index 24.1 ± 20.3 (1.7–90.9).

      Beyond the per-subject AHI, each study carries **20,789 sleep epochs** scored
      in 30 s to Rechtschaffen and Kales rules and **3,428 respiratory events**
      annotated with onset, duration, oxygen desaturation, snoring, arousal and
      heart-rate change — all by the same experienced sleep technologist.

      All 102 shipped files verify against the release's own `SHA256SUMS.txt`, all
      25 records pass every ECGBench quality check, and `clean/` therefore equals
      `original/`. Read the next four sections before trusting that last sentence.

  - type: description
    title: "The annotations are stamped in polysomnogram time, and the Holter's clock is a placeholder"
    body: |
      **This is the thing to understand before using this dataset for anything
      event-level.** The respiratory events carry a time of day and the sleep
      stages are 30 s epochs from polysomnogram onset. Both line up with the `.rec`
      header, whose start time equals `SubjectDetails.xls`'s "PSG Start Time" in
      all 25 records. The `_lifecard.edf` headers do **not**: they read 09:01:17 to
      09:48:29 on 01.01.06, rising monotonically in filename order about a minute
      apart. Those are archive timestamps, and the landing page says as much —
      *"The recording dates and times are not available."*

      Taken at face value, that makes every one of the 3,428 events and 20,789
      epochs unusable with the ECG this dataset is catalogued for.

      ECGBench recovers the offset for **24 of the 25 records** by cross-correlating
      heart rate between the two recordings: median-RR heart rate at 1 Hz, smoothed
      over 60 s, from the polysomnogram's own ECG channel against each Holter
      channel, at every lag leaving at least 90% overlap. The Holter was fitted
      **17.6 to 132.2 minutes before** the polysomnogram started.

      | | Records |
      |---|---|
      | Offset recovered | **24** of 25 |
      | …passing both reliability thresholds (r ≥ 0.70, third-to-third spread ≤ 30 s) | **22** |
      | …with all three Holter channels agreeing on the lag to within 20 s | 24 |
      | No offset (`ucddb028`, whose Holter is another subject's — see below) | 1 |

      The spread column is the instrument that matters: the offset is refitted
      independently on the first, middle and last third of the night, and a real
      constant offset does not move between thirds while a spurious correlation
      peak does. **22 records move by 3 s or less.** The two that fail are
      `ucddb023` (r = 0.74, spread 46 s), which is genuinely uncertain — only its
      middle third correlates above 0.7 — and `ucddb013` (r = 0.40, spread 63 s),
      which fails for a milder reason: its first third is unusable signal and fits
      −60 s at r = 0.18, while its middle and last thirds **both** fit +3 s at
      r = 0.97. Check `psg_offset_reliable` rather than assuming, and recompute the
      whole table if you want to:

      ```python
      from ecgbench.labels.ucddb import verify_psg_alignment
      verify_psg_alignment("/path/to/ucddb/1.0.0/")
      ```

      `respiratory_events()` and `sleep_stages()` apply the offset for you and
      return a `holter_secs` column that indexes straight into `window=`.

  - type: description
    title: "ucddb028's ECG belongs to ucddb014, and nothing upstream says so"
    body: |
      `ucddb014_lifecard.edf` and `ucddb028_lifecard.edf` differ in exactly **four
      bytes** — the EDF start-time field, 09:25:37 against 09:48:29 — and their
      20,782,080-byte signal payloads are **bit-identical**. Their polysomnograms,
      sleep stages, respiratory events and demographics are all different, so these
      are two genuinely different men (56, AHI 36; 50, AHI 46) sharing one Holter
      recording.

      The alignment search confirms it from an independent direction:

      | Polysomnogram | Holter | Best offset | r |
      |---|---|---|---|
      | ucddb014 | ucddb014 | 3432 s | **0.941** |
      | ucddb014 | **ucddb028** | **3432 s** | **0.940** |
      | ucddb028 | ucddb014 | 7174 s | −0.005 |
      | ucddb028 | ucddb028 | −20 s | 0.010 |

      `ucddb028`'s waveform matches `ucddb014`'s night, at the same offset and the
      same correlation as `ucddb014`'s own file, and does not match its own
      subject's night at all.

      **Both records are kept**, because each is an official record with its own
      official annotations and dropping one would silently diverge from the
      release. Two things follow instead. `patient_id_column` is
      `recording_group`, not a subject id, and it merges the pair into
      `"ucddb014+ucddb028"` so the shared waveform cannot land on both sides of a
      split — 25 records, **24 groups**. And `waveform_matches_subject` is `False`
      for `ucddb028`, because its AHI of 46 labels a night belonging to a man whose
      AHI was 36. For record-level supervised work, filter it out:

      ```python
      df = df[df["waveform_matches_subject"]]
      ```

      A second, smaller duplication *is* documented upstream: in `ucddb002`
      "only two distinct ECG signals were recorded; the second ECG signal was also
      used as the third signal". Verified — its channels 2 and 3 are equal at every
      one of its 3,525,120 samples, and `n_distinct_leads` reports 2. So the
      database holds **74 distinct ECG channels**, not 75.

  - type: description
    title: "The first 67–119 seconds of every record are not ECG"
    body: |
      **`window=(0, n)` is the wrong window for this dataset**, and it fails
      silently. Every Holter file opens with a calibration block — a two-level
      2 Hz square wave alternating between digital 1843 and 2253, i.e. 4.5006 and
      5.5018 mV, so **1.0012 mV peak to peak**: the instrument's 1 mV calibration
      pulse. Nothing in the release documents it.

      Over the shortest block (67 s) the samples are **byte-identical across all
      25 records and all three channels**, so a naive first-N-samples window
      returns the same array for every record in the database and none of it is
      anybody's heart. The block runs **67.0 s (`ucddb006`) to 119.0 s
      (`ucddb027`)**, every length a whole number of seconds, and `ucddb014` and
      `ucddb028` agree at 87.0 s as their shared waveform requires.

      ```python
      from ecgbench.labels.ucddb import ECG_STARTS_AT_SAMPLE   # 15232 = 119.0 s
      ECGDataset("ucddb", split="train", window=(ECG_STARTS_AT_SAMPLE, 3840), ...)
      ```

      `ECG_STARTS_AT_SAMPLE` is the first sample past the *longest* block, so it
      is safe for every record; `CALIBRATION_SAMPLES` has the per-record length if
      you want to start earlier on a specific one, and `calibration_samples` is a
      label column. Recompute both — it reads only the first 15 minutes of each
      file, so it takes seconds:

      ```python
      from ecgbench.labels.ucddb import verify_calibration_block
      verify_calibration_block("/path/to/ucddb/1.0.0/")
      ```

      Positions derived from `psg_offset_secs` are unaffected: the recovered
      offsets are 1053 s to 7932 s, all well past every calibration block, so a
      window cut around a scored respiratory event lands in real ECG.

  - type: description
    title: "The ECG rides a 5 mV pedestal, and no quality check can see the duplicates"
    body: |
      All 25 records pass every ECGBench check and `clean/` equals `original/`.
      That is arithmetically correct and substantively incomplete, so here is what
      the validation report cannot tell you.

      **Every Holter channel declares digital 0–4095 mapping to physical 0–10 mV**,
      so the baseline sits at mid-scale — measured mean 5.00 mV — and the signal is
      unipolar by construction. ECGBench applies EDF's declared calibration
      verbatim, as any EDF reader does: an offset is not something a reader should
      silently remove, and `signal_unit_scale` is a multiplier that could not
      express it anyway. Subtract the median yourself if you want a centred trace.

      One consequence is that `amplitude_range_mv` is **[0.0, 10.001] — the ADC
      span, not a physiologic range**. Both rails are attained: the float32 minimum
      across all 25 records is exactly 0.0 (11 records touch digital 0) and the
      maximum exactly 10.0 (`ucddb015` channel 1 touches 4095). The upper bound
      carries a thousandth of a millivolt of slack so a platform whose float32
      rounding differs cannot exclude the record the bound was computed from; the
      lower carries none, because a negative physical value could only come from a
      corrupt read. What the check still catches is a reader that forgot the EDF
      calibration and returned raw digital counts. What it does **not** do is
      police amplitude plausibility.

      Neither duplication above is visible to any check either. `ucddb002`'s third
      channel is a perfectly good ECG — just not a third one — and `ucddb028`'s
      waveform is a perfectly good overnight Holter belonging to somebody else.
      Read `n_distinct_leads` and `waveform_matches_subject`, not the report.

  - type: description
    title: "25 records, 24 groups, and a stratification label coarser than the grade"
    body: |
      Folds are grouped on `recording_group`, which is the record name for 23
      records and `"ucddb014+ucddb028"` for the duplicated pair. With ten folds
      that is 2–3 records each; no fold is empty, and no group straddles a fold.

      **The stratification label is deliberately coarser than the clinical grade.**
      The four AHI severity classes split the database **normal 1 / mild 10 /
      moderate 6 / severe 8**, and a class of one subject cannot be spread over ten
      folds — scikit-learn warns and then puts that subject wherever it lands.
      `stratify_class` pools at the standard moderate-or-severe cut point (AHI ≥
      15), giving 14 records against 11, or **13 recording groups against 11** once
      the pair merges. Both clear the fold count.

      So `label_column` (`ahi_severity`) and the stratification axis are different
      quantities, on purpose. **Train on `psg_ahi` or `ahi_severity`**; the pooled
      class exists to build the partition.

      With the default mapping that gives train = folds 1–8 (21 records), **val =
      fold 9 (`ucddb021`, `ucddb025`) and test = fold 10 (`ucddb008`,
      `ucddb027`)**. Two subjects is not an evaluation set; use `split=None` with
      `fold_numbers=[...]` for real cross-validation.

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page is recomputed from the shipped files, which all
      verify against the release's own `SHA256SUMS.txt`.

      | Quantity | Landing page | Recomputed | Note |
      |---|---|---|---|
      | Subjects | 25 (21 M, 4 F) | 25 (21 M, 4 F) | agrees; numbered 002–028, with 004 and 016 absent |
      | Holter records | 25 | 25 | but only **24 distinct waveforms** — see above |
      | Age | 50 ± 10, range 28–68 | 50.0 ± 9.5, range 28–68 | agrees |
      | BMI | 31.6 ± 4.0, range 25.1–42.5 | 31.6 ± 4.0, range 25.1–42.5 | agrees |
      | AHI | 24.1 ± 20.3, range 1.7–90.9 | 24.2 ± 20.3, range 2–91 | `SubjectDetails.xls` rounds to whole numbers |
      | Sampling rate | not stated | 128 Hz, all 75 channels | |
      | Holter duration | not stated | 7.52–8.68 h, **203.4 h** total | |
      | Respiratory events | not stated | **3,428** | 9–436 per record |
      | Scored epochs | not stated | **20,789** | 30 s each |

      **The parsing behind every event and epoch count is checkable, and was
      checked.** Counting apneas and hypopneas out of `_respevt.txt` and dividing
      by sleep time from `_stage.txt` recovers the shipped "PSG AHI" to within 1.0
      for **23 of 25** records and within 3.9 for all 25 — the two outliers are the
      two highest indices in the database, `ucddb025` (94.8 recomputed against 91
      shipped) and `ucddb028` (48.2 against 46). The same files recover "Sleep
      Efficiency (%)" to within 0.5 points for all 25. `ahi_recomputed` and
      `sleep_efficiency_recomputed_pct` ship beside the originals so the
      disagreement is visible rather than hidden.

      Two things the release documents but does not contain, and one it contains
      but does not document. Sleep-stage codes **6 (Artifact) and 7
      (Indeterminate) never occur**; an undocumented code **8** occurs 15 times, 11
      in `ucddb008` and 4 in `ucddb024`. Counting 8 as non-sleep is what reproduces
      `ucddb008`'s shipped 64% sleep efficiency (64.3% excluding it against 65.8%
      including), so that is how it is counted, and `n_epochs_undocumented` reports
      it separately.

      **The lead names come from the landing page, not the files.** Every one of
      the 25 headers labels its channels `chan 1`, `chan 2`, `chan 3` and names no
      electrode. The page says *"Three-channel Holter ECGs (V5, CC5, V5R)"*, and
      `lead_names` is that sentence's order. Nothing in the release corroborates
      it, so treat `leads=["V5R"]` as a channel position with a probable name
      rather than an anatomically certain lead.

      The respiratory-event breakdown, by the scorer's own categories:

      | Event | Count | Counts toward AHI |
      |---|---|---|
      | Obstructive hypopnea (`HYP-O`) | 1,433 | yes |
      | Central hypopnea (`HYP-C`) | 1,076 | yes |
      | Central apnea (`APNEA-C`) | 343 | yes |
      | Obstructive apnea (`APNEA-O`) | 216 | yes |
      | Mixed apnea (`APNEA-M`) | 136 | yes |
      | Mixed hypopnea (`HYP-M`) | 114 | yes |
      | Periodic breathing (`PB`) | 91 | no |
      | Equivocal (`POSSIBLE`) | 19 | no |
      | **Total** | **3,428** | 3,318 counted |

      Excluding `PB` and `POSSIBLE` is not a guess — it is what reproduces the
      shipped AHI. And the sleep-stage totals across the database: wake 4,707,
      REM 3,016, stage 1 3,403, stage 2 6,985, stage 3 673, stage 4 1,990,
      undocumented 15.

  - type: table
    title: "The 25 records, recomputed from the files"
    headers: ["Rec", "Sex", "Age", "BMI", "AHI", "Grade", "Hours", "Cal s", "A+H", "Epochs", "Offset s", "r", "Spread s", "Fold"]
    rows:
      - ["ucddb014", "M", "56", "29.0", "36", "severe", "7.52", "87", "182", "774", "3432", "0.941", "2", "1 (train)"]
      - ["ucddb018", "M", "35", "26.3", "2", "normal", "8.40", "104", "9", "822", "5280", "0.974", "3", "1 (train)"]
      - ["ucddb028", "M", "50", "30.1", "46", "severe", "7.52", "87", "198", "721", "—", "—", "—", "1 (train)"]
      - ["ucddb010", "M", "38", "39.3", "34", "severe", "8.15", "109", "233", "907", "1837", "0.979", "1", "2 (train)"]
      - ["ucddb011", "M", "51", "28.6", "8", "mild", "8.17", "101", "35", "900", "2186", "0.974", "2", "2 (train)"]
      - ["ucddb019", "M", "49", "30.9", "16", "moderate", "8.28", "87", "104", "852", "3985", "0.872", "2", "2 (train)"]
      - ["ucddb005", "M", "65", "32.4", "13", "mild", "8.10", "97", "57", "826", "4163", "0.816", "28", "3 (train)"]
      - ["ucddb013", "F", "62", "34.2", "16", "moderate", "8.52", "115", "65", "811", "5043", "0.401", "63", "3 (train)"]
      - ["ucddb020", "M", "52", "34.0", "15", "moderate", "7.85", "77", "73", "752", "5422", "0.837", "1", "3 (train)"]
      - ["ucddb002", "M", "54", "33.9", "23", "moderate", "7.65", "116", "124", "748", "4761", "0.971", "1", "4 (train)"]
      - ["ucddb015", "M", "28", "29.0", "6", "mild", "8.62", "80", "37", "916", "3032", "0.975", "1", "4 (train)"]
      - ["ucddb023", "F", "68", "32.7", "39", "severe", "8.68", "117", "191", "861", "2699", "0.742", "46", "4 (train)"]
      - ["ucddb006", "M", "52", "30.2", "31", "severe", "8.45", "67", "187", "808", "5888", "0.928", "15", "5 (train)"]
      - ["ucddb022", "M", "34", "29.3", "7", "mild", "7.83", "81", "27", "787", "4447", "0.928", "1", "5 (train)"]
      - ["ucddb007", "M", "47", "25.1", "12", "mild", "8.25", "74", "73", "813", "4845", "0.947", "1", "6 (train)"]
      - ["ucddb009", "M", "52", "31.3", "12", "mild", "8.05", "101", "76", "925", "1053", "0.966", "1", "6 (train)"]
      - ["ucddb012", "M", "51", "30.4", "25", "moderate", "8.25", "102", "151", "864", "3286", "0.976", "1", "6 (train)"]
      - ["ucddb017", "M", "53", "37.8", "12", "mild", "7.63", "75", "68", "789", "3505", "0.916", "1", "7 (train)"]
      - ["ucddb024", "M", "54", "33.8", "24", "moderate", "8.35", "103", "154", "908", "2586", "0.919", "0", "7 (train)"]
      - ["ucddb003", "M", "48", "31.8", "51", "severe", "8.08", "94", "305", "882", "2464", "0.977", "2", "8 (train)"]
      - ["ucddb026", "M", "49", "27.4", "14", "mild", "7.82", "79", "84", "838", "2565", "0.879", "1", "8 (train)"]
      - ["ucddb021", "F", "41", "33.6", "13", "mild", "8.43", "94", "80", "913", "2381", "0.935", "3", "9 (val)"]
      - ["ucddb025", "M", "52", "42.5", "91", "severe", "8.22", "86", "433", "711", "7932", "0.918", "0", "9 (val)"]
      - ["ucddb008", "F", "63", "28.4", "5", "mild", "8.37", "97", "19", "768", "3885", "0.952", "8", "10 (test)"]
      - ["ucddb027", "M", "45", "28.1", "55", "severe", "8.25", "119", "353", "893", "2522", "0.970", "1", "10 (test)"]

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset
      from ecgbench.labels.ucddb import (
          ECG_STARTS_AT_SAMPLE,          # 15232 — the first sample past every
          respiratory_events,            #         record's calibration block
          sleep_stages,
      )

      # window= is not optional here: a record is 3.46M to 4.00M samples across 3
      # leads, so a whole one is ~40 MB. It is pushed into the EDF reader as a seek
      # to the data record it starts in. AND IT MUST NOT START AT 0: the first
      # 67-119 s of every record is a 1 mV calibration square wave.
      ds = ECGDataset(
          "ucddb",
          split="train",
          labels=True,
          window=(ECG_STARTS_AT_SAMPLE, 3840),   # 30 s of ECG, from 119.0 s in
          data_path="/path/to/ucddb/1.0.0/",
      )

      len(ds)                                    # 21
      sample = ds[0]
      sample["signal"].shape                     # (3, 3840)
      sample["record_id"]                        # 'ucddb002'
      sample["labels"]["recording_group"]        # 'ucddb002'
      sample["labels"]["psg_ahi"]                # 23.0
      sample["labels"]["ahi_severity"]           # 'moderate'
      sample["labels"]["n_apnea_hypopnea"]       # 124
      sample["labels"]["psg_offset_secs"]        # 4761.0   <- the Holter started first
      sample["labels"]["calibration_samples"]    # 14848    <- 116.0 s of it, on this record
      sample["labels"]["n_distinct_leads"]       # 2        <- ucddb002's chan 3 copies chan 2
      float(sample["signal"].mean())             # 5.040    <- mV: the EDF pedestal, not a bug

      # The intended use: cut a window around a scored respiratory event. holter_secs
      # already has psg_offset_secs applied, so it indexes the ECG directly.
      events = respiratory_events("/path/to/ucddb/1.0.0/", "ucddb002")
      apnea = events[events["event_type"] == "APNEA-O"].iloc[0]
      apnea["holter_secs"]                       # 10145.0
      start = int((apnea["holter_secs"] - 30) * 128)                     # 1294720
      event = ECGDataset("ucddb", split="train", window=(start, 60 * 128),
                         data_path="/path/to/ucddb/1.0.0/")
      # -> (3, 7680), 4.20 to 6.64 mV

      # Sleep stage at the same moment, on the same clock.
      stages = sleep_stages("/path/to/ucddb/1.0.0/", "ucddb002")
      stages[stages["holter_secs"] <= apnea["holter_secs"]].iloc[-1]["stage_name"]   # 's1'

      # None of V5, CC5, V5R is a standard 12-lead name, so select by name.
      ECGDataset("ucddb", split="train", window=(ECG_STARTS_AT_SAMPLE, 3840),
                 leads=["CC5"],
                 data_path="/path/to/ucddb/1.0.0/")[0]["signal"].shape   # (1, 3840)

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # No flags: 24 recording groups over 25 records make ten folds of 2-3.
      # The first run reads SubjectDetails.xls, the 25 stage files, the 25
      # respiratory-event files and the 25 EDF headers, and caches the result as
      # ecgbench_metadata.csv in the dataset root — so that root must be writable.
      # Reading the .xls needs xlrd: pip install 'ecgbench[xls]'.
      ecgbench splits --dataset ucddb --data-path /path/to/ucddb/1.0.0/

  - type: links
    title: "Links"
    links:
      - label: "PhysioNet — ucddb 1.0.0"
        url: "https://physionet.org/content/ucddb/1.0.0/"
      - label: "Dataset DOI — 10.13026/C26C7D"
        url: "https://doi.org/10.13026/C26C7D"
      - label: "Example script — examples/load_ucddb.py"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_ucddb.py"
---
