---
slug: "vitaldb-arrhythmia-database"
name: "VitalDB Arrhythmia Database"
category: "one-lead"
order: 8
status: "completed"
source_url: "https://physionet.org/content/vitaldb-arrhythmia/1.0.0/"
url_label: "physionet.org"
format: "beat & rhythm annotations for VitalDB Lead II · no raw ECGs · 500 Hz"
patients: "473"
records: "482 annotated cases"
access: "open"
license: "CC BY 4.0"
origin_institution: "Seoul National University Hospital"
origin_country: "South Korea"
leads: 1
paper_title: "An Anesthesiologist-Validated Large-Scale Intraoperative Arrhythmia Dataset with Beat and Rhythm Labels"
paper_doi: "https://doi.org/10.1038/s41597-026-07076-8"
search_keywords: "vitaldb arrhythmia south korea seoul intraoperative lead ii surgical anesthesia annotations beat rhythm no waveforms afib ventricular tachycardia"

sections:
  - type: description
    title: "Overview"
    body: |
      Intraoperative arrhythmia, annotated by the people who manage it. Five
      anesthesiologists reviewed screened stretches of Lead II from 482 surgical
      cases at Seoul National University Hospital, labelling every beat and every
      rhythm segment, with each segment read independently by at least two of them
      (reported Cohen's kappa 0.930 ± 0.130). Public ECG arrhythmia data is
      overwhelmingly ambulatory Holter or ICU telemetry; this is the operating
      theatre, where surgical stimulus, anesthetic agents and autonomic swings
      produce a different mix of rhythms and a different noise profile.

      **The release ships no ECG waveforms.** What you download from PhysioNet is
      a `metadata.csv` and 482 per-case annotation CSVs. The signals live in the
      public [VitalDB](https://vitaldb.net/) project and are fetched by `case_id`:

      ```python
      import vitaldb
      vals = vitaldb.load_case(1018, ['SNUADC/ECG_II'], 1/500)
      ```

      Both halves are open — CC BY 4.0 here, and VitalDB needs no credentialing —
      so this is a packaging split, not an access restriction. But it does mean
      the annotations and the samples arrive by different routes.

  - type: description
    title: "How ECGBench integrates it — no config, no splits"
    body: |
      **There is deliberately no `vitaldb_arrhythmia` config, and no
      `ecgbench splits --dataset vitaldb_arrhythmia`.** Nothing in the package is a
      signal file: `signal_format` has nothing to name, `signal_path_columns` has
      nothing to resolve against `data_path`, and `validate_dataset` — which reads
      every record off disk to check leads, length, flat lines and amplitude — has
      nothing to read. Fold CSVs whose signal-path column pointed at a network call
      would not be reproducible artefacts.

      So ECGBench exposes it as an **annotation provider**: you load the tables,
      and pull the waveform for a case from VitalDB when you need samples.

      ```python
      from ecgbench.labels.vitaldb_arrhythmia import load_beats, load_cases

      root = "/path/to/vitaldb-arrhythmia/1.0.0/"
      cases = load_cases(root)                       # 482 cases, 473 patients
      beats = load_beats(root, beats_only=True)      # 658,874 classified beats
      ```

      This is the same treatment
      [PTB-XL+](ptb-xl-plus.html) and the
      [ECG eye-tracking study](eye-tracking-dataset-for-12-lead-ecg-interpretation.html)
      get, for the same reason: a release carrying no recordings of its own gets no
      fold assignment of its own.

      **Any split you define must group on `subjectid`, not `case_id`.** The 482
      cases come from 473 distinct patients — eight patients contributed two cases
      and one contributed three — so case-level folds put the same patient on both
      sides of the boundary. `load_cases()` carries `subjectid` for exactly this.

  - type: description
    title: "The annotations are windows, not whole surgeries"
    body: |
      Each case contributes **one contiguous screened window**, not its full
      anesthetic. The windows average 1,109 s (median 1,198 s, range 139–2,991 s)
      and begin anywhere from 2 s to 33,628 s into the recording, because the
      authors' screening model picked out arrhythmia-candidate stretches rather
      than annotating everything.

      Two consequences that bite immediately:

      - **`time_second` is an offset into the whole VitalDB recording**, not into
        the annotated segment. An R-peak at `t` sits at sample `int(t * 500)` of
        what `vitaldb.load_case` returns. Slicing that array from `0` gets you a
        stretch carrying no labels at all.
      - **The dataset is not a continuous record of every beat**, so absence of an
        arrhythmia annotation outside a window is not evidence of sinus rhythm.
        Train and evaluate inside the windows; `case_window()` returns each one.

  - type: table
    title: "Rhythm labels — recomputed from the annotation files"
    headers: ["Label in files", "Meaning", "Cases", "Beats"]
    rows:
      - ["`N`", "Normal Sinus Rhythm", "370", "408,420"]
      - ["`AFIB/AFL`", "Atrial Fibrillation / **Flutter**", "111", "163,270"]
      - ["`Patterned Ventricular Ectopy`", "Patterned Ventricular Ectopy", "109", "24,069"]
      - ["`SVTA`", "Supraventricular Tachyarrhythmia", "109", "6,416"]
      - ["`VT`", "Ventricular Tachyarrhythmia", "88", "1,598"]
      - ["`Patterned Atrial Ectopy`", "Patterned Atrial Ectopy", "85", "20,326"]
      - ["`SND`", "Sinus Node Dysfunction", "66", "23,141"]
      - ["`WAP/MAT`", "Wandering Atrial Pacemaker / Multifocal Atrial Rhythm", "26", "10,132"]
      - ["`AVB`", "Atrioventricular Block", "10", "4,323"]
      - ["`Unclassifiable`", "Unclassifiable", "6", "199"]
      - ["`Noise`", "*(signal quality, not a rhythm)*", "250", "0 — no beats annotated"]

  - type: description
    title: "Reading those labels"
    body: |
      Every case count and beat count above matches the published summary table
      **exactly**, recomputed with `load_beats()` over all 482 files.

      Three things the table does not say on its face:

      - **Cases are multi-label and the columns do not sum to 482.** A case appears
        on every row whose rhythm occurs anywhere in its window; `rhythm_classes`
        in `metadata.csv` lists them. Note that `"N"` is a substring of `"Noise"`,
        so `str.contains("N")` mislabels the 250 noise-bearing cases —
        `load_cases()` provides `rhythm_class_list` already split.
      - **`AFIB/AFL` includes atrial flutter**, though the published table calls the
        row "Atrial Fibrillation". Do not treat it as pure AF.
      - **`Noise` is a signal-quality verdict, not a rhythm** — it marks stretches
        where artefact hides the QRS complexes themselves, so its 10,098 rows carry
        no `beat_type` at all. Excluding it is what makes the paper's "10 distinct
        rhythm categories" come out at ten. The separate `bad_signal_quality` flag
        is the milder case: QRS visible, P and T obscured, rhythm still readable.

  - type: description
    title: "About those counts"
    body: |
      All 485 shipped files were verified against the release's own
      `SHA256SUMS.txt` before anything here was computed, so every discrepancy
      below is upstream rather than download damage.

      **Patients, not cases, is 473.** The landing page reports 482 for both. The
      shipped `subjectid` column resolves the 482 cases to 473 patients:

      | Figure | Published | Recomputed | Diff |
      |---|---|---|---|
      | Annotated cases | 482 | 482 | — |
      | Distinct patients | 482 | **473** | −9 |

      Derivation: `metadata.csv`, `df["subjectid"].nunique()`. Eight patients
      contributed two cases and one contributed three. This is the figure a split
      must group on.

      **`total_beats` counts annotation rows, not beats.** It equals the row count
      of each case's annotation file for all 482 cases, and those rows sum to
      676,250 — but 17,376 of them annotate no heartbeat (every `Noise` row, plus
      rhythm-only rows and twelve misfiled boundary markers). Only **658,874**
      classify a beat:

      | Quantity | Value |
      |---|---|
      | Annotation rows (`sum(total_beats)`) | 676,250 |
      | Rows classifying a beat | **658,874** |
      | N / S / U / V / P | 439,458 / 184,203 / 18,972 / 16,234 / 7 |

      The abstract's "over 660,000 individually annotated heartbeats" is the row
      count; the beat count falls just under it. `load_beats()` adds `is_beat` so
      the two never get confused, and `beats_only=True` keeps only real beats.

      **`beat_type` has an undocumented fifth value.** The paper names four classes
      (Normal, Supraventricular, Ventricular, Unclassifiable). A `P` also occurs —
      7 beats across cases 708, 1018, 3433 and 3631. ECGBench surfaces it as
      written rather than folding it into `U`, because inferring "paced" from a
      single letter would put an invented label on real data.

      **The per-rhythm durations are not reproducible; the beat counts are.** The
      published duration column sums to 734,525 s — the abstract's "734,528 seconds
      of continuous ECG" — but the annotated windows shipped here span
      **534,398 s** in total, and collapsing the annotations into rhythm segments
      accounts for 526,987 s of that:

      | Figure | Published | Recomputed | Diff |
      |---|---|---|---|
      | Total annotated ECG | 734,528 s | **534,398 s** | −200,130 s |
      | Sum of per-rhythm durations | 734,525 s | 526,987 s (segments) | −207,538 s |

      Derivation: `analysis_end_time_sec − analysis_start_time_sec` summed over
      `metadata.csv`, and `rhythm_segments()` summed over all 482 cases. Since every
      beat and case count agrees to the unit, the annotation content is the same
      data the paper describes — the durations were measured over intervals that
      overlap one another instead of partitioning the windows. Treat the published
      seconds-per-rhythm as an upper bound, and derive durations from
      `rhythm_segments()` if you need figures that add up.

  - type: table
    title: "Three shipped schemas, and other file-level traps"
    headers: ["What", "Where", "What breaks", "ECGBench's handling"]
    rows:
      - ["Boundary markers in `beat_type`", "case **2453** only", "A release-wide `value_counts` reports `Start` and `End` as beat classes", "Moved into `bad_signal_quality_label`, numbered `Start1`/`End1`… as every other case does"]
      - ["Extra `caseid` column", "case **2453** only", "Column-count checks and positional reads", "Dropped"]
      - ["Last two columns swapped", "case **3828** only", "Positional reads put the label text in `bad_signal_quality`", "Read by name"]
      - ["Repeated `time_second`", "111 cases, 250 rows", "`set_index(\"time_second\")` silently misaligns", "Preserved; the positional index is the key"]
      - ["Empty `rhythm_label`", "333 cases, 4,258 rows", "Usually one beat inside a labelled run — a gap, not a boundary", "Kept as `NaN`, never forward-filled; its own segment"]
      - ["Unpaired `StartN`/`EndN`", "17 cases", "Dropping the Start hides a bad-quality stretch", "Closed at the last annotated sample, `closed=False`, logged"]
      - ["`age` is `\">89\"`", "2 cases", "Reading the column as a number yields `NaN`", "Numeric `age_years` plus an `age_censored` flag"]

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench.labels.vitaldb_arrhythmia import (
          bad_signal_intervals,
          case_window,
          load_annotations,
          load_beats,
          load_cases,
          rhythm_segments,
      )

      root = "/path/to/vitaldb-arrhythmia/1.0.0/"

      # No ECGDataset here: the release ships no waveforms, so there is no config,
      # no fold CSVs on the Hub, and nothing for `ecgbench splits` to partition.
      cases = load_cases(root)
      print(len(cases), cases["subjectid"].nunique())
      # 482 473          <- group any split on subjectid, not case_id

      # Every beat in the release, normalised across the three shipped schemas.
      beats = load_beats(root, beats_only=True)
      print(len(beats))
      # 658874
      print(beats["beat_type"].value_counts().to_dict())
      # {'N': 439458, 'S': 184203, 'U': 18972, 'V': 16234, 'P': 7}

      # One case: rhythm runs collapsed into segments with durations that add up.
      segments = rhythm_segments(root, 1018)
      print(len(segments), round(segments["duration_second"].sum(), 1))
      # 41 989.5
      print(segments["rhythm_label"].value_counts(dropna=False).to_dict())
      # {'N': 21, 'VT': 9, <NA>: 8, 'SVTA': 3}

      # Stretches the annotators flagged as too noisy to read.
      print(len(bad_signal_intervals(root, 1018)))
      # 4

      # The waveform is not in this package — fetch it from VitalDB by case_id.
      start, end = case_window(root, 1018)
      print(round(start, 1), round(end, 1))
      # 1523.4 2541.3

      # import vitaldb
      # ecg = vitaldb.load_case(1018, ['SNUADC/ECG_II'], 1/500)['SNUADC/ECG_II']
      # window = ecg[int(start * 500):int(end * 500)]   # the labelled stretch
      # An R-peak at t seconds is sample int(t * 500) of `ecg`, not of `window`.

  - type: links
    title: "Links"
    items:
      - label: "PhysioNet — VitalDB Arrhythmia Database v1.0.0"
        url: "https://physionet.org/content/vitaldb-arrhythmia/1.0.0/"
      - label: "VitalDB — where the waveforms live"
        url: "https://vitaldb.net/"
      - label: "Paper — Scientific Data 2026"
        url: "https://doi.org/10.1038/s41597-026-07076-8"
      - label: "Authors' usage notebook (vitaldb/arrdb)"
        url: "https://github.com/vitaldb/arrdb/"
---
