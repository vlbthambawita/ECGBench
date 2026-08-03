---
slug: "leipzig-heart-center-ecg-database"
name: "Leipzig Heart Center ECG Database"
category: "12-lead-physionet"
order: 12
status: "completed"
source_url: "https://physionet.org/content/leipzig-heart-center-ecg/1.0.0/"
url_label: "physionet.org"
format: "12-lead + intracardiac EGM · 14-20 ch · 78 s-2.5 h · 977 Hz"
patients: "39"
records: "39"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Leipzig Heart Center"
origin_country: "Germany"
leads: 12
paper_title: "Leipzig Heart Center ECG-Database: Arrhythmias in children and patients with congenital heart disease"
paper_doi: "https://doi.org/10.13026/7a4j-vn37"
search_keywords: "leipzig heart center germany intracardiac electrogram paediatric children congenital heart disease avrt avnrt tetralogy of fallot supraventricular tachycardia electrophysiological study ablation"

sections:
  - type: description
    title: "Overview"
    body: |
      **39 recordings from 39 subjects, made during electrophysiological studies**
      at the Leipzig Heart Center — and unusually for this catalogue, every record
      carries the 12-lead surface ECG *and* the intracardiac electrograms from the
      catheters, sampled together at 977 Hz.

      The cohort is the point. Two groups, deliberately chosen for conditions the
      big adult datasets barely contain:

      - **29 children** (`x001`–`x0029`, ages 5–18) referred for ablation of
        supraventricular tachycardia — 13 with AV nodal reentrant tachycardia, 16
        with AV reentrant tachycardia including 5 with Wolff-Parkinson-White and one
        with permanent junctional reciprocating tachycardia. The children's records
        also give the accessory-pathway location.
      - **10 adults** (`x100`–`x109`, ages 21–64) with **repaired Tetralogy of
        Fallot**, 8 of them with ventricular tachycardia.

      Two cardiologists annotated every beat — one annotating, the second checking
      and correcting — using LightWAVE. The result is **118,214 annotations** over
      18.5 hours, with beat class, tachycardia mechanism, rhythm segment and signal
      quality all recorded. Signals are the raw CardioLab export, filtered
      0.05–100 Hz at acquisition and not touched since.

      Licence is **ODC-By 1.0** and access is open, so ECGBench publishes the fold
      CSVs to the Hub and they download automatically.

  - type: description
    title: "Records are not all 12-lead — read this before indexing a channel"
    body: |
      The README describes two layouts: children with the 12-lead ECG plus 5
      coronary-sinus, 1 right-ventricular-apex and 1 ablation channel (19), and
      adults with the ECG plus RVA and ABL, with the coronary sinus catheter "in
      some studies". **The files hold six layouts and four different channel
      counts.** Verified against all 39 headers:

      | Channels | Layout after the 12 ECG leads | Records |
      |---|---|---|
      | 19 | `ABL12 RVA12 CS12 CS34 CS56 CS78 CS90` | 27 |
      | 14 | `ABL12 RVA12` | 5 (`x102 x105 x107 x108 x109`) |
      | 18 | `RVA12 CS12…CS90` — **no `ABL12`** | 4 (`x002 x005 x0015 x0029`) |
      | 20 | `ABL12 ABL_uni RVA12 CS12…CS90` | 1 (`x0023`) |
      | 20 | `ART ABL12 RVA12 CS12…CS90` | 1 (`x0028`) |
      | 19 | `ABL12 CS12…CS90 RVA12` — **`RVA12` last** | 1 (`x100`) |

      Three consequences:

      1. **Only channels 0–11 are stable.** The 12-lead ECG is the same channel in
         the same position in all 39 records. Channel index 12 is `ABL12` in 28
         records, `RVA12` in 4 and `ART` in one.
      2. **`ABL_uni` and `ART` are documented nowhere** in the release — not in the
         README, not in `dataset_info.csv`. `ART` is most likely an arterial
         pressure trace, but the release does not say, so ECGBench does not guess.
      3. **Four children's records carry no ablation channel at all**, against the
         README's "every recording of the children includes … an ablation catheter".

      So `config.lead_names` declares the **12 surface ECG leads and nothing else**,
      deliberately: there is no dataset-wide intracardiac order for
      `ECGDataset(leads=…)` to resolve against. To reach an intracardiac channel,
      look its name up in that record's own header:

      ```python
      from ecgbench.labels.leipzig_heart_center_ecg import channel_index

      names = labels["channel_names"]            # 'I|II|…|ABL12|RVA12|CS12|…'
      channel_index(names, "RVA12")              # 13 in most records, 18 in x100
      channel_index(names, "CS12")               # None in the 14-channel records
      ```

      And pass `leads=` if you want a homogeneous batch — without it a batch mixes
      14-, 18-, 19- and 20-channel tensors.

  - type: description
    title: "Length varies by two orders of magnitude"
    body: |
      Records run from **77.7 s** (`x0027`) to **2 h 30 m 32 s** (`x003`), median
      699 s. Two things follow.

      `validation.expected_samples` is deliberately **empty** — a truncation check
      would fail almost every record — and `window=(start, length)` is how you batch
      at all. **60 s (58,620 samples at 977 Hz) is the longest window that fits every
      record**; anything longer raises `WindowOutOfRangeError` on `x0027`.

      Recording length also correlates with case complexity, so a naive per-beat
      pooling is dominated by a handful of long children's studies: `x001`, `x003`,
      `x004` and `x006` alone contribute 61,806 of the 118,214 annotations.

  - type: table
    title: "Diagnoses, and what folds are built on"
    headers: ["Diagnosis (shipped)", "Records", "Family (fold label)", "Records"]
    rows:
      - ["`AVNRT`", "13", "`AVNRT`", "13"]
      - ["`AVRT`", "10", "`AVRT`", "16"]
      - ["`AVRT-WPW`", "5", "↳ incl. WPW and PJRT", ""]
      - ["`AVRT-PJRT`", "1", "", ""]
      - ["`TOF with VT`", "8", "`TOF`", "10"]
      - ["`TOF without VT`", "1", "↳ all three presentations", ""]
      - ["`TOF with nsVT`", "1", "", ""]

  - type: description
    title: "The fold label is a coarsening — do not train on it"
    body: |
      The shipped `diagnosis` has **seven classes over 39 records, three of them
      singletons**. Ten folds cannot hold that, so the label loader derives a
      `diagnosis_family` — the leading token, giving AVRT (16), AVNRT (13) and TOF
      (10) — and folds are stratified on that. The result is even: every one of the
      ten folds holds 3–4 records with all three families represented.

      `stratify_class` is **for fold construction only**. It throws away whether an
      AVRT had a manifest accessory pathway and whether an adult's VT was sustained,
      which is the very thing the adult cohort is interesting for. Train on
      `diagnosis`, and better still on the beat-level `tachy_*` columns: a child
      referred for AVNRT still contributes AFIB and VT beats, and those are recorded.

      Because the families map cleanly onto the cohorts — all TOF are adults — the
      fold stratification also spreads children and adults evenly without being
      asked to.

      **With 39 records, prefer cross-validation to the default 8/1/1 layout**,
      which leaves 3 records in test and 4 in val. Pass `split=None` with
      `fold_numbers=[…]` to select folds directly.

  - type: table
    title: "Annotations — 118,214 in all"
    headers: ["Symbol", "Count", "Share of the 113,924", "Meaning"]
    rows:
      - ["`N`", "50,973", "44.74%", "normal (sinus) beat — 13,786 of them carry the `N-Prex` pre-excitation marker"]
      - ["`X`", "29,477", "25.87%", "tachycardia beat; every one names its mechanism in an aux string"]
      - ["`/`", "13,779", "12.09%", "paced beat — 8,031 atrial, 5,748 ventricular"]
      - ["`R`", "12,023", "10.55%", "complete right bundle branch block"]
      - ["`A`", "3,469", "3.05%", "premature atrial (2,786 pre-excited)"]
      - ["`V`", "2,133", "1.87%", "premature ventricular"]
      - ["`J`", "1,710", "1.50%", "premature junctional"]
      - ["`F` `a` `f` `L` `b` `j`", "360", "0.32%", "fusion, aberrated atrial, paced-normal fusion, cLBBB, AV block 1°, junctional escape"]
      - ["**tabulated total**", "**113,924**", "**100%**", "the figure the release reports"]
      - ["`Q`", "1,824", "—", "unclassifiable — **not** in the README's table"]
      - ["`~`", "228", "—", "signal-quality change — not a beat at all"]
      - ["`+`", "2,238", "—", "rhythm-segment markers"]
      - ["**file total**", "**118,214**", "—", "what `wfdb.rdann` actually returns"]

  - type: table
    title: "Which tachycardia each X beat was"
    headers: ["Aux string", "Beats", "Records", "Meaning"]
    rows:
      - ["`AVRT`", "17,599", "16", "atrioventricular reentrant tachycardia"]
      - ["`AVNRT`", "8,097", "13", "AV nodal reentrant tachycardia"]
      - ["`VT`", "1,914", "16", "ventricular tachycardia"]
      - ["`AFIB`", "1,235", "6", "atrial fibrillation"]
      - ["`avnrt`", "231", "6", "aberrated AVNRT (lower case)"]
      - ["`AVNRT+BII`", "161", "2", "AVNRT with 2° AV block"]
      - ["`AFL`", "109", "1", "atrial flutter"]
      - ["`IVR`", "82", "4", "accelerated idioventricular rhythm"]
      - ["`EAT`", "28", "1", "ectopic atrial tachycardia"]
      - ["`avrt`", "21", "2", "aberrated AVRT (lower case)"]

  - type: description
    title: "About those counts"
    body: |
      Recomputed from the files. **All 126 files match the release's own
      `SHA256SUMS.txt`**, so everything below is the data as published rather than a
      damaged local copy.

      **The 113,924 beats reproduce exactly**, and so does every published
      percentage. The abstract's breakdown — 33% normal sinus, 12% sinus with
      pre-excitation, 11% cRBBB, 15% AVRT, 8% AVNRT, 12% paced (7% atrial, 5%
      ventricular), 7% premature — recomputes as 32.6%, 12.1%, 10.6%, 15.4%, 7.5%,
      12.1% (7.1%/5.0%) and 6.7%. Nothing to correct.

      **But 113,924 is not the number of annotations in the files; 118,214 is.** The
      published figure counts exactly the beat classes the README tabulates. It
      omits 1,824 `Q` (unclassifiable) beats, 228 `~` signal-quality change markers
      — which are not beats, and which the `ANNOTATORS` file does mention — and
      2,238 `+` rhythm markers. So the release is right about what it counts; a
      naive `len(annotation.symbol)` is 3.8% higher. ECGBench exposes `n_beats`
      (113,924), `n_unclassifiable`, `n_quality_marks`, `n_rhythm_changes` and
      `n_annotations` separately for that reason.

      **The total recording time does not reconcile.** The abstract reports 1,075.85
      minutes. The headers give **1,107.02 minutes** (66,421 s, 18.45 h), and the
      shipped `ecg_duration` values agree with them to 1,107.07 minutes. The
      31.2-minute gap is almost exactly the length of `x005` (30.9 min): the total
      excluding `x005` is 1,076.09 minutes, still 14 s off. `x005` is also the one
      record whose `ecg_duration` disagrees with its header — 0:30:58.974 against
      1,856.169 s, a 2.8 s difference, where the other 38 agree to within 0.01 s. So
      the published total appears to omit `x005`, but we cannot confirm that from
      the release. The recomputed 18.45 hours is what the entry above reports.

      The min and max **do** match: 0:01:17 to 2:30:31, recomputed as 77.66 s and
      9,031.75 s.

      **One age is malformed.** `x007` ships `age` = `.14.3`, which no float parser
      accepts. The loader strips the single leading `.` — giving 14.3, in range for
      this cohort — logs a warning, and keeps the verbatim string in `age_raw` so
      the repair is visible rather than assumed.

      **`dataset_info.csv` has one row wrong.** Its `PVC` row reads
      `PVC,PVC,V,Premature ventricular beats`, i.e. symbol `PVC` with aux string
      `V`, while the README and the data both use symbol `V` with no aux string. No
      annotation anywhere carries the symbol `PVC`.

      **The children's CSV column is spelled `ap_loacation`.** ECGBench exposes it
      as `ap_location`; the typo is the source's. It is empty for the 13 children
      with AVNRT, which has no accessory pathway — that is correct, not missing data.

  - type: description
    title: "Signal quality: amplitude clipping, and why the range is ±52 mV"
    body: |
      **The signals are clipped at the recording amplifier's rail**, and the rail
      differs per channel and per record: ±10.24 mV on most channels, ±20.48 mV on
      `x007`, ±51.20 mV on `x002`'s `RVA12`, ±49.83 mV on `x0023`'s `ABL_uni`. Six
      records reach the rail on the **surface ECG** itself — `x001`, `x002`, `x003`,
      `x004`, `x006` and `x109` — with `x001` clipped on 11 of its 12 ECG channels.

      `amplitude_range_mv` is therefore set to **±52.0**, which is *not* a
      physiologic bound and is not comparable with the other datasets' ranges. These
      records mix surface ECG with intracardiac electrograms, and
      `check_amplitude_outlier` applies one range to every channel: at ±10 mV, **24
      of the 39 records fail** — not because anything is wrong with them, but
      because an intracardiac electrogram is not a surface-ECG amplitude. ±52 sits
      just above the widest rail measured, so it excludes nothing while still
      catching a record that is broken outright.

      **Use the cardiologist's own signal-quality annotations instead.** The 228 `~`
      markers, exposed as `n_quality_marks`, appear in 10 records — including all six
      of the clipped ones — which makes them the better quality signal here.

      Validation is otherwise clean: **0 NaN samples, 0 absent channels, 0 corrupt
      headers**. Exactly **one record is excluded** from `clean/`: `x0027`, whose
      `ABL12` channel is a constant 0.076 mV for all 77.7 s — a disconnected
      ablation catheter, caught by `flat_line`. So `clean/` holds 38 records and
      `original/` 39.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset
      from ecgbench.labels.leipzig_heart_center_ecg import ECG_LEADS, channel_index

      # window= is mandatory: records run 77.7 s to 2.5 h. leads= makes the batch
      # homogeneous — without it you get 14-, 18-, 19- and 20-channel tensors.
      ds = ECGDataset("leipzig_heart_center_ecg", split="train",
                      data_path="/path/to/leipzig-heart-center-ecg/1.0.0/",
                      window=(0, 9770),          # 10 s at 977 Hz; 58620 is the max
                      leads=list(ECG_LEADS),     # the 12 surface leads
                      labels=True)

      len(ds)                              # 31  (of 38 in clean/)
      ds[0]["signal"].shape                # (12, 9770)
      ds[0]["record_id"]                   # 'x001'
      ds[0]["labels"]["diagnosis"]         # 'AVRT'          <- train on this
      ds[0]["labels"]["diagnosis_family"]  # 'AVRT'          <- folds only
      ds[0]["labels"]["ap_location"]       # 'right posteroseptal'
      ds[0]["labels"]["age"]               # 6.6
      ds[0]["labels"]["n_signals"]         # 19  -- but 14, 18 or 20 elsewhere
      ds[0]["labels"]["n_beats"]           # 15937

      # Reaching an intracardiac channel: by name, from THIS record's header.
      names = ds[0]["labels"]["channel_names"]
      channel_index(names, "RVA12")        # 13   (18 in x100, None if absent)
      channel_index(names, "CS12")         # 14   (None in the 14-channel records)

      # labels_df is aligned positionally with metadata_df and carries a
      # RangeIndex, so re-index it by record name for per-record lookups.
      df = ds.labels_df.copy()
      df.index = ds.metadata_df["record_name"].to_numpy()
      df["diagnosis"].value_counts().to_dict()
      # {'AVNRT': 10, 'AVRT': 8, 'TOF with VT': 6, 'AVRT-WPW': 4,
      #  'AVRT-PJRT': 1, 'TOF without VT': 1, 'TOF with nsVT': 1}

      # Beat-level tachycardia, which is richer than the subject-level diagnosis:
      int(df["tachy_AFIB"].sum())          # 1235 beats, in 5 records of this split
      int(df["n_quality_marks"].sum())     # 214  <- the quality signal to use

      # 39 records: prefer cross-validation to the 3-record default test split.
      cv = ECGDataset("leipzig_heart_center_ecg", split=None, fold_numbers=[1, 2, 3],
                      data_path="/path/to/leipzig-heart-center-ecg/1.0.0/",
                      window=(0, 9770), leads=list(ECG_LEADS))
      len(cv)                              # 12

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/leipzig-heart-center-ecg/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/7a4j-vn37" }
      - { label: "WFDB Python package (needed to read the .atr annotations)", url: "https://wfdb.readthedocs.io/" }
---
