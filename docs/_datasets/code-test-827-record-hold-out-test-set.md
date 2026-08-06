---
slug: "code-test-827-record-hold-out-test-set"
name: "CODE-test (827-record hold-out test set)"
category: "12-lead-other"
order: 6
status: "completed"
source_url: "https://doi.org/10.5281/zenodo.3765780"
url_label: "zenodo.org"
format: "12-lead · 7–10 s padded to 10.24 s · 400 Hz · HDF5"
patients: "827"
records: "827"
access: "open"
license: "CC BY 4.0"
origin_institution: "Universidade Federal de Minas Gerais / TNMG"
origin_country: "Brazil"
leads: 12
paper_title: "Ribeiro et al., Nature Communications, 2020"
paper_doi: "https://doi.org/10.1038/s41467-020-15432-4"
search_keywords: "code test 827 zenodo hdf5 h5py brazil minas gerais tnmg ribeiro hold-out evaluation annotator agreement cardiologist residents medical students gold standard dnn"

sections:
  - type: description
    title: "Overview"
    body: |
      827 twelve-lead ECGs from 827 distinct patients of the CODE cohort,
      published as the evaluation set of Ribeiro et al. 2020. It is one of the
      smallest datasets in this catalogue and one of the most useful, because of
      how heavily it is annotated.

      **Seven independent readings of every record.** Six binary abnormalities —
      1dAVb, RBBB, LBBB, SB, AF, ST — were labelled independently by two
      cardiologists, with a third senior specialist adjudicating every
      disagreement to produce the gold standard. The same 827 tracings were then
      annotated separately by two 4th-year cardiology residents, two 3rd-year
      emergency residents and two 5th-year medical students, and scored by the
      paper's own neural network. ECGBench exposes all seven side by side, so
      reader-agreement analysis needs no extra plumbing.

      **This is an evaluation set, and ECGBench still gives it ten folds.** That
      is the framework applying its convention uniformly, not a recommendation.
      The release's intended use is all 827 records as a single hold-out set —
      see the loading snippet below for the one-line way to get exactly that.
      Train on [CODE-15%]({{ site.baseurl }}/datasets/code-15-pct-telehealth-network-of-minas-gerais-15-pct-subset.html),
      not on eight tenths of a test set.

      **Nothing in the release has an identifier.** `ecg_tracings.hdf5` holds one
      `(827, 4096, 12)` array and nothing else; `attributes.csv` and the seven
      annotation CSVs each hold 827 unkeyed rows aligned to it by position
      alone — "the i-th line corresponds to the i-th tracing", per the bundled
      README. So ECGBench's `record_id` **is** the row index, 0–826, and every
      source file is refused unless it has exactly 827 rows. A positional join
      against a table of the wrong length does not partially match; it silently
      mislabels everything.

      **Its lead order is not standard.** The augmented leads run
      `aVL, aVF, aVR`, so `signal[3]` is aVL. Its sibling CODE-15% — same cohort,
      same 400 Hz, same array layout — uses the standard `aVR, aVL, aVF`. Select
      leads by name if you touch both.

  - type: table
    title: "Every annotator's readings, side by side"
    headers: ["Annotator", "1dAVb", "RBBB", "LBBB", "SB", "AF", "ST", "≥1 finding", "Multi-label"]
    rows:
      - ["**gold_standard**", "28", "34", "30", "16", "13", "37", "**146**", "12"]
      - ["cardiologist1", "30", "35", "28", "13", "10", "30", "134", "12"]
      - ["cardiologist2", "26", "34", "30", "13", "14", "34", "138", "13"]
      - ["cardiology_residents", "21", "38", "27", "18", "13", "31", "137", "11"]
      - ["emergency_residents", "36", "27", "27", "17", "10", "37", "143", "11"]
      - ["medical_students", "43", "35", "29", "16", "21", "34", "149", "28"]
      - ["dnn", "30", "38", "30", "18", "10", "38", "147", "17"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the shipped files, not
      copied from the paper. The local copy's `data.zip` matches Zenodo's own
      md5 (`cef8fe457abf4b8f66a34dc1269b4ced`) exactly, so it is the v1.0.3
      release unmodified. **827 records** and **827 patients** agree with the
      paper.

      Four things the table above will mislead you about if left unqualified:

      **cardiologist1 and cardiologist2 are not independent of the gold
      standard.** The gold standard *is* their agreement, with a third specialist
      breaking the ties. Scoring either against it measures how often that
      reader was overruled, not inter-rater reliability between strangers.

      **The three non-expert rows are each two people.** Every one of the
      residents' and students' sets was produced by a pair who annotated half
      the records each, and the release does not say which record went to which
      annotator. A gap between `medical_students` and `gold_standard` is partly
      a difference in training level and partly a difference between two
      individuals; nothing in the data can separate the two.

      **It is multi-label, so the rows do not sum to the "≥1 finding" column.**
      12 records carry more than one gold-standard finding, which is why the six
      class counts total 158 against 146 records.

      **681 records (82.3%) carry none of the six — and that is not "normal".**
      This release publishes no normal flag at all, unlike CODE-15%, so the
      negative class here means strictly "none of these six findings" and
      nothing more.

      Demographics: ages run **17–97** (mean 54.6) and the sample is **61.2%
      women** (506 F / 321 M). That is not the parent CODE-15% balance, which is
      40.3% men — worth knowing before treating this as a representative sample
      of the cohort.

      The rarest classes are very small in absolute terms: **AF appears in 13
      records and SB in 16**. Any per-class metric computed here has wide
      confidence intervals however the folds fall.

  - type: description
    title: "Where the files come from"
    body: |
      One Zenodo download, `data.zip` (218 MB), which extracts to a `data/`
      directory. **`--data-path` must be that directory**, not its parent — it is
      the one holding `ecg_tracings.hdf5`, `attributes.csv` and `annotations/`.
      Getting this wrong is the commonest slip, and ECGBench's error message
      says so.

      `download_url` is `null` in the config even though a single URL exists,
      because the splitter generates the metadata CSV that the config names.

      Two facts about the waveforms, both verified from the arrays rather than
      taken from the README:

      **The samples are already in millivolts.** The bundled README says the
      signals are "at the scale 1e-4V: so it should be multiplied by 1000 in
      order to obtain the signals in V", which cannot both be true — the first
      clause puts a median R wave at 0.15 mV and the second at 1,480 V. Measured:
      the median per-record peak is **3.51 mV** and the median lead-II R
      amplitude **1.48 mV**. `signal_unit_scale` is therefore 1.0.

      **The lead order really is the non-standard one the README states.**
      Checked across all 827 records: taking channels 0 and 1 as I and II,
      channel 3 matches `I − II/2` (aVL), channel 4 matches `II − I/2` (aVF) and
      channel 5 matches `−(I+II)/2` (aVR), each to a median relative error under
      2.3%, while every other assignment is off by more than 150%.

      Every record is stored as 4,096 samples at 400 Hz. The underlying
      acquisitions were 7 s (2,800 samples) or 10 s (4,000), zero-padded
      symmetrically to that length, so a window into the middle of the record is
      not guaranteed to be all signal.

  - type: table
    title: "Validation summary (400 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "827", "all records, with is_valid + quality_issues"]
      - ["clean", "808", "97.7% pass rate"]
      - ["excluded", "19", "14 missing_leads, 5 amplitude_outlier"]

  - type: description
    title: "About the 19 excluded records"
    body: |
      **14 records have a lead recorded as exactly zero throughout** — a dead
      channel, caught by `missing_leads`. One lead each; no record has two.

      **5 records exceed the amplitude range**, across 29 lead-level failures.

      `amplitude_range_mv` here is `[-20, 20]`, **not** ECGBench's usual
      `[-10, 10]`, and the same value is used for CODE-15% so the two releases
      are cleaned to one standard. The reason is the cohort: these are telehealth
      recordings made by non-specialist staff in primary-care units, so they
      carry far more electrode artefact and baseline wander than a hospital
      dataset. Measured over all 827 records, the exclusion rate would be 21.5%
      at ±5 mV, 6.7% at ±10, 0.6% at ±20 and 0.2% at ±30, against a median
      per-record peak of 3.51 mV. ±10 would discard 55 legitimate recordings from
      an 827-record evaluation set.

      No record fails any other check: no NaN samples, no flat leads, no
      unreadable arrays.

      Fold membership is identical between `original/` and `clean/` — `clean/` is
      a row subset, not a re-split.

  - type: table
    title: "Splits"
    headers: ["Split", "Folds", "clean", "original"]
    rows:
      - ["train", "1–8", "646", "663"]
      - ["val", "9", "82", "82"]
      - ["test", "10", "80", "82"]

  - type: description
    title: "How the folds were made"
    body: |
      Ten folds, stratified on the rarest gold-standard abnormality each record
      carries (`NONE` for the 681 carrying none), with **no patient grouping** —
      the release states one record per patient and ships no patient identifier,
      so there is nothing to group on and nothing that needs it.

      Rarest-wins matters more here than anywhere else in ECGBench. AF appears in
      13 of 827 records, barely above the ten a ten-fold stratified split
      requires; any reduction that let a commoner class win would leave folds
      with no AF example at all. As built, AF lands 1–2 records per fold and SB
      1–2 per fold, and fold sizes run 82–83 records.

      Again: these folds exist so the tooling is uniform across the catalogue.
      For the use the release was published for, take all ten.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      unzip data.zip          # creates data/

      # --data-path is the data/ directory, not its parent.
      ecgbench splits --dataset code_test --data-path /path/to/code-test/data/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      # pip install ecgbench[torch,hdf5]   <- h5py is needed for this dataset
      from ecgbench import ECGDataset

      # WHAT YOU PROBABLY WANT: all 827 records as one hold-out set, which is
      # what the release was published for. split=None selects by fold alone.
      ds = ECGDataset(
          "code_test",
          split=None,
          fold_numbers=list(range(1, 11)),
          data_path="/path/to/code-test/data/",
          labels=True,
      )
      len(ds)                                     # 808  (827 with version="original")

      # The record id IS the row index into the tracings array. The release
      # assigns no identifier of any kind.
      ds[0]["record_id"]                          # 0
      ds[0]["signal"].shape                       # (12, 4096)

      # The gold standard, unprefixed — the evaluation target.
      ds[0]["labels"]["abnormality_codes"]        # '' — none of the six
      ds[0]["labels"]["n_abnormalities"]          # 0

      # ...and every other annotator, prefixed. This is the point of the
      # release, so here is a record they actually disagree about: the gold
      # standard and the DNN call LBBB, cardiologist 1 and the students do not.
      ds[1]["labels"]["abnormality_codes"]                  # 'LBBB'
      ds[1]["labels"]["cardiologist1_abnormality_codes"]    # ''
      ds[1]["labels"]["medical_students_abnormality_codes"] # ''
      ds[1]["labels"]["dnn_abnormality_codes"]              # 'LBBB'

      # 72 of the 808 records in this set have at least one such disagreement
      # among those four readings.

      # NON-STANDARD lead order: signal[3] is aVL, not aVR.
      ds.config.lead_names
      # ['I','II','III','aVL','aVF','aVR','V1','V2','V3','V4','V5','V6']

      # CODE-15% stores the SAME cohort with the standard order, so index-based
      # access crosses three leads between the two. Select by name instead:
      both = ECGDataset(
          "code_test",
          split=None,
          fold_numbers=list(range(1, 11)),
          data_path="/path/to/code-test/data/",
          leads=["aVR", "aVL", "aVF"],
      )
      both[0]["signal"].shape                     # (3, 4096)

      # Samples are already millivolts, despite the README's "1e-4V".
      ds.config.signal_unit_scale                 # 1.0

      # The conventional train/val/test division is there too, if you want it —
      # but see the note above before training on it.
      train = ECGDataset("code_test", split="train",
                         data_path="/path/to/code-test/data/")
      len(train)                                  # 646

  - type: links
    title: "References"
    items:
      - { label: "Zenodo record (the copy ECGBench was verified against)", url: "https://doi.org/10.5281/zenodo.3765780" }
      - { label: "Ribeiro et al., Nature Communications 11, 1760 (2020)", url: "https://doi.org/10.1038/s41467-020-15432-4" }
      - { label: "antonior92/automatic-ecg-diagnosis — the authors' companion code", url: "https://github.com/antonior92/automatic-ecg-diagnosis" }
---
