---
slug: "code-15-pct-telehealth-network-of-minas-gerais-15-pct-subset"
name: "CODE-15% (Telehealth Network of Minas Gerais, 15% subset)"
category: "12-lead-other"
order: 5
status: "completed"
source_url: "https://doi.org/10.5281/zenodo.4916206"
url_label: "zenodo.org"
format: "12-lead · 10.24 s · 400 Hz · HDF5"
patients: "233,770"
records: "345,779"
access: "open"
license: "CC BY 4.0"
origin_institution: "Telehealth Network of Minas Gerais (TNMG)"
origin_country: "Brazil"
leads: 12
paper_title: "Ribeiro et al., Nature Communications, 2020"
paper_doi: "https://doi.org/10.1038/s41467-020-15432-4"
search_keywords: "code 15 percent telehealth minas gerais brazil tnmg zenodo hdf5 h5py deep learning ribeiro mortality follow-up patient-grouped largest"

related:
  - slug: "code-test-827-record-hold-out-test-set"
    relation: "same_cohort"
    shares_records: false
    verified: true
    note: >
      Both are drawn from the CODE cohort collected by the same telehealth
      network, but they hold different recordings — checked against the
      waveforms, not the documentation, since CODE-test ships no identifiers to
      compare. Each of the 827 CODE-test tracings was matched against all
      345,779 CODE-15% tracings on a padding-invariant signature of leads I and
      II; zero matched at any tight threshold, against a positive control in
      which 300 known-present records all matched exactly. So CODE-15% is a
      legitimate training set for a model evaluated on CODE-test. The two do
      differ in limb-lead order — CODE-15% is standard, CODE-test is not — so
      select leads by name when using both.

sections:
  - type: description
    title: "Overview"
    body: |
      345,779 twelve-lead ECGs from 233,770 patients, and **the largest dataset
      in this catalogue**. It is an openly licensed 15% sample of the CODE
      cohort gathered by the Telehealth Network of Minas Gerais (TNMG), a
      Brazilian public telehealth service that reports ECGs recorded by
      non-specialist staff in primary-care units across the state.

      **A record is a row, not a file.** 18 HDF5 parts each hold one
      `(N, 4096, 12)` array, so a signal reference names a row as well as a
      file — `exams_part0.hdf5:tracings:417`. Loading needs
      `pip install ecgbench[hdf5]`; nothing else about the API changes.

      **Six binary abnormality labels ship** — 1dAVb, RBBB, LBBB, SB, ST, AF —
      derived from the TNMG's own cardiologist reports, alongside a `normal_ecg`
      flag, age, sex, a neural-network age estimate, and **mortality follow-up**
      for 233,647 records. That last is unusual in an open ECG release and is
      what makes the dataset usable for survival work as well as classification.

      **The label trap is the important thing on this page.** 308,004 records
      carry none of the six abnormalities, but only 134,657 records are flagged
      normal. The remaining **173,347 have some finding this six-class
      vocabulary does not name** — see "About those counts" below.

      **Patients repeat heavily.** 66,929 of the 233,770 patients contributed
      more than one recording, one of them 38. ECGBench's folds are grouped on
      `patient_id`, so no patient appears in two folds.

      Its limb-lead order is the standard `I, II, III, aVR, aVL, aVF`, which was
      **checked rather than assumed** — its sibling
      [CODE-test]({{ site.baseurl }}/datasets/code-test-827-record-hold-out-test-set.html),
      from the same cohort at the same sampling rate, uses a different one.

  - type: table
    title: "Label distribution"
    headers: ["Class", "Records", "% of 345,779", "Note"]
    rows:
      - ["RBBB", "9,672", "2.80%", "right bundle branch block"]
      - ["ST", "7,584", "2.19%", "sinus tachycardia"]
      - ["AF", "7,033", "2.03%", "atrial fibrillation"]
      - ["LBBB", "6,026", "1.74%", "left bundle branch block"]
      - ["1dAVb", "5,716", "1.65%", "1st-degree AV block"]
      - ["SB", "5,605", "1.62%", "sinus bradycardia"]
      - ["**≥1 of the six**", "**37,775**", "**10.92%**", "3,671 carry more than one"]
      - ["`normal_ecg`", "134,657", "38.94%", "explicitly flagged normal"]
      - ["**neither**", "**173,347**", "**50.13%**", "**not flagged, and not normal**"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the shipped `exams.csv`,
      whose md5 matches Zenodo's published value
      (`0107516d3f63864498fb77d15799cc95`) exactly. The headline figures —
      **345,779 records** and **233,770 patients** — agree with the release
      description.

      **An empty label list is not a normal ECG, and this is the mistake the
      dataset invites.** The six flags name six specific findings. Half the
      release (173,347 records, 50.1%) is neither flagged with any of them nor
      flagged `normal_ecg`: those recordings have something else — an axis
      deviation, an old infarct, a nonspecific ST change — that this vocabulary
      cannot express. A model trained on the six flags alone silently treats all
      173,347 as confident negatives for every class. ECGBench exposes
      `normal_ecg` alongside `abnormality_codes` precisely so the two cases stay
      distinguishable, and the fold stratification separates `NORMAL` from
      `OTHER` for the same reason. Restrict your negatives to `normal_ecg` if
      you need clean ones.

      **The class table is multi-label and does not sum to 37,775.** 3,671
      records carry more than one of the six, so the six class counts total
      41,636 against 37,775 records.

      **`nn_predicted_age` is a model output, not an observation.** It is an age
      estimated from the tracing by a neural network. Exposed because it ships;
      training against it is training against another model.

      **Missing mortality follow-up means "not followed up", not "survived".**
      `death` and `timey` are blank for 112,132 records and ship as an
      object-dtype column mixing `True`/`False` with NaN. ECGBench exposes
      `death` as a nullable boolean and adds an explicit `has_followup` flag,
      because reading NaN as `False` converts 112,132 unknowns into 112,132
      survivors. Of the 233,647 records with follow-up, **8,341 died**.

      Demographics: ages run **17–100** (mean 53.2) and **40.3% of records are
      from men**.

      **The class label is not a fresh expert read of each tracing.** The six
      flags were derived from the reporting cardiologists' free text by a
      combination of text mining and the automatic Minnesota coding, so they
      inherit that pipeline's error rate as well as the reporters'.

  - type: description
    title: "Where the files come from"
    body: |
      Zenodo serves 19 separate files — `exams.csv` plus 18 `exams_part*.zip`
      archives of roughly 2.7 GB each, expanding to 66 GB of HDF5. There is no
      single URL to auto-download from, so pass `--data-path`.

      Three things about the layout that a naive reader gets wrong, all of which
      ECGBench's splitter handles and checks:

      **Every part has one more row than it has records.** Parts 0–16 hold
      20,001 rows for 20,000 records and part 17 holds 5,780 for 5,779. The
      extra is an all-zero padding row with `exam_id` 0 that appears in no CSV.

      **`exams.csv` is not in file order.** Its `trace_file` column says which
      part holds a record but not where in it, and its rows within a part do not
      follow that part's own `exam_id` dataset. The row index has to come from
      that dataset; taking it from the CSV's row number mislabels almost
      everything.

      **The samples are already in millivolts.** The sibling release's README
      claims a scale of "1e-4V … multiplied by 1000 in order to obtain the
      signals in V", which is self-contradictory — the first clause puts a
      median R wave at 0.17 mV and the second at 1,750 V. Measured here: the
      median per-record peak is **4.27 mV** and the median lead-II R amplitude
      **1.75 mV**. `signal_unit_scale` is 1.0.

      Zenodo publishes checksums only for the `.zip` archives, so an extracted
      copy cannot be checked against the provider directly. ECGBench verifies it
      structurally instead, on every run: all 18 arrays must have the expected
      `(N, 4096, 12)` shape, and each part's `exam_id` set must equal the set
      `exams.csv` assigns to it. A mismatch raises rather than dropping records
      quietly.

      The lead order was likewise derived from the arrays rather than taken on
      trust, over 1,200 records spanning parts 0, 5, 11 and 17: `III = II − I`,
      `aVR = −(I+II)/2`, `aVL = I − II/2` and `aVF = II − I/2` all hold to a
      median relative error under 2%, while every other assignment is off by
      more than 140%.

  - type: table
    title: "Validation summary (400 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "345,779", "all records, with is_valid + quality_issues"]
      - ["clean", "337,238", "97.53% pass rate"]
      - ["excluded", "8,541", "6,798 amplitude_outlier, 1,773 missing_leads, 62 flat_line"]

  - type: description
    title: "About the excluded records"
    body: |
      `amplitude_range_mv` here is `[-20, 20]`, **not** ECGBench's usual
      `[-10, 10]`, and the same value is used for CODE-test so the two releases
      of one cohort are cleaned to a single standard.

      The reason is the cohort itself. These are telehealth recordings made by
      non-specialist staff on portable tele-electrocardiographs in primary-care
      units, so they carry considerably more electrode artefact and baseline
      wander than a hospital dataset: the **median per-record peak is 4.27 mV**,
      against 1.74 mV for the hospital-collected
      [SPH]({{ site.baseurl }}/datasets/shandong-provincial-hospital-ecg-database-sphdb.html).
      Measured over 20,000 records of part 3, the exclusion rate would be 38.7%
      at ±5 mV, 11.8% at ±10, 6.5% at ±15, 2.0% at ±20 and 0.15% at ±30. ±10
      would discard roughly 40,000 legitimate recordings; ±20 keeps the
      exclusion near the 1–2% the rest of the catalogue sits at while still
      catching the railed leads — the worst record in that sample peaks at
      281 mV.

      What that produced over all 345,779 records: **6,798 records fail
      amplitude_outlier** (10,161 lead-level failures — most have one or two bad
      leads, not twelve), **1,773 fail missing_leads** with a lead recorded as
      exactly zero throughout, and **62 fail flat_line**. Only 91 records fail
      more than one check. Nothing fails `nan_values` or `truncated_signal`:
      every array is complete and every record is exactly 4,096 samples.

      Fold membership is identical between `original/` and `clean/` — `clean/`
      is a row subset, not a re-split — so a model trained on `clean/` can be
      scored against `original/` for the same fold without re-partitioning.

  - type: table
    title: "Splits"
    headers: ["Split", "Folds", "clean", "original"]
    rows:
      - ["train", "1–8", "269,733", "276,624"]
      - ["val", "9", "33,735", "34,578"]
      - ["test", "10", "33,770", "34,577"]

  - type: description
    title: "How the folds were made"
    body: |
      Ten folds, **grouped on `patient_id`** and stratified on the rarest of the
      six abnormalities each record carries, with the unflagged records split
      into `NORMAL` and `OTHER` rather than pooled — those two are different
      things, as above.

      Rarest-wins is used because the release ranks nothing. Taking the
      first-listed flag instead would starve the small classes in favour of
      RBBB.

      What that produced, measured on the output:

      - **no patient spans two folds** — zero of 233,770, in both versions;
      - fold sizes 34,577–34,578 in `original/` and 33,686–33,787 in `clean/`;
      - every stratification class lands within one record of an even tenth in
        every fold — the normal fraction is 38.94% in all ten, against 38.94%
        overall.

      **One caveat that patient grouping cannot fix.** The source contains a
      small number of byte-identical duplicate recordings filed under *different*
      `patient_id`s — 47 non-degenerate records among part 0's 20,000 (0.24%),
      in groups that sometimes span patients. Grouping on `patient_id` cannot
      see those, so a very small amount of duplicate-record leakage between
      folds is possible. It comes from the source, is documented here rather
      than silently accepted, and is far below the leakage that ignoring
      `patient_id` altogether would cause.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # All 18 parts must be extracted alongside exams.csv.
      for f in exams_part*.zip; do unzip -n "$f"; done

      ecgbench splits --dataset code15 --data-path /path/to/code-15/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      # pip install ecgbench[torch,hdf5]   <- h5py is needed for this dataset
      from ecgbench import ECGDataset

      ds = ECGDataset(
          "code15",
          split="train",
          data_path="/path/to/code-15/",
          window=(48, 4000),       # the 10 s inside the symmetric zero padding
          labels=True,
      )

      len(ds)                                      # 269733
      ds[0]["signal"].shape                        # (12, 4000)
      ds[0]["record_id"]                           # 26

      # THE TRAP: an empty list is not a normal ECG. Read both.
      ds[0]["labels"]["abnormality_codes"]         # '' — none of the six
      ds[0]["labels"]["normal_ecg"]                # False -> some OTHER finding
      ds[0]["labels"]["stratify_class"]            # 'OTHER'  (folds only)

      # Mortality follow-up, with its missingness intact. death is None for a
      # record that was never followed up, not False.
      ds[0]["labels"]["death"]                     # None
      ds[0]["labels"]["has_followup"]              # False — "not followed up"

      # Standard lead order — but its sibling CODE-test is NOT standard, so
      # select by name if you use both releases.
      ds.config.lead_names
      # ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

      both = ECGDataset("code15", split="train", data_path="/path/to/code-15/",
                        leads=["aVR", "aVL", "aVF"])
      both[0]["signal"].shape                      # (3, 4096)

      # Samples are already millivolts, despite the sibling README's "1e-4V".
      ds.config.signal_unit_scale                  # 1.0

  - type: links
    title: "References"
    items:
      - { label: "Zenodo record (the copy ECGBench was verified against)", url: "https://doi.org/10.5281/zenodo.4916206" }
      - { label: "Ribeiro et al., Nature Communications 11, 1760 (2020)", url: "https://doi.org/10.1038/s41467-020-15432-4" }
      - { label: "antonior92/automatic-ecg-diagnosis — the authors' companion code", url: "https://github.com/antonior92/automatic-ecg-diagnosis" }
---
