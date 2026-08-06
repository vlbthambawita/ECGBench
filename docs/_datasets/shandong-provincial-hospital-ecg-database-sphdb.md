---
slug: "shandong-provincial-hospital-ecg-database-sphdb"
name: "Shandong Provincial Hospital ECG Database (SPHDB)"
category: "12-lead-other"
order: 4
status: "completed"
source_url: "https://doi.org/10.6084/m9.figshare.c.5779802.v1"
url_label: "figshare.com"
format: "12-lead · 10–56 s · 500 Hz · HDF5"
patients: "24,666"
records: "25,770"
access: "open"
license: "CC BY 4.0"
origin_institution: "Shandong Provincial Hospital"
origin_country: "China"
leads: 12
paper_title: "Liu et al., Scientific Data, 2022"
paper_doi: "https://doi.org/10.1038/s41597-022-01403-5"
search_keywords: "shandong provincial hospital sphdb sph ecg hdf5 h5py figshare mendeley china aha acc hrs standardized diagnostic statements multi-label patient-grouped"

sections:
  - type: description
    title: "Overview"
    body: |
      25,770 twelve-lead clinical ECGs from 24,666 patients recorded at Shandong
      Provincial Hospital between August 2019 and August 2020 — the largest
      **single-source** ECG dataset in this catalogue, and the only one whose
      labels are a published clinical standard rather than a vocabulary invented
      for the release.

      Three things make it distinctive:

      **The labels are AHA/ACC/HRS standardised diagnostic statements.** 44
      primary statements across 11 categories, each optionally qualified by one
      or more of 15 modifiers, so a record's ground truth reads as
      `60+310;147` — frequent ventricular premature complexes, plus T-wave
      abnormality. The vocabulary is closed and fully documented in the shipped
      `code.csv`; every code in every one of the 25,770 records resolves against
      it. 3,724 records (14.45%) carry more than one statement.

      **It is the only HDF5 dataset in ECGBench.** Each record is one `.h5` file
      holding a single `(12, N)` float16 array named `ecg`, already in
      millivolts. Loading needs `pip install ecgbench[hdf5]`; nothing else about
      the API changes, and `window=` is pushed into h5py's own slicing so a 56 s
      record decodes only the samples you ask for.

      **Patients repeat, and the folds account for it.** 1,066 of the 24,666
      patients contributed between two and five recordings, covering 2,170
      records. ECGBench's folds are grouped on `patient_id`, so no patient
      appears in two folds — verified, not assumed.

      Record length varies from 10 s to 56 s in 39 distinct lengths, and the
      metadata's `N` column gives it exactly for every record, so nothing has to
      open a signal file to learn a length. Age (18–95) and sex are complete for
      all 25,770 rows with no sentinels and no blanks.

  - type: table
    title: "AHA statement categories"
    headers: ["Category", "Meaning", "Statements", "Records carrying ≥1"]
    rows:
      - ["A", "Normal interpretation", "1", "13,905"]
      - ["C", "Sinus node rhythms", "3", "4,643"]
      - ["D", "Supraventricular premature/escape complexes", "4", "622"]
      - ["E", "Supraventricular arrhythmias", "3", "787"]
      - ["F", "Ventricular premature complexes", "1", "1,067"]
      - ["H", "AV conduction", "9", "370"]
      - ["I", "Intraventricular/intra-atrial conduction", "6", "2,195"]
      - ["J", "Axis and voltage", "3", "612"]
      - ["K", "Chamber hypertrophy and enlargement", "3", "229"]
      - ["L", "ST, T and U abnormalities", "7", "5,125"]
      - ["M", "Myocardial infarction", "4", "260"]
      - ["", "**total**", "**44**", "**29,815 statement-record pairs**"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the shipped `metadata.csv`
      joined against `code.csv`, and from the 25,770 HDF5 arrays themselves. The
      headline figures agree with the paper: **25,770 records**, **24,666
      patients**, **55.36% male** (14,265 M / 11,505 F), **14.45%
      multi-statement** (3,724 records) and **46.04% carrying an abnormality**
      (11,865 records).

      Three things worth knowing about how those numbers were derived:

      **The category table is multi-label and does not sum to 25,770.** It counts
      every category a record carries, so the 29,815 statement-record pairs
      exceed the record total by 4,045 — the second and later statements of the
      3,724 multi-statement records. Categories B, G and N of the AHA standard
      have no statements in this release at all.

      **"Normal" is a code, and two records list it twice.** 13,905 records carry
      primary code 1 (Normal ECG) and nothing else. Only 13,903 have an
      `AHA_Code` cell equal to the string `1`, because `A02322` and `A05000` read
      `1;1` — the same statement repeated. ECGBench deduplicates the primary-code
      list, which is what makes `is_normal` give 13,905 and reproduce the paper's
      46.04% abnormal exactly; a string comparison misses by two.

      **There is no primary diagnosis, so ECGBench does not pretend to have
      one.** Nothing in the release ranks a record's statements. The order is
      *nearly* a numeric sort of the primary code — 24,961 of 25,770 records are
      in ascending order, and 2,915 of the 3,724 multi-statement ones — so the
      first statement is neither reliably a sort artefact nor a documented
      priority. The loader's single-label reduction is therefore called
      `stratify_code`, not `primary_code`: it takes the globally **rarest** code
      each record carries, exists to make stratified folds well defined, and is
      not ground truth. Train on `aha_primary_codes`.

      One small disagreement with the paper: it gives the age range as 18–100,
      where the shipped maximum is **95**. The minimum, 18, matches — this is an
      adults-only cohort and nothing here supports paediatric use.

  - type: description
    title: "Where the files come from"
    body: |
      The release ships four files: `metadata.csv`, `code.csv`,
      `records.tar.gz` (2.3 GB) and two PDFs documenting the coding rules.
      figshare serves them as separate downloads rather than one archive, so
      there is no single URL to auto-download from — pass `--data-path`.

      **`records.tar.gz` is an uncompressed tar despite the name.** Extract it
      with `tar -xf records.tar.gz`; `tar -xzf` fails with *"not in gzip
      format"*. It expands to `records/A00001.h5` … one file per record.

      No checksums ship, so authenticity was established by internal consistency
      instead. On the copy ECGBench was built against:

      - all **25,770** HDF5 arrays have shape `(12, N)` with `N` equal to the
        metadata's own `N` column — not one disagreement;
      - every metadata ID has a file and every file has a metadata row, in both
        directions, with no extras on either side;
      - every AHA code in every record resolves against `code.csv` — the
        vocabulary is closed;
      - the lead order was derived from the signals rather than taken on trust:
        `III = II − I`, `aVR = −(I+II)/2`, `aVL = I − II/2` and `aVF = II − I/2`
        all hold to under 2% relative RMS error, confirming the standard
        `I, II, III, aVR, aVL, aVF, V1…V6`.

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "25,770", "all records, with is_valid + quality_issues"]
      - ["clean", "25,447", "98.7% pass rate"]
      - ["excluded", "323", "all 323 fail amplitude_outlier and nothing else"]

  - type: description
    title: "About the 323 excluded records"
    body: |
      `amplitude_range_mv` is ECGBench's standard `[-10, 10]` here, and the
      exclusions are genuine artefacts rather than a badly chosen threshold: the
      **median per-record peak amplitude across the whole dataset is 1.74 mV**,
      so a record reaching past 10 mV has a railed or artefact-dominated lead.

      - **1,080 lead-level failures over 323 records.** 100 records have a single
        bad lead; 55 have all twelve.
      - **The precordials dominate** — V6 (201 records), V3 (200), V5 (181), V4
        (169) — consistent with chest-electrode contact problems rather than
        anything physiological.
      - **The magnitudes are extreme, not marginal.** 200 records peak between
        10 and 20 mV, 66 between 20 and 50, 12 between 50 and 100, and **45
        exceed 100 mV**. The worst reaches 1,590 mV.

      No record fails any other check: no NaN samples, no dead or flat leads, no
      unreadable files, in all 25,770.

      Fold membership is identical between `original/` and `clean/` — `clean/` is
      a row subset, not a re-split — so a model trained on `clean/` can be scored
      against `original/` for the same fold without re-partitioning.

  - type: table
    title: "Splits"
    headers: ["Split", "Folds", "clean", "original"]
    rows:
      - ["train", "1–8", "20,364", "20,616"]
      - ["val", "9", "2,542", "2,577"]
      - ["test", "10", "2,541", "2,577"]

  - type: description
    title: "How the folds were made"
    body: |
      Ten folds, **grouped on `patient_id`** and stratified on `stratify_code`,
      the rarest AHA primary code each record carries. The 9 codes with fewer
      than ten records (49 records in total) are pooled into an `OTHER` bucket
      first, so ten-fold stratification stays well defined.

      What that produced, measured on the output:

      - **no patient spans two folds** — zero, in both versions;
      - fold sizes 2,535 to 2,556 records;
      - the normal fraction is 53.7–54.0% in every fold, against 53.96% overall;
      - the rare codes are spread across folds rather than concentrated in one.

      Rarest-first stratification is used because there is no primary diagnosis
      to stratify on instead. It keeps all 44 codes representable, where taking
      the first listed statement collapses to 40 classes and leaves 12 below ten
      records rather than 9.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # records.tar.gz is an UNCOMPRESSED tar despite the name.
      tar -xf records.tar.gz

      ecgbench splits --dataset sph --data-path /path/to/SPH/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      # pip install ecgbench[torch,hdf5]   <- h5py is needed for this dataset
      from ecgbench import ECGDataset

      # Records run 10-56 s, so a fixed window must fit the SHORTEST one.
      # window= is pushed into h5py's slicing, so a 56 s record decodes 10 s.
      ds = ECGDataset(
          "sph",
          split="train",
          data_path="/path/to/SPH/",
          window=(0, 5000),        # first 10 s at 500 Hz
          labels=True,
      )

      len(ds)                                        # 20364
      ds[0]["signal"].shape                          # (12, 5000)
      ds[0]["record_id"]                             # 'A00001'
      ds[0]["labels"]["patient_id"]                  # 'S00001'
      ds[0]["labels"]["aha_code"]                    # '22;23'  — as shipped
      ds[0]["labels"]["aha_primary_codes"]           # '22;23'  — ground truth
      ds[0]["labels"]["aha_primary_descriptions"]    # 'Sinus bradycardia;Sinus arrhythmia'
      ds[0]["labels"]["aha_primary_categories"]      # 'C'
      ds[0]["labels"]["aha_modifier_codes"]          # ''  — this record has none
      ds[0]["labels"]["duration_seconds"]            # 10.0  — the FULL record
      ds[0]["labels"]["age"], ds[0]["labels"]["sex"] # 55, 'M'

      # Multi-label: build the 44-way target from aha_primary_codes, not from
      # stratify_code (a rarest-code reduction that exists only for the folds).
      from ecgbench.labels.sph import load_code_table
      codes = load_code_table("/path/to/SPH/")
      primaries = list(codes.index[~codes["is_modifier"]])
      carried = ds[0]["labels"]["aha_primary_codes"].split(";")
      y = {c: int(c in carried) for c in primaries}
      sum(y.values())                                # 2

      # Standard lead order and standard spelling — derived from the signals.
      ds.config.lead_names
      # ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

      # Samples are already millivolts, so signal_unit_scale is 1.0.
      ds.config.signal_unit_scale                    # 1.0

  - type: links
    title: "References"
    items:
      - { label: "figshare collection (the copy ECGBench was verified against)", url: "https://doi.org/10.6084/m9.figshare.c.5779802.v1" }
      - { label: "Liu et al., Scientific Data 9, 272 (2022)", url: "https://doi.org/10.1038/s41597-022-01403-5" }
      - { label: "Mendeley Data mirror", url: "https://data.mendeley.com/datasets/z5sh7pwypd/2" }
---
