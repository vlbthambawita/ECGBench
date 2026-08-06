---
slug: "cpsc-2018-china-physiological-signal-challenge-2018"
name: "CPSC 2018 (China Physiological Signal Challenge 2018)"
category: "12-lead-other"
order: 1
status: "completed"
source_url: "http://2018.icbeb.org/Challenge.html"
url_label: "icbeb.org"
format: "12-lead · 6–144 s · 500 Hz · WFDB (.hea/.mat)"
patients: "—"
patients_class: "count-na"
records: "6,877"
access: "open"
license: "CC BY 4.0"
origin_institution: "11 hospitals (ICBEB, Nanjing)"
origin_country: "China"
leads: 12
paper_title: "Liu et al., J. Med. Imaging Health Inform., 2018"
paper_doi: "https://doi.org/10.1166/jmihi.2018.2442"
search_keywords: "cpsc 2018 china physiological signal challenge icbeb nanjing arrhythmia multi-label snomed rbbb af pac pvc std ste kaggle"

sections:
  - type: description
    title: "Overview"
    body: |
      The public training set of the **China Physiological Signal Challenge
      2018**: 6,877 twelve-lead recordings from 11 Chinese hospitals, 500 Hz,
      labelled with one normal and eight abnormal classes. The challenge's own
      2,954-record test set was never released and the organisers say it will
      stay private, so this is the whole publicly available dataset.

      Two things make it awkward in a way most 12-lead resting datasets are not:

      **Record length varies by a factor of 24.** 6 s to 144 s, median 12 s,
      1,650 distinct lengths. Only 2,416 of 6,877 records are the familiar 10 s.
      Any fixed window has to fit the shortest record — `window=(0, 3000)`, 6 s
      at 500 Hz — and a model handed raw lengths can partly read the class off
      the duration, because the premature-beat classes are systematically the
      longest recordings.

      **It is multi-label, but the primary label did not survive.** 470 records
      carry two classes and 6 carry three. CPSC's original `REFERENCE.csv`
      distinguished First, Second and Third label; the WFDB copy everyone
      actually uses sorted each record's codes by class index and dropped the
      distinction. See "About those counts" for what that costs.

      **This dataset is contained whole in Challenge 2020 and 2021.** All 6,877
      waveform files are byte-identical to their counterparts there, under the
      same names. Training on either challenge and evaluating here is testing on
      training data.

  - type: table
    title: "The nine classes"
    headers: ["#", "Class", "SNOMED-CT", "Records (any label)", "Published (first label)", "Diff"]
    rows:
      - ["1", "Normal (`NSR`)", "426783006", "918", "918", "0"]
      - ["2", "Atrial fibrillation (`AF`)", "164889003", "1,221", "1,098", "+123"]
      - ["3", "First-degree AV block (`IAVB`)", "270492004", "722", "704", "+18"]
      - ["4", "Left bundle branch block (`LBBB`)", "164909002", "236", "207", "+29"]
      - ["5", "Right bundle branch block (`RBBB`)", "59118001", "1,857", "1,695", "+162"]
      - ["6", "Premature atrial contraction (`PAC`)", "284470004", "616", "556", "+60"]
      - ["7", "Premature ventricular contraction (`PVC`)", "164884008", "700", "672", "+28"]
      - ["8", "ST-segment depression (`STD`)", "429622005", "869", "825", "+44"]
      - ["9", "ST-segment elevation (`STE`)", "164931005", "220", "202", "+18"]
      - ["", "**total class-record pairs**", "", "**7,359**", "**6,877**", "**+482**"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 6,877 shipped headers.
      The **6,877 records** and the **3,699 male / 3,178 female** split match the
      challenge's published figures exactly. Three things do not.

      **The class table above shows two different quantities, and the difference
      is fully explained.** The challenge's Table 1 is labelled *"according to the
      'First label' annotations"* and sums to 6,877 — one class per record. The
      "any label" column counts every class a record carries and sums to 7,359.
      The gap is exactly **482 = 470 × 1 + 6 × 2**, the second and third labels of
      the 476 multi-label records. Neither column is wrong; they answer different
      questions. **Use the any-label column** — the data is multi-label and
      `dx` exposes all of it.

      **The primary diagnosis is not recoverable from the shipped files, so
      ECGBench does not pretend to have one.** The challenge page documents
      `A0043` as First label 5 (RBBB), Second label 2 (AF). Its header reads
      `#Dx: 164889003,59118001` — AF first. The conversion sorted every `#Dx`
      list into CPSC class-index order, which is why the first code cannot be
      read as a primary diagnosis and why the loader's single-label reduction is
      named `stratify_dx`, not `primary_dx`. It exists to make stratified folds
      well defined; do not train on it.

      **The published 60 s maximum is wrong for the shipped release.** Table 1
      gives min 6.00 s, max 60.00 s, mean 15.79 s over the training set. The files
      run **6.0 s to 144.0 s**, mean 15.95 s, with **27 records (0.39%) longer
      than 60 s** — the longest being `A4133` at 144 s. Clipping the durations at
      60 s reproduces the published mean of 15.79 (15.84), so the published table
      is consistent with a copy in which those 27 records were shorter or
      excluded. Size any window to the 6 s minimum regardless.

      Two further details, both about age:

      - **5 records carry no age** (`A0608`, `A1549`, `A1876`, `A2299`, `A5990`)
        and **4 more carry the sentinel `-1`**. Filter both out before computing
        any age statistic; `ecgbench.labels.cpsc_2018.AGE_SENTINELS` names the
        second case.
      - **This copy keeps exact ages above 89 where PhysioNet's does not.** 125
        records give an age over 89, up to 104. PhysioNet's Challenge 2020
        re-release of the same waveforms rails every such age to 92, changing the
        value in 104 of the 125. The waveforms are identical; only these header
        fields differ.

  - type: description
    title: "Where the files come from"
    body: |
      **The official icbeb.org download links are dead.** ECGBench was built and
      verified against the
      [Kaggle mirror](https://www.kaggle.com/datasets/physionet/china-physiological-signal-challenge-in-2018),
      a flat `Training_WFDB/` directory of 6,877 `.hea`/`.mat` pairs. That mirror
      is not the original MATLAB release: it is the WFDB conversion prepared for
      the PhysioNet/CinC Challenge 2020, so labels arrive as SNOMED-CT codes in
      the header `#Dx` field and no `REFERENCE.csv` ships.

      The mirror publishes no checksums, so authenticity was established by
      comparison instead. All 6,877 `.mat` files are **byte-identical** to
      `training/cpsc_2018/` in PhysioNet's Challenge 2020 v1.0.2, and that
      release's entire 13,761-file `cpsc_2018` subtree verifies against its own
      published `SHA256SUMS.txt` (13,761 of 13,761 match). Only the headers
      differ, and only in formatting and the 104 railed ages described above.

      If you have a Challenge 2020 or 2021 download already, you have this
      dataset — point `--data-path` at a directory holding a `Training_WFDB/`
      with those records in it.

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "6,877", "all records, with is_valid + quality_issues"]
      - ["clean", "6,760", "98.3% pass rate"]
      - ["excluded", "117", "99 with out-of-range amplitudes, 32 with a dead lead (14 with both)"]

  - type: description
    title: "About the 117 excluded records"
    body: |
      `amplitude_range_mv` is ECGBench's standard `[-10, 10]` here, and the
      exclusions it produces are genuine defects rather than a badly chosen
      threshold:

      - **50 records are clipped at the 16-bit rail.** At a gain of 1000 adu/mV
        the rail sits at ±32.767 mV, and these records reach it exactly — whole
        leads pinned, samples genuinely lost.
      - **49 records have excursions between 10 and 30 mV**, an order of
        magnitude past any physiological QRS but short of saturation.
      - **32 records have at least one lead that is entirely NaN or all-zero**,
        most often lead V6 (13 records). 14 of these also fail the amplitude
        check.

      Fold membership is identical between `original/` and `clean/` — `clean/` is
      a row subset, not a re-split — so a model trained on `clean/` can be scored
      against `original/` for the same fold without re-partitioning.

      Folds are **stratified on the rarest class each record carries**, which
      leaves nine classes with the smallest at 220 records; nothing needs pooling
      into an `OTHER` bucket, unlike Challenge 2020. **Folds are not grouped by
      patient, because no patient identifiers ship.** The challenge describes
      6,877 recordings from 11 hospitals and never mentions repeat patients, but
      nothing in the files proves one record per patient either, so this page
      leaves the patient count blank rather than guessing.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset cpsc_2018 --data-path /path/to/CPSC_2018/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Records run 6-144 s, so a fixed window must fit the SHORTEST one.
      # window= is pushed into the wfdb reader, so the 144 s records decode 6 s
      # rather than all of it.
      ds = ECGDataset(
          "cpsc_2018",
          split="train",
          data_path="/path/to/CPSC_2018/",
          window=(0, 3000),        # first 6 s at 500 Hz
          labels=True,
      )

      len(ds)                                    # 5408
      ds[0]["signal"].shape                      # (12, 3000)
      ds[0]["record_id"]                         # 'A0001'
      ds[0]["labels"]["dx"]                      # '59118001'
      ds[0]["labels"]["dx_abbreviations"]        # 'RBBB'
      ds[0]["labels"]["dx_names"]                # 'Right bundle branch block'
      ds[0]["labels"]["duration_seconds"]        # 15.0  — the FULL record
      ds[0]["labels"]["age"], ds[0]["labels"]["sex"]   # '74', 'Male'

      # Multi-label: build the 9-way target from dx_abbreviations, not from
      # stratify_dx_abbreviation (which is a rarest-class reduction for folds).
      from ecgbench.labels.cpsc_2018 import CPSC_CLASSES
      names = [abbr for _, _, abbr, _ in CPSC_CLASSES]
      y = [int(n in ds[0]["labels"]["dx_abbreviations"].split(",")) for n in names]
      dict(zip(names, y))
      # {'NSR': 0, 'AF': 0, 'IAVB': 0, 'LBBB': 0, 'RBBB': 1,
      #  'PAC': 0, 'PVC': 0, 'STD': 0, 'STE': 0}

      # Standard lead order and standard spelling — verified in all 6,877 headers.
      ds.config.lead_names
      # ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

  - type: links
    title: "References"
    items:
      - { label: "Challenge page (downloads are dead)", url: "http://2018.icbeb.org/Challenge.html" }
      - { label: "Liu et al., JMIHI 2018", url: "https://doi.org/10.1166/jmihi.2018.2442" }
      - { label: "Kaggle mirror (the copy ECGBench was verified against)", url: "https://www.kaggle.com/datasets/physionet/china-physiological-signal-challenge-in-2018" }
      - { label: "PhysioNet/CinC Challenge 2020 (contains these records)", url: "https://physionet.org/content/challenge-2020/1.0.2/" }
---
