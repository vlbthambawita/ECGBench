---
slug: "mhd-effect-on-12-lead-ecgs-in-mri-scanners"
name: "MHD Effect on 12-Lead ECGs in MRI Scanners"
category: "12-lead-physionet"
order: 14
status: "completed"
source_url: "https://physionet.org/content/mhd-effect-ecg-mri/1.0.0/"
url_label: "physionet.org"
format: "12-lead + 3-lead · 24 s–12 min · 1,024 Hz"
patients: "26"
records: "53"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Otto-von-Guericke University of Magdeburg"
origin_country: "Germany"
leads: 12
paper_title: "Krug et al., CinC 2017"
paper_doi: "https://doi.org/10.13026/05td-jn37"
search_keywords: "mhd mri ecg germany magdeburg otto guericke magnetohydrodynamic 1t 3t 7t tesla scanner signal separation qrs patient monitoring"

sections:
  - type: description
    title: "Overview"
    body: |
      53 ECG records from 26 healthy volunteers, recorded **inside 1T, 3T and 7T
      MRI scanners** at Otto-von-Guericke University Magdeburg between 2011 and
      2017, all at 1,024 Hz.

      Inside the bore, the **magnetohydrodynamic (MHD) effect** — blood ions moving
      through the scanner's static B0 field — induces a voltage that superimposes
      on the ECG. It is large enough to bury the P wave, ST segment and T wave,
      which is why ECG-based patient monitoring and cardiac gating are hard during
      MRI exams. Recordings were made **without imaging running**, so no switched
      gradients or RF fields contribute and the only distortion present is the MHD
      effect itself.

      **The distortion is the subject of the dataset, not an artefact to remove at
      validation time.** ECGBench's `amplitude_range_mv` is set to ±35 mV for
      exactly this reason: a conventional ±10 mV range would exclude 16 of the 53
      records for being precisely what they are meant to be.

      10 records are **reference ECGs taken outside the scanner** for 7 of the
      subjects, standing in for the in-bore ground truth that cannot be measured.
      They are a stationarity assumption, not a simultaneous recording — heart rate
      and morphology genuinely differ between the two acquisitions.

      There is **no diagnosis to predict**. Every subject was free of known cardiac
      disease, and the 14,950 manually annotated QRS complexes carry no beat
      classification (the release is explicit that no normal/ectopic distinction was
      made). This is a signal-processing benchmark — ECG/MHD separation and QRS
      detection under distortion — not a classification one.

      One scanner is unlike the others: the 1T **Philips Panorama HFO** is an
      open-bore design whose B0 is *vertical* (back to chest) rather than
      horizontal, which changes the MHD morphology and magnitude. Its records were
      made prone and supine to invert the effect, while every horizontal-B0 scanner
      was used head-first and feet-first, supine.

  - type: table
    title: "Acquisition conditions"
    headers: ["Condition", "Records", "Subjects", "QRS", "Minutes"]
    rows:
      - ["3T", "31", "23", "10,454", "154.6"]
      - ["7T", "10", "5", "2,853", "47.4"]
      - ["1T", "2", "1", "95", "1.1"]
      - ["reference (outside bore)", "10", "7", "1,548", "23.5"]
      - ["**total**", "**53**", "**26**", "**14,950**", "**226.6**"]

  - type: description
    title: "How big is the MHD effect? Load it and look"
    body: |
      Measured over the first 24.4 s of every record in the `train` split, with
      ECGBench doing the reading:

      | Condition | Amplitude range (mV) |
      |---|---|
      | reference (outside bore) | −0.88 … +3.09 |
      | 3T | **−31.05** … +14.22 |
      | 7T | −18.06 … **+19.91** |

      The reference recordings sit in the normal physiological range. Inside the
      bore the same subjects' signals swing an order of magnitude further, past both
      devices' nominal input ranges (±6 mV for the 12-lead Holter, ±2.4 mV for the
      3-lead monitor). That excursion is the MHD voltage plus each channel's
      baseline offset — and it is the reason the dataset exists.

  - type: table
    title: "Two ECG devices, two channel layouts"
    headers: ["Device", "Lead configuration", "Channels", "Resolution", "Input range", "Records"]
    rows:
      - ["Getemed CM 3000 Holter", "Diagnostic 12 lead ECG", "I II III aVR aVL aVF V1–V6", "12 bit", "±6 mV", "39"]
      - ["MIPM Tesla M3 monitor (MRI-conditional)", "Reduced Einthoven Triangle", "I II III", "24 bit", "±2.4 mV", "14"]

  - type: description
    title: "Records are not all 12-lead, and not one length"
    body: |
      Exactly two channel layouts ship, and they agree only on channels **0–2**:

      ```
      I II III aVR aVL aVF V1 V2 V3 V4 V5 V6    39 records
      I II III                                  14 records
      ```

      `lead_names` declares the 12-lead layout, so `leads=["I"]`, `["II"]` and
      `["III"]` resolve on every record while anything past III raises on the 14
      three-lead records, naming the record and its true channel count rather than
      returning the wrong physical channel. Without a `leads=` filter a batch mixes
      `(12, N)` and `(3, N)` tensors — which `ecg_collate_fn` will hand you, but a
      model will not accept. The label loader exposes `lead_config`, `n_signals` and
      `channel_names` so you can filter first.

      Note the confound: **all 14 three-lead records were recorded at 3T**, so lead
      configuration is not independent of field strength.

      Length varies by a factor of 30 — **24.4 s to 722.7 s** (25,000 to 740,001
      samples), median 191.9 s. `expected_samples` is therefore deliberately empty,
      and batching needs a fixed `window=` sized to the *shortest* record:
      `window=(0, 25000)`. Anything larger raises `WindowOutOfRangeError` on
      `ECGMRI3T02Ff` and `ECGMRI3T02Out`.

  - type: description
    title: "About those counts — the release contradicts itself"
    body: |
      All 163 shipped files were verified against the release's own
      `SHA256SUMS.txt` before any figure here was computed — all OK. So everything
      below is an upstream property, not download damage.

      **There are 53 records, not 43.** The README, the PhysioNet page and the 2017
      CinC paper all state 43 records / 23 subjects / 203 minutes. `RECORDS` lists
      53, 53 exist, and they total 226.6 minutes. The release evidently grew after
      publication and the prose was never updated. Every figure ECGBench publishes
      is recomputed from the files.

      | Figure | Release says | Recomputed | Diff |
      |---|---|---|---|
      | Records | 43 | **53** | +10 |
      | Subjects | 23 | **26** (by demographics) | +3 |
      | Total duration | 203 min | **226.6 min** | +23.6 |
      | Mean age | 27.1 ± 3.2 y | **24.6 y** (range 18–30) | −2.5 |
      | Mean weight | 73.8 ± 13.1 kg | **72.5 kg** (45–98) | −1.3 |
      | Mean height | 181.7 ± 10.5 cm | **179.4 cm** (158–193) | −2.3 |

      **Subject numbers in the filenames are scoped per scanner, so they are not a
      patient ID.** `ECGMRI1T01` is Male/27y/75kg/190cm; `ECGMRI3T01` is
      Female/29y/60kg/165cm. Different people, same number. Worse, three filename
      slots belong to people who were recorded in more than one scanner:

      ```
      Male/27y/75kg/190cm    ->  1T01, 3T02, 7T05   (8 records)
      Female/29y/60kg/165cm  ->  3T01, 7T04         (6 records)
      ```

      Grouping folds on the filename number would put one person's 3T record in
      train and their 7T record in test — textbook leakage in a dataset whose entire
      purpose is comparing one subject across field strengths. Since the release
      ships **no subject identifier at all**, ECGBench derives `subject_key` from
      the one identifying thing the headers do carry — the sex/age/weight/height
      tuple — collapsing 29 filename slots into 26 people and reuniting those two.
      Folds are grouped on it, and the released folds put all 8 records of
      `Male/27y/75kg/190cm` in fold 10 and all 6 of `Female/29y/60kg/165cm` in
      fold 3.

      Two honest limits on that key, both documented in
      `ecgbench/labels/mhd_effect_ecg_mri.py`:

      - **It can over-merge.** Two different volunteers with identical sex, age,
        weight and height would become one group. That direction is safe — it costs
        a little fold balance and creates no leakage.
      - **It can still under-merge, and 26 is not the release's 23.** Collection
        spanned 2011–2017, so one person's recorded weight or age can differ between
        sessions. `Male/27y/75kg/190cm` and `Male/30y/75kg/190cm` (3T09) are
        plausibly the same person three years apart. Neither 23 nor any other count
        is reproducible from the files by a stated rule, so ECGBench reports the 26
        it can defend. If your work turns on exact subject identity, contact the
        author.

      **Two records contradict themselves or their filename.** Both values are
      exposed and a flag marks the disagreement; ECGBench does not silently pick a
      winner.

      - `ECGMRI3T01Hf` — filename says head-first, its header says
        `Positon in the scanner:Feet first (Ff)`. Flagged by
        `position_disagrees`. The filename is the likelier of the two (subject 3T01's
        record set is Ff/Hf/Out, a deliberate protocol, and the README documents the
        naming convention), but nothing in the release settles it.
      - `ECGMRI1T01Out` — position is `Outside the scanner`, yet its field strength
        reads `1T` and B0 `Vertical`, where the other 9 reference records read
        `Outside the scanner` for all three. So filtering references on the header's
        field-strength string silently misses one of the ten. Flagged by
        `reference_header_agrees`; use `condition` or `is_reference`, which are
        derived from the filename and catch all 10.

      **No record is a breath-hold recording.** The README says breath-hold
      protocols "are noted in the header files"; all 53 headers say
      `Spontaneous respiration`. The column is exposed anyway, so a future release
      that adds them will show up rather than being assumed away.

      **A fourth scanner ships that the README never mentions.** `ECGMRI3T02Ff` and
      `ECGMRI3T02Out` were recorded on a **Philips Achiva**; every other 3T record
      used a Siemens Magnetom Skyra. Both declare 3T, so this is extra detail rather
      than a contradiction — but a per-field-strength analysis that assumes one
      scanner per field strength is wrong.

      The header's `Magnetic field strength` is also not a clean numeric field: it
      reads `Outside the scanner` for 9 of the 10 reference records. ECGBench
      therefore derives two numeric columns — `field_strength_T` (0 outside the
      bore, else 1/3/7: what the subject was exposed to) and `scanner_field_T`
      (1/3/7 even for a reference: which session it belongs to) — and keeps the raw
      string as `field_strength_header`.

  - type: table
    title: "Validation summary (1,024 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "53", "all records, with is_valid + quality_issues"]
      - ["clean", "53", "100% pass rate — no record excluded"]
      - ["excluded", "0", "no NaN samples, no flat channels; peak |amplitude| 31.05 mV, inside the ±35 range set for MHD distortion"]

  - type: description
    title: "Folds are grouped by subject, and the tail concentrates"
    body: |
      Folds use `StratifiedGroupKFold` grouped on `subject_key` and stratified on
      `condition`. **No subject spans a fold or a split** — verified against the
      released folds, including both cross-scanner subjects.

      With 26 subjects over 10 folds the split is inherently coarse: folds hold 3–10
      records and the default mapping gives **37 train / 6 val / 10 test**. Two
      consequences worth knowing before you quote a per-condition result:

      - **All 1T records land in one fold.** Both belong to a single subject, who is
        also `3T02` and `7T05`, so that whole group sits in fold 10 (test) and the
        train split contains no 1T record at all. Any 1T conclusion rests on one
        person regardless of how you split.
      - **Rare conditions are deliberately not pooled.** Collapsing 1T, 7T and
        reference into one bucket would balance the folds by destroying the only
        distinction the dataset is about.

      For per-condition work, rotate folds with `split=None, fold_numbers=[...]`
      rather than using the default mapping.

  - type: code
    title: "Getting the data"
    language: bash
    body: |
      # ~265 MB zip, genuinely public -- no PhysioNet credentials needed.
      wget https://physionet.org/static/published-projects/mhd-effect-ecg-mri/mhd-effect-on-12-lead-ecgs-in-mri-scanners-1.0.0.zip
      unzip mhd-effect-on-12-lead-ecgs-in-mri-scanners-1.0.0.zip

      # Note: it expands to a directory named after the full title, not the slug:
      cd influence-of-the-mhd-effect-on-12-lead-and-3-lead-ecgs-recorded-in-1t-to-7t-mri-scanners-1.0.0

      # Verify before trusting any figure -- all 163 files should report OK.
      sha256sum -c SHA256SUMS.txt

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # Writes ecgbench_metadata.csv into the dataset root on first run: the release
      # ships no metadata table, and the validation engine re-reads that file from
      # disk. The dataset root must be writable.
      ecgbench splits --dataset mhd_effect_ecg_mri --data-path /path/to/mhd-effect-ecg-mri/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the HuggingFace Hub by default; only the waveforms
      # need to be local.
      ds = ECGDataset(
          "mhd_effect_ecg_mri",
          split="train",
          data_path="/path/to/mhd-effect-ecg-mri/1.0.0/",
          labels=True,
      )

      len(ds)                                   # 37
      ds[0]["record_id"]                        # 'ECGMRI3T01Ff'
      ds[0]["signal"].shape                     # (12, 334001)  -- 326.2 s at 1024 Hz

      # The label is the acquisition condition, not a diagnosis:
      ds[0]["labels"]["condition"]               # '3T'
      ds[0]["labels"]["field_strength_T"]        # 3   (0 for a reference recording)
      ds[0]["labels"]["position"]                # 'Feet first'
      ds[0]["labels"]["mr_scanner"]              # 'Siemens Magnetom Skyra'
      ds[0]["labels"]["lead_config"]             # 'Diagnostic 12 lead ECG'
      ds[0]["labels"]["n_qrs"]                   # 361 manually annotated QRS complexes
      ds[0]["labels"]["subject_key"]             # 'Female/29years/60kg/165cm'
      ds[0]["labels"]["scanner_subject_slot"]    # '3T01'  -- per SCANNER, not a patient ID
                                                 # (this subject is also 7T04)

      # Length varies 30x and channel count varies, so batching needs both a
      # window and a lead filter. 25,000 samples is the shortest record.
      batchable = ECGDataset(
          "mhd_effect_ecg_mri",
          split="train",
          data_path="/path/to/mhd-effect-ecg-mri/1.0.0/",
          leads=["I", "II", "III"],   # the only channels present in every record
          window=(0, 25000),          # 24.4 s -- fits ECGMRI3T02Ff, the shortest
      )
      batchable[0]["signal"].shape              # (3, 25000)

      # Pull the reference (outside-the-bore) recordings, which are the closest
      # thing to undistorted ground truth:
      refs = ds.labels_df["is_reference"]
      int(refs.sum())                           # 6 of 37 in the train split

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/mhd-effect-ecg-mri/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/05td-jn37" }
      - { label: "Example script", url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_mhd_effect_ecg_mri.py" }
---
