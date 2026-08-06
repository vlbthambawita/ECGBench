---
slug: "ningbo-first-hospital-ecg-database-idiopathic-ventricular-arrhythmia"
name: "Ningbo First Hospital ECG Database (Idiopathic Ventricular Arrhythmia)"
category: "12-lead-other"
order: 3
status: "completed"
source_url: "https://doi.org/10.6084/m9.figshare.c.4668086.v2"
url_label: "figshare.com"
format: "12-lead · 2.9–59 s · 2000 Hz · CSV"
patients: "334"
records: "334"
access: "open"
license: "CC BY 4.0"
origin_institution: "Chapman University; Ningbo First Hospital, Zhejiang University"
origin_country: "China / USA"
leads: 12
paper_title: "Zheng et al., Scientific Data, 2020"
paper_doi: "https://doi.org/10.1038/s41597-020-0440-8"
search_keywords: "ningbo first hospital ecg idiopathic ventricular arrhythmia iva ot-va rvot lvot pvc vt figshare china chapman ablation ep workmate 2000 hz alphabetical lead order"

sections:
  - type: description
    title: "Overview"
    body: |
      334 twelve-lead ECGs recorded **during** catheter ablation, one per patient,
      each labelled with the outflow tract the ablation proved the arrhythmia came
      from. That is what makes 334 records worth having: the ground truth is
      invasive, so the intended task is to predict the origin from the surface ECG
      *before* the procedure, which is a decision an operator actually has to make.

      Every patient had a **successful** ablation, defined as absence of
      spontaneous or induced outflow-tract arrhythmia 30 minutes after the last
      energy delivery, and patients whose frequent ectopy recurred within six
      months were excluded. Only single-source arrhythmias are included.

      Three properties will surprise anyone who assumes this looks like a resting
      12-lead dataset:

      **The lead order is alphabetical.** The CSV header reads
      `aVF,aVL,aVR,I,II,III,V1…V6`. So `signal[0]` is aVF, not lead I, and
      `signal[4]` is lead II, not aVL. Use `leads=[...]` to select by name — see
      the snippet below.

      **The samples carry no declared unit.** The CSVs hold bare integers, and
      neither the paper nor figshare states a gain; the paper's own Figure 8 plots
      the raw counts with a MATLAB ×10⁴ multiplier and no y-axis unit. ECGBench
      supplies a measured estimate rather than leaving the tensors uncalibrated —
      see "About the amplitude scale".

      **The sampling rate is 2000 Hz**, the highest in this catalogue and four
      times the usual 500 Hz, because the recordings come from an EP-lab
      acquisition system (Abbott EP-WorkMate) rather than a diagnostic cart. A
      10 s window is 20,000 samples here. Record length runs 2.9 s to 59.3 s in
      317 distinct lengths over 334 records, because the providers cut an excerpt
      containing both sinus beats and ectopy out of each procedure recording.

  - type: table
    title: "Anatomic origin (the label)"
    headers: ["Tract", "Sub-site", "Records", "Paper's Table 2"]
    rows:
      - ["RVOT", "LC — left cusp", "71", "71"]
      - ["RVOT", "PosteriorSeptal", "32", "32"]
      - ["RVOT", "AC — anterior cusp", "29", "29"]
      - ["RVOT", "AnteriorSeptal", "28", "28"]
      - ["RVOT", "FreeWall", "28", "28"]
      - ["RVOT", "RC — right cusp", "24", "24"]
      - ["RVOT", "RVOTOther", "6", "45"]
      - ["RVOT", "*(blank)*", "39", "—"]
      - ["RVOT", "**subtotal**", "**257**", "**257**"]
      - ["LVOT", "LCC — left coronary cusp", "39", "39"]
      - ["LVOT", "AMC — aortomitral continuity", "18", "18"]
      - ["LVOT", "RCC — right coronary cusp", "7", "7"]
      - ["LVOT", "LCC-RCC Ommisure", "7", "7"]
      - ["LVOT", "Summit", "5", "5"]
      - ["LVOT", "*(blank)*", "1", "1 (as “NA”)"]
      - ["LVOT", "**subtotal**", "**77**", "**77**"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the shipped `Diagnosis.xlsx`
      and the 334 CSV files. Most agree with the paper exactly: **334 patients**,
      **257 RVOT (77%) / 77 LVOT (23%)**, **104 male / 230 female**, and the
      per-class male split (65 of 257 RVOT, 39 of 77 LVOT). Two figures do not.

      **The 40 blank sub-sites are explained by the paper, but ECGBench does not
      fill them in.** Table 2 assigns 45 RVOT patients to "RVOTOther" and one
      LVOT patient to "NA". The shipped spreadsheet has 6 explicit `RVOTOther`
      plus 39 blanks among the RVOT patients, and the single LVOT blank —
      6 + 39 = 45, and the arithmetic closes on both tracts. The inference is
      almost certainly right, but it is *our* inference, not the providers', so
      the cells stay blank in the labels and the table above shows both columns.

      **The presentation type disagrees, and the shipped file wins.** Table 1
      reports **325 frequent PVC / 9 sustained VT** (RVOT 251/6, LVOT 74/3);
      `Diagnosis.xlsx` reads **329 PVC / 5 VT** (RVOT 254/3, LVOT 75/2). Four
      patients are VT in the paper and PVC in the file. There is no changelog, and
      figshare shows a single revision, so which is correct cannot be established
      from the release. ECGBench reports the file. **Do not quote the paper's 9.**

      **No age ships at all.** The paper reports a cohort mean of 46.1 ± 13.1
      years and per-tract means, but `Diagnosis.xlsx` has five columns —
      HospitalID, Type, LeftRight, Sublocation, Gender — and none of them is age.
      It is not recoverable from the data.

      **Sub-site is not usable as a split target.** 12 values over 334 patients,
      five of them under ten cases, plus 40 blanks. Ten-fold stratification is
      not well defined on it, so ECGBench stratifies on `left_right` and exposes
      `sublocation` for analysis.

      **Sex leaks the label.** 230 of 334 patients are female overall, but 192 of
      257 RVOT cases against 38 of 77 LVOT ones. Sex alone separates the classes
      better than chance, so a model handed it as a feature can learn it instead
      of the ECG.

  - type: description
    title: "About the amplitude scale"
    body: |
      **`signal_unit_scale` for this dataset is an ECGBench estimate, not a
      declared value.** The release ships integers with no unit anywhere: the
      paper states the sampling rate and the acquisition system and never states
      a gain, and its Figure 8 plots the counts directly.

      Rather than hand users uncalibrated tensors — which would make
      `units="mV"`, `amplitude_range_mv` and any cross-dataset comparison
      meaningless — the scale was measured. Method: median lead-II R-peak
      prominence (Butterworth 0.5–40 Hz, `scipy.signal.find_peaks`) over all 334
      records, compared **sex for sex** against `sph`, whose samples are
      millivolts by declaration. Sex matters because this cohort is 69% female and
      QRS amplitude differs by sex.

      | Comparison | Implied counts per mV |
      |---|---|
      | Female-matched | 14,029 |
      | Male-matched | 17,111 |
      | **Adopted: 2¹⁴** | **16,384** (`signal_unit_scale = 6.1035e-05`) |

      16,384 sits between the two estimates and is the nearest binary-clean gain
      of the kind acquisition systems export. **Treat it as accurate to roughly
      ±20%**: every waveform's shape is exact, its absolute millivolt calibration
      is not. The same measurement confirmed the 2000 Hz rate independently —
      R-peak intervals give a median 81.5 bpm at 2000 Hz, which would be an
      impossible 20 bpm at 500 Hz.

      If you would rather calibrate it yourself, divide the millivolt values by
      `6.1035e-05` to recover the shipped integers exactly, or pass `units="uV"`
      and scale from there.

  - type: description
    title: "Two directories, and why only one is canonical"
    body: |
      The release ships the same 334 recordings twice: `PVCVTRawECGData/` (raw
      integers) and `PVCVTECGData/` (wavelet-denoised, `coif5` with a SURE
      threshold, per the providers' [MATLAB
      script](https://github.com/zheng120/PVCVTECGDenoising)). ECGBench points at
      the **raw** copy, matching `chapman_shaoxing`, which likewise takes
      `ECGData/` over `ECGDataDenoised/`.

      The denoised copy is not a drop-in substitute, for two measured reasons:

      - **The denoiser ran on each lead independently, so the leads no longer
        agree with each other.** `III − (II − I)` is under 1% of III's RMS in the
        raw files and up to 14% in the denoised ones; the Goldberger relations
        degrade similarly. Anything that derives one lead from others, or checks
        internal consistency, will see the difference.
      - **106 of the 334 denoised files are shorter than their raw
        counterparts**, by up to 7×, even though the published denoising script
        preserves row count exactly. The two directories are therefore *not
        sample-aligned*, and a window computed on one does not transfer to the
        other.

      `ecgbench.labels.ningbo_iva` exposes the denoised path as
      `signal_path_denoised` for anyone who wants it anyway.

  - type: table
    title: "Validation summary (2000 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "334", "all records, with is_valid + quality_issues"]
      - ["clean", "334", "100% pass rate — nothing excluded"]
      - ["excluded", "0", "no NaN, no dead leads, no unreadable files"]

  - type: description
    title: "How the folds were made"
    body: |
      Ten folds stratified on `left_right`, which is already one label per
      patient, so nothing is reduced. **No patient grouping is configured, and
      none is needed:** `HospitalID` is simultaneously the record and the patient
      identifier — 334 patients, 334 files, an exact one-to-one match in both
      directions — so `patient_id_column` stays null rather than asserting a
      grouping nothing exercised.

      What that produced: folds of 33–34 records, each holding 25–26 RVOT and 7–8
      LVOT, against the 77/23% cohort ratio. Fold membership is identical between
      `original/` and `clean/`, which here is trivially true since nothing was
      excluded.

      With 33 records per fold, single-split results on this dataset are noisy.
      Use the ten folds as cross-validation — `ECGDataset(split=None,
      fold_numbers=[...])` selects by fold across the default layout — rather
      than quoting one 33-record test set.

      `amplitude_range_mv` is left at `[-10, 10]` deliberately even though the
      largest sample anywhere in the release is 9.45 estimated mV, so the check is
      a guard rather than a filter. Tightening it would turn a 20% error in the
      estimated scale into hundreds of spurious exclusions.

  - type: description
    title: "Not the same Ningbo dataset as the one in ecg-arrhythmia"
    body: |
      Ningbo First Hospital contributed to **two** unrelated public releases, and
      they are easy to confuse. The PhysioNet `ecg-arrhythmia` dataset (catalogued
      here as *Chapman-Shaoxing Arrhythmia*) merges the Chapman-Shaoxing and
      Ningbo cohorts into 45,152 records, of which 34,905 are Ningbo's — routine
      diagnostic 10 s ECGs at 500 Hz under `JS…` names.

      This dataset is a different thing entirely: 334 intra-procedural EP-lab
      recordings at 2000 Hz, of variable length, named by hospital number. No
      recording can be shared between the two. Whether the same *patients* appear
      in both is unverifiable — the releases publish no common identifier, and
      this one publishes no age or date to join on — so the overlap edge on the
      Chapman-Shaoxing Arrhythmia page stays `verified: false`.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset ningbo_iva --data-path /path/to/Ningbo_IVA/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Records run 2.9-59.3 s at 2000 Hz, so a fixed window must fit the
      # SHORTEST one (5,791 samples). window= is pushed into the csv reader, so
      # the 59 s records parse 5,000 rows instead of 118,642.
      ds = ECGDataset(
          "ningbo_iva",
          split="train",
          data_path="/path/to/Ningbo_IVA/",
          window=(0, 5000),        # first 2.5 s at 2000 Hz
          labels=True,
      )

      len(ds)                                       # 268
      ds[0]["signal"].shape                         # (12, 5000)
      ds[0]["record_id"]                            # 1000364  (an int: HospitalID)
      ds[0]["labels"]["left_right"]                 # 'RVOT'  — ablation-proven
      ds[0]["labels"]["sublocation"]                # 'AC'
      ds[0]["labels"]["arrhythmia_type"]            # 'PVC'
      ds[0]["labels"]["sex"]                        # 'female'  (sex_code: 'F')

      # LEAD ORDER IS ALPHABETICAL — signal[0] is aVF, not lead I.
      ds.config.lead_names
      # ['aVF','aVL','aVR','I','II','III','V1','V2','V3','V4','V5','V6']

      # Ask by name to get the standard order a cross-dataset model expects.
      std = ECGDataset(
          "ningbo_iva",
          split="train",
          data_path="/path/to/Ningbo_IVA/",
          window=(0, 5000),
          leads=["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"],
      )
      # As shipped, signal[0] is aVF; reordered it is signal[5].
      (ds[0]["signal"][0] == std[0]["signal"][5]).all()   # True
      (ds[0]["signal"][3] == std[0]["signal"][0]).all()   # True  (both lead I)

      # The stored integers have no declared unit; the mV scale is an ESTIMATE.
      ds.config.signal_unit_scale                   # 6.1035e-05
      ds[0]["signal"].abs().max()                   # 1.041   (estimated mV)
      # units="uV" gives 1041.1, and dividing the mV values by 6.1035e-05
      # recovers the shipped integers (peak 17057) exactly.

  - type: links
    title: "References"
    items:
      - { label: "figshare collection (the copy ECGBench was verified against)", url: "https://doi.org/10.6084/m9.figshare.c.4668086.v2" }
      - { label: "Zheng et al., Scientific Data 7, 98 (2020)", url: "https://doi.org/10.1038/s41597-020-0440-8" }
      - { label: "Providers' denoising script (MATLAB)", url: "https://github.com/zheng120/PVCVTECGDenoising" }
---
