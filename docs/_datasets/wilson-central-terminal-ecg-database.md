---
slug: "wilson-central-terminal-ecg-database"
name: "Wilson Central Terminal ECG Database"
category: "12-lead-physionet"
order: 15
status: "completed"
source_url: "https://physionet.org/content/wctecgdb/1.0.1/"
url_label: "physionet.org"
format: "37 ch (I/II/III + V1–V6 + LA/RA/LL + UV1–UV6, raw & filtered, + WCT) · 10 s · 800 Hz"
patients: "92"
records: "540"
access: "open"
license: "ODC Attribution"
origin_institution: "MARCS Institute, Western Sydney Univ.; Campbelltown Hospital"
origin_country: "Australia"
leads: 37
paper_title: "Machines, 2016"
paper_doi: "https://doi.org/10.3390/machines4040018"
search_keywords: "wilson central terminal australia western sydney campbelltown unipolar leads precordial reconstruction wct 800 hz"

sections:
  - type: description
    title: "Overview"
    body: |
      540 ten-second segments from **92 patients admitted to Campbelltown
      Hospital, Sydney**, recorded by the MARCS Institute at Western Sydney
      University with the left-arm, right-arm and left-leg electrode potentials
      brought out individually — so that the **Wilson Central Terminal could be
      measured rather than assumed**.

      The WCT is the reference V1–V6 are measured against. Conventional
      electrocardiography treats it as 0 V; this release exists because it is not,
      and the authors report WCT amplitudes reaching **241% of lead II**. With the
      WCT measured, the six chest electrodes can be referenced against a true zero
      instead, giving the **true unipolar chest leads UV1–UV6** that ship alongside
      the conventional V1–V6. `V = UV − WCT` holds by construction, which is what
      makes the release usable both ways: to quantify the error the assumption
      introduces, and to develop methods that reconstruct or correct it.

      So the **WCT channel is the point of this dataset, not an artefact to filter
      out**, and the "37 leads" are not a 37-lead ECG — see the next section.

      The cohort is clinical and elderly: mean age 65.2 (SD 12.1, range 41–94), 27
      of 92 patients female, mostly with a history of cardiac disease. Each patient
      contributed **1 to 31 segments**, which is why folds are grouped on patient.

  - type: description
    title: "37 channels: 18 raw, 18 filtered, and the WCT"
    body: |
      This is the only dataset in ECGBench that ships **every channel twice**, and
      the only 12-lead-family dataset with **no aVR, aVL or aVF at all**. Channel
      order is identical in all 540 headers:

      | Index | Channels | What they are |
      |---|---|---|
      | 0–8 | `I-Raw`, `II-Raw`, `III-Raw`, `V1-Raw`…`V6-Raw` | conventional leads, raw |
      | 9–11 | `LA-Raw`, `RA-Raw`, `LL-Raw` | the three limb electrode potentials, raw |
      | 12–17 | `UV1-Raw`…`UV6-Raw` | true unipolar chest leads, raw |
      | 18–26 | `I`, `II`, `III`, `V1`…`V6` | the same conventional leads, filtered |
      | 27–29 | `LA`, `RA`, `LL` | the same limb potentials, filtered |
      | 30–35 | `UV1`…`UV6` | the same unipolar chest leads, filtered |
      | 36 | `WCT` | the Wilson Central Terminal — filtered only |

      "Filtered" means **DC removal plus a 0.05–150 Hz band-pass**. There is no
      `WCT-Raw`.

      Three practical consequences:

      - **Do not mix the two families in one tensor.** They are two preprocessing
        states of the same recording. Index 18 is filtered lead I; index 0 is raw
        lead I. Select by name (`leads=[...]`), never by index.
      - **The raw limb and unipolar channels carry several mV of DC offset**, which
        is exactly what the DC-removal filter takes out — in `patient001/seg01` the
        raw `LA`/`RA`/`LL` and `UV1`–`UV6` channels sit around −5 mV while their
        filtered counterparts are centred near zero. That is why
        `amplitude_range_mv` is a deliberately wide ±20 rather than the usual ±10:
        an unreferenced potential is not a lead voltage.
      - **aVR, aVL and aVF must be derived** if you want them:
        `aVR = −(I + II)/2`, `aVL = I − II/2`, `aVF = II − I/2`.

      Also unusual: **800 Hz, and 8001 samples** — 10.00125 s, not a round number.
      All 540 records are identical in geometry, so the truncation check is exact.

  - type: description
    title: "Eight records contain synthesised precordial channels"
    body: |
      Eight segments from five patients (007, 008, 010, 014, 031) have one or two
      precordial channels that were **reconstructed as `V = UV − WCT` rather than
      recorded**, because the recorded channel was unusable. The release notes this
      per record in a `#Reconstruct Precordials:` header comment, and ECGBench
      exposes it as `reconstructed_precordials` (a list) and
      `has_reconstructed_precordials`.

      | Record | Synthesised channels |
      |---|---|
      | `patient007_seg01`, `patient007_seg02` | V2, V2-raw |
      | `patient007_seg03` | V1, V1-raw |
      | `patient008_seg01`, `patient008_seg02` | V1, V1-raw, V2, V2-raw |
      | `patient010_seg01`, `patient014_seg01`, `patient031_seg01` | V2, V2-raw |

      **Exclude them when evaluating precordial reconstruction**, which is this
      dataset's headline use. On those channels `V = UV − WCT` holds *exactly* by
      construction, so a method that estimates V from UV and WCT is being scored
      against its own output.

      ```python
      from ecgbench import load_labels

      labels = load_labels("wctecgdb", "/path/to/wctecgdb/1.0.1/")
      measured = labels[~labels["has_reconstructed_precordials"]]
      len(labels), len(measured)          # (540, 532)
      ```

  - type: table
    title: "Admission diagnosis groups — counted two ways"
    headers: ["Group", "Patients", "Records", "Note"]
    rows:
      - ["Myocardial infarction", "28 (30.4%)", "137 (25.4%)", "NSTEMI 21 patients, STEMI 4"]
      - ["Atrial fibrillation or flutter", "14 (15.2%)", "99 (18.3%)", "10 distinct strings"]
      - ["Other or non-cardiac", "14 (15.2%)", "69 (12.8%)", "chest pain, gastritis, urosepsis, PE, mitral stenosis, syncope"]
      - ["Angina or coronary artery disease", "12 (13.0%)", "72 (13.3%)", "stable angina 6 patients"]
      - ["Not reported", "10 (10.9%)", "38 (7.0%)", "the release's own \"not reported\""]
      - ["Other tachyarrhythmia", "6 (6.5%)", "41 (7.6%)", "VT 3 patients, SVT 3"]
      - ["Cardiomyopathy or heart failure", "5 (5.4%)", "68 (12.6%)", "**5.4% of patients, 12.6% of records**"]
      - ["Bradyarrhythmia or conduction block", "3 (3.3%)", "16 (3.0%)", "sinus bradycardia 2, complete heart block 1"]
      - ["**Total**", "**92**", "**540**", "single-label per patient, so both columns sum"]

  - type: description
    title: "About those counts"
    body: |
      All 1,082 shipped files were verified against the release's own
      `SHA256SUMS.txt` before any figure here was computed — all OK. Record and
      patient counts match the landing page exactly at **540 and 92**, as do the 27
      female patients, the mean age of 65.23 (SD 12.12), the 10 patients with
      unreported diagnoses and the 8 reconstructed segments. There is nothing to
      reconcile.

      Every figure above is **recomputed from the 540 `.hea` files**, which are the
      only place any label exists — the release ships no metadata table (just the
      `.dat`/`.hea` pairs under `patient001/`…`patient092/`, plus `RECORDS`,
      `SHA256SUMS.txt` and `LICENSE.txt`). ECGBench parses the `#Age:`, `#Sex:`,
      `#Diagnosis report:` and `#Reconstruct Precordials:` comment lines; see
      `ecgbench/labels/wctecgdb.py`.

      **The two count columns are the same information weighted differently, and
      the difference is large.** Age, sex and diagnosis are constant within a
      patient (verified across all 540 headers), while segment counts run 1–31
      (mean 5.9, median 4). Five patients contribute 132 records — **24.4% of the
      dataset** — and 18 patients contribute one each. Cardiomyopathy or heart
      failure is 5.4% of patients but 12.6% of records. Any per-record rate you
      quote is a statement about segment counts as much as about patients, so
      **weight by patient or group by `patient_id`**.

      **The diagnosis is an admission label, not a waveform label.** It records why
      the patient came in, not what these ten seconds show: a segment in the "Other
      tachyarrhythmia" group need not contain a tachycardia. There is **no beat,
      rhythm or interval annotation anywhere in this release**.

      **The group column is a reduction, and the reduction is a judgement call.**
      The headers hold **43 distinct free-text strings over 92 patients** (40 after
      correcting the misspellings), 28 of them held by a single patient — so the
      strings cannot be stratified on directly. `diagnosis_group` reduces them to
      the 8 groups above via an explicit, auditable map in
      `ecgbench/labels/wctecgdb.py`, which **raises rather than guessing** on a
      string it does not know. Six patients' strings name two conditions (e.g.
      `Non ST segment elevation myocardial infarction (NSTEMI)- rapid Atrial
      fibrillation`), and the map picked one. Train on `diagnosis`, not
      `diagnosis_group`.

      Three transcription traps live in these header lines, all handled and
      regression-tested:

      - **The headers are not UTF-8.** The dash in "ST segment elevation" is byte
        `0xA0`, a Windows-1252 non-breaking space, so a strict UTF-8 read raises and
        `errors="replace"` silently yields `ST�segment�elevation`. ECGBench decodes
        cp1252 and folds the NBSPs; `diagnosis_raw` keeps the original bytes.
      - **Some strings are misspelt or inconsistently cased**: `Atypica chest
        pain`, `Type 2 Myocaridal infarctoin`, `Congestive Cardic failure (CCF)`,
        and `sinus bradycardia` alongside `Sinus bradycardia`. Left alone, each
        variant becomes its own class. `diagnosis` is corrected, `diagnosis_raw` is
        verbatim.
      - **`not reported` is a value, not a blank.** 10 patients / 38 records.
        Exposed as `diagnosis_reported = False`, because "nothing recorded" and
        "column missing" are different facts.

  - type: table
    title: "Validation summary (800 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "540", "all records, with is_valid + quality_issues"]
      - ["clean", "540", "100% pass rate — no record excluded"]
      - ["excluded", "0", "no NaN samples, no flat or all-zero channel, uniform 37 × 8001"]
      - ["amplitude extremes", "−16.51 … +9.23 mV", "over all 540 × 37 channels; `amplitude_range_mv` is ±20"]

  - type: description
    title: "140 records have a channel clipped at the acquisition rail"
    body: |
      Every record passes validation, but "valid" is not "unclipped". **128,802
      samples across the release sit at exactly ±9.2250 mV**, the acquisition rail:
      140 of the 540 records have at least one channel pinned there, 61 of them for
      a plateau of 10 ms or longer, the longest 227.5 ms in `patient069/seg07`
      `UV1-Raw`.

      It is concentrated where you would expect — the **unreferenced** channels,
      before DC removal: `UV2-Raw` (83 records), `UV6-Raw` (46), `UV1-Raw` (44),
      `UV3-Raw` (39), `UV5-Raw` (32), `UV4-Raw` (24), `LA-Raw` (20), `RA-Raw` (13),
      and single figures for the rest. This is saturation of the raw unipolar and
      limb-potential channels, i.e. a property of the recording setup, not download
      damage — all 1,082 files match the release's `SHA256SUMS.txt`.

      `amplitude_range_mv` is therefore ±20, wide enough not to reject the rail as
      an outlier. **If clipped samples matter for your method, test for the rail
      yourself:**

      ```python
      import numpy as np, wfdb
      sig = wfdb.rdrecord("patient069/seg07").p_signal          # (8001, 37)
      clipped = np.abs(np.abs(sig) - 9.2250) < 1e-3
      clipped.sum(), clipped.any(axis=0).nonzero()[0]           # (1486, array([3, 12]))  -- V1-Raw, UV1-Raw
      ```

      Only 4 records exceed ±10 mV, and three of them are exactly the synthesised
      channels of patients 008 and 031 (`V1-Raw` down to −16.51 mV in
      `patient008/seg01`): `V = UV − WCT` is a difference of two channels, so it can
      overshoot a rail that neither input can pass.

  - type: description
    title: "Folds are grouped by patient, and that is not optional here"
    body: |
      Fold boundaries follow **patients**, not records: `StratifiedGroupKFold` on
      `patient_id`, stratified on `diagnosis_group`. Without grouping, a patient with
      31 near-identical ten-second windows of one admission would land on both sides
      of every split — and since the diagnosis is constant within a patient, the
      label would leak outright. Verified on the shipped release: **no patient spans
      two folds**.

      Two consequences of grouping 92 patients into 10 folds:

      - **Fold sizes vary** — 7–11 patients and 49–58 records per fold rather than a
        uniform 54, because segment counts run 1–31. The default 8/1/1 mapping gives
        **439 train / 49 val / 52 test**.
      - **Stratification is approximate**, and quantifiably so: three groups hold
        3–6 patients, fewer than the 10 folds. `Bradyarrhythmia or conduction block`
        (3 patients) reaches only 3 folds, and **the test fold contains no
        bradyarrhythmia and no cardiomyopathy record at all**. `Other
        tachyarrhythmia` lands 13 records in test against 18 in train, because one
        high-segment patient carries it. `StratifiedGroupKFold` does not emit the
        "least populated class" warning `StratifiedKFold` does, so the silence is
        not evidence of balance — check the per-fold table if a group matters to you.

      For a stricter cross-validation harness, rotate the held-out fold rather than
      using the default 8/1/1 mapping:

      ```python
      from ecgbench import ECGDataset

      # Hold out fold 3, train on the rest. With split=None each sample's
      # ["split"] reports that record's own default split.
      train = ECGDataset("wctecgdb", split=None,
                         fold_numbers=[1, 2, 4, 5, 6, 7, 8, 9, 10],
                         data_path="/path/to/wctecgdb/1.0.1/")
      held  = ECGDataset("wctecgdb", split=None, fold_numbers=[3],
                         data_path="/path/to/wctecgdb/1.0.1/")
      ```

  - type: code
    title: "Getting the data"
    language: bash
    body: |
      # The ~300 MB zip is genuinely public -- no PhysioNet credentials needed.
      wget https://physionet.org/static/published-projects/wctecgdb/wilson-central-terminal-ecg-database-1.0.1.zip
      unzip wilson-central-terminal-ecg-database-1.0.1.zip

      # Verify before trusting any figure -- all 1,082 files should report OK.
      cd wilson-central-terminal-ecg-database-1.0.1 && sha256sum -c SHA256SUMS.txt

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # Writes ecgbench_metadata.csv into the dataset root on first run: the
      # release ships no metadata table, and the validation engine re-reads that
      # file from disk. The dataset root must be writable.
      ecgbench splits --dataset wctecgdb --data-path /path/to/wctecgdb/1.0.1/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the HuggingFace Hub by default; only the waveforms
      # need to be local.
      ds = ECGDataset(
          "wctecgdb",
          split="train",
          data_path="/path/to/wctecgdb/1.0.1/",
          labels=True,
      )

      len(ds)                                    # 439
      ds[0]["signal"].shape                      # (37, 8001)  -- 10 s at 800 Hz
      ds[0]["record_id"]                         # 'patient001_seg01'

      # Patient-level admission label, repeated across that patient's segments:
      ds[0]["labels"]["patient_id"]              # 'patient001'
      ds[0]["labels"]["age"], ds[0]["labels"]["sex"]        # (46, 'M')
      ds[0]["labels"]["diagnosis"]               # 'Non ST segment elevation myocardial infarction (NSTEMI)'
      ds[0]["labels"]["diagnosis_group"]         # 'Myocardial infarction'  -- the stratification label

      # 540 records, 92 patients: group before quoting any rate. labels_df is
      # aligned POSITIONALLY with metadata_df and carries a RangeIndex.
      ds.labels_df["patient_id"].nunique()       # 75 patients in this split
      ds.labels_df.groupby("patient_id").size().max()       # 31 segments

      # Pick ONE preprocessing family, by name. Index 0 is raw lead I and index 18
      # is filtered lead I -- they are the same signal before and after the
      # 0.05-150 Hz band-pass, not two leads.
      filtered = ECGDataset("wctecgdb", split="train",
                            leads=["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"],
                            data_path="/path/to/wctecgdb/1.0.1/")
      filtered[0]["signal"].shape                # (9, 8001)

      # The Wilson Central Terminal itself -- the channel the rest of ECG practice
      # assumes is 0 V.
      wct = ECGDataset("wctecgdb", split="train", leads=["WCT", "II"],
                       data_path="/path/to/wctecgdb/1.0.1/")
      sig = wct[0]["signal"]
      (sig[0].max() - sig[0].min()) / (sig[1].max() - sig[1].min())   # tensor(0.6670)

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/wctecgdb/1.0.1/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/f73z-an96" }
      - { label: "Paper (Machines, 2016)", url: "https://doi.org/10.3390/machines4040018" }
      - { label: "Example script", url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_wctecgdb.py" }
---
