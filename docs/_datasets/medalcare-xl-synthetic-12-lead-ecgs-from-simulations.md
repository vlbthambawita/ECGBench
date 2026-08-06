---
slug: "medalcare-xl-synthetic-12-lead-ecgs-from-simulations"
name: "MedalCare-XL (Synthetic 12-Lead ECGs from Simulations)"
category: "12-lead-other"
order: 10
status: "completed"
source_url: "https://doi.org/10.5281/zenodo.8068944"
url_label: "zenodo.org"
format: "12-lead · 10 s · 500 Hz · CSV (leads in rows, no header) · raw/noise/filtered variants"
patients: "0 (synthetic)"
records: "16,842"
access: "open"
license: "CC BY 4.0"
origin_institution: "Medical Univ. of Graz; KIT; PTB; Univ. of Edinburgh"
origin_country: "Austria / Germany / UK"
leads: 12
paper_title: "Gillette et al., Scientific Data, 2023"
paper_doi: "https://doi.org/10.1038/s41597-023-02416-4"
search_keywords: "medalcare xl synthetic simulation electrophysiological in silico zenodo austria germany uk graz kit ptb edinburgh myocardial infarction bundle branch block atrial fibrosis simulated transposed csv"
patients_class: "count-na"

sections:
  - type: description
    title: "Overview"
    body: |
      16,842 twelve-lead ECGs that no heart ever produced. Every record is the
      output of a multi-scale electrophysiological simulation: atrial and
      ventricular models were run independently over anatomical meshes, composed
      into 10 s rhythms at 500 Hz, and projected onto torso models to give a
      surface 12-lead trace. Eight conditions ship, one of them subdivided eight
      ways.

      That makes it the odd one out in this catalogue, and worth being precise
      about why. **The label is exact by construction** — it is the setting the
      simulator was given, so there is no reader disagreement, no comorbidity, no
      borderline case and no diagnostic uncertainty. A model that separates these
      classes perfectly has learned to separate simulator configurations, which is
      a weaker claim than separating patients. Used as intended — pre-training,
      augmentation, controlled ablations where you need to vary one physiological
      parameter and hold the rest fixed — the exactness is the point. Used as a
      benchmark on its own, it will overstate.

      Three practical things to know before loading it:

      **The CSVs are transposed relative to every other CSV dataset here.** Each
      file is 12 rows × 5000 columns with **no header row** — one row per lead,
      samples running along the row. `chapman_shaoxing` and `ningbo_iva` are the
      other way round, samples in rows under a header naming the leads. ECGBench
      gives this its own `signal_format`, `csv_lead_rows`, because reading one
      layout with the other's reader returns a plausibly-shaped array of the wrong
      thing rather than raising.

      **Each record ships three times.** `<n>_raw.csv` is the noise-free simulator
      output, `<n>_noise.csv` adds superimposed noise, and `<n>_filtered.csv`
      applies the release's 0.5–150 Hz order-3 Butterworth pair. These are one
      record in three renderings, not three records. ECGBench wires up
      **filtered**, the closest of the three to a recorded ECG; the other two are
      one string substitution away and the label loader exposes both.

      **The split is the authors' own, not a generated one.** Their
      train/validation/test directories become folds 1, 2 and 3, so `--n-folds`
      has no effect here and there are three fold CSVs rather than ten. See "The
      split guarantee has one hole" below for the one place it does not do what
      the README says.

  - type: table
    title: "Class breakdown (all 16,842 records, recomputed from the files)"
    headers: ["Condition", "Subclass", "Train", "Val", "Test", "Total"]
    rows:
      - ["Normal sinus rhythm", "sinus", "900", "200", "200", "1,300"]
      - ["AV block", "avblock", "900", "200", "200", "1,300"]
      - ["Left bundle branch block", "lbbb", "900", "200", "200", "1,300"]
      - ["Right bundle branch block", "rbbb", "898", "200", "200", "1,298"]
      - ["Left atrial enlargement", "lae", "1,040", "130", "130", "1,300"]
      - ["Fibrotic atrial cardiomyopathy", "fam", "1,040", "130", "130", "1,300"]
      - ["Interatrial conduction block", "iab", "994", "124", "126", "1,244"]
      - ["Myocardial infarction", "LAD_0.3", "900", "200", "200", "1,300"]
      - ["", "LAD_1.0", "900", "200", "200", "1,300"]
      - ["", "LCX_0.3_ant", "450", "150", "100", "700"]
      - ["", "LCX_0.3_post", "450", "100", "100", "650"]
      - ["", "LCX_1.0_ant", "400", "100", "100", "600"]
      - ["", "LCX_1.0_post", "450", "100", "100", "650"]
      - ["", "RCA_0.3", "900", "200", "200", "1,300"]
      - ["", "RCA_1.0", "900", "200", "200", "1,300"]
      - ["", "**MI subtotal**", "**5,350**", "**1,250**", "**1,200**", "**7,800**"]
      - ["**All**", "", "**12,022**", "**2,434**", "**2,386**", "**16,842**"]

  - type: description
    title: "About those counts"
    body: |
      **The title says 16,900 and what ships is 16,842.** Both the paper's title
      and the Zenodo title read "16,900 … electrocardiograms". The v1.3 deposit
      holds 16,842, and there is no changelog explaining the difference.

      | Figure | Published | ECGBench (v1.3) | Diff |
      |---|---|---|---|
      | Records | 16,900 | **16,842** | −58 |

      The recomputed figure was derived three independent ways, which all agree:

      - 16,848 `*_raw.csv` files under `WP2_largeDataset_Noise/`, **minus the 6 in
        `mi/examples/`**, which are figure illustrations sitting outside the
        `<split>/run_<model>/` layout and are byte-distinct from every record in
        `mi/*/test/run_S62/`;
      - 16,842 `*_AtrialParameters.txt` files;
      - 16,842 `*_VentricularParameters.txt` files.

      The shortfall is concentrated in **iab**, the only class that does not reach
      a round total (1,244 against 1,300 for its peers). 13 of the 186 run
      directories have gaps in their file numbering — `iab/train/run_S66` holds
      107 files numbered up to `000130`, and `iab/test/run_S62` is missing numbers
      17, 71, 72 and 75 out of 130. ECGBench enumerates records from the files
      that exist rather than from a range, so a gap costs a record rather than
      producing a broken path.

      **Class counts are single-label and sum to the record total**, unlike the
      multi-label datasets in this catalogue. Every record carries exactly one
      condition and none is unlabelled. The 15-class `pathology_subclass` above is
      also what the config stratifies on, so the split table and the stratification
      label are the same quantity here — which is not true of most pages in this
      catalogue.

      **Myocardial infarction is 46.3% of the release** and each remaining
      condition is 7.4–7.7%. That is an authoring choice about how finely to
      subdivide MI, not a prevalence.

      **The local copy was checksum-verified before any figure above was
      computed.** `MedalCare-XL.zip` has md5 `96497fcbc5c443bec1280f2033836776`,
      matching Zenodo's published value, and the extracted tree is an exact
      bidirectional match with the archive listing — 84,416 files, 0 missing,
      0 extra.

  - type: description
    title: "The split guarantee has one hole"
    body: |
      The release README states that ECGs "calculated with the same anatomical
      model but different electrophysiological parameters are only present in one
      of the test, validation and training datasets but never in multiple."

      **That holds within each condition and fails across them.** Ventricular
      model `S64` is the *test*-side model for sinus, AV block, LBBB, RBBB and
      every MI subclass — and simultaneously a *train*-side model for the three
      atrial conditions (fam, iab, lae). `S67` does the same for validation
      against train. Eleven of the thirteen models sit in exactly one fold; those
      two straddle.

      This is not a name collision. Checked at the parameter level:
      `sinus/test/run_S64` and `fam/train/run_S64` agree on **all 84** ventricular
      parameters that are constant within a run directory — same ionic model, same
      conductivities, same action-potential durations, same fibre angles. It is one
      ventricular model appearing on both sides of the split.

      **No records are shared, though**, so this is not duplicate-row leakage.
      Two independent checks:

      - No two records in those directories have the same full ventricular
        parameter vector — 0 matches over 100 × 130 pairs.
      - A waveform signature over a 600 ms QRS-centred window across all 12 leads
        puts the nearest cross-directory pair at distance **1.13**, against a
        within-directory median of **3.78**. The signature detects exact duplicates
        at distance 0.0 (verified on a self-match), so the null result is evidence
        of absence rather than of an insensitive measure.

      **What it means in practice:** if you train a classifier that spans the
      atrial conditions (fam/iab/lae) and the ventricular ones (everything else),
      the same ventricular anatomy appears in both your training and your test
      set. Within either arm alone, the guarantee holds as stated. `model_id` is
      exported in every fold CSV precisely so this is checkable — and regroupable —
      by hand rather than being something you have to take on trust.

  - type: table
    title: "Ventricular model by fold"
    headers: ["Model", "Folds it appears in", "Note"]
    rows:
      - ["S62", "3 (test)", ""]
      - ["S63", "2 (val)", ""]
      - ["S64", "**1 and 3**", "test for sinus/avblock/lbbb/rbbb/mi, train for fam/iab/lae"]
      - ["S65–S66", "1 (train)", ""]
      - ["S67", "**1 and 2**", "val for sinus/avblock/lbbb/rbbb/mi, train for fam/iab/lae"]
      - ["S68–S74", "1 (train)", ""]

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "16,842", "all records, with is_valid + quality_issues"]
      - ["clean", "16,810", "99.81% pass rate"]
      - ["excluded", "32", "all amplitude_outlier — see below"]

  - type: description
    title: "32 excluded records, and why it is not a units bug"
    body: |
      All 32 exclusions fail `amplitude_outlier`, and **30 of the 32 fail on V1**
      (the other two on V2 and V3). The magnitudes are not marginal: 10.1 to
      **879.6 mV**, median 25.7 mV, where a random 400-record sample of the release
      peaks at a median of **2.09 mV** with a 99th percentile of **5.30 mV**. There
      is a wide empty gap between the bulk of the release and the failures, so the
      `[-10, 10]` mV threshold is not slicing into a continuum.

      Two things rule out a configuration mistake on ECGBench's side. First, the
      blowups are already present in the **raw** simulator output, before the noise
      and filtering stages ever ran, so they are a forward-solution artefact rather
      than something the release's processing introduced. Second, if the unit scale
      were wrong, every record would fail, not 0.19% of them.

      V1 is the precordial electrode closest to the ventricular surface, which is
      where a torso mesh node sitting too near the source would produce exactly
      this. The exclusions span six of the eight conditions (24 of the 32 are MI
      records), so it is not a property of one class.

      They are not deleted — `version="original"` returns all 16,842 with
      `is_valid` and `quality_issues` marking them, and
      `output/medalcare_xl/validation_report.json` names each one and the lead and
      range that failed.

  - type: description
    title: "The simulation parameters are the real ground truth"
    body: |
      Every record ships two plain-text parameter files alongside it:
      `<n>_AtrialParameters.txt` (~21 keys) and `<n>_VentricularParameters.txt`
      (~105 keys), holding the ionic model, tissue conductivities, regional
      conduction velocities, action-potential durations, fibre angles, stimulus
      sites, ischaemic-region geometry, the atrial and torso meshes and all ten
      electrode positions. For a simulated dataset these *are* the ground truth —
      the pathology label is a summary of them.

      `ecgbench.labels.medalcare_xl.load_simulation_parameters` returns them as a
      frame indexed by record id, with `atrial.` and `ventricular.` prefixes
      because the two files reuse key names (`im.name`, `G.torso`) and an
      unprefixed concat would silently drop one of each. It is **opt-in**:
      `labels=True` does not read them, because that would be 33,684 file opens on
      every dataset construction. Pass `record_ids=` to restrict it to a split.

      The key set is ragged, and that is information rather than damage:

      | Condition | Parameter difference |
      |---|---|
      | MI | adds 14 `isch[0].*` keys — the ischaemic region's size, position and velocities |
      | LBBB / RBBB | drop the `stim[*]` block; removing stimulus sites is *how* the block is created |
      | LAE | drops the 4 `cv_t.*` regional atrial conduction velocities |

      So the union is 119 ventricular and 21 atrial keys, of which 70 and 17 appear
      in every condition. A NaN means the condition has no such parameter, not that
      a value is missing.

      One thing ECGBench deliberately does **not** read: the per-run-directory
      `siginfo.csv`. For fam, iab and lae its `info2` column holds a foreign
      simulation id rather than the record number, and in 13 of the 186 run
      directories it has more rows than there are records — so joining it means
      guessing at row order, on exactly the tables where the guess is least
      checkable. The parameter files carry the same anatomy (`geo.atria`,
      `geo.torso`) keyed by record number, so nothing is lost by ignoring it.

  - type: description
    title: "How the folds were made"
    body: |
      **Not generated — adopted.** The release's own `train/`, `validation/` and
      `test/` directories become folds 1, 2 and 3, giving 12,022 / 2,434 / 2,386
      (71.4% / 14.5% / 14.2%, matching the README's stated ~70/15/15). Because the
      splits are predefined, `--n-folds` has no effect and there are three fold
      CSVs, not ten.

      `patient_id_column` is set to `model_id` — the ventricular simulation model,
      `S62`–`S74`. **There are no patients here**, and rather than leave the field
      null this points at the grouping unit that plays the same role: the thing the
      authors defined the split around, and the one whose reuse is worth being able
      to see. Read "The split guarantee has one hole" above before assuming it is
      fold-disjoint; two of the thirteen models are not.

      Fold membership is identical between `original/` and `clean/` — `clean/` is a
      row subset of 16,810, not a re-split.

      With one train fold and one test fold there is no ten-fold rotation to run
      here. If you want cross-validation, regroup by `model_id`, which every fold
      CSV carries.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # Point --data-path at the INNER directory: the release extracts to a
      # MedalCare-XL/MedalCare-XL/ nesting, and the one you want holds
      # WP2_largeDataset_Noise/. It must be writable — the release ships no
      # metadata table, so ECGBench generates ecgbench_metadata.csv there.
      ecgbench splits --dataset medalcare_xl \
        --data-path /path/to/MedalCare-XL/MedalCare-XL/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the Hub; data_path points at your local signals.
      # labels=True needs `ecgbench splits` to have run once — the release ships
      # no metadata table, so the labels come from the CSV that run generates.
      ds = ECGDataset(
          "medalcare_xl",
          split="train",
          data_path="/path/to/MedalCare-XL/MedalCare-XL/",
          labels=True,
      )

      len(ds)                                    # 11996   (clean; original: 12022)
      ds[0]["signal"].shape                      # (12, 5000)   10 s at 500 Hz
      ds[0]["record_id"]                         # 'avblock_train_S65_000001'
      ds[0]["labels"]["pathology"]               # 'avblock'
      ds[0]["labels"]["pathology_name"]          # 'AV block'
      ds[0]["labels"]["pathology_subclass"]      # 'avblock'   (15-class label)
      ds[0]["labels"]["mi_subclass"]             # nan  — blank for non-MI records
      ds[0]["labels"]["model_id"]                # 'S65'  — the simulation model
      ds[0]["signal"].abs().max()                # 1.9783  (millivolts, no scaling)

      # The CSVs are 12 rows x 5000 columns with NO header — the transpose of
      # every other CSV dataset here. That is what csv_lead_rows means.
      ds.config.signal_format                    # 'csv_lead_rows'
      ds.config.lead_names
      # ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

      # Standard lead order, so leads= is about selection rather than repair.
      lead = ECGDataset("medalcare_xl", split="train",
                        data_path="/path/to/MedalCare-XL/MedalCare-XL/",
                        leads=["II", "V1", "V5"])
      lead[0]["signal"].shape                    # (3, 5000)

      # Every record is exactly 5000 samples, so window= is optional here —
      # useful for cutting batch size, not for making batching possible.
      win = ECGDataset("medalcare_xl", split="train",
                       data_path="/path/to/MedalCare-XL/MedalCare-XL/",
                       window=(3000, 2000))     # the last 4 s
      win[0]["signal"].shape                     # (12, 2000)

      # Samples are already millivolts; units="uV" scales by 1000.
      uv = ECGDataset("medalcare_xl", split="train",
                      data_path="/path/to/MedalCare-XL/MedalCare-XL/",
                      units="uV")
      uv[0]["signal"].abs().max()                # 1978.3

  - type: code
    title: "The other two signal variants, and the simulation parameters"
    language: python
    body: |
      import numpy as np
      from ecgbench.labels import load_labels
      from ecgbench.labels.medalcare_xl import load_simulation_parameters

      root = "/path/to/MedalCare-XL/MedalCare-XL/"
      labels = load_labels("medalcare_xl", root)

      # ECGBench wires up the filtered variant; the other two are columns here.
      labels.loc["sinus_train_S65_000001", ["signal_path",
                                            "signal_path_raw",
                                            "signal_path_noise"]]

      # One record in three renderings — same simulation, different processing.
      row = labels.loc["sinus_train_S65_000001"]
      for column in ("signal_path_raw", "signal_path_noise", "signal_path"):
          signal = np.loadtxt(f"{root}/{row[column]}", delimiter=",")
          print(column, signal.shape, round(float(np.abs(signal).max()), 3))

      # The parameters that produced each record. Opt-in: it reads two text
      # files per record, so restrict it to the split you care about.
      params = load_simulation_parameters(
          root, "medalcare_xl", record_ids=list(labels.index[:200])
      )
      params["ventricular.im.name"].iloc[0]      # 'MitchellSchaeffer'
      params["atrial.im.name"].iloc[0]           # 'Courtemanche'
      params["atrial.geo.atria"].iloc[0]         # 'cn617_g043'  — the atrial mesh
      params["atrial.geo.torso"].iloc[0]         # 'torsoID16'
      # Units travel with the value rather than being coerced away:
      params["atrial.cv_t.BulkTissue"].iloc[0]   # '591mm/s'
      # 14 isch[0].* columns, populated for MI records and NaN elsewhere.

  - type: links
    title: "References"
    items:
      - { label: "Zenodo deposit v1.3 (the copy ECGBench was verified against)", url: "https://doi.org/10.5281/zenodo.8068944" }
      - { label: "Gillette et al., Scientific Data 10, 531 (2023)", url: "https://doi.org/10.1038/s41597-023-02416-4" }
      - { label: "Example script: examples/load_medalcare_xl.py", url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_medalcare_xl.py" }
---
