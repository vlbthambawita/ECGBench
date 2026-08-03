---
slug: "ptb-xl-plus"
name: "PTB-XL+"
category: "12-lead-physionet"
order: 2
status: "completed"
source_url: "https://physionet.org/content/ptb-xl-plus/1.0.1/"
url_label: "physionet.org"
format: "features & annotations for PTB-XL · no raw ECGs"
patients: "18,869"
records: "21,799"
access: "open"
license: "CC BY 4.0"
origin_institution: "Karlsruhe Institute of Technology"
origin_country: "Germany"
leads: 12
paper_title: "PTB-XL+: A Comprehensive ECG Feature Dataset"
paper_doi: "https://doi.org/10.1038/s41597-023-02153-8"
search_keywords: "ptb-xl+ karlsruhe germany features snomed 12sl unig ecgdeli median beats fiducial points derived"

related:
  - slug: "ptb-xl"
    relation: "derived_from"
    shares_records: true
    verified: true
    note: >
      An annotation and feature layer over PTB-XL's own records, with no raw ECGs of
      its own. Verified from the files: the statements tables and the ecgdeli feature
      table each cover exactly PTB-XL v1.0.3's 21,799 ecg_ids with none missing and
      none extra, and every record of PTB-XL's official train split joins
      (17,376 of 17,376). Because the records are PTB-XL's, ECGBench publishes no
      separate fold assignment for PTB-XL+ — you use PTB-XL's official folds and join
      these columns onto them. Never treat the two as independent datasets: any
      train/test split that puts a recording's waveform on one side and its PTB-XL+
      features on the other is the same recording twice.

sections:
  - type: description
    title: "Overview"
    body: |
      PTB-XL+ is a **companion release, not a standalone dataset**: it ships no raw
      ECGs. It annotates the same 21,799 recordings PTB-XL holds, keyed by PTB-XL's
      own `ecg_id`, and its value is that it gives several independent opinions
      about each recording:

      - **`labels/ptbxl_statements.csv`** — PTB-XL's cardiologist-assigned SCP
        statements, extended with SNOMED CT concept ids.
      - **`labels/12sl_statements.csv`** — statements from the Marquette 12SL
        commercial algorithm, likewise SNOMED-mapped. Comparing human against
        algorithm on identical recordings is much of the point.
      - **`features/{unig,ecgdeli,12sl}_features.csv`** — 748, 531 and 782 measured
        features from three independent providers (University of Glasgow, the KIT
        ECGdeli toolbox, Marquette 12SL), with `features/feature_description.csv`
        mapping equivalent columns between them.
      - **`fiducial_points/ecgdeli/`** — 283,326 per-lead WFDB annotation files.
      - **`median_beats/{12sl,unig}/`** — derived single-beat waveforms. See the
        defects below; ECGBench exposes these as paths only.

      Everything is CC BY 4.0, so unlike MIMIC-IV-ECG there is no redistribution
      question here.

  - type: description
    title: "How ECGBench integrates it — no separate splits"
    body: |
      **There is deliberately no `ptbxl_plus` config, and no
      `ecgbench splits --dataset ptbxl_plus`.** Every PTB-XL+ row is a PTB-XL
      record, so generating a ten-fold partition for it would create a *second*
      ECGBench-blessed split over recordings that `ptbxl` already partitions. A user
      who trained on PTB-XL's folds and evaluated on PTB-XL+'s would be testing on
      training data, with both partitions carrying ECGBench's imprimatur. Rather
      than create that trap and then warn about it, we do not create it.

      Instead PTB-XL+ is a **label and feature provider**: load PTB-XL on its own
      official folds and join these columns onto it.

      ```python
      from ecgbench import ECGDataset
      from ecgbench.labels.ptbxl_plus import load_ptbxl_plus

      ds = ECGDataset("ptbxl", split="train",
                      data_path="/path/to/ptb-xl/1.0.3/", labels=True)
      plus = load_ptbxl_plus("/path/to/ptb-xl-plus/1.0.1/", features=("unig",))
      joined = plus.reindex(ds.metadata_df["ecg_id"].values)   # 100% match
      ```

      You need **both** downloads: PTB-XL+ has no waveforms.

  - type: table
    title: "What ships, and how much of PTB-XL it covers"
    headers: ["Artefact", "Rows / files", "Columns", "Coverage of PTB-XL's 21,799"]
    rows:
      - ["`labels/ptbxl_statements.csv`", "21,799", "3", "complete"]
      - ["`labels/12sl_statements.csv`", "21,799", "4", "complete"]
      - ["`features/ecgdeli_features.csv`", "21,799", "531", "complete"]
      - ["`features/12sl_features.csv`", "21,799", "782", "complete; key hidden at col 145"]
      - ["`features/unig_features.csv`", "21,795", "748", "**4 records missing**"]
      - ["`median_beats/unig/`", "21,794", "—", "5 missing"]
      - ["`median_beats/12sl/`", "20,914", "—", "**885 missing**"]
      - ["`fiducial_points/ecgdeli/`", "283,326 `.atr`", "—", "~13 per record"]
      - ["`labels/snomed_description.csv`", "287", "8", "vocabulary, not per record"]
      - ["`features/feature_description.csv`", "195", "10", "feature dictionary"]

  - type: description
    title: "About those counts"
    body: |
      Recomputed from the shipped files, all of which were verified against the
      release's own `SHA256SUMS.txt` — so the four defects below are upstream, not
      download damage.

      **1. `12sl_features.csv` hides its key column mid-table.** `ecg_id` is
      column **145 of 783**, sitting between `QRS_Area_aVF` and `P_On_Global`
      rather than at the front. Inspect the first or last few columns — the obvious
      thing to do with a 783-column table — and you will conclude it has no key at
      all. Locate the key by name, never by position.

      **2. Neither 12SL table is sorted by `ecg_id`.** Both `12sl_features.csv` and
      `12sl_statements.csv` run `1, 21803, 21804, 21805, 21806, …` — the same order
      as each other, but not ascending. So joining by row position, or assuming
      sorted ids, attaches values to the wrong recordings. (We confirmed the two
      files' key columns are identical in order, so a positional join between
      *them* happens to work — but nothing else should rely on it.)

      **3. Every `median_beats/12sl/*.hea` is unreadable by `wfdb.rdrecord`**
      (300 of 300 sampled). The record line reads
      `ge_median_beats_wfdb/00001_medians 12 500 600` — a stale producer-side
      directory prefix — and wfdb rejects the `/`, raising
      `HeaderSyntaxError: invalid syntax in record line`. The `unig` headers are
      clean. The two providers also pad record stems differently: `00001_medians`
      for 12sl, `000001_medians` for unig.

      **4. unig median-beat amplitudes are about 1000x too large.** They decode to
      600 samples x 12 leads at 500 Hz — a 1.2 s averaged beat, correctly — but span
      roughly −1361 to +602 against a declared `5.9756(-1557)/mV` gain, which is
      microvolts rather than millivolts.

      Because of 3 and 4, ECGBench returns median beats as *paths*
      (`median_beat_path()`) and never decodes them: it will not hand back a signal
      whose units it cannot state. The feature and statement tables, which are what
      the dataset is mostly for, are unaffected.

      **Two label sets, two different quantities.** `ptbxl_scp_codes` is
      cardiologist-assigned with per-statement likelihoods; `12sl_statements` is a
      commercial algorithm's output. They use different vocabularies, mapped into a
      shared 287-concept SNOMED set of which 176 appear in both. Do not treat either
      as ground truth for the other.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset
      from ecgbench.labels.ptbxl_plus import load_features, load_ptbxl_plus

      # Waveforms and folds come from PTB-XL; annotations from PTB-XL+.
      ds = ECGDataset("ptbxl", split="train",
                      data_path="/path/to/ptb-xl/1.0.3/", labels=True)
      plus = load_ptbxl_plus("/path/to/ptb-xl-plus/1.0.1/", features=("unig",))

      plus.shape                         # (21799, 755)
      joined = plus.reindex(ds.metadata_df["ecg_id"].values)
      joined.notna().any(axis=1).sum()   # 17376 of 17376 -- a complete join

      joined.iloc[0]["ptbxl_scp_codes"]      # [('NORM', 100.0), ('LVOLT', 100.0), ('SR', 100.0)]
      joined.iloc[0]["12sl_statements"]      # ['NSR', 'NML']
      joined.iloc[0]["unig_QRS_Dur_Global"]  # 86.0 ms
      joined.iloc[0]["unig_QT_Int_Global"]   # 410.0 ms

      # The 12sl table is keyed for you from the statements file, in file order:
      load_features("/path/to/ptb-xl-plus/1.0.1/", "12sl").index[:5]
      # Index([1, 21803, 21804, 21805, 21806]) -- deliberately not ascending

      # features= is empty by default: the three providers together are >2000
      # columns, and prefix=True keeps their many shared names from colliding.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/ptb-xl-plus/1.0.1/" }
      - { label: "Paper (Scientific Data, 2023)", url: "https://doi.org/10.1038/s41597-023-02153-8" }
      - { label: "PTB-XL — the recordings this annotates", url: "https://physionet.org/content/ptb-xl/1.0.3/" }
---
