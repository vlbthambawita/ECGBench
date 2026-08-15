---
slug: "edgar-experimental-data-and-geometric-analysis-repository"
name: "EDGAR (Experimental Data & Geometric Analysis Repository)"
category: "bspm"
order: 1
status: "completed"
source_url: "https://edgar.sci.utah.edu/"
url_label: "edgar.sci.utah.edu"
format: "24 ECGI experiments · body-surface + epicardial/endocardial/intramural · 54–2223 electrodes · 500–2048 Hz · MATLAB"
patients: "20 subjects (13 human, 5 animal, 2 simulated)"
records: "2,943"
access: "open"
license: "Free registration"
origin_institution: "SCI Institute & CVRTI, University of Utah (curator); ten contributing groups"
origin_country: "Multi-national (USA / Czechia / Germany / Spain / Canada / Netherlands / New Zealand / France / Slovakia / Switzerland)"
leads: "bspm"
paper_title: "Aras et al., J Electrocardiol, 2015"
paper_doi: "https://doi.org/10.1016/j.jelectrocard.2015.08.008"
search_keywords: "edgar bspm body surface potential mapping ecgi electrocardiographic imaging inverse problem utah sci institute cvrti geometry torso ct mri simulation human canine porcine pacing site localization carto sock cage plunge needle transmembrane charles pstov prague kit karlsruhe dalhousie valencia nijmegen bordeaux maastricht auckland ep solutions matlab mat"

sections:
  - type: description
    title: "Overview"
    body: |
      EDGAR is the reference open repository for **electrocardiographic imaging
      (ECGI)**: experiments that record the body surface and the heart surface
      *at the same time*, with matched geometry, so an inverse solution can be
      checked against what the heart was actually doing.

      **It is not one dataset, and ECGBench does not pretend otherwise.** The
      portal holds **26 distinct experiments** contributed by ten institutions
      between 2002 and 2022, each with its own archive layout, MATLAB variable
      names, electrode array, sampling rate and unit convention. 24 of them ship
      time signals: **2,943 recordings from 20 subjects** — 13 human, 5 animal
      and 2 simulated anatomies. Electrode counts run from 54 to 2,223 and rates
      from 500 to 2,048 Hz. Every figure below is an aggregate over experiments
      that were never designed to be pooled; the per-record `experiment`,
      `subject_id`, `recording_surface`, `electrode_array`, `n_leads` and
      `sampling_rate_hz` columns are what you should actually filter on.

      **This is the first MATLAB dataset in ECGBench.** `signal_format: "mat"`
      was added for it, and its signal paths are not paths — they are
      `<file>.mat:<variable>:<orientation>:<unit>`, because a MATLAB container
      declares none of those three reliably and two of them are wrong in the
      files. See "The signal path is not a path" below.

      **The recordings have to be unpacked before anything can read them.** EDGAR
      publishes only zips — 291 of them — so `ecgbench splits` extracts the 24
      authoritative archives into `ecgbench_extracted/` on first run and every
      signal path starts there. Only the `.mat` members under a signal directory
      are written: the CT and MRI volumes are 7,524 DICOM files and 10 of the
      repository's 11 GB, and nothing in ECGBench reads them.

      What makes EDGAR worth the trouble is the **ground truth**. 2,157 of the
      2,943 recordings are body-surface maps of a heart being paced from a site
      whose x/y/z position was measured with CARTO, which is the reference task
      for non-invasive pacing-site localisation. Another 445 are the cardiac
      half of a simultaneous pair — epicardial socks and cages, endocardial
      catheters, plunge needles — which is what an inverse solution is scored
      against.

      2,936 of 2,943 records pass every ECGBench quality check. The seven
      exclusions are all real and all explicable; see "What validation caught".

  - type: description
    title: "The portal cross-posts whole archives, and the wrong ones"
    body: |
      **Read this before downloading anything by hand.** EDGAR is a WordPress
      site, and WordPress stores one upload per filename per month. A dataset
      post that links a generically named `Interventions.zip` therefore gets
      whichever dataset uploaded that name first — and several of them do.

      Verified by SHA-256 over all **12,212** shipped `.mat` members:

      | Post | What its link actually serves |
      |---|---|
      | Afib … Valencia_pat2 → `Interventions.zip` | Charles-PSTOV-pat3's 594 BSPM recordings, byte-identical |
      | KIT … `TMV_FEM` → `Docs`/`Interventions`/`Meshes.zip` | Dalhousie-2006-01-05's |
      | Ischemia torso tank (Utah-02-05-15) → `Docs_Bordeaux…zip` | Bordeaux's documentation |
      | KIT-2020-SimVentrPacings → **every data link** | The KIT-20 *clinical* dataset |

      On top of that, **seven of the 33 portal posts are re-publications** of
      another post — the 2016 archive and the 2025 re-issue of the same
      experiment both appear in the listing (`dalhousie-2006-01-05`,
      `valencia_pat1`, `valencia_pat2` and four `sim-*` posts).

      ECGBench reads **one uniquely titled archive per experiment**, listed in
      `ecgbench.labels.edgar.EXPERIMENTS`. Those 24 archives cover all 2,943
      recordings **exactly once, with no overlap and nothing left over** — which
      is the check that the curation is complete, and it is asserted by
      `ecgbench.labels.edgar.verify_archive_coverage()`.

      Two of the 26 experiments contribute nothing as a result:

      - **`bratislava_2020_p034`** publishes only documentation and geometry. Its
        README describes Run1 (RV apex pacing at 100 bpm) and Run2 (spontaneous
        PVCs) as 131-channel recordings; neither is offered for download.
      - **`kit-2020-simventrpacings-fivesourcemodels`** has the broken links
        above, so the five simulated source models it describes are not
        downloadable. Only its `Documentation-8.pdf` is unique to the post.

      Four further files are 404 on EDGAR's own server (`Charles_PSTOV-12-07-27`
      and `-28` full zips, `images.zip`, `Interventions_Dalhousie-2006-01-05.zip`).
      None costs any data here: the per-module archives cover the same content.

  - type: description
    title: "The signal path is not a path"
    body: |
      Every signal reference in EDGAR's fold CSVs has four parts:

      ```
      ecgbench_extracted/dalhousie_2006/Interventions/BSPM/6105d35e_39.120avg.mat:bspm:sl:uV
                         └ experiment ┘└──────── member ────────┘ └var┘ └┘  └┘
                                                                    orient  unit
      ```

      All four are written explicitly, so nothing is inferred at load time and
      the fold table records exactly what was decided. Each exists because the
      files do not say:

      **`<variable>`** — 22 distinct names across the release. EDGAR's own
      standard says `ts`, but its contributors also ship `bspm`, `ECG`, `EGM`,
      `ens`, `eps`, `pots`, `lichaampots`, `heartleadpots`, and one variable per
      simulated pacing site (`Simulation_04_LVLAT`).

      **`<orientation>`** — `ls` is (leads, samples), `sl` its transpose. **This
      cannot be inferred from the shape.** KIT's `TMV_FEM` simulations are 2,223
      leads by 225 samples and Dalhousie's averaged beats are 1,142 samples by
      120 leads, so "leads are the shorter axis" is wrong in both directions.
      Dalhousie is the one transposed experiment, established from its own
      `bad_leads` field (lead indices up to 120) and its `avg_beats_mtx`, which
      is (beats, 120).

      **`<unit>`** — `mV` or `uV`. EDGAR mixes both across contributors, several
      files declare nothing at all, and **two experiments declare the wrong one**:
      Valencia pat1 and pat2 set `ECG.units = 'mV'` on samples reaching 5350,
      which would be five volts on a body surface. Their own `Docs/Readme.txt`
      says *"the units are microV"*, which is what ECGBench applies. Every record
      carries `declared_unit`, `unit_applied` and `unit_source`, so the
      disagreement is visible rather than silently corrected; those four records
      are the only one in the release.

  - type: description
    title: "One `potvals` field, three different quantities"
    body: |
      EDGAR's struct puts the recording in a field called `potvals`. So do its
      derived maps, and so does a physical quantity that is not a potential.

      **Derived maps.** Utah-10-03-02 stores 570 activation/recovery-interval
      maps of shape (leads, 3) and 570 QRS/QRST/ST/ST80/STT integral maps of
      shape (leads, 5) in the same field as its 570 recordings — the integral
      maps mislabelled `unit = 'ms'`. They are excluded. The separation is clean
      rather than a judgement call: no derived map in the release has more than
      **5** frames and no real recording has fewer than **145**.

      **An inverse solution.** Maastricht ships `heartpots.mat`, which its README
      states are *"NOT measured, but reconstructed from the body-surface
      potentials (with a Tikhonov zeroth order regularization method)"*. Those
      two files are excluded; including a method's output in a benchmark of
      measurements would be a trap of our own making. The same README flags an
      unresolved gain factor on that experiment's *measured* epicardial
      recordings, so their absolute amplitudes should not be trusted either.

      **Transmembrane voltages.** KIT's two TMV source models are simulated
      membrane voltages on a source mesh — resting level exactly −84 mV in every
      one of the 16 records — not extracellular potentials. They are kept, with
      `recording_surface: transmembrane` of their own, so that nobody trains on
      them as though they were electrograms.

  - type: table
    title: "The 24 experiments that ship signals, recomputed from the files"
    headers: ["Experiment", "Subject", "Species", "Setting", "n", "Electrodes", "Hz", "Samples", "Unit", "Surfaces", "Fold"]
    rows:
      - ["charles_pat1", "charles_pstov_pat1", "human", "human_clinical", "944", "120", "2000", "246–364", "mV", "torso 944", "1 (train)"]
      - ["charles_pat3", "charles_pstov_pat3", "human", "human_clinical", "594", "120", "2000", "250–341", "mV", "torso 594", "2 (train)"]
      - ["charles_pat2", "charles_pstov_pat2", "human", "human_clinical", "589", "120", "2000", "250–353", "mV", "torso 589", "6 (train)"]
      - ["utah_2010_sock", "utah_dog_2010_03_02", "dog", "torso_tank", "570", "192–480", "1000", "419–801", "mV", "torso 190 + epicardium 190 + intramural 190", "3 (train)"]
      - ["utah_2002_cage", "utah_dog_2002_05_15", "dog", "torso_tank", "58", "192–599", "1000", "452–750", "mV", "torso 29 + epicardium 29", "9 (val)"]
      - ["dalhousie_2006", "dalhousie_6105", "human", "human_clinical", "54", "120", "2000", "377–1636", "uV", "torso 54", "4 (train)"]
      - ["kit20_clinical", "kit_subject20", "human", "human_clinical", "39", "63", "1000", "145–201", "mV", "torso 39", "5 (train)"]
      - ["kit20_sim_ep_endoepi", "kit_subject20", "simulated", "simulation", "16", "163–502", "—", "200–272", "mV", "torso 8 + epicardium 8", "5 (train)"]
      - ["nijmegen_2004", "nijmegen_ppd2", "human", "human_clinical", "13", "65", "—", "9999", "mV", "torso 13", "7 (train)"]
      - ["kit20_sim_ep_peri", "kit_subject20", "simulated", "simulation", "8", "502", "—", "200–272", "mV", "epicardium 8", "5 (train)"]
      - ["kit20_sim_tmv_endoepi", "kit_subject20", "simulated", "simulation", "8", "502", "—", "200–272", "mV", "transmembrane 8", "5 (train)"]
      - ["kit20_sim_tmv_fem", "kit_subject20", "simulated", "simulation", "8", "2223", "—", "200–272", "mV", "transmembrane 8", "5 (train)"]
      - ["auckland_2012", "auckland_pig_2012_06_05", "pig", "insitu_animal", "6", "171–256", "1200–2048", "2930–13717", "mV", "torso 2 + epicardium 2 + endocardium 2", "8 (train)"]
      - ["bordeaux_2016", "bordeaux_pig_exp16", "pig", "torso_tank", "6", "108–128", "2048", "45056–55296", "mV", "torso 3 + epicardium 3", "10 (test)"]
      - ["utah_2018_tank", "utah_canine_2018_08_09", "dog", "torso_tank", "6", "192–256", "1000", "220–244", "mV", "torso 3 + epicardium 3", "4 (train)"]
      - ["valencia_sim", "valencia_sim_08_01_2014", "simulated", "simulation", "6", "771–2048", "500", "4001", "mV", "torso 3 + endocardium 3", "7 (train)"]
      - ["maastricht_2015", "maastricht_dog2", "dog", "insitu_animal", "4", "65–140", "2048", "515–593", "uV", "torso 2 + epicardium 2", "8 (train)"]
      - ["epsol_24", "ep_solutions_pt_24", "human", "human_clinical", "2", "220", "1000", "192–238", "mV", "torso 2", "9 (val)"]
      - ["epsol_26", "ep_solutions_pt_26", "human", "human_clinical", "2", "192", "1000", "223–253", "mV", "torso 2", "9 (val)"]
      - ["epsol_27", "ep_solutions_pt_27", "human", "human_clinical", "2", "229–230", "1000", "216–245", "mV", "torso 2", "9 (val)"]
      - ["epsol_33", "ep_solutions_pt_33", "human", "human_clinical", "2", "164", "1000", "171–217", "mV", "torso 2", "10 (test)"]
      - ["epsol_36", "ep_solutions_pt_36", "human", "human_clinical", "2", "173–177", "1000", "179–182", "mV", "torso 2", "10 (test)"]
      - ["valencia_pat1", "valencia_pat1", "human", "human_clinical", "2", "54–62", "2034.5", "15191", "uV", "torso 1 + endocardium 1", "10 (test)"]
      - ["valencia_pat2", "valencia_pat2", "human", "human_clinical", "2", "54–73", "2034.5", "12440", "uV", "torso 1 + endocardium 1", "7 (train)"]

  - type: description
    title: "About those counts"
    body: |
      The EDGAR paper (Aras et al. 2015) describes the repository at launch and
      gives no per-experiment record counts, so there is no published table to
      disagree with. Everything above is recomputed from the shipped archives on
      2026-08-10, and the derivation is:

      - one row per 2-D matrix found in a `potvals` field (or in one of the three
        bare-array variables two contributors use) under a signal directory of
        the experiment's authoritative archive;
      - excluding matrices with fewer than 20 time frames (the 1,140 derived
        maps), Maastricht's reconstructed potentials, and the shared BSPM set
        that three of the four KIT simulation posts re-link;
      - deduplicated by content, so a cross-posted archive is counted once.

      **`recording_surface` totals:** torso 2,485 · epicardium 245 · intramural
      190 · transmembrane 16 · endocardium 7.

      **Two counts that will look odd and are correct.** `kit_subject20` is one
      subject across **five** experiments — the KIT-20 clinical study and all
      four KIT simulations, which were computed on that subject's own anatomy, so
      they share a fold. And `kit20_sim_ep_endoepi` has 16 records where its three
      siblings have 8, because its archive is the only one of the four that
      bundles the family's shared 8-run body-surface set.

  - type: description
    title: "The folds are patient-safe, not equal"
    body: |
      Folds are grouped on `subject_id`, which is the guarantee that matters:
      **no subject spans two folds**, so a model cannot memorise one patient's
      torso and be scored on that same patient's other pacing sites.

      What ten folds cannot do is make them equal. Four subjects hold **92%** of
      the recordings — `charles_pstov_pat1` alone has 944 of 2,943 — and five
      subjects have two records each. So the fold sizes are 944 / 594 / 570 /
      589 / 79 / 60 / 21 / 10 / 64 / 12, and the default fold-10 test split is
      **12 records**, not a tenth of the release.

      That is what a repository of 20 experiments looks like. The alternative —
      splitting one subject's 944 recordings across train and test — would make
      pacing-site localisation look solved. For a different question, pass
      `split=None` with `fold_numbers=[...]` and group them yourself.

      **Stratification is deliberately coarser than the label.** Two of the five
      surfaces come from a single subject each (all 190 intramural recordings are
      one dog's plunge needles; all 16 transmembrane runs are one simulated
      anatomy), and a class living in one patient group cannot be spread over ten
      folds. Measured with `StratifiedGroupKFold(10)` over the real table:
      stratifying on `recording_surface` leaves **all ten** folds missing at least
      one class, while stratifying on body-surface vs cardiac-surface leaves
      three. So the fold builder uses that binary and `recording_surface` stays
      the label you train on.

  - type: description
    title: "What validation caught"
    body: |
      **2,936 of 2,943 records pass every check.** The seven exclusions are all
      real properties of the recordings, not decode failures:

      | Records | Issue | What it is |
      |---|---|---|
      | 4 | `missing_leads` + `nan_values` | Auckland marks disconnected electrodes with NaN — 13 to 30 whole channels per record, 0 partially-NaN channels. This is the convention Maastricht's README describes as the intended fix for non-connected electrodes. |
      | 1 | `missing_leads` | One dead electrode in Maastricht's LV-apex-paced epicardial recording. |
      | 2 | `flat_line` | Valencia basket-catheter channels that lost contact with the atrial wall. The README says such channels "were culled from the data"; evidently not all of them. |

      No record fails `amplitude_outlier`, `truncated_signal` or
      `corrupt_header`.

      **`amplitude_range_mv` here is a corruption guard, not a physiological
      range**, and that is forced by the content. Measured over all 2,943
      records, body-surface potentials span [−29.14, +19.18] mV, epicardial
      [−281.55, +49.61], intramural [−103.65, +81.79], simulated transmembrane
      [−84.00, +26.96] and simulated endocardial [−901.35, +670.05]. No single
      bound can mean "physiologically plausible" for all five, so the configured
      one is the union of the attained ranges with a millivolt of slack. Judge
      amplitude plausibility per `recording_surface`, not from the validation
      report.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # The fold CSVs come from the Hub; data_path points at your local EDGAR
      # mirror, which must already have been unpacked by one `ecgbench splits` run.
      ds = ECGDataset(
          "edgar",
          split="train",
          labels=True,
          data_path="/path/to/EDGAR/",
      )
      len(ds)                       # 2861
      s = ds[0]
      s["record_id"]                # 'auckland_2012__Interventions_epi_pacing_Endocardium_epiPacing'
      s["signal"].shape             # torch.Size([256, 2930])
      s["labels"]["experiment"]     # 'auckland_2012'
      s["labels"]["subject_id"]     # 'auckland_pig_2012_06_05'   (species 'pig')
      s["labels"]["recording_surface"]  # 'endocardium'
      s["labels"]["electrode_array"]    # 'EnSite LV catheter'
      s["labels"]["n_leads"]            # 256
      s["labels"]["sampling_rate_hz"]   # 1200.0
      s["labels"]["unit_applied"]       # 'mV'

      # FILTER BEFORE YOU TRAIN. This split alone mixes 16 experiments, 12
      # subjects and electrode counts from 54 to 2223, so a DataLoader over it
      # raises on the first batch.
      df = ds.labels_df
      df["recording_surface"].value_counts().to_dict()
      # {'torso': 2440, 'epicardium': 210, 'intramural': 190,
      #  'transmembrane': 16, 'endocardium': 5}

      # 2157 of these 2861 records carry the paced site's CARTO coordinates —
      # the ground truth for non-invasive pacing-site localisation.
      df[df["pacing_site_x"].notna()].shape[0]      # 2157

      # A batchable subset: one experiment, one electrode count, windowed to its
      # shortest record. window= is pushed into the reader and pickles cleanly,
      # unlike a cropping lambda transform.
      from torch.utils.data import DataLoader, Subset
      from ecgbench import ecg_collate_fn

      paced = ECGDataset("edgar", split="train", labels=True, window=(0, 246),
                         data_path="/path/to/EDGAR/")
      keep = [i for i, r in enumerate(paced.metadata_df["record_id"])
              if r.startswith("charles_pat1__")]
      batch = next(iter(DataLoader(Subset(paced, keep), batch_size=4,
                                   collate_fn=ecg_collate_fn)))
      batch["signal"].shape         # torch.Size([4, 120, 246])

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # The first run UNPACKS 24 archives into ecgbench_extracted/ (~4.4 GB, ten
      # seconds), opens every recording for its shape, rate, declared unit and
      # bad-lead count, joins the CARTO pacing-site tables, and caches the result
      # as ecgbench_metadata.csv in the dataset root — so that root must be
      # writable. Reading MATLAB files needs scipy: pip install 'ecgbench[mat]'.
      #
      # No flags: 20 subjects over ten folds.
      ecgbench splits --dataset edgar --data-path /path/to/EDGAR/

  - type: links
    title: "Links"
    links:
      - label: "EDGAR portal (free registration)"
        url: "https://edgar.sci.utah.edu/"
      - label: "Consortium for ECG Imaging"
        url: "https://www.ecg-imaging.org/edgar-database"
      - label: "Paper — Aras et al., J Electrocardiol 48(6):975-981 (2015)"
        url: "https://doi.org/10.1016/j.jelectrocard.2015.08.008"
      - label: "Example script — examples/load_edgar.py"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_edgar.py"
      - label: "Curated experiment table — ecgbench/labels/edgar.py"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/ecgbench/labels/edgar.py"
---
