---
slug: "echonext"
name: "EchoNext"
category: "12-lead-physionet"
order: 20
status: "completed"
source_url: "https://physionet.org/content/echonext/1.1.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 250 Hz · NumPy arrays, z-scored"
patients: "36,286"
records: "100,000 (82,543 split)"
access: "restricted"
license: "PhysioNet Restricted Health Data License 1.5.0"
origin_institution: "Columbia University Irving Medical Center"
origin_country: "USA — New York, NY"
leads: 12
paper_title: "Poterucha et al., Nature, 2025"
paper_doi: "https://doi.org/10.1038/s41586-025-09227-0"
search_keywords: "echonext columbia usa new york echocardiography structural heart disease npy numpy restricted zscore"

sections:
  - type: description
    title: "Overview"
    body: |
      EchoNext pairs 100,000 12-lead ECGs from Columbia University Irving Medical
      Center and Allen Hospital with the findings of a **matched
      echocardiogram**. The task it defines is inferring structural heart disease
      from the ECG alone — the composite `shd_moderate_or_greater_flag` is
      positive for 52% of the training split, with ten per-condition flags
      (reduced LVEF, wall thickening, valve regurgitation and stenosis, pulmonary
      pressure, effusion) underneath it.

      Waveforms are 10 s at 250 Hz, shipped **preprocessed**: median-filtered per
      lead, clipped at the 0.1st and 99.9th percentiles, and standardised with a
      dataset-wide mean and standard deviation. They arrive as four NumPy arrays
      of shape `(N, 1, 2500, 12)` — one per split — rather than as per-record
      files, alongside `echonext_metadata_100k.csv` and preprocessed tabular
      features.

  - type: table
    title: "The ECGBench partition"
    headers: ["Split", "Records", "Patients", "SHD prevalence", "Notes"]
    rows:
      - ["train", "72,475", "26,218", "0.524", "up to 146 ECGs for one patient"]
      - ["val", "4,626", "4,626", "0.430", "exactly one ECG per patient"]
      - ["test", "5,442", "5,442", "0.426", "exactly one ECG per patient"]
      - ["**total**", "**82,543**", "**36,286**", "**0.512**", "82,539 after validation"]
      - ["*excluded*", "*17,457*", "*(4,618)*", "*0.568*", "*`no_split` — see below*"]

  - type: description
    title: "Three things that make this dataset unlike the others"
    body: |
      **1. Records are rows, not files.** ECGBench's `npy` signal format exists for
      this release. A record's signal path is
      `EchoNext_test_waveforms.npy:417` — the file *and* the row — and reads are
      memory-mapped, so pulling one record out of the 17 GB training array takes a
      few milliseconds rather than loading the array. The release ships no
      per-record paths at all, so `EchoNextSplitter` generates
      `ecgbench_metadata.csv` carrying them.

      **2. The samples are z-scores, and cannot be made millivolts.** The
      publisher standardised the waveforms with a mean and SD they did not
      release, on top of a nonlinear median filter and percentile clip. No
      `signal_unit_scale` recovers physical units. The config therefore declares
      `signal_units: zscore`, and ECGBench refuses to pretend otherwise:
      `ECGDataset(units="uV")` raises `UnitConversionError` rather than
      multiplying dimensionless numbers by 1000, `ds.units` reports `"zscore"`,
      and `amplitude_outlier` — whose thresholds are millivolts — is skipped.

      **3. The 17,457 `no_split` records are excluded, and that is a leakage
      fix.** They are the non-latest ECGs of patients whose latest ECG is in val
      or test, and the publisher excludes them from training. ECGBench's exporter
      maps any fold outside the split mapping to `"train"`, so keeping them would
      have put **2,499 test patients'** and **2,119 val patients'** earlier
      recordings into the training set. With them dropped, the three splits are
      patient-disjoint — verified from the exported fold CSVs, 0 patients shared
      between any pair. The records remain available through
      `ecgbench.labels.echonext`, which covers all 100,000; they are simply not
      part of the partition. Their 4,618 patients are bracketed in the table above
      because every one of them already appears in val or test — excluding these
      records removes no patient from the cohort, only their earlier recordings.

  - type: description
    title: "A 0 label does not always mean negative"
    body: |
      Every flag is 0/1 with no nulls, so `notna()` reports the label columns 100%
      populated. But the echo measurement each flag was thresholded from is
      **absent for a large minority of records, and in every such case the flag
      reads 0** — checked across all seven measurable conditions, no record has a
      positive flag with a missing value. Prevalence therefore shifts sharply once
      you restrict to records that were actually measured:

      | Flag | Unmeasured | Prevalence, all | Prevalence, measured |
      |---|---|---|---|
      | `tr_max_gte_32_flag` | 54,996 | 0.102 | **0.227** |
      | `pasp_gte_45_flag` | 43,424 | 0.190 | **0.336** |
      | `pericardial_effusion_moderate_large_flag` | 11,823 | 0.030 | 0.034 |
      | `lvef_lte_45_flag` | 8,944 | 0.239 | 0.262 |
      | `shd_moderate_or_greater_flag` | — | 0.522 | composite, no single measure |

      A model trained on `tr_max_gte_32_flag` as shipped is learning "measured and
      high" against "low **or never imaged**", and the second class is largely an
      artefact of who received a full study. `load_labels()` emits a
      `<flag>_measured` boolean beside every per-condition flag; mask on it before
      computing a prevalence or a loss. The composite gets no mask, having no
      single measurement behind it.

      The ordinal severity columns also carry a **`presumed none`** level distinct
      from `none` (6,561 records for aortic stenosis) — the report parser inferring
      an absence rather than measuring one. ECGBench keeps them separate.

  - type: description
    title: "About those counts, and a defect in the release's README"
    body: |
      All 11 shipped files — including the 17 GB training array — were verified
      against the release's own `SHA256SUMS.txt` before any figure here was
      computed, so what follows is upstream rather than download damage.

      **The tabular feature arrays are documented in the wrong column order.** The
      README lists `EchoNext_<split>_tabular_features.npy` as
      `sex, ventricular_rate, atrial_rate, pr_interval, qrs_duration,
      qt_corrected, age_at_ecg`. It is actually
      `sex, **age_at_ecg**, ventricular_rate, atrial_rate, pr_interval,
      qrs_duration, qt_corrected` — age is column **1**, not column 6, and
      everything between shifts down one. Recovered by rank-correlating each array
      column against the metadata: the corrected order gives Spearman **1.000** on
      all 100,000 rows of all four splits, while the documented order gives
      0.05–0.32. Anyone following the README trains with age labelled as
      ventricular rate. `ecgbench.labels.echonext.TABULAR_FEATURE_COLUMNS` carries
      the true order.

      Two smaller discrepancies. The README calls the metadata file
      `EchoNext_metadata_100k.csv`; the shipped and checksummed file is lowercase
      `echonext_metadata_100k.csv`, and the capitalised name does not exist. And
      it documents `sex` as "0 = female, 1 = male", which is true of the array but
      not of the CSV, where the column holds the strings `male`/`female`.

      **Validation.** 4 records of 82,543 fail, all in train, all `flat_line` — a
      lead of near-zero variance. That is the only check that fires;
      `amplitude_outlier` is skipped because the samples are not millivolts.

      **Lead order is not stated anywhere in the release** — there are no headers,
      just a 12-wide axis. Inferred from the signals themselves: Einthoven's
      `III = II − I` and the three Goldberger relations hold to a residual SD of
      0.13–0.19 against a signal SD of 0.92, where deliberately wrong pairings give
      1.06–1.56. Not exact, because the per-lead median filter and clip are
      nonlinear, but the standard order is unambiguous.

  - type: description
    title: "Splits are generated, not downloaded"
    body: |
      EchoNext is released under the **PhysioNet Restricted Health Data License
      1.5.0**, whose clause 3 reads "The LICENSEE will not share access to
      PhysioNet restricted data with anyone else." Fold CSVs carry identifiers
      only, but those identifiers are still data derived under that agreement and
      the ECGBench Hub repository is public and ungated. So `publish_fold_csvs` is
      `false`: `ecgbench upload` refuses the dataset before any network call, and
      `ECGDataset` raises `SplitsNotPublishedError` quoting the regeneration
      command rather than a bare 404.

      The partition is distributed as a **recipe** instead. It is reproducible
      because it is the publisher's own assignment, read from the `split` column —
      no seed is involved.

      ```bash
      # 1. Generate — about 3.5 minutes, dominated by validating 82,543 records
      ecgbench splits --dataset echonext --data-path /path/to/echonext/1.1.0/

      # 2. Verify it is the canonical partition, not merely a plausible one
      python -c "from ecgbench import verify_splits; \
                 print(verify_splits('echonext', 'output/echonext')['ok'])"

      # 3. Copy the fold tree next to the arrays so metadata_source='local' finds it
      cp -r output/echonext/clean output/echonext/original /path/to/echonext/1.1.0/
      ```

      Step 2 compares against `ecgbench/data/manifests/echonext.json`, which ships
      with the package: seed, record counts, input checksums and a `fold_digest`
      over the whole record-to-fold mapping. The metadata checksum in it is the
      release's own published value, so a filtered copy is named as the cause
      rather than producing a silently different split.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset
      from ecgbench.labels.echonext import load_labels

      # metadata_source="local" is required: the splits are not on the Hub.
      ds = ECGDataset("echonext", split="test", version="clean",
                      data_path="/path/to/echonext/1.1.0/",
                      metadata_source="local", labels=True)

      len(ds)                              # 5442
      ds[0]["signal"].shape                # torch.Size([12, 2500])  -- 10 s @ 250 Hz
      ds.units                             # 'zscore'  -- NOT millivolts
      ds[0]["signal"].min(), ds[0]["signal"].max()   # (-6.829, 5.922)

      ds[0]["labels"]["shd_moderate_or_greater_flag"]      # 1
      ds[0]["labels"]["tr_max_gte_32_flag_measured"]       # True

      # Refuses rather than silently scaling dimensionless numbers by 1000:
      ECGDataset("echonext", split="test", units="uV", ...)
      # UnitConversionError: This dataset's samples are stored as 'zscore', not a
      # physical unit, so they cannot be converted to 'uV'. ...

      # Lead order was inferred from Einthoven's law, so leads= works by name.
      ECGDataset("echonext", split="test", leads=["II", "V5"], ...)[0]["signal"].shape
      # torch.Size([2, 2500])

      # Mask before computing a prevalence: an unmeasured echo reads as a 0 flag.
      labels = load_labels("/path/to/echonext/1.1.0/")
      labels["tr_max_gte_32_flag"].mean()                            # 0.102
      m = labels[labels["tr_max_gte_32_flag_measured"]]
      m["tr_max_gte_32_flag"].mean()                                 # 0.227

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page (credentialed)", url: "https://physionet.org/content/echonext/1.1.0/" }
      - { label: "Paper (Nature, 2025)", url: "https://doi.org/10.1038/s41586-025-09227-0" }
      - { label: "Example script", url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_echonext.py" }
---
