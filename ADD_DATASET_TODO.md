# Adding a Dataset to ECGBench — TODO Checklist

Reusable plan for adding a new dataset end-to-end (config → splits → upload).
Copy this file (or duplicate the relevant sections) when starting on a new
dataset, and tick items off as you go. The phases are roughly sequential, but
phase 1 (catalogue) is independent and can happen at any time.

**Substitute `<slug>` throughout** (lowercase, underscores, no dashes — e.g.
`ptbxl`, `chapman_shaoxing`, `mimic_iv_ecg`). The slug must match the YAML
filename and the registered splitter name.

---

## Phase 0 — Discovery (before writing anything)

- [ ] Locate the dataset's **official source URL**, **license**, **citation**, **DOI**.
- [ ] Confirm signal **format** (`wfdb` / `edf` / `csv` / `mat` / `numpy` / `hdf5`).
- [ ] Confirm **leads**, **duration (s)**, **sampling rate(s)**, **default rate**.
- [ ] Download a small subset locally and inspect the **metadata CSV**:
  - record ID column name
  - patient ID column (or confirm one-record-per-patient, set `null`)
  - signal-path column(s) per sampling rate — note any **prefix** that must be prepended (cf. Chapman's `ECGData/`)
  - label column name and **format** (`single` / `comma_separated` / `dict_string` / `json`)
- [ ] Decide stratification: `direct`, `superclass_mapping` (needs a mapping CSV), or `custom_function` (needs a custom splitter — see Phase 3).
- [ ] Decide if **predefined splits** exist (e.g. PTB-XL `strat_fold`). If yes, note the column and which fold values go to train/val/test.
- [ ] Sanity-check **expected samples = duration_s × sampling_rate** per rate.
- [ ] Note any quirks for the PR description (credentialed access, weird encodings, missing leads in some records, etc.).

## Phase 1 — Catalogue entry (optional but recommended)

- [ ] Add a row to `ecgbench/data/ecg_datasets.csv` so `ecgbench.search()` / `list_datasets()` surface the dataset. Keep column order consistent with existing rows.
- [ ] Verify it shows up: `python -c "import ecgbench; print(ecgbench.get_dataset('<Name>'))"`.

A catalogue entry is *not* required to run splits/validation — the YAML config drives the pipeline — but datasets without one are invisible to discovery APIs.

## Phase 2 — Config YAML

- [ ] `cp ecgbench/data/configs/_template.yaml ecgbench/data/configs/<slug>.yaml`
- [ ] Fill in **Identity** block (`name`, `slug`, `version`, `url`, `download_url`, `license`, `description`, `citation`, `doi`, `creators`). `download_url` should be a direct zip/tar.gz URL or `null` if the source needs credentialed access.
- [ ] Fill in **Signal Properties**.
- [ ] Fill in **File Structure**: `metadata_csv`, separator, `record_id_column`, `patient_id_column`, `signal_path_columns` (rate → column).
- [ ] Fill in **Labels** (`label_column`, `label_format`).
- [ ] Fill in **stratification** block (and provide `mapping_source` + `superclass_column` if using `superclass_mapping`).
- [ ] Fill in **predefined_splits** if applicable.
- [ ] Fill in **validation.expected_samples** for each declared sampling rate.
- [ ] Fill in **croissant** block (`keywords`, `rai_data_collection`, `rai_data_biases`, `rai_personal_sensitive_info`).
- [ ] Smoke-test the config loads: `python -c "from ecgbench import load_config; print(load_config('<slug>'))"`.

## Phase 3 — Splitter strategy

Decide which path applies and do **one**:

- [ ] **Generic path (default).** No code needed — `GenericSplitter` is the fallback. Use this if the metadata CSV can be read as-is and `label_column` works directly for stratification.
- [ ] **Custom path.** Required if any of the following are true:
  - signal paths need transformation (prefix, suffix, joined columns) → see `chapman.py`
  - labels need decoding (dict-strings, superclass mapping) → see `ptbxl.py`
  - multiple metadata files need to be joined
  - records need filtering before splitting

  If custom:
  - [ ] Create `ecgbench/splitting/strategies/<slug>.py`
  - [ ] Subclass `DatasetSplitter`, decorate the class with `@register("<slug>")` — the slug here **must match the config slug**, since that's how the registry looks it up.
  - [ ] Implement `load_metadata()` and `get_stratification_labels()`. Override other hooks only if necessary.
  - [ ] Import the module in `ecgbench/splitting/strategies/__init__.py` so the `@register` side-effect runs.
  - [ ] Verify: `python -c "from ecgbench.splitting import get_splitter; print(type(get_splitter('<slug>')))"` — should NOT print `GenericSplitter`.

## Phase 4 — Run the pipeline

- [ ] Dry-run on local data (auto-download if `download_url` is set; otherwise pass `--data-path`):
  ```bash
  ecgbench splits --dataset <slug> --data-path /path/to/<slug>/
  ```
- [ ] Inspect `output/<slug>/`:
  - `original/` — fold CSVs covering all records, with `is_valid` and `quality_issues` columns.
  - `clean/` — fold CSVs with valid records only.
  - `validation_report.json` — totals and per-check failure counts.
  - `croissant.json` — Croissant 1.1 metadata.
- [ ] Confirm fold CSVs contain **only** the minimal columns (record ID, patient ID, signal paths, fold, split). Full metadata stays in the source CSV — if you see extra columns leaking through, fix the exporter, not the config.
- [ ] Confirm fold counts roughly match `n_folds=10` distribution and that **patients do not span folds** (if `patient_id_column` is set).
- [ ] Spot-check `validation_report.json` for unexpected check failures — high `truncated_signal` counts usually mean `expected_samples` is wrong; high `corrupt_header` counts usually mean `signal_format` or path prefix is wrong.
- [ ] (Optional) Standalone Croissant regeneration:
  ```bash
  ecgbench croissant --dataset <slug> --splits-dir output/<slug>/clean/ --version clean
  ```

## Phase 5 — Tests

- [ ] Add a config-loading test (or extend an existing parametrised one) in `tests/test_config.py` covering `<slug>`.
- [ ] If you wrote a custom splitter, add a unit test under `tests/test_splitting.py` using synthetic data from `tests/conftest.py` patterns. Cover at minimum: `load_metadata` shape, label distribution, patient grouping if applicable.
- [ ] Run the full suite: `pytest`.
- [ ] Run lint/format: `ruff check ecgbench/ && black ecgbench/`.

## Phase 6 — HuggingFace Hub upload (optional)

- [ ] Ensure `HF_TOKEN` is set (env var or `.env`).
- [ ] Upload:
  ```bash
  ecgbench upload --data-dir output/ --datasets <slug>
  ```
- [ ] Verify a logged-out user can load it:
  ```python
  from ecgbench import ECGDataset
  ds = ECGDataset("<slug>", split="train", data_path="/path/to/<slug>/")
  print(ds[0]["signal"].shape)
  ```

## Phase 7 — Wrap up

- [ ] Update `README.md` "Dataset Catalogue" section if applicable.
- [ ] Add an entry to `docs/_datasets/` if datasets are documented individually there.
- [ ] Commit: config, optional splitter, catalogue row, tests, any docs. Keep generated output (`output/`) out of the commit.
- [ ] Open a PR with: source URL, license, record count, validation pass rate, and whether a custom splitter was needed and why.

---

## Common gotchas

- **Slug mismatch.** Config filename, `slug:` field inside the YAML, and `@register("...")` argument must all be identical. A mismatch silently falls back to `GenericSplitter` (or fails to find the config).
- **Path prefixes.** `signal_path_columns` values must resolve relative to `data_path`. If the source CSV stores bare filenames but signals live in a subdirectory, fix it in the splitter's `load_metadata` — don't ship a config that only works when the user pre-rewrites paths.
- **Predefined splits are 1-indexed.** Fold numbers in `predefined_splits.fold_mapping` follow the same 1..N convention as generated folds.
- **`expected_samples` per rate.** Every rate listed in `sampling_rates` should have a key in `validation.expected_samples`, or `truncated_signal` will fire spuriously.
- **`amplitude_range_mv`.** Units are millivolts. Datasets stored in microvolts or ADC counts will trip `amplitude_outlier` en masse — convert in the splitter or adjust the range deliberately.
- **Heavy deps stay lazy.** Do not add `import wfdb` / `import torch` / `import mlcroissant` at module top-level in any file imported by `ecgbench/__init__.py`'s eager path. Import inside functions instead.
