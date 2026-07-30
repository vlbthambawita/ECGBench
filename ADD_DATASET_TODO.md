# Adding a Dataset to ECGBench — TODO Checklist

Reusable plan for adding a new dataset end-to-end (config → splits → upload).
Copy this file (or duplicate the relevant sections) when starting on a new
dataset, and tick items off as you go. The phases are roughly sequential, but
phase 1 (catalogue) is independent and can happen at any time.

**There are two slug namespaces — do not use one where the other belongs.**

| | Form | Example | Must match |
|---|---|---|---|
| `<config-slug>` | lowercase, **underscores** | `ptbxl`, `chapman_shaoxing`, `mimic_iv_ecg` | the YAML filename, the `slug:` field inside it, and the `@register("...")` argument |
| `<catalogue-slug>` | lowercase, **dashes** | `ptb-xl`, `chapman-shaoxing`, `mimic-iv-ecg` | the `docs/_datasets/<catalogue-slug>.md` filename and the `slug:` field in its front matter |

Phases 2–6 use `<config-slug>`. Phase 1 (catalogue) uses `<catalogue-slug>`.
The two are unrelated strings — nothing maps one to the other mechanically, so
pick both up front and keep them straight.

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
- [ ] Write down the **headline figures the paper/landing page claims** — record count, patient count, per-class breakdown — *before* you look at the data. These are what you will check the shipped files against in Phase 1.
- [ ] Check for a **changelog** in the dataset root (`*changelog*`, `CHANGES`, release notes on the landing page). It is the authoritative explanation when the shipped version disagrees with the paper.
- [ ] Note any quirks for the PR description (credentialed access, weird encodings, missing leads in some records, etc.).

## Phase 1 — Catalogue entry (optional but recommended)

The catalogue is **one Markdown file per dataset** — YAML front matter holds the
row fields. There is no CSV. The same file is consumed twice: by
`catalogue.py` (`_entry_from_meta`) for the Python discovery API, and by Jekyll
as the `datasets` collection powering the website. A field only one side knows
about is a field that silently does nothing on the other.

- [ ] Create `docs/_datasets/<catalogue-slug>.md`, copying the front matter of a comparable existing entry (e.g. `ptb-xl.md`).
- [ ] Set the identity fields: `slug` (must equal the filename), `name`, `source_url`, `url_label`, `format`, `patients`, `records`, `access`, `license`, `origin_institution`, `origin_country`, `leads`, `paper_title`, `paper_doi`, `search_keywords`.
- [ ] Set `category` to one of the six values fixed by `_CATEGORY_ORDER` in `catalogue.py` — `12-lead-physionet`, `12-lead-other`, `two-lead`, `one-lead`, `three-lead`, `bspm`. It must also match an `id:` in `docs/_data/tables.yml`, or the row renders in no table.
- [ ] Set `status` to one of the keys of `docs/_data/statuses.yml` — `not_started`, `implementing`, `completed`, `needs_review`.
- [ ] Set `order` (int) — controls sort position within the category.
- [ ] Add a `sections:` list for the detail page. Each entry's `type` must have a matching partial in `docs/_includes/sections/`: `description`, `table`, `code`, `links`, `notebook`, `plot`.
- [ ] Verify it shows up: `python -c "import ecgbench; print(ecgbench.get_dataset('<Name>'))"`.
- [ ] Verify the count went up: `python -c "import ecgbench; print(len(ecgbench.list_datasets()))"`.

A catalogue entry is *not* required to run splits/validation — the YAML config drives the pipeline — but datasets without one are invisible to discovery APIs and to the website.

### Every published figure must be recomputed, and disagreements written down

**Do not copy record counts, patient counts or class breakdowns from the paper.**
Papers describe the version the authors had; PhysioNet reissues datasets. Recompute
every number from the files you actually downloaded, put the recomputed value in
the entry, and — when it differs from the published one — add a note giving both
figures and the reason.

- [ ] Recompute `records` and `patients` from the metadata (`len(df)`, `df[patient_id].nunique()`) and compare with the Phase 0 figures.
- [ ] Recompute any **per-class breakdown** you put in a `table` section, and state the derivation in the entry (which column, which mapping file, which filter) so a reader can reproduce it.
- [ ] If any figure differs, add a `description` section — conventionally titled "About those counts" — carrying **all** of:
  - the recomputed value and the published value, side by side (a `Diff` column in the table works well);
  - the version each belongs to;
  - the cause, cited to the changelog if there is one (e.g. "v1.0.3 dropped 38 duplicate records, see `ptbxl_v103_changelog.txt`");
  - the exact derivation used to recompute.
- [ ] State whether class counts are **multi-label**, and if so that they do not sum to the record total — give both sums.
- [ ] Say how many records fall into **no class** at all, and what the splitter labels them.
- [ ] If the splitter's stratification label is a *different quantity* from the breakdown table (single-label vs multi-label, or a different mapping source), say so explicitly and point readers at the metadata join for training targets. A table that looks like ground truth but isn't is worse than no table.

`docs/_datasets/ptb-xl.md` is the worked example: the paper's v1.0.1 counts, the
recomputed v1.0.3 counts, the 38 dropped duplicates that explain the gap, and the
note that `PTBXLSplitter`'s hardcoded code map has drifted from the shipped
`scp_statements.csv`.

## Phase 2 — Config YAML

- [ ] `cp ecgbench/data/configs/_template.yaml ecgbench/data/configs/<config-slug>.yaml`
- [ ] Fill in **Identity** block (`name`, `slug`, `version`, `url`, `download_url`, `license`, `description`, `citation`, `doi`, `creators`). `download_url` should be a direct zip/tar.gz URL or `null` if the source needs credentialed access.
- [ ] Fill in **Signal Properties**.
- [ ] Fill in **File Structure**: `metadata_csv`, separator, `record_id_column`, `patient_id_column`, `signal_path_columns` (rate → column).
- [ ] Fill in **Labels** (`label_column`, `label_format`).
- [ ] Fill in **stratification** block (and provide `mapping_source` + `superclass_column` if using `superclass_mapping`).
- [ ] Fill in **predefined_splits** if applicable — **and set `has_predefined_splits: true`**. `engine.py` gates on `config.has_predefined_splits and config.predefined_splits`, so a fully-filled `predefined_splits` block with the flag left at `false` is silently ignored and folds get generated instead.
- [ ] Fill in **validation**: `expected_leads`, `expected_samples` (one key per declared sampling rate), the `checks` list, and `amplitude_range_mv`.
- [ ] Fill in **croissant** block (`keywords`, `rai_data_collection`, `rai_data_biases`, `rai_personal_sensitive_info`).
- [ ] Smoke-test the config loads: `python -c "from ecgbench import load_config; print(load_config('<config-slug>'))"`.

## Phase 3 — Splitter strategy

Decide which path applies and do **one**:

- [ ] **Generic path (default).** No code needed — `GenericSplitter` is the fallback. Use this if the metadata CSV can be read as-is and `label_column` works directly for stratification.
- [ ] **Custom path.** Required if any of the following are true:
  - signal paths need transformation (prefix, suffix, joined columns) → see `chapman.py`
  - labels need decoding (dict-strings, superclass mapping) → see `ptbxl.py`
  - multiple metadata files need to be joined
  - records need filtering before splitting

  If custom:
  - [ ] Create `ecgbench/splitting/strategies/<config-slug>.py`
  - [ ] Subclass `DatasetSplitter`, decorate the class with `@register("<config-slug>")` — the slug here **must match the config slug**, since that's how the registry looks it up.
  - [ ] Implement the two abstract methods, `load_metadata()` and `get_stratification_labels()`. Override other hooks only if necessary. `get_splitter()` instantiates with no arguments, so keep `__init__` argument-free.
  - [ ] Import the module in `ecgbench/splitting/strategies/__init__.py` so the `@register` side-effect runs.
  - [ ] Verify: `python -c "from ecgbench.splitting import get_splitter; print(type(get_splitter('<config-slug>')))"` — should NOT print `GenericSplitter`.

## Phase 4 — Run the pipeline

- [ ] Dry-run on local data (auto-download if `download_url` is set; otherwise pass `--data-path`):
  ```bash
  ecgbench splits --dataset <config-slug> --data-path /path/to/<config-slug>/
  ```
- [ ] Inspect `output/<config-slug>/` — the tree `export.py` and `cli/splits.py` actually produce:
  ```
  output/<config-slug>/
    validation_report.json            # only top-level artefact
    {original,clean}/
      folds.csv                       # every record + fold + default_split
      croissant.json                  # one PER VERSION, not top-level
      {train,val,test}/fold_<N>.csv   # N is 1-indexed
  ```
- [ ] Confirm `folds.csv` exists in both versions and that fold CSVs sit under `train/`, `val/`, `test/` — not loose in the version directory. With the default folds 1–8 → train, 9 → val, 10 → test, expect 8 CSVs in `train/` and one each in `val/` and `test/`.
- [ ] Confirm the exported columns match the version, per `_minimal_columns()` in `export.py`:
  - `clean/` — record ID, patient ID (if configured), signal paths, `fold`, `default_split`. Nothing else.
  - `original/` — the same **plus `is_valid` and `quality_issues`**. These two are intentional here; do not "fix" the exporter for them.

  Anything beyond that list is real metadata leakage — full metadata stays in the source CSV. Fix the exporter, not the config.
- [ ] Confirm fold membership is identical between `original/` and `clean/` (`clean/` is a row subset, not a re-split), that fold counts roughly match the `n_folds=10` distribution, and that **patients do not span folds** (if `patient_id_column` is set).
- [ ] Spot-check `validation_report.json` for unexpected check failures — high `truncated_signal` counts usually mean `expected_samples` is wrong; high `corrupt_header` counts usually mean `signal_format` or path prefix is wrong.
- [ ] (Optional) Standalone Croissant regeneration — `--splits-dir` points at the **version** directory, and the file lands inside it:
  ```bash
  ecgbench croissant --dataset <config-slug> --splits-dir output/<config-slug>/clean/ --version clean
  ```

## Phase 5 — Tests

- [ ] Add a `test_load_<config-slug>_config()` function to `tests/test_config.py`, alongside the existing `test_load_ptbxl_config` / `test_load_chapman_config`. These are hand-written per dataset, not parametrised — there is no table to extend.
- [ ] If you wrote a custom splitter, add a unit test under `tests/test_splitting.py` using synthetic data from `tests/conftest.py` patterns. Cover at minimum: `load_metadata` shape, label distribution, patient grouping if applicable.
- [ ] Run the full suite: `pytest`. Note that `conftest.py` builds `DatasetConfig` objects **in Python, not from the shipped YAML**, so a green suite does not prove your new YAML parses — the Phase 2 `load_config()` smoke-test is what covers that.
- [ ] Confirm optional-extra tests actually ran rather than skipping (`torch` for `test_dataset.py`, `mlcroissant` for `test_croissant.py`/`test_cli.py`): install `.[dev]` and check `pytest -rs` output for unexpected skips.
- [ ] Run lint/format: `ruff check ecgbench/ && black ecgbench/`. There is no CI test/lint job — local is the only gate.

## Phase 6 — HuggingFace Hub upload (optional)

- [ ] Ensure `HF_TOKEN` is set (env var or `.env`).
- [ ] Dry-run first to see the file list without pushing: add `--dry-run`.
- [ ] Upload:
  ```bash
  ecgbench upload --data-dir output/ --datasets <config-slug>
  ```
- [ ] Verify on the Hub that the tree is prefixed by the dataset slug — `<config-slug>/<version>/folds.csv` and `<config-slug>/<version>/<split>/fold_<N>.csv`. This prefix is what `ECGDataset(metadata_source="hf")` fetches with `hf_hub_download`; a missing or wrong prefix fails only at load time.
- [ ] Verify an **anonymous** user can load it. `ECGDataset` defaults to `metadata_source="hf"`, so this exercises the real download path — but only if no token is picked up. Unset the token and use a scratch cache so a warm cache can't mask a missing upload:
  ```bash
  env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN HF_HOME="$(mktemp -d)" python - <<'PY'
  from ecgbench import ECGDataset
  for version in ("clean", "original"):        # default is "clean" — check both
      ds = ECGDataset("<config-slug>", split="train", version=version,
                      data_path="/path/to/<config-slug>/")
      print(version, len(ds), ds[0]["signal"].shape)
  PY
  ```
  `data_path` points at the local **signal** files; the fold CSVs come from the Hub.

## Phase 7 — Wrap up

- [ ] Cross-check the catalogue entry's `records`/`patients` against the pipeline's own output (`original.total` in the `ecgbench splits` summary). If they disagree, one of them is wrong — usually the config filters or drops rows you did not expect.
- [ ] Update the `README.md` "Dataset Catalogue" section if applicable.
- [ ] Flip `status:` to `completed` in `docs/_datasets/<catalogue-slug>.md` (created back in Phase 1 — don't create a second entry here).
- [ ] Commit: config YAML, optional splitter + its `strategies/__init__.py` import, catalogue Markdown entry, tests, any docs. Keep generated output (`output/`) out of the commit.
- [ ] Remember `docs/**` changes trigger the Pages and HF Space deploys on `main`, while Python-only changes deploy nothing until a `v*` tag is pushed.
- [ ] Open a PR with: source URL, license, record count, validation pass rate, and whether a custom splitter was needed and why.

---

## Common gotchas

- **Slug mismatch.** Config filename, `slug:` field inside the YAML, and `@register("...")` argument must all be identical. A mismatch silently falls back to `GenericSplitter` (or fails to find the config).
- **Two slug namespaces.** The catalogue slug is dashed (`ptb-xl`), the config slug underscored (`ptbxl`). Naming the Markdown file after the config slug breaks the catalogue and the website; naming the YAML after the catalogue slug breaks `load_config()`.
- **`has_predefined_splits` is a separate gate.** Filling the `predefined_splits` block is not enough — `engine.py` also requires `has_predefined_splits: true`. Left at `false`, your carefully specified splits are silently discarded in favour of generated folds.
- **Catalogue `category` is closed-vocabulary and cross-referenced.** It must be one of `_CATEGORY_ORDER` in `catalogue.py` *and* an `id:` in `docs/_data/tables.yml`. An unrecognised value sorts to the end of the Python listing and renders in no website table. Likewise `status` must be a key of `docs/_data/statuses.yml`.
- **Front matter has two consumers.** `catalogue.py:_entry_from_meta` and the Liquid templates in `docs/_layouts/`, `docs/_includes/`. A new field needs handling on both sides, or it silently does nothing on one of them.
- **`original/` fold CSVs carry `is_valid` + `quality_issues`; `clean/` does not.** That asymmetry is deliberate (`_minimal_columns(include_quality=...)`) — not metadata leakage.
- **Path prefixes.** `signal_path_columns` values must resolve relative to `data_path`. If the source CSV stores bare filenames but signals live in a subdirectory, fix it in the splitter's `load_metadata` — don't ship a config that only works when the user pre-rewrites paths.
- **Predefined splits are 1-indexed.** Fold numbers in `predefined_splits.fold_mapping` follow the same 1..N convention as generated folds.
- **`expected_samples` per rate.** Every rate listed in `sampling_rates` should have a key in `validation.expected_samples`, or `truncated_signal` will fire spuriously.
- **`amplitude_range_mv`.** Units are millivolts. Datasets stored in microvolts or ADC counts will trip `amplitude_outlier` en masse — convert in the splitter or adjust the range deliberately.
- **Published figures rarely match the shipped version.** PhysioNet reissues datasets and papers are not revised. PTB-XL v1.0.3 dropped 38 duplicate/triplicate records relative to the v1.0.1 the paper describes, so every superclass count is 6-17 records smaller. Copying the paper's table produces figures nobody can reproduce. Recompute, then note both values and the reason — see the Phase 1 subsection above.
- **A class breakdown is not the stratification label.** They can differ in cardinality (multi-label counts vs one label per record) *and* in mapping source. `PTBXLSplitter` maps SCP codes with a hardcoded dict rather than the shipped `scp_statements.csv`, and the two have drifted apart. If your dataset page shows a breakdown, say which quantity it is.
- **Heavy deps stay lazy.** Do not add `import wfdb` / `import torch` / `import mlcroissant` at module top-level in any file imported by `ecgbench/__init__.py`'s eager path. Import inside functions instead.
