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
- [ ] Confirm signal **format**. `wfdb` and `csv` are implemented; anything else raises `NotImplementedError` in `_load_signal` and needs a branch there (in **both** `validation/engine.py` and `dataset.py` — they each have a copy).
- [ ] Confirm the **units of the stored samples** and set `signal_unit_scale` so they reach ECGBench as millivolts (µV → `0.001`). Read one file and check the peak amplitude: a QRS peaking near 1000 is microvolts, near 1.0 is millivolts.
- [ ] Confirm **leads**, **duration (s)**, **sampling rate(s)**, **default rate**.
- [ ] Record the **lead names in file order** into `lead_names:`, spelled as the source spells them. Read a header or CSV column row — never assume the standard order. `ECGDataset(leads=...)` cannot work without it, and two of the four implemented datasets deviate: MIMIC-IV-ECG stores aVF before aVL, PTB-XL spells them AVR/AVL/AVF.
- [ ] Download a small subset locally and inspect the **metadata CSV**:
  - record ID column name
  - patient ID column (or confirm one-record-per-patient, set `null`)
  - signal-path column(s) per sampling rate — note any **prefix** that must be prepended (cf. Chapman's `ECGData/`)
  - label column name and **format** (`single` / `comma_separated` / `dict_string` / `json`)
- [ ] Decide stratification: `direct`, `superclass_mapping` (needs a mapping CSV), or `custom_function` (needs a custom splitter — see Phase 3).
- [ ] Decide if **predefined splits** exist (e.g. PTB-XL `strat_fold`). If yes, note the column and which fold values go to train/val/test.
- [ ] Sanity-check **expected samples = duration_s × sampling_rate** per rate.
- [ ] Record the **minimum record length** and whether length is uniform. Two things depend on it: whether `expected_samples` can be set at all (see Phase 2), and what `window=` a user can safely apply — `ECGDataset(window=(start, length))` raises `WindowOutOfRangeError` on any record shorter than `start + length`, so the example script and the dataset page should quote a window that fits **every** record.
- [ ] Write down the **headline figures the paper/landing page claims** — record count, patient count, per-class breakdown — *before* you look at the data. These are what you will check the shipped files against in Phase 1.
- [ ] **Verify the files against the release's own checksums** (`SHA256SUMS.txt`, `md5sums.txt`) before trusting any figure, especially for the metadata/label CSVs. A local copy may have been filtered, deduplicated or renamed by whoever downloaded it, and a filtered file under the official name is invisible until your counts disagree with the paper. MIMIC-IV-ECG is the worked example: the shipped `machine_measurements.csv` had been replaced by a 789,481-row subset of the real 800,035-row file, with the original renamed `_original.csv`; only the checksum revealed which was authentic. Compute figures from the verified file, and name the **official** filename in the config regardless of what the local copy is called.
- [ ] Check for a **changelog** in the dataset root (`*changelog*`, `CHANGES`, release notes on the landing page). It is the authoritative explanation when the shipped version disagrees with the paper.
- [ ] Check whether the metadata is really a **CSV**. `.xlsx` needs converting — `validate_dataset` re-reads `metadata_csv` from disk with `pandas.read_csv`, so an in-memory conversion is not enough. Convert in the acquisition script, and have the splitter generate a normalised CSV (see `chapman.py`).
- [ ] If the source is **several files** rather than one archive, `download_url` cannot express it — write an acquisition script under `examples/download_<name>.py` that md5-verifies each file (see `download_chapman_figshare.py`).
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
- [ ] Add a `related:` block if the dataset overlaps any other — see below.
- [ ] Verify it shows up: `python -c "import ecgbench; print(ecgbench.get_dataset('<Name>'))"`.
- [ ] Verify the count went up: `python -c "import ecgbench; print(len(ecgbench.list_datasets()))"`.

A catalogue entry is *not* required to run splits/validation — the YAML config drives the pipeline — but datasets without one are invisible to discovery APIs and to the website.

### Declare overlaps with other datasets

Datasets in this catalogue are not independent: challenge sets bundle other
datasets, demo subsets are carved out of full releases, derived releases re-label
the same recordings. A user who trains on one and evaluates on another has a
contaminated test set and no warning. The `related:` block is how that gets said.

```yaml
related:
  - slug: "other-catalogue-slug"   # must be an existing docs/_datasets/<slug>.md
    relation: "contains"           # contains | subset_of | derived_from |
                                   # has_derivative | same_cohort | sibling_release
    shares_records: true           # do the two hold any of the SAME recordings?
    verified: true                 # was the overlap checked against the data files?
    note: >
      What overlaps, how much, and what a user must not do because of it.
```

- [ ] **Declare each relationship once, on one side only.** `catalogue.py` derives the inverse (`contains` ↔ `subset_of`, `derived_from` ↔ `has_derivative`, `same_cohort` and `sibling_release` are symmetric) and the website recomputes it in Liquid. Declaring both directions double-counts on the site and `pytest` fails.
- [ ] Set `shares_records` honestly. It is the field that flags leakage; a shared *cohort* with different recordings is `false`.
- [ ] Set `verified: true` **only** if you checked the overlap against the actual data files. Documentation and papers are `false`. Say which in the note.
- [ ] Give every `shares_records: true` edge a note saying what a user must not do — a warning with no explanation is not actionable, and a test enforces the note.
- [ ] Quantify the overlap when you can, and record the join key or the reason there isn't one. The MIMIC demo overlaps the full release in 658 of 659 records but their `study_id`s are disjoint, so a naive key comparison reports 0% — exactly the trap a note prevents.
- [ ] Run `pytest tests/test_catalogue.py` — it checks that every slug resolves, every relation is in the vocabulary, every edge is mirrored, and the Python and website edge counts agree.

Worked examples: `mimic-iv-ecg.md` (verified from the files, with the study_id
caveat), `physionet-cinc-challenge-2021.md` (unverified, taken from the challenge
description), `chapman-shaoxing-arrhythmia.md` (partial overlap, quantified).

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
recomputed v1.0.3 counts, and the 38 dropped duplicates that explain the gap.

## Phase 2 — Config YAML

- [ ] `cp ecgbench/data/configs/_template.yaml ecgbench/data/configs/<config-slug>.yaml`
- [ ] Fill in **Identity** block (`name`, `slug`, `version`, `url`, `download_url`, `license`, `description`, `citation`, `doi`, `creators`). `download_url` should be a direct zip/tar.gz URL or `null` if the source needs credentialed access.
- [ ] Fill in **Signal Properties** — including the three fields that are easy to skip and silently wrong:
  - `signal_format` (`wfdb` or `csv`);
  - `signal_unit_scale`, so samples reach ECGBench as millivolts (`0.001` for microvolts). Everything downstream — `amplitude_range_mv`, `units=`, the tensors users train on — assumes mV;
  - `lead_names`, in the order the **files** store them and spelled as the source spells them. `ECGDataset(leads=...)` is unusable without it.
- [ ] Fill in **File Structure**: `metadata_csv`, separator, `record_id_column`, `patient_id_column`, `signal_path_columns` (rate → column).
- [ ] Fill in **Labels** (`label_column`, `label_format`).
- [ ] Fill in **stratification** block (and provide `mapping_source` + `superclass_column` if using `superclass_mapping`).
- [ ] Fill in **predefined_splits** if applicable — **and set `has_predefined_splits: true`**. `engine.py` gates on `config.has_predefined_splits and config.predefined_splits`, so a fully-filled `predefined_splits` block with the flag left at `false` is silently ignored and folds get generated instead.
- [ ] Fill in **validation**: `expected_leads`, `expected_samples` (one key per declared sampling rate), the `checks` list, and `amplitude_range_mv`. If record length genuinely varies, leave `expected_samples` **empty** rather than guessing a value — see the gotcha below — and say so in a comment, as `ptbdb.yaml` and `challenge2021.yaml` do.
- [ ] Fill in **croissant** block (`keywords`, `rai_data_collection`, `rai_data_biases`, `rai_personal_sensitive_info`).
- [ ] Smoke-test the config loads: `python -c "from ecgbench import load_config; print(load_config('<config-slug>'))"`.

## Phase 2b — Labels

Users get labels through `ECGDataset(labels=True)` or `load_labels(slug, data_path)`.
Both dispatch on the `labels:` block you fill in here, so **a dataset without this
block returns no ground truth** — the fold CSVs never carry labels.

- [ ] Find where the labels actually live. It is often *not* `metadata_csv`: PTB-XL needs `scp_statements.csv` as well, and `ecg_arrhythmia`'s labels only exist in the metadata CSV the splitter generates from WFDB headers.
- [ ] Fill in the `labels:` block — `source_csv`, `join_column` (the column holding record IDs, which may be named differently from `record_id_column`), and `columns` (or `null` for everything but the join key).
- [ ] **If the dataset genuinely has no labels, say so** with `available: false` and an `unavailable_reason` that points at where labels *could* come from (the full release, another module, a linked dataset). `labels=True` then raises `LabelsUnavailableError` quoting that reason. Silently returning empty columns is not acceptable.
- [ ] Decide declarative vs module:
  - **Declarative (default).** A column select plus a join. No Python.
  - **Module** — `ecgbench/labels/<config-slug>.py` exposing `load_labels(data_path, config) -> DataFrame` indexed by the record ID — when labels need decoding, a taxonomy join, several source files, or a derived column. Register it in `_custom_loaders()` in `ecgbench/labels/__init__.py`.
- [ ] Expose the **full** label hierarchy, not just what stratification needs — raw codes, superclasses *and* subclasses, report/note text, demographics. Users select from the dict; they cannot recover what you dropped.
- [ ] **Make the label loader the single source of truth.** If the splitter needs a stratification label, it must derive it from this loader (attach the column in `load_metadata`), never re-implement the mapping. PTB-XL previously had two copies that drifted apart — that is the bug this rule prevents.
- [ ] Say in the docstring whether labels are **multi-label**, and how many records carry none. If you provide a single-label reduction for stratification, name it clearly (`primary_*`), document how ties break, and tell users not to train on it.
- [ ] Smoke-test: `python -c "from ecgbench import load_labels; df = load_labels('<config-slug>', '<path>'); print(df.shape); print(df.head())"`.
- [ ] Check the record IDs join. `load_labels` raises on duplicate IDs, and `ECGDataset` raises when *nothing* matches, but a partial match only logs a warning — confirm the matched count is what you expect.
- [ ] Add an example script (Phase 5b) and a test in `tests/test_labels.py`.

**Labels are never uploaded to the Hub.** Only fold CSVs are. That is partly
practical (the Hub tree is identifiers) and partly licensing: redistributing
labels is fine for CC-BY sources and *not* for credentialed ones such as
MIMIC-IV. So `labels=True` requires a local copy of the source dataset, and the
missing-file error must name the file and say where to get it.

## Phase 3 — Splitter strategy

Decide which path applies and do **one**:

- [ ] **Generic path (default).** No code needed — `GenericSplitter` is the fallback. Use this if the metadata CSV can be read as-is and `label_column` works directly for stratification.
- [ ] **Custom path.** Required if any of the following are true:
  - signal paths need transformation (prefix, suffix, joined columns) → see `chapman.py`
  - multiple metadata files need to be joined
  - records need filtering before splitting
  - a stratification label must be derived — but derive it by calling the Phase 2b label loader and attaching the column in `load_metadata`, as `ptbxl.py` does. Do not re-implement a mapping the label loader already owns.

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

- [ ] Add a `test_load_<config-slug>_config()` function to `tests/test_config.py`, alongside the existing `test_load_ptbxl_config` / `test_load_chapman_config`. These are hand-written per dataset, not parametrised — there is no table to extend. Assert `signal_format`, `signal_unit_scale` and `lead_names` explicitly: those three are silent when wrong.
- [ ] Add the dataset to `TestShippedLeadNames` in `tests/test_dataset.py`, which asserts the declared lead order against what the files hold.
- [ ] If you wrote a custom splitter, add a unit test under `tests/test_splitting.py` using synthetic data from `tests/conftest.py` patterns. Cover at minimum: `load_metadata` shape, label distribution, patient grouping if applicable.
- [ ] Run the full suite: `pytest`. Note that `conftest.py` builds `DatasetConfig` objects **in Python, not from the shipped YAML**, so a green suite does not prove your new YAML parses — the Phase 2 `load_config()` smoke-test is what covers that.
- [ ] Confirm optional-extra tests actually ran rather than skipping (`torch` for `test_dataset.py`, `mlcroissant` for `test_croissant.py`/`test_cli.py`): install `.[dev]` and check `pytest -rs` output for unexpected skips.
- [ ] Add label tests to `tests/test_labels.py`: the declarative block parses, the source columns resolve, and — for datasets with no labels — that `available: false` carries a reason. Fixtures build tiny source CSVs in `tmp_path`; no real data.
- [ ] Run lint/format: `ruff check ecgbench/ && black ecgbench/`. There is no CI test/lint job — local is the only gate.

### Phase 5b — Example script

Every dataset with a config gets `examples/load_<config-slug>.py`. These are the
only end-to-end runnable documentation ECGBench has, and they are what catches
API drift — a stale example is how the broken PTB-XL snippet survived.

- [ ] Copy the closest existing example (`load_ptbxl.py` for rich labels, `load_ecg_arrhythmia.py` for multi-label codes, `load_mimic_iv_ecg_demo.py` for a dataset with none).
- [ ] Show: config summary, `len(dataset)`, one sample's keys and label fields, the label distribution over the split, one batch through `DataLoader` + `ecg_collate_fn`, and how to turn labels into a target tensor.
- [ ] Surface the dataset's own gotchas in the output — non-standard lead order, raw codes with no acronym, labels requiring a prior pipeline run.
- [ ] **Run it and paste nothing you did not see.** Numbers in comments must come from a real run.
- [ ] Use `dataset.labels_df` for split-level statistics rather than iterating the Dataset — iterating decodes every signal.
- [ ] Print the dataset's `lead_names`, and if the order is non-standard, demonstrate `leads=[...]` selecting by name so a reader sees the fix rather than just the warning (see `load_mimic_iv_ecg_demo.py`).
- [ ] If the source is not in millivolts, show `units="uV"` returning the original scale, so the `signal_unit_scale` conversion is visible rather than implicit (see `load_chapman_shaoxing.py`).
- [ ] If records are **long or variable-length**, batch them with `window=(start, length)` rather than a cropping `transform`. It is read at load time, so only those samples are decoded (~106 ms to ~8 ms per record on `incartdb`), and it survives `DataLoader(num_workers>0)` under the `spawn` start method, where a lambda `transform` raises `PicklingError`. See `load_incartdb.py`, `load_ptbdb.py`, `load_challenge2021.py`.

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
- [ ] Update `README.md` if the dataset adds a capability the docs do not yet mention — a new `signal_format`, a units quirk, an unusual label shape. The parameter table and the "Leads and units" table both list per-dataset facts that go stale.
- [ ] Add the dataset's lead order to the table in the README's **Leads and units** section.
- [ ] Update the "Loading with ECGBench" `code` section in the catalogue entry to show `labels=True`, and `leads=`/`units=`/`window=` where the dataset has a quirk worth demonstrating (`window=` whenever records are long or variable-length). **Run the snippet before pasting it** — a stale example on a dataset page is how the broken PTB-XL one survived.
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
- **`expected_samples` per rate — omitting a rate disables the check, it does not make it fire.** `check_truncated_signal` returns `[]` when `expected_samples.get(rate)` is None, so a missing key silently skips validation for that rate. That is the correct escape hatch for genuinely variable-length datasets (PTBDB records differ in length), but for fixed-length data a forgotten key means the check never runs and nobody notices.
- **`expected_leads` is declared but never checked.** Every config sets it, `_config_to_dict` ships it to the subprocess, and no function in `CHECK_REGISTRY` reads it. Do not rely on it to catch a record with the wrong number of signals — nothing does today.
- **`amplitude_range_mv`.** Units are millivolts. Datasets stored in microvolts or ADC counts trip `amplitude_outlier` on every record — set `signal_unit_scale` (0.001 for µV) rather than widening the range. The figshare Chapman release is the worked example: raw values run to ±2750, which is ±2.75 mV.
- **A path fix-up that lives only in the splitter is a bug, not a fix.** `validate_dataset` re-reads `metadata_csv` from disk and rebuilds paths from the raw column, so it never sees what `load_metadata` changed in memory. Chapman shipped this way for months and every record failed `corrupt_header`. Write the normalised frame to disk as the config's `metadata_csv`, the way `chapman.py` and `ecg_arrhythmia.py` now do.
- **Published figures rarely match the shipped version.** PhysioNet reissues datasets and papers are not revised. PTB-XL v1.0.3 dropped 38 duplicate/triplicate records relative to the v1.0.1 the paper describes, so every superclass count is 6-17 records smaller. Copying the paper's table produces figures nobody can reproduce. Recompute, then note both values and the reason — see the Phase 1 subsection above.
- **A class breakdown is not the stratification label.** They differ in cardinality: the breakdown counts every class a record carries, the stratification label is one class per record. Say which quantity a table shows. And derive both from the same loader — PTB-XL once had a hardcoded splitter map that drifted from the shipped `scp_statements.csv` (465 records in OTHER instead of 411) precisely because there were two sources.
- **Labels never reach the batch unless you fill in `labels:`.** Fold CSVs are identification-only by design, so a dataset with a perfect config and no `labels:` block silently returns no ground truth. If the data genuinely has none, declare `available: false` with a reason rather than leaving the block out.
- **A single-label reduction of multi-label data is a trap.** Name it `primary_*`, document how ties break, and say plainly it is for stratification only. In PTB-XL 10.8% of records have tied superclasses, so the "primary" class is partly an artefact of the tie-break rule.
- **`ecgbench_metadata.csv`-style generated sources mean labels depend on pipeline order.** `ecg_arrhythmia` labels only exist after `ecgbench splits` has run once, because that is what scans the WFDB headers. Say so in the example script, and let `LabelSourceMissingError` name the file.
- **Overlapping datasets are the norm, not the exception.** Around a third of the catalogue sits in a family — CinC challenges bundle PTB-XL and CPSC-2018, CODE's subsets come out of CODE-full, MIMIC demo out of MIMIC full. Adding a dataset without checking whether it overlaps an existing one ships a silent leakage trap. Check before you write the entry.
- **`shares_records` cannot be inferred from IDs alone.** The MIMIC demo and the full release hold the same 659 recordings but renumber `study_id` into a disjoint range and truncate timestamps to the minute, so comparing keys says 0% overlap while the truth is 99.8%. When IDs disagree, try a natural key (subject + timestamp) before concluding anything.
- **Lead order is not a given.** `config.leads` is a count, not an order. `signal[4]` is aVL in PTB-XL, Chapman and ecg_arrhythmia, but aVF in MIMIC-IV-ECG. Fill in `lead_names:` from the files so `ECGDataset(leads=["aVL"])` returns the same physical lead everywhere; a model trained across datasets without it silently crosses two leads.
- **Heavy deps stay lazy.** Do not add `import wfdb` / `import torch` / `import mlcroissant` at module top-level in any file imported by `ecgbench/__init__.py`'s eager path. Import inside functions instead.
- **A fold lives in exactly one split, so `fold_numbers` alone cannot cross splits.** `ECGDataset(split="train", fold_numbers=[9])` is an error, because fold 9 was exported under `val/`. For custom cross-validation pass `split=None` with `fold_numbers`, which selects by fold from `folds.csv` and ignores the default split layout. With `split=None` each sample's `["split"]` reports that record's own default split, not one name for the set.
- **Prefer `window=(start, length)` over a cropping `transform`, and never a lambda.** Both give the same tensor, but `window=` is pushed into the reader (`sampfrom`/`sampto` for wfdb, `skiprows`/`max_rows` for csv), so on long records it is the difference between decoding 44 MB and 0.25 MB — `incartdb` drops from ~106 ms to ~8 ms per record. A lambda `transform` additionally breaks `DataLoader(num_workers>0)` under the `spawn` start method (macOS and Windows default) with `PicklingError`; `window=` is plain data and pickles fine. Order is `window` → `leads` → `units` → `transform`.
- **A fixed window does not fit a variable-length dataset.** `window=` raises `WindowOutOfRangeError` for any record shorter than `start + length`, naming the record and its true length. `cpsc_2018` runs 6-144 s and `ptbdb` 32-120 s, so quote a window sized to the **shortest** record in examples and on dataset pages, and note the minimum length in Phase 0.
- **A metadata column that is 100% populated is not necessarily complete.** Machine-generated measurements often encode "not measurable" as an integer rail rather than a blank, so `notna().mean()` reports 1.0 and every summary statistic is silently wrong. MIMIC-IV-ECG uses `29999` for unmeasurable wave timings (230,323 records for `p_end` alone), `32767`/`-32768` for axes and `65535` for RR interval. Check the min/max of every numeric column against a physiologic range, not just its null count. Convert sentinels to NaN in the label loader — that is lossless when the source has no genuine blanks — and document which values you treated as sentinels, because a reader cannot otherwise tell a converted value from a missing one.
- **Verify a local copy before quoting any figure from it.** See the Phase 0 checksum item: the label CSV is the file most likely to have been quietly filtered, and it is the one every published count depends on.
- **Windowing is a read-time adapter, like `leads=` and `units=`.** It never touches the source files, the exported fold CSVs or validation — `validate_dataset` always reads whole records, so a record excluded for a railed lead stays excluded even if your window never reaches that lead. `validation/engine.py` has its own copy of `_load_signal` **without** the window parameter, and that is deliberate; do not "fix" it.
