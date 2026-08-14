# Adding a Dataset to ECGBench — TODO Checklist

Reusable plan for adding a new dataset end-to-end (config → splits → upload).
Copy this file (or duplicate the relevant sections) when starting on a new
dataset, and tick items off as you go. The phases are roughly sequential, but
phase 1 (catalogue) is independent and can happen at any time.

Two phases are non-negotiable and sit at either end. **Phase −1, reading the
requesting GitHub issue and all of its comments, always runs first** — the local
data path and other constraints live in the comments, not the body. **Phase 7, the
HuggingFace upload, is required and always runs last** — a dataset that is not on
the Hub 404s for every user, because `ECGDataset` defaults to fetching fold CSVs
from there.

---

## Phase −1 — Read the whole GitHub issue, comments included

Datasets are requested as issues labelled `DATASET - Datasets to be added`, and
**the issue body is usually not the whole request.** The body is often a single
link to the dataset's own catalogue page; the local path to an already-downloaded
copy, the acquisition notes, the licence caveat and any "use this version, not
that one" correction arrive as **comments**. Reading only the body means either
re-downloading gigabytes that are already on disk, or building against the wrong
copy.

- [ ] **Read the issue body *and* every comment before touching anything.** Treat the
  comments as part of the specification, not as commentary on it.
- [ ] **Find the local data path and use it.** It is conventionally given in a comment
  as `Local path: /...` — e.g. issue #19 (CiPA) carried only a catalogue link in the
  body and `Local path: /global/D1/homes/vajira/data/SEARCH/physionet.org/files/ecgcipa`
  in its one comment. Pass it as `--data-path` throughout; do not re-download a copy
  that is already on disk. (Finding a public download URL is a *separate* question,
  answered in Phase 2 — `download_url` must stay `null` whenever the splitter
  generates `metadata_csv`, however public the release is.)
- [ ] **Note the exact version directory.** The path in the comment is often the
  dataset root *above* the version (`.../ecgcipa`, containing `1.0.0/`). The
  `--data-path` the pipeline wants is the version directory, and `version:` in the
  config must match it.
- [ ] **Use the GitHub REST API to read comments, not a page fetch.**

  ```bash
  gh issue view <N> --repo vlbthambawita/ECGBench --comments   # if gh is installed
  # gh is often absent; the API needs no auth for a public repo:
  curl -s https://api.github.com/repos/vlbthambawita/ECGBench/issues/<N>/comments
  curl -s https://api.github.com/repos/vlbthambawita/ECGBench/issues/<N>
  ```

  **A summarising fetch of the issue's HTML page is not good enough** and will not
  announce that it failed. Fetching `github.com/.../issues/19` returned
  "Comments: None visible in the provided content" for an issue that had one
  comment holding the only copy of the data path. The API returns comments as JSON;
  use it.
- [ ] **Carry the rest of the comment thread into the PR description.** Credentialed
  access, a superseded version, a known-bad file, a request to keep something
  unpublished — anything stated in a comment is a decision you are accountable for
  having read, and Phase 0's licence and distribution-policy items may depend on it.
- [ ] **A local copy is not a verified copy.** Whoever downloaded it may have
  filtered, renamed or partially fetched it. Run the release's own checksums before
  computing a single figure from it — see the `SHA256SUMS.txt` item in Phase 0,
  which is the reason that item exists.
- [ ] Close the issue from the commit (`... and close #<N>`), matching the existing
  history.

---

**Then check the dataset actually has its own recordings.** If it is a feature,
annotation or label layer over another dataset's records — PTB-XL+ over PTB-XL, say —
most of this checklist does not apply and generating splits for it is actively harmful.
Jump to "Derived and annotation-only datasets" below.

**There are two slug namespaces — do not use one where the other belongs.**

| | Form | Example | Must match |
|---|---|---|---|
| `<config-slug>` | lowercase, **underscores** | `ptbxl`, `chapman_shaoxing`, `mimic_iv_ecg` | the YAML filename, the `slug:` field inside it, and the `@register("...")` argument |
| `<catalogue-slug>` | lowercase, **dashes** | `ptb-xl`, `chapman-shaoxing`, `mimic-iv-ecg` | the `docs/_datasets/<catalogue-slug>.md` filename and the `slug:` field in its front matter |

Phases 2–7 use `<config-slug>`. Phase 1 (catalogue) uses `<catalogue-slug>`.
The two are unrelated strings — nothing maps one to the other mechanically, so
pick both up front and keep them straight.

---

## Phase 0 — Discovery (before writing anything)

- [ ] **Does it contain its own recordings?** Count the signal files and compare the metadata's record ids against every dataset already in the catalogue. A release whose records are another dataset's — an annotation, feature or relabelling layer — must NOT get its own config or splits; see "Derived and annotation-only datasets" below. This is the first question because getting it wrong means building a second partition of a dataset ECGBench already partitions.
- [ ] Locate the dataset's **official source URL**, **license**, **citation**, **DOI**.
- [ ] Confirm signal **format**. `wfdb`, `csv`, `csv_lead_rows`, `opensignals`, `npy`, `hdf5` and `edf` are implemented; anything else raises `NotImplementedError` in `_load_signal` and needs a branch there (in **both** `validation/engine.py` and `dataset.py` — they each have a copy). For `edf`, note that every channel of a file must share one sampling rate: a polysomnogram mixing ECG at 128 Hz with oximetry at 8 Hz has no single `(leads, samples)` array and `_read_edf` raises rather than reshaping it (`ucddb` ships both shapes).
- [ ] Confirm the **units of the stored samples** and set `signal_unit_scale` so they reach ECGBench as millivolts (µV → `0.001`). Read one file and check the peak amplitude: a QRS peaking near 1000 is microvolts, near 1.0 is millivolts.
- [ ] Confirm **leads**, **duration (s)**, **sampling rate(s)**, **default rate**.
- [ ] Record the **lead names in file order** into `lead_names:`, spelled as the source spells them. Read a header or CSV column row — never assume the standard order. `ECGDataset(leads=...)` cannot work without it, and two of the four implemented datasets deviate: MIMIC-IV-ECG stores aVF before aVL, PTB-XL spells them AVR/AVL/AVF.
- [ ] Download a small subset locally and inspect the **metadata CSV**:
  - record ID column name
  - patient ID column (or confirm one-record-per-patient, set `null`) — **and do not read "there is no column" as "one record per patient"**. Compare the record count against the patient count the paper or landing page states; if they differ, the grouping exists and is merely unshipped, and `null` is a leak. Recover it from per-record demographics if any ship (`apnea_ecg`), or from the waveforms themselves (`szdb`: median-beat correlation against each record's own split-half self-control, validated against *two* counts the paper states)
  - signal-path column(s) per sampling rate — note any **prefix** that must be prepended (cf. Chapman's `ECGData/`)
  - label column name and **format** (`single` / `comma_separated` / `dict_string` / `json`)
- [ ] Decide stratification: `direct`, `superclass_mapping` (needs a mapping CSV), or `custom_function` (needs a custom splitter — see Phase 3).
- [ ] Decide if **predefined splits** exist (e.g. PTB-XL `strat_fold`). If yes, note the column and which fold values go to train/val/test.
- [ ] Sanity-check **expected samples = duration_s × sampling_rate** per rate.
- [ ] **Check what the first samples of a record actually are.** Instrument calibration blocks are not marked in any header, and `window=(0, n)` on one returns a square wave rather than an ECG — silently, and identically for every record in the release. `ucddb` is the worked case: all 25 Holter files open with 67 to 119 s of a 1 mV 2 Hz pulse, byte-identical across records over the shortest block, so a first-N-samples window returns the same array for the whole database. The cheap test is to read the same early window from two records and compare (`np.array_equal`) — two unrelated ECGs never match. Record the per-record block length and expose the first safe sample, as `ecgbench.labels.ucddb.CALIBRATION_SAMPLES` and `ECG_STARTS_AT_SAMPLE` do.
- [ ] Record the **minimum record length** and whether length is uniform. Two things depend on it: whether `expected_samples` can be set at all (see Phase 2), and what `window=` a user can safely apply — `ECGDataset(window=(start, length))` raises `WindowOutOfRangeError` on any record shorter than `start + length`, so the example script and the dataset page should quote a window that fits **every** record.
- [ ] Write down the **headline figures the paper/landing page claims** — record count, patient count, per-class breakdown — *before* you look at the data. These are what you will check the shipped files against in Phase 1.
- [ ] **Verify the files against the release's own checksums** (`SHA256SUMS.txt`, `md5sums.txt`) before trusting any figure, especially for the metadata/label CSVs. A local copy may have been filtered, deduplicated or renamed by whoever downloaded it, and a filtered file under the official name is invisible until your counts disagree with the paper. MIMIC-IV-ECG is the worked example: the shipped `machine_measurements.csv` had been replaced by a 789,481-row subset of the real 800,035-row file, with the original renamed `_original.csv`; only the checksum revealed which was authentic. Compute figures from the verified file, and name the **official** filename in the config regardless of what the local copy is called.
- [ ] Check for a **changelog** in the dataset root (`*changelog*`, `CHANGES`, release notes on the landing page). It is the authoritative explanation when the shipped version disagrees with the paper.
- [ ] Check whether the metadata is really a **CSV**. `.xlsx` needs converting — `validate_dataset` re-reads `metadata_csv` from disk with `pandas.read_csv`, so an in-memory conversion is not enough. Convert in the acquisition script, and have the splitter generate a normalised CSV (see `chapman.py`).
- [ ] If the source is **several files** rather than one archive, `download_url` cannot express it — write an acquisition script under `examples/download_<name>.py` that md5-verifies each file (see `download_chapman_figshare.py`).
- [ ] **Decide the distribution policy now, from the licence** — it changes Phase 2 and Phase 7. Open licences (CC-BY, CC BY-SA, ODC-By, public domain) → fold CSVs get published. **Credentialed or restricted** → they do not; see "Restricted and credentialed datasets" below before writing the config.
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
- [ ] Set `status` to one of the keys of `docs/_data/statuses.yml` — `not_started`, `implementing`, `completed`, `needs_review`, `unavailable`. Use `unavailable` only when the **source** has withdrawn the data (as KURIAS-ECG's authors did); it describes their side, not ours, so none of the phases below apply and the page should say why.
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
- [ ] For a **credentialed or restricted** source, set `publish_fold_csvs: false` and a `no_publish_reason` that names the agreement and gives the regeneration command. Both are enforced: `ecgbench upload` refuses the dataset, and `ECGDataset` raises `SplitsNotPublishedError` quoting the reason instead of a bare 404. Leave both unset for an openly licensed source — the default publishes.
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
    validation_report.json            # top-level artefacts
    manifest.json                     # seed, input checksums, fold digest
    {original,clean}/
      folds.csv                       # every record + fold + default_split
      croissant.json                  # one PER VERSION, not top-level
      {train,val,test}/fold_<N>.csv   # N is 1-indexed
  ```
- [ ] Confirm `folds.csv` exists in both versions and that fold CSVs sit under `train/`, `val/`, `test/` — not loose in the version directory. With the default folds 1–8 → train, 9 → val, 10 → test, expect 8 CSVs in `train/` and one each in `val/` and `test/`.
- [ ] **If the release has fewer records or fewer patients than 10, set `n_folds` in the config rather than passing `--n-folds`.** `StratifiedKFold` and `StratifiedGroupKFold` both raise once `n_splits` exceeds the record count, and `StratifiedGroupKFold` produces *silently empty* folds once it merely exceeds the number of patient groups — so a 5-patient release generates two empty folds at `n_folds=7` and says nothing. `szdb` is the worked case (`n_folds: 5`, one subject per fold). It has to live in the config because `manifest.json` hashes the partition into `fold_digest`, so a user who forgot the flag would get a different digest with no explanation.
- [ ] Confirm the exported columns match the version, per `_minimal_columns()` in `export.py`:
  - `clean/` — record ID, patient ID (if configured), signal paths, `fold`, `default_split`. Nothing else.
  - `original/` — the same **plus `is_valid` and `quality_issues`**. These two are intentional here; do not "fix" the exporter for them.

  Anything beyond that list is real metadata leakage — full metadata stays in the source CSV. Fix the exporter, not the config.
- [ ] Confirm fold membership is identical between `original/` and `clean/` (`clean/` is a row subset, not a re-split), that fold counts roughly match the `n_folds=10` distribution, and that **patients do not span folds** (if `patient_id_column` is set).
- [ ] Check `manifest.json`: the `inputs` checksums should match the provider's own (`SHA256SUMS.txt`), `split.random_state` should be recorded, and the two `fold_digest` values should differ only because `clean/` is a row subset. This file is what lets anyone confirm they reproduced your partition.
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

## Phase 6 — Wrap up

- [ ] Cross-check the catalogue entry's `records`/`patients` against the pipeline's own output (`original.total` in the `ecgbench splits` summary). If they disagree, one of them is wrong — usually the config filters or drops rows you did not expect.
- [ ] Update `README.md` if the dataset adds a capability the docs do not yet mention — a new `signal_format`, a units quirk, an unusual label shape. The parameter table and the "Leads and units" table both list per-dataset facts that go stale.
- [ ] Add the dataset's lead order to the table in the README's **Leads and units** section.
- [ ] Update the "Loading with ECGBench" `code` section in the catalogue entry to show `labels=True`, and `leads=`/`units=`/`window=` where the dataset has a quirk worth demonstrating (`window=` whenever records are long or variable-length). Leave the final **run-and-paste** until after Phase 7, because the upload changes how the snippet loads — see the note there.
- [ ] Flip `status:` to `completed` in `docs/_datasets/<catalogue-slug>.md` (created back in Phase 1 — don't create a second entry here).
- [ ] Commit (after Phase 7): config YAML, optional splitter + its `strategies/__init__.py` import, catalogue Markdown entry, tests, any docs. Keep generated output (`output/`) out of the commit.
- [ ] Remember `docs/**` changes trigger the Pages and HF Space deploys on `main`, while Python-only changes deploy nothing until a `v*` tag is pushed.
- [ ] Open a PR with: source URL, license, record count, validation pass rate, and whether a custom splitter was needed and why.

## Phase 7 — HuggingFace Hub upload (REQUIRED, and do it last)

**This is the final step of the task, not an optional extra.** A dataset whose splits
are not on the Hub fails for every user, because `ECGDataset` defaults to
`metadata_source="hf"` — the symptom is a bare `RemoteEntryNotFoundError: 404 ...
<slug>/clean/folds.csv`. Do not leave this for someone else to run.

**Branch on the licence first.**

- **Openly licensed** (CC-BY, CC BY-SA, ODC-By, public domain) → upload, following the
  checklist below.
- **Credentialed or restricted** → do NOT upload. The dataset should already carry
  `publish_fold_csvs: false` from Phase 2, which makes `ecgbench upload` refuse it.
  Follow "Restricted and credentialed datasets" below instead, then return here only
  for the last item (confirm no page claims the splits are downloadable).

Publication is effectively irreversible — caches and mirrors persist after deletion —
so when in doubt do not push, and ask the dataset owner.

- [ ] Ensure `HF_TOKEN` is set (env var or a `.env` in the working directory).
- [ ] Dry-run first to see the file list without pushing: add `--dry-run`. Expect 25
  files per dataset (2 versions x (folds.csv + 10 fold CSVs + croissant.json), plus
  `validation_report.json`).
- [ ] Upload:
  ```bash
  ecgbench upload --data-dir output/ --datasets <config-slug>
  ```
- [ ] Verify on the Hub that the tree is prefixed by the dataset slug —
  `<config-slug>/<version>/folds.csv` and
  `<config-slug>/<version>/<split>/fold_<N>.csv`. This prefix is what
  `ECGDataset(metadata_source="hf")` fetches with `hf_hub_download`; a missing or
  wrong prefix fails only at load time.
- [ ] Verify an **anonymous** user can load it, both versions. `ECGDataset` defaults to
  `metadata_source="hf"`, so this exercises the real download path — but only if no
  token is picked up. Unset the token and use a scratch cache so a warm cache cannot
  mask a missing upload:
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
- [ ] **Now finalise the catalogue snippet** (the Phase 6 item you deferred): drop any
  `metadata_source="local"`, run the snippet exactly as it now reads, and paste the
  values that run produced. **`ds[0]` is not the same record under `hf` and `local`** —
  HF mode filters `folds.csv` while local mode concatenates the per-split fold files,
  so the row order differs and every quoted sample value changes. This has bitten
  twice; re-run, do not translate.
- [ ] Confirm no dataset page still says the splits are "not on the Hub yet":
  `grep -rn 'metadata_source="local"\|not on the Hub' docs/_datasets/`.


### Restricted and credentialed datasets

Worked example: `mimic_iv_ecg`. Publishing its fold CSVs would have put 800,035
`study_id`s and 161,352 `subject_id`s, derived from a source under the PhysioNet
Credentialed Health Data Use Agreement, onto a public ungated repo. Instead the split
is distributed as a **recipe** and regenerated by each user, so no identifiers leave
the credentialed environment at all.

Everything through Phase 6 is unchanged. The differences are:

- [ ] **Config (Phase 2).** Set `publish_fold_csvs: false` and a `no_publish_reason`
  naming the agreement and containing the exact regeneration command. That string is
  what `ECGDataset` and `ecgbench upload` quote back, so write it for a stranger.
- [ ] **Do not upload.** The guard in `cli/upload.py` raises `PermissionError` before
  any network call, so this is enforced rather than remembered. Do not work around it.
- [ ] **Ship a reference manifest.** Copy the `manifest.json` that `ecgbench splits`
  wrote into `ecgbench/data/manifests/<config-slug>.json`. It is a few hundred bytes of
  checksums and counts — no identifiers — and it is what lets a user prove their local
  regeneration is the canonical partition rather than merely a plausible one.
- [ ] **Check the manifest before shipping it.** Its `inputs` checksums must match the
  provider's own published values, so a user with a clean download matches and a user
  with a filtered copy does not. If they disagree, you generated the split from a
  non-canonical input and must regenerate.
- [ ] **Verify the round trip** on your own run:
  ```python
  from ecgbench import verify_splits
  verify_splits("<config-slug>", "output/<config-slug>")   # raises on mismatch
  ```
- [ ] **Dataset page (Phase 6).** Add a section saying the splits are generated rather
  than downloaded and why; give the three-step generate → verify → copy recipe; and use
  `metadata_source="local"` in the loading snippet, since the default `"hf"` now raises
  `SplitsNotPublishedError`.
- [ ] Never put labels, report text or clinical columns in a manifest or a fold CSV.
  The rule that fold CSVs are identifier-only is what makes the open datasets
  redistributable at all; a manifest is checksums and counts only.

**What is safe to publish, in decreasing order of caution.** None of this is legal
advice — the agreement governs, and some prohibit publishing derived data at all:

| Artefact | Contains | Publishable for a credentialed source? |
|---|---|---|
| Waveforms | the data | never |
| Labels / report text | clinical content | never |
| Fold CSVs | record + patient identifiers | no — this is the case that motivated the policy |
| Manifest | seed, checksums, counts, one digest | yes: no identifiers, and the digest is not invertible |
| Config | column names, rates, thresholds | yes |

A plain hash of the identifiers is **not** a safe middle ground: identifier spaces are
small and enumerable (100,000 sequential IDs is seconds of brute force), so a published
hash is reversible and republishes the identifiers in effect. Pseudonymous fold tables
would need a keyed HMAC whose key never reaches the public repo — and the signal-path
column would have to be dropped too, because paths like
`files/p1376/p13767422/s40000162/40000162` embed the identifiers directly.

### Derived and annotation-only datasets

Worked example: **PTB-XL+** (`ecgbench/labels/ptbxl_plus.py`, issue #13). It ships no raw
ECGs at all — it is 3 feature tables, 2 statement tables, derived median beats and
283,326 fiducial-point files, all keyed by PTB-XL's own `ecg_id`.

**Do not give such a dataset a config, a splitter, or a fold assignment.** Every one of
its rows is another dataset's record, so `ecgbench splits` would produce a *second*
ten-fold partition over recordings that dataset already partitions — and both would
carry ECGBench's imprimatur. A user who trained on one and evaluated on the other would
be testing on training data. The `related:` graph exists to warn about overlap that
upstream providers created; there is no reason to manufacture more of it inside the
project.

Integrate it as a **label/feature provider** instead:

- [ ] **Confirm the records really are the other dataset's**, from the files. Compare id
  sets both ways (missing *and* extra), and join against a real split to get a match
  rate. For PTB-XL+: the statements and ecgdeli tables cover PTB-XL v1.0.3's 21,799
  `ecg_id`s exactly, and 17,376 of 17,376 records of PTB-XL's train split join.
- [ ] Write `ecgbench/labels/<slug>.py` exposing loaders that return frames **indexed by
  the host dataset's record id**, so a user can `reindex`/`join` onto the host's existing
  folds. Do **not** register it in `_custom_loaders()` — that dict maps *config* slugs to
  loaders, and this dataset has no config.
- [ ] Keep provider tables separate and offer a combined frame with **prefixed columns**.
  Independent providers reuse feature names, so an unprefixed concat silently overwrites;
  raise on duplicates rather than letting the last one win.
- [ ] **Find the key column by name, and check every column — not the first and last
  few.** PTB-XL+'s `12sl_features.csv` keeps `ecg_id` at column **145 of 783**, buried
  among the features. Scanning the head and tail of a wide table suggests it has no key,
  which is wrong and sends you building a positional join you do not need. Print
  `list(df.columns)` and search it.
- [ ] **Never assume row order, even when a key exists.** Both PTB-XL+ 12SL tables run
  `1, 21803, 21804, …` — not ascending. If a table genuinely lacks a key, recover it from
  an aligned file in *file* order, verify the alignment against an independent measure of
  the same quantity (e.g. one provider's heart rate against another's RR interval), and
  refuse to guess when row counts disagree.
- [ ] **Record per-artefact coverage, not one record count.** Derived releases are
  ragged: PTB-XL+ has 21,799 rows of statements, 21,795 unig features, 21,794 unig median
  beats and 20,914 12sl median beats. A single "records:" figure hides that, so put a
  coverage table on the dataset page.
- [ ] **Do not expose a derived waveform you cannot state the units of.** PTB-XL+'s
  median beats fail twice: every `12sl` header is unreadable by `wfdb.rdrecord` (a stale
  `ge_median_beats_wfdb/` prefix in the record line, which wfdb rejects), and the `unig`
  amplitudes are ~1000x their declared `/mV` gain. Return paths, not arrays, and say why.
- [ ] **Catalogue entry (Phase 1).** Set `format:` to describe what it actually is
  (`"features & annotations for PTB-XL · no raw ECGs"`), declare the `derived_from` edge
  with `verified: true` once checked, and make the note say *that no separate split is
  published and why* — that is the actionable consequence for a reader.
- [ ] Add a section to the page explaining the integration, with a runnable join snippet,
  and state that **both** downloads are needed.
- [ ] **Phase 4 and Phase 7 do not apply**: there is nothing to validate (no signals of
  its own) and nothing to upload (no fold CSVs). Say so in the PR rather than leaving a
  reviewer wondering.
- [ ] Example script under `examples/load_<slug>.py` that loads the **host** dataset and
  joins this one onto it, so the intended usage is the thing that is executable.
- [ ] Tests in `tests/test_labels.py`: the positional-join key recovery, the refusal on
  mismatched row counts, prefix collision handling, and an assertion that **no config
  exists** for the slug — that last one is what stops someone "helpfully" adding one
  later.

---

## Common gotchas

- **The GitHub issue's comments hold the local data path, and a page fetch hides them.** The issue body is typically one link to the catalogue page; `Local path: /...` is a comment. Read comments through the REST API (`curl .../issues/<N>/comments`) or `gh issue view --comments` — summarising the issue's HTML page reported "no comments visible" on an issue that had one, with no error, and the path in it was the only copy. See Phase −1.
- **A zero-padded record id is destroyed by the CSV round-trip unless the config says so.** `pandas.read_csv` reads a column of digits as int64, so a record named `00735` comes back as `735` — the id stops matching the source, the label join misses, and `data_path / "735"` is not a file, so every record fails `corrupt_header` while the traceback blames the signal files. Set `zero_padded_identifiers: true` (see `afdb.yaml`), which routes every metadata and fold-CSV read through `config.identifier_dtypes()`. It is opt-in because forcing it on changes `record_id` from int to str for six existing datasets. You are unlikely to have to remember it: `export_splits` refuses to write a zero-padded identifier from a config that left it `false`. Check for it in Phase 0 while inspecting the metadata CSV — `df[record_id].astype(str).str.startswith("0").any()` is the whole test.
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
- **A bound set to the exact hardware rail fails if any record actually reaches it, because signals are loaded as float32.** `_load_signal` casts to `np.float32`, and `check_amplitude_outlier` compares the result against the float64 bound from the YAML. Most rails are safely unrepresentable in the harmless direction, but not all: `chfdb`'s chf15 ECG2 has an `adc_zero` of −70, so its rail is (2047+70)/200 = **10.585 mV**, float32 stores that as 10.585000038146973, and a bound of `10.585` excluded the very record it was computed from — one record of 15, for 12 samples out of 36 million. Give any *attained* rail a thousandth of a millivolt of slack (`10.586`) and say why in the config. Check the direction before adding slack: `float32(-10.24)` is −10.239999771, which rounds toward zero and so cannot trip a lower bound. This is invisible until you read the excluded-record list, because the reported figure is rounded (`max_10.59`) and looks like a genuine outlier.
- **A per-channel `baseline` makes the rail asymmetric, and only one channel may have one.** Compute the bound as the union of every channel's rail, not from `adc_zero` alone. 29 of `chfdb`'s 30 channels sit at [−10.24, 10.235] mV and the thirtieth at [−9.89, 10.585].
- **A path fix-up that lives only in the splitter is a bug, not a fix.** `validate_dataset` re-reads `metadata_csv` from disk and rebuilds paths from the raw column, so it never sees what `load_metadata` changed in memory. Chapman shipped this way for months and every record failed `corrupt_header`. Write the normalised frame to disk as the config's `metadata_csv`, the way `chapman.py` and `ecg_arrhythmia.py` now do.
- **Published figures rarely match the shipped version.** PhysioNet reissues datasets and papers are not revised. PTB-XL v1.0.3 dropped 38 duplicate/triplicate records relative to the v1.0.1 the paper describes, so every superclass count is 6-17 records smaller. Copying the paper's table produces figures nobody can reproduce. Recompute, then note both values and the reason — see the Phase 1 subsection above.
- **A class breakdown is not the stratification label.** They differ in cardinality: the breakdown counts every class a record carries, the stratification label is one class per record. Say which quantity a table shows. And derive both from the same loader — PTB-XL once had a hardcoded splitter map that drifted from the shipped `scp_statements.csv` (465 records in OTHER instead of 411) precisely because there were two sources.
- **Labels never reach the batch unless you fill in `labels:`.** Fold CSVs are identification-only by design, so a dataset with a perfect config and no `labels:` block silently returns no ground truth. If the data genuinely has none, declare `available: false` with a reason rather than leaving the block out.
- **A single-label reduction of multi-label data is a trap.** Name it `primary_*`, document how ties break, and say plainly it is for stratification only. In PTB-XL 10.8% of records have tied superclasses, so the "primary" class is partly an artefact of the tie-break rule.
- **`ecgbench_metadata.csv`-style generated sources mean labels depend on pipeline order.** `ecg_arrhythmia` labels only exist after `ecgbench splits` has run once, because that is what scans the WFDB headers. Say so in the example script, and let `LabelSourceMissingError` name the file.
- **A dataset with no recordings of its own must not get splits.** Feature, annotation and relabelling layers (PTB-XL+ over PTB-XL) key on the host dataset's record ids. Generating folds for them creates a second ECGBench partition of the same recordings, which is a leakage trap of our own making. Integrate as a label provider — see "Derived and annotation-only datasets" — and assert in a test that no config exists for the slug.
- **A wide table can hide its key column in the middle.** PTB-XL+'s `12sl_features.csv` has 783 columns with `ecg_id` at position 145, so inspecting the first and last few — the natural move on a 783-column table — suggests there is no key at all. Search `list(df.columns)` by name. And do not assume row order even once you have the key: both PTB-XL+ 12SL tables run `1, 21803, 21804, …`, not ascending, so a positional join is wrong.
- **Overlapping datasets are the norm, not the exception.** Around a third of the catalogue sits in a family — CinC challenges bundle PTB-XL and CPSC-2018, CODE's subsets come out of CODE-full, MIMIC demo out of MIMIC full. Adding a dataset without checking whether it overlaps an existing one ships a silent leakage trap. Check before you write the entry.
- **`shares_records` cannot be inferred from IDs alone.** The MIMIC demo and the full release hold the same 659 recordings but renumber `study_id` into a disjoint range and truncate timestamps to the minute, so comparing keys says 0% overlap while the truth is 99.8%. When IDs disagree, try a natural key (subject + timestamp) before concluding anything.
- **Lead order is not a given.** `config.leads` is a count, not an order. `signal[4]` is aVL in PTB-XL, Chapman and ecg_arrhythmia, but aVF in MIMIC-IV-ECG. Fill in `lead_names:` from the files so `ECGDataset(leads=["aVL"])` returns the same physical lead everywhere; a model trained across datasets without it silently crosses two leads.
- **Heavy deps stay lazy.** Do not add `import wfdb` / `import torch` / `import mlcroissant` at module top-level in any file imported by `ecgbench/__init__.py`'s eager path. Import inside functions instead.
- **A fold lives in exactly one split, so `fold_numbers` alone cannot cross splits.** `ECGDataset(split="train", fold_numbers=[9])` is an error, because fold 9 was exported under `val/`. For custom cross-validation pass `split=None` with `fold_numbers`, which selects by fold from `folds.csv` and ignores the default split layout. With `split=None` each sample's `["split"]` reports that record's own default split, not one name for the set.
- **Prefer `window=(start, length)` over a cropping `transform`, and never a lambda.** Both give the same tensor, but `window=` is pushed into the reader (`sampfrom`/`sampto` for wfdb, `skiprows`/`max_rows` for csv), so on long records it is the difference between decoding 44 MB and 0.25 MB — `incartdb` drops from ~106 ms to ~8 ms per record. A lambda `transform` additionally breaks `DataLoader(num_workers>0)` under the `spawn` start method (macOS and Windows default) with `PicklingError`; `window=` is plain data and pickles fine. Order is `window` → `leads` → `units` → `transform`.
- **A fixed window does not fit a variable-length dataset.** `window=` raises `WindowOutOfRangeError` for any record shorter than `start + length`, naming the record and its true length. `cpsc_2018` runs 6-144 s and `ptbdb` 32-120 s, so quote a window sized to the **shortest** record in examples and on dataset pages, and note the minimum length in Phase 0.
- **A metadata column that is 100% populated is not necessarily complete.** Machine-generated measurements often encode "not measurable" as an integer rail rather than a blank, so `notna().mean()` reports 1.0 and every summary statistic is silently wrong. MIMIC-IV-ECG uses `29999` for unmeasurable wave timings (230,323 records for `p_end` alone), `32767`/`-32768` for axes and `65535` for RR interval. Check the min/max of every numeric column against a physiologic range, not just its null count. Convert sentinels to NaN in the label loader — that is lossless when the source has no genuine blanks — and document which values you treated as sentinels, because a reader cannot otherwise tell a converted value from a missing one.
- **`ds[0]` is not the same record under `metadata_source="hf"` and `"local"`.** HF mode downloads `folds.csv` and filters it by split; local mode concatenates the per-split `fold_<N>.csv` files. The row order differs, so every sample value quoted in a docs snippet changes when you switch modes. Generate the numbers the same way the snippet loads, and re-run rather than translating — this has produced wrong example values on two dataset pages already.
- **Verify a local copy before quoting any figure from it.** See the Phase 0 checksum item: the label CSV is the file most likely to have been quietly filtered, and it is the one every published count depends on.
- **`window=(0, n)` may return no ECG at all.** A Holter recorder writes its calibration pulse into the file ahead of the signal, and nothing in the header or the landing page says where the pulse ends. All 25 `ucddb` records open with 67-119 s of a 1 mV 2 Hz square wave that is byte-identical across the release, so the natural first window returns the same non-ECG array for every record and looks like a working loader. Compare an early window between two records before quoting any figure from one; see the Phase 0 item above.
- **Windowing is a read-time adapter, like `leads=` and `units=`.** It never touches the source files, the exported fold CSVs or validation — `validate_dataset` always reads whole records, so a record excluded for a railed lead stays excluded even if your window never reaches that lead. `validation/engine.py` has its own copy of `_load_signal` **without** the window parameter, and that is deliberate; do not "fix" it.
