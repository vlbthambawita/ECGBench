# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECGBench is a config-driven library for reproducible ECG benchmark datasets. It has four major subsystems:

1. **Config system** — every dataset described by a single YAML file (`ecgbench/data/configs/`). Adding a dataset requires zero Python for standard cases.
2. **Validation engine** — pre-validates every ECG record, producing `original` (all records + quality flags) and `clean` (valid only) versions.
3. **Splitting framework** — strategy-pattern splitters producing deterministic 10-fold stratified splits with patient-level grouping. PTB-XL, Chapman-Shaoxing, and a generic config-driven fallback.
4. **Croissant metadata** — programmatic MLCommons Croissant 1.1 JSON-LD generation and validation via `mlcroissant`.

Plus a **catalogue** of 64 ECG datasets (pure Python, no heavy deps), a unified **PyTorch Dataset**, auto-download, and HuggingFace Hub integration.

**Catalogue ≠ implemented datasets.** The catalogue lists all 64 surveyed datasets (`docs/_datasets/*.md`); only datasets with a YAML config in `ecgbench/data/configs/` can actually be validated/split/loaded (currently `ptbxl`, `chapman_shaoxing`). The two use *different slug namespaces*: catalogue slugs are dashed and match the Markdown filename (`ptb-xl`), config slugs are underscored and match the YAML filename and the `@register()` splitter name (`ptbxl`). Don't assume one maps mechanically to the other.

## Development Setup

```bash
uv pip install -e ".[dev]"
```

## Common Commands

```bash
# Lint & format
ruff check ecgbench/
black ecgbench/

# Tests — no real ECG data needed; conftest builds numpy arrays + DataFrames (no WFDB files)
pytest
pytest tests/test_config.py -v          # single module
pytest -k "test_split" -v               # by name pattern
pytest tests/test_cli.py::test_run_upload_dry_run_counts_files -v   # single test

# Sanity-check the catalogue without heavy deps
python -c "import ecgbench; print(len(ecgbench.list_datasets())); print(ecgbench.get_dataset('PTB-XL'))"

# Full pipeline: validate + split + Croissant
ecgbench splits --dataset ptbxl --data-path /path/to/ptb-xl/1.0.3/

# Standalone Croissant generation
ecgbench croissant --dataset ptbxl --splits-dir output/ptbxl/clean/ --version clean

# Upload to HuggingFace Hub (requires HF_TOKEN in .env or env var)
ecgbench upload --data-dir output/ --datasets ptbxl
```

The same pipelines are importable from Python as `ecgbench.run_splits`,
`ecgbench.run_croissant`, and `ecgbench.run_upload` (see `ecgbench/cli/`).

## Code Style

- Python 3.10+ — use modern type hints (`str | None`, `list[int]`, no `typing` imports for builtins)
- Line length: 100 (both ruff and black)
- Ruff rules: E, F, I, N, W
- Use `dataclasses` over plain dicts for structured data
- Use `pathlib.Path` everywhere, never raw string paths

## Architecture

### Config System (`ecgbench/config.py` + `ecgbench/data/configs/`)
`DatasetConfig` dataclass is the typed representation of a YAML config. All modules accept `DatasetConfig`, never raw dicts. `load_config(slug)` parses YAML and validates required fields. Nested dataclasses: `CreatorInfo`, `StratificationConfig`, `ValidationConfig`, `PredefinedSplitConfig`, `CroissantConfig`.

### Catalogue (`ecgbench/catalogue.py`)
Loads one Markdown file per dataset (YAML front matter holds the row fields) from `docs/_datasets/<slug>.md` in the repo, or `ecgbench/_datasets/` in an installed wheel (hatch force-includes `docs/_datasets` → `ecgbench/_datasets`, see `pyproject.toml`). Cached with `functools.cache`, returns `CatalogueEntry` dataclass instances. No heavy deps — always importable. **To add/edit a catalogue dataset, edit the Markdown front matter, not a CSV.**

Those same `.md` files are a Jekyll collection powering the website, so front matter is consumed by both `catalogue.py` (`_entry_from_meta`) and Liquid templates (`docs/_layouts/dataset.html`, `docs/_includes/`). Adding a front-matter field means updating both sides. Valid `category` values are fixed by `_CATEGORY_ORDER` in `catalogue.py` and must match a table id in `docs/_data/tables.yml`; valid `status` values are the keys of `docs/_data/statuses.yml` (`not_started`, `implementing`, `completed`, `needs_review`).

**`status:` is not a reliable signal of implementation state.** 63 of the 64 entries are `not_started` and only `ptb-xl` is `completed` — including `chapman-shaoxing-arrhythmia`, which has a working config *and* a registered splitter. Check `ecgbench/data/configs/` to find out what actually runs. Note also that two catalogue entries (`chapman-shaoxing-arrhythmia.md`, `chapman-shaoxing-ecg-database-10-646-patients.md`) describe source datasets served by the single `chapman_shaoxing` config — another reason the two namespaces don't map one-to-one.

### Website (`docs/`)
Jekyll site (`docs/_config.yml`, `baseurl: /ECGBench`) with `index.html` rendering catalogue tables from `docs/_data/{tables,columns,statuses}.yml` + the `datasets` collection, plus one detail page per dataset via `_layouts/dataset.html`. Table cells render through one partial per column (`_includes/cells/<column>.html`); dataset detail pages are driven by the `sections:` list in front matter, one partial per section type (`_includes/sections/`: `description`, `table`, `code`, `links`, `notebook`, `plot`). There is no CSV download — tabular access is the Python catalogue only.

### CLI (`ecgbench/cli/`)
`_main.py` builds the root parser and dispatches via `args.func`. Each subcommand module follows the same contract: a public `run_X(...)` function (the Python API, pure kwargs, no argparse), a private `_cli_run(args)` adapter, and `add_subparser(subparsers)` that ends with `p.set_defaults(func=_cli_run)`. A new subcommand needs all three plus registration in `_main.py` and re-export in `cli/__init__.py`.

### Validation (`ecgbench/validation/`)
- `checks.py` — individual check functions (`check_missing_leads`, `check_nan_values`, `check_truncated_signal`, `check_flat_line`, `check_corrupt_header`, `check_amplitude_outlier`) registered in `CHECK_REGISTRY`.
- `engine.py` — `validate_dataset()` runs checks in parallel via `ProcessPoolExecutor`, returns `ValidationResult` with `original_df` (all records + `is_valid`/`quality_issues` columns) and `clean_df`.
- `report.py` — generates `validation_report.json`.

### Splitting (`ecgbench/splitting/`)
- `base.py` — `DatasetSplitter` ABC + `SplitResult` dataclass (with `.train`, `.val`, `.test` properties and `get_kfold_split()`).
- `engine.py` — `split_dataset()` dispatches to `StratifiedGroupKFold` (patient-aware) or `StratifiedKFold`, or reads predefined splits. Folds are 1-indexed.
- `strategies/` — `@register("slug")` decorated splitters. `PTBXLSplitter` (SCP superclass mapping), `ChapmanSplitter`, `GenericSplitter` (config-driven fallback).
- `export.py` — writes `original/` and `clean/` fold CSVs with **minimal columns only** (record ID, patient ID, signal paths, `fold`, `default_split`), plus `is_valid`/`quality_issues` in `original/` only (`_minimal_columns(include_quality=...)`). Full metadata stays in the original dataset CSV.
- `registry.py` — splitter lookup with `GenericSplitter` fallback.

Three behaviours of `split_dataset()` that aren't obvious from its signature:

- **The default split mapping is derived from `n_folds`, not fixed.** `_split_grouped`/`_split_simple` compute `train=range(1, n_folds-1)`, `val=[n_folds-1]`, `test=[n_folds]`. The documented 8/1/1 layout is just what `n_folds=10` produces; `--n-folds 5` yields train=[1,2,3], val=[4], test=[5].
- **`--n-folds` is silently ignored for predefined splits.** `_split_predefined` takes folds from the dataset's own column and the mapping from YAML, so the flag has no effect on PTB-XL.
- **Nothing verifies patient grouping or fold balance.** `_split_predefined` partitions only on the fold column yet returns `group_column=config.patient_id_column` in `SplitResult`, so provenance asserts patient-awareness that was never checked. There is no leakage assertion and no stratification-balance measure anywhere in the pipeline. Separately, `random_state` is a default arg (`42`) that `run_splits` never passes, so the seed is neither CLI-configurable nor recorded in any output artefact.

### Croissant (`ecgbench/croissant.py`)
Generates Croissant 1.1 JSON-LD using `mlcroissant` (optional dep, lazy import). Includes SHA-256 hashes for all CSVs.

### Download (`ecgbench/download.py`)
`resolve_data_path()` is the single entry point for locating dataset files. Auto-downloads to `~/.ecgbench/datasets/<slug>/` if no local path given.

### Dataset (`ecgbench/dataset.py`)
Single `ECGDataset` class loading any dataset via config. `metadata_source="hf"` (default) downloads fold CSVs from HuggingFace Hub; `"local"` reads from disk. `ecg_collate_fn` handles heterogeneous batches.

**`metadata_source="local"` does not read the `output/` tree that `export_splits` writes.** `data_path` is a single argument serving two roles: the signal root used by `__getitem__`, and — in local mode — the splits root. `_load_from_local` probes `data_path/<version>/<split>/`, `data_path/<split>/`, then `data_path/<version>/folds.csv`, `data_path/folds.csv`. So local mode only works if the fold CSVs sit *inside* the raw dataset directory; pointing `data_path` at `output/<slug>/` breaks signal loading, and pointing it at the raw data breaks metadata loading. Either copy the fold CSVs into the data directory or use `metadata_source="hf"`.

**Read-time adapters — `window=`, `leads=`, `units=`, `transform=` — shape the returned tensor only.** They never touch source files, exported fold CSVs, or validation, which always reads whole records. Order is `window` → `leads` → `units` → `transform`. `window=(start, length)` is pushed into the reader (`sampfrom`/`sampto` for wfdb, `skiprows`/`max_rows` for csv), so it avoids decoding what it discards — ~13x on `incartdb`'s 1800 s records — and it raises `WindowOutOfRangeError` naming the record and its true length when it does not fit. `validation/engine.py` keeps its own window-less copy of `_load_signal`; that asymmetry is deliberate.

**Fold selection has two modes.** `fold_numbers` with a named `split` reads `<split>/fold_<N>.csv`, so the folds must belong to that split (1-8 train, 9 val, 10 test). `split=None` requires `fold_numbers` and instead filters the master `folds.csv` by fold alone — the only way to express custom cross-validation — and then `sample["split"]` reports each record's own `default_split` rather than one name.

### Public API (`ecgbench/__init__.py`)
Catalogue and config imports are eager (lightweight). Everything else (`ECGDataset`, validation, splitting, croissant, download, `run_*` pipelines) is lazy-imported via `__getattr__` so `import ecgbench` doesn't pull in torch/wfdb/mlcroissant. A new public symbol that needs a heavy dep must be added to both `_LAZY_IMPORTS` and `__all__` — never imported at module top level.

## On-disk and Hub Layouts

Both are conventions hard-coded across `export.py`, `cli/upload.py`, and `dataset.py` — keep them in sync when changing any one of them.

```
output/<config-slug>/                 # local, from `ecgbench splits`
  validation_report.json
  {original,clean}/
    folds.csv                         # every record + fold + default_split
    {train,val,test}/fold_<N>.csv     # N is 1-indexed
    croissant.json
```

Default split assignment is folds 1–8 → `train`, 9 → `val`, 10 → `test`, so `train/` holds 8 fold CSVs and `val/`, `test/` one each. Fold membership is identical between `original/` and `clean/`; `clean/` is a row subset.

HuggingFace dataset repo `vlbthambawita/ECGBench` mirrors that tree with the dataset slug as top-level prefix: `<slug>/<version>/<split>/fold_<N>.csv` and `<slug>/<version>/folds.csv`. This is what `ECGDataset(metadata_source="hf")` fetches with `hf_hub_download`.

The repo id is **overridable on upload but hard-coded on download**: `run_upload(hf_repo_id=...)` takes it as a parameter with a `--hf-repo-id` CLI flag, while `ECGDataset._load_from_hf` assigns `repo_id = "vlbthambawita/ECGBench"` inline with no override. Forking to a different Hub repo therefore requires a source edit in `dataset.py`.

## Derived datasets get no config

A release whose records belong to another dataset — a feature, annotation or
relabelling layer — must not get a config, a splitter or a fold assignment. Generating
folds for it would create a second ECGBench partition of recordings an existing config
already partitions, so a user could train on one and evaluate on the other. **PTB-XL+**
is the worked case: `ecgbench/labels/ptbxl_plus.py` exposes its statements and features
indexed by PTB-XL's `ecg_id`, and there is deliberately **no `ptbxl_plus` config** —
`tests/test_labels.py::TestPTBXLPlusLabels::test_there_is_deliberately_no_ptbxl_plus_config`
enforces that. Such a module is *not* registered in `labels/__init__.py:_custom_loaders()`,
because that dict maps config slugs to loaders and there is no config.

Its loader also carries a trap worth knowing before touching that file:
`12sl_features.csv` keeps `ecg_id` at **column 145 of 783**, buried among the features,
so a glance at the header suggests the table has no key — and neither 12SL table is
sorted by `ecg_id` (both run `1, 21803, 21804, …`). The loader keys by column name and
keeps a row-order fallback, which refuses to guess when row counts disagree, only in case
a future release drops the column. See "Derived and
annotation-only datasets" in `ADD_DATASET_TODO.md`.

## Distribution policy — not every dataset's splits are published

`DatasetConfig.publish_fold_csvs` (default `True`) decides whether a dataset's fold
CSVs go to the public HuggingFace repo. Fold CSVs are identifiers only, but for a
**credentialed or restricted** source those identifiers are still data derived under a
use agreement, and the repo is public and ungated. `mimic_iv_ecg` sets it `False`; it is
the only such dataset today.

The policy is enforced in both directions, not left to whoever runs the command:
`cli/upload.py` raises `PermissionError` before any network call, and
`ECGDataset._load_from_hf` raises `SplitsNotPublishedError` — quoting
`no_publish_reason`, which must therefore contain the regeneration command — instead of
letting the user hit a bare 404.

Such datasets are distributed as a **recipe**: `ecgbench splits` reproduces the
canonical partition locally because fold assignment is a pure function of the input
table and a fixed seed (`random_state=42`, recorded in `SplitResult.split_metadata`).

`ecgbench/manifest.py` is what makes that trustworthy. `run_splits` writes
`manifest.json` into the output directory for **every** dataset, holding the seed, fold
count, a SHA-256 per input file, record counts, and a `fold_digest` — a hash over the
whole record-to-fold mapping in canonical order, so two runs agree iff they produced the
same partition. For unpublished datasets a reference copy ships in
`ecgbench/data/manifests/<slug>.json`, and `verify_splits(slug, output_dir)` compares
the two, naming the differing input file on mismatch (the usual cause: a filtered local
copy, exactly the MIMIC `machine_measurements.csv` case).

Adding another restricted dataset: see "Restricted and credentialed datasets" in
`ADD_DATASET_TODO.md`, which also tabulates what may and may not be published.

## Testing

Tests never touch real ECG data or the network. `tests/conftest.py` builds `DatasetConfig` objects directly in Python (not from YAML — so config fixtures can drift from the shipped YAML), synthetic numpy signal arrays per failure mode (`synthetic_signal_bad_nan`, `_flat`, `_truncated`, `_amplitude_outlier`, …), mock metadata DataFrames, and `tmp_splits_dir`, a full `{original,clean}/{train,val,test}/fold_N.csv` + `folds.csv` tree in `tmp_path`. The `CHECK_REGISTRY` checks are tested against arrays, not files, so they involve no WFDB I/O. Loader-level tests do: `tmp_wfdb_signal_dataset` and `tmp_csv_signal_dataset` each build five real 12x5000 records plus a fold tree, so `_load_signal`, `window=`, `leads=`, `units=` and fold selection run against actual files.

Three coverage gaps to know before trusting a green run:

- **`validate_dataset()` is still untested end-to-end** — the parallel `ProcessPoolExecutor` path in `validation/engine.py` never runs in the suite, so anything touching it needs a manual smoke run against real data. `_load_signal()` and `ECGDataset.__getitem__` *are* covered against real files by the `tmp_wfdb_signal_dataset` and `tmp_csv_signal_dataset` fixtures (12x5000 records written with `wfdb.wrsamp`, values encoding lead and sample index so a windowed or shifted read cannot pass by accident).
- **`tmp_splits_dir` uses folds 1–5** (train=[1,2,3], val=[4], test=[5]), not the production 1–8/9/10 convention documented above. A test passing on this fixture proves nothing about the real fold mapping.
- **`TestECGDatasetLocal`, `TestECGDatasetLabels` and `TestLeadSelectionAndUnits` bypass the constructor** via `ECGDataset.__new__` and assign attributes by hand. Two consequences: `__init__` is not covered by them, and every new read-time attribute must be added to their helpers or `__getitem__` raises `AttributeError` there (this is how `window` broke them). `TestWindowedLoading` and `TestFoldSelection` do go through the real `__init__`, so `resolve_data_path`, split/version/window validation and `signal_col` resolution are covered there.

Optional-extra tests guard with `pytest.importorskip` (`torch` in `test_dataset.py`, `mlcroissant` in `test_croissant.py`/`test_cli.py`) — a base install silently skips them, so install `.[dev]`. HF upload is exercised via `monkeypatch` + `--dry-run`.

## Adding a New Dataset

1. Copy `ecgbench/data/configs/_template.yaml` to `<slug>.yaml`, fill in fields
2. Run `ecgbench splits --dataset <slug> --data-path /path/to/data/`
3. If custom logic needed, create `ecgbench/splitting/strategies/<slug>.py` with `@register("<slug>")` — and import it in `ecgbench/splitting/strategies/__init__.py`, otherwise the decorator never runs and `get_splitter()` falls back to `GenericSplitter`
4. Add/update the catalogue entry at `docs/_datasets/<dashed-slug>.md` (set `status: completed`)
5. Run `pytest`

## Versioning & Release

Version derived from git tags via `hatch-vcs`. Push a `v*` tag to trigger PyPI publish (Trusted Publishing) and HF Space deploy.

## CI/CD (GitHub Actions)

- `deploy-pages.yml` — GitHub Pages deploy of `docs/`, **only when `docs/**` changes** on `main`
- `deploy-hf-space.yml` — HF Space (`vlbthambawita/ECGBench`, `sdk: static`) on `docs/**` changes or `v*` tags. HF Spaces don't run Jekyll, so the workflow builds `_site/` itself, patches `baseurl: ""` (Spaces serve from root, Pages from `/ECGBench/`), writes `_site/_version.json`, and uploads with `delete_patterns=["*"]`
- `publish-pypi.yml` — PyPI publish on `v*` tags via Trusted Publishing

There is no CI test/lint job — run `pytest` and `ruff`/`black` locally before pushing. Python-only changes deploy nothing until a `v*` tag is pushed.

## Environment Variables

- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) — for HuggingFace Hub upload/download. Set in `.env` (see `.env.example`).

## Reference Docs

- `ECGBench_architecture/ARCHITECTURE.md` — end-to-end flow in Mermaid diagrams; every box names a real function/class, so it doubles as a map into the source.
- `ADD_DATASET_TODO.md` — full per-dataset checklist (discovery → catalogue → config → splitter → splits → tests → upload), with a Gotchas section covering the silent-failure traps. Verified against the code; use it as the authoritative procedure rather than the condensed "Adding a New Dataset" steps above.
- `DATASET_ANALYSIS_PLAN.md` — design doc for the per-dataset analysis scripts (statistical tables + optional Plotly HTML report). **Not yet implemented, and its §0 decisions are contested** — it proposes replacing `ecgbench splits` with unpackaged `scripts/analyse_<slug>.py` files. Read §0 before acting on it.
- `ecgbench_expenctation.txt` — the requirements the analysis plan derives from, plus a second (unimplemented) set covering loader-side preprocessing, lead filters, and per-dataset metadata filters.
- `CLI_PLAN.md`, `webiste_paln_todo.md` — design notes for the CLI and the Markdown-driven website; historical, both already implemented.
