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

# Tests — no real ECG data needed; tests/conftest.py synthesises WFDB signals + configs
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

### Croissant (`ecgbench/croissant.py`)
Generates Croissant 1.1 JSON-LD using `mlcroissant` (optional dep, lazy import). Includes SHA-256 hashes for all CSVs.

### Download (`ecgbench/download.py`)
`resolve_data_path()` is the single entry point for locating dataset files. Auto-downloads to `~/.ecgbench/datasets/<slug>/` if no local path given.

### Dataset (`ecgbench/dataset.py`)
Single `ECGDataset` class loading any dataset via config. `metadata_source="hf"` (default) downloads fold CSVs from HuggingFace Hub; `"local"` reads from disk. `ecg_collate_fn` handles heterogeneous batches.

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

HuggingFace dataset repo `vlbthambawita/ECGBench` (default in both `dataset.py` and `cli/upload.py`) mirrors that tree with the dataset slug as top-level prefix: `<slug>/<version>/<split>/fold_<N>.csv` and `<slug>/<version>/folds.csv`. This is what `ECGDataset(metadata_source="hf")` fetches with `hf_hub_download`.

## Testing

Tests never touch real ECG data or the network. `tests/conftest.py` builds `DatasetConfig` objects directly in Python (not from YAML — so config fixtures can drift from the shipped YAML), synthetic numpy signal arrays per failure mode (`synthetic_signal_bad_nan`, `_flat`, `_truncated`, `_amplitude_outlier`, …), mock metadata DataFrames, and `tmp_splits_dir`, a full `{original,clean}/{train,val,test}/fold_N.csv` + `folds.csv` tree in `tmp_path`. Checks are tested against arrays, not files, so no WFDB I/O is involved.

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
- `CLI_PLAN.md`, `webiste_paln_todo.md` — design notes for the CLI and the Markdown-driven website; historical, both already implemented.
