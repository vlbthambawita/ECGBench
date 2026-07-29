# Dataset Analysis Scripts — Implementation Plan

Plan for the **"SCRIPTS to analyse given dataset"** capability described in
`ecgbench_expenctation.txt`:

> 1. If user inputs a dataset path, these scripts (one per dataset) will analyse the
>    dataset and produce record IDs to identify the records from original datasets and
>    put into fold csv files, and will produce statistical tables which are relevant to
>    data analysis.
> 2. If user wants, user can enable a parameter to generate html based data analysis
>    report with rich plots.

Companion docs: `ADD_DATASET_TODO.md` (per-dataset onboarding), `CLI_PLAN.md` (CLI
contract), `ECGBench_architecture/ARCHITECTURE.md` (end-to-end flow).

---

## 0. Locked decisions

| # | Decision | Consequence |
|---|---|---|
| D1 | **The analysis pipeline replaces `ecgbench splits`.** Fold export moves into the analysis pipeline; the `splits` subcommand becomes a deprecated alias. | Breaking change to the documented CLI. `run_splits()` stays importable as a legacy shim. |
| D2 | **One standalone script per dataset** — `scripts/analyse_<config-slug>.py`, run as `python scripts/analyse_ptbxl.py --data-path ...`. Shared helpers live in `scripts/_common.py` on a best-effort basis. | Per-dataset behaviour is readable in one file. Some duplication is accepted by design. |
| D3 | **HTML report = Plotly, single self-contained file.** New `[analysis]` extra pinning `plotly>=5.18` + `jinja2>=3.1`; `include_plotlyjs="inline"` so `report.html` opens offline with zero network requests. | ~3–5 MB report. Interactive hover/zoom on distributions and signal previews. |

### Risks created by D1 + D2 — resolve before Phase 1

**R1 — `scripts/` is not shipped in the wheel.** `pyproject.toml` packages only
`["ecgbench"]`, so a `pip install ecgbench` user would have no way to regenerate fold
CSVs once `ecgbench splits` is deprecated. Mitigation options (pick one, tick it):

- [ ] **M1a (recommended)** — `scripts/_common.py` holds the whole pipeline; add
      `scripts/**` to `[tool.hatch.build.targets.sdist].include` **and** keep
      `ecgbench.run_splits` working as a thin legacy wrapper so installed users are never
      stranded. Per-dataset scripts remain the documented, authoritative entry point.
- [ ] **M1b** — Accept it: analysis is a maintainer-only, repo-clone workflow. Fold CSVs
      reach users exclusively through the HuggingFace Hub, which is already how
      `ECGDataset(metadata_source="hf")` works by default.
- [ ] **M1c** — Later promotion: build in `scripts/`, and once the API settles, move
      `_common.py` into `ecgbench/analysis/` and have the scripts import from there.

**R2 — 64 datasets × copy-pasted stats logic.** The catalogue lists 64 datasets. To keep
duplication bounded, every script must stay a **thin, declarative** file: config
overrides + dataset-specific tables + one `main()` that calls `_common.run_analysis(...)`.
Target ≤120 lines per script. If a script grows past that, the logic belongs in
`_common.py`.

**Invariants that must not break** (they are hard-coded across `splitting/export.py`,
`cli/upload.py`, and `dataset.py`):

- `output/<slug>/{original,clean}/folds.csv` and `{train,val,test}/fold_<N>.csv`, N 1-indexed.
- Folds 1–8 → `train`, 9 → `val`, 10 → `test`; `clean/` is a strict row subset of `original/`
  with identical fold membership.
- Minimal columns only (record ID, patient ID, signal paths, `fold`, `default_split`,
  plus `is_valid`/`quality_issues` in `original/`).
- `validation_report.json` at the dataset root.

New analysis artefacts go into a **new `analysis/` subdirectory** so the existing tree is
untouched.

---

## 1. Target UX

```bash
# Minimum: folds + statistical tables
python scripts/analyse_ptbxl.py --data-path /data/ptb-xl/1.0.3/

# With the rich HTML report
python scripts/analyse_ptbxl.py --data-path /data/ptb-xl/1.0.3/ --html

# Fast iteration: skip per-record signal reads entirely
python scripts/analyse_ptbxl.py --data-path /data/ptb-xl/1.0.3/ \
    --no-signal-stats --skip-validation

# Full signal pass (no sampling) at 100 Hz, 16 workers
python scripts/analyse_ptbxl.py --data-path /data/ptb-xl/1.0.3/ \
    --signal-sample all --sampling-rate 100 --max-workers 16 --html
```

### Shared flags (defined once in `_common.build_parser()`)

| Flag | Default | Purpose |
|---|---|---|
| `--data-path` | auto-download via `resolve_data_path()` | Dataset root. |
| `--output-dir` | `output/<slug>/` | Root of all artefacts. |
| `--html` | off | Emit `analysis/report.html`. |
| `--sampling-rate` | `config.default_sampling_rate` | Which rate to validate/analyse. |
| `--n-folds` | `10` | Passed straight to `split_dataset()`. |
| `--signal-sample` | `2000` | Records to read for signal stats; `all` for every record. Seeded, stratified by label. |
| `--no-signal-stats` | off | Metadata + split stats only — no WFDB I/O. |
| `--hr-estimate` | off | Approximate heart-rate distribution (opt-in, see §4.3). |
| `--max-workers` | `4` | `ProcessPoolExecutor` width, matching `validation/engine.py`. |
| `--seed` | `42` | Sampling + split determinism. |
| `--skip-validation` | off | Same semantics as today's `run_splits`. |
| `--skip-croissant` | off | Same semantics as today's `run_splits`. |
| `--emit-catalogue-sections` | off | Print paste-ready YAML `sections:` for `docs/_datasets/<slug>.md` (see §7). |

---

## 2. Output layout

```
output/<config-slug>/
  validation_report.json              # unchanged
  original/                           # unchanged — folds.csv, train/val/test, croissant.json
  clean/                              # unchanged
  analysis/                           # NEW
    analysis.json                     # machine-readable, everything below in one file
    report.html                       # only with --html; self-contained
    stats/
      dataset_summary.csv             # one row: counts, rates, leads, duration
      metadata_completeness.csv       # per column: dtype, non-null, %missing, n_unique
      demographics.csv                # age/sex summaries + implausible-value flags
      label_distribution.csv          # label -> count, %, n_patients
      label_cooccurrence.csv          # label x label matrix (multi-label datasets)
      records_per_patient.csv         # histogram of records/patient
      fold_counts.csv                 # fold -> n_records, n_patients, original/clean
      fold_label_distribution.csv     # fold x label counts + % deviation from global prior
      stratification_quality.csv      # per label: max abs deviation across folds, chi2
      patient_leakage.csv             # patient_id -> folds (empty file == pass)
      signal_lead_stats.csv           # lead x {min,max,mean,std,p1,p50,p99,ptp}
      signal_record_stats.csv         # per sampled record, one row (long-form source data)
      signal_quality.csv              # clipping/flat/NaN/baseline/powerline aggregates
      quality_checks.csv              # per check: records_failed, total_issues, %
```

Everything under `analysis/` is derived and regenerable; nothing else reads it, so it is
safe to add without touching `cli/upload.py` or `dataset.py`.

---

## 3. File-by-file architecture

```
scripts/
  _common.py                  # shared pipeline, CLI, orchestration      (~350 lines)
  _stats_metadata.py          # metadata / demographics / label tables    (~250)
  _stats_signal.py            # per-record signal features + aggregation  (~250)
  _stats_splits.py            # fold balance, leakage, stratification      (~150)
  _report.py                  # Plotly figures + Jinja2 HTML renderer     (~350)
  _templates/report.html.j2   # single Jinja2 template
  analyse_ptbxl.py            # per-dataset script                        (~110)
  analyse_chapman_shaoxing.py # per-dataset script                        (~90)
  _template_analyse.py        # copy-me starting point for dataset #3..64
```

### `_common.py` — the pipeline

```python
def run_analysis(
    slug: str,
    *,
    data_path: Path | None = None,
    output_dir: Path | None = None,
    html: bool = False,
    sampling_rate: int | None = None,
    n_folds: int = 10,
    signal_sample: int | str = 2000,
    signal_stats: bool = True,
    hr_estimate: bool = False,
    max_workers: int = 4,
    seed: int = 42,
    skip_validation: bool = False,
    skip_croissant: bool = False,
    extra_tables: Callable[[pd.DataFrame, DatasetConfig], dict[str, pd.DataFrame]] | None = None,
    extra_sections: list[ReportSection] | None = None,
) -> AnalysisResult: ...
```

Ordered stages — the first four are exactly today's `run_splits()`, reused unchanged:

1. `load_config(slug)` → `resolve_data_path(...)`
2. `get_splitter(slug)` → `load_metadata()` → `get_stratification_labels()`
3. `validate_dataset(...)` (unless `--skip-validation`)
4. `split_dataset(...)` → `export_splits(...)` → `save_croissant(...)`
5. **NEW** metadata stats — `_stats_metadata.compute(df, labels, config)`
6. **NEW** split stats — `_stats_splits.compute(split_result, val_result, config)`
7. **NEW** signal stats — `_stats_signal.compute(...)` (skipped by `--no-signal-stats`)
8. **NEW** `extra_tables(...)` hook → dataset-specific tables merged in
9. **NEW** write `stats/*.csv` + `analysis.json`
10. **NEW** if `--html`: `_report.render(...)` → `report.html`

`AnalysisResult` is a dataclass: `tables: dict[str, pd.DataFrame]`, `summary: dict`,
`split_stats: dict`, `paths: dict[str, Path]`. Making stats return DataFrames (not dicts)
means the CSV writer, the JSON writer, and the HTML renderer all consume one
representation.

### Per-dataset script shape

```python
#!/usr/bin/env python3
"""Dataset analysis for PTB-XL."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_parser, run_analysis, print_summary   # noqa: E402

SLUG = "ptbxl"


def extra_tables(df, config):
    """PTB-XL-specific tables: SCP superclass + subclass breakdown, device, site."""
    from ecgbench.splitting.strategies.ptbxl import SCP_TO_SUPERCLASS, _parse_scp_codes
    ...
    return {"scp_superclass": ..., "scp_subclass": ..., "device_distribution": ...}


def main(argv=None) -> int:
    args = build_parser(SLUG, description=__doc__).parse_args(argv)
    result = run_analysis(SLUG, extra_tables=extra_tables, **vars(args))
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The `sys.path` insert keeps the scripts runnable from a clone with no install step and no
`scripts/__init__.py`. Note it in each script's docstring — it is the one bit of magic.

---

## 4. Statistics catalogue

The concrete deliverable of requirement 1 ("statistical tables which are relevant to data
analysis"). Every table below is written as CSV **and** embedded in `analysis.json`.

### 4.1 Metadata & demographics (`_stats_metadata.py`)

- **`dataset_summary`** — n_records, n_patients, records/patient (mean, median, max),
  leads, duration_s, sampling rates, signal format, source version, license.
- **`metadata_completeness`** — for every column in the source metadata CSV: dtype,
  non-null count, `%` missing, `n_unique`, 3 example values. Catches the silent
  "column exists but is 98% empty" trap.
- **`demographics`** — age: n, mean, std, min, p25, median, p75, max, `n_missing`,
  `n_implausible` (`age <= 0` or `age > 120` — PTB-XL encodes >89 as `300`, so this must
  be reported, not silently clipped); sex: counts + `%` per category, plus `n_missing`;
  height/weight/BMI when present.
- **`records_per_patient`** — distribution table (1, 2, 3, 4–5, 6–10, >10 records) with
  record and patient counts. Drives the case for patient-level grouping.
- **`label_distribution`** — per label: n_records, `%`, n_patients. Label parsing keys off
  `config.label_format` (`single`, `comma_separated`, `dict_string`, `json`) so this is
  generic across datasets.
- **`label_cooccurrence`** — square label×label co-occurrence matrix; only emitted when
  `label_format` is multi-label. Add `label_cardinality` (labels per record: 0, 1, 2, 3+)
  and explicitly count `n_records_with_zero_labels`.
- **Cross-tabs** — label × sex, label × age-decade. Cheap, and the first thing a reviewer
  asks for.

### 4.2 Split / fold quality (`_stats_splits.py`)

- **`fold_counts`** — per fold: n_records, n_patients, n_excluded, original vs clean.
- **`fold_label_distribution`** — fold × label counts, plus each cell's percentage-point
  deviation from the global prior.
- **`stratification_quality`** — per label: max abs deviation across folds, chi-square
  statistic + p-value for uniformity (`scipy` is not a dependency — compute chi-square
  from `numpy` and report the statistic and dof; skip the p-value rather than adding a dep).
- **`patient_leakage`** — any `patient_id` appearing in more than one fold. **An empty
  table is the pass condition**; a non-empty one is a hard failure and must exit non-zero
  (a leaked patient invalidates every benchmark number downstream).
- **`quality_checks`** — per check from `ValidationResult.summary`: records_failed,
  total_issues, `%` of dataset, and the same broken down per fold (so a check failing
  disproportionately in one fold is visible).

### 4.3 Signal statistics (`_stats_signal.py`)

Reuses `validation/engine._load_signal()` and the same `ProcessPoolExecutor` pattern.
Sampling defaults to 2000 records (stratified by label, seeded by `--seed`) because a
full pass over PTB-XL at 500 Hz is minutes of I/O; `--signal-sample all` forces
everything. **The sampled record IDs are written to `analysis.json`** so any number in
the report is reproducible.

Per record, per lead:

- `n_samples`, `sampling_rate`, actual duration; lead count and lead-name consistency
  vs `config.validation.expected_leads`.
- Amplitude: `min`, `max`, `mean`, `std`, `p1`, `p50`, `p99`, `ptp`.
- `n_nan`, `frac_flat` (rolling-window std below eps), `frac_clipped`
  (`|x| >= config.validation.amplitude_range_mv` bound).
- `baseline_drift` — std of a moving-average-detrended signal (`numpy` convolution; no
  `scipy`).
- `powerline_ratio` — FFT energy in 49–51 Hz and 59–61 Hz over total energy
  (`numpy.fft.rfft`). Flags mains interference and hints at the recording region.
- Optional (`--hr-estimate`) — approximate heart rate from a derivative-plus-threshold
  peak detector on lead II. Label it **approximate** everywhere it surfaces; it is a
  data-exploration aid, not a clinical measurement.

Aggregated into:

- **`signal_record_stats`** — one row per sampled record (long-form source for all plots).
- **`signal_lead_stats`** — per lead: mean/std/percentiles of the per-record values.
- **`signal_quality`** — dataset-level aggregates: `%` records with any clipping, any flat
  lead, any NaN; distribution of `baseline_drift` and `powerline_ratio`.

### 4.4 Provenance (in `analysis.json`)

`dataset`, `source_version`, `ecgbench_version`, `generated_at` (UTC ISO-8601), SHA-256 of
the config YAML, the resolved `data_path`, every CLI parameter, `seed`, the sampled record
IDs, and SHA-256 of every emitted CSV — including the fold CSVs, mirroring what
`croissant.py` already does. Two runs with the same inputs must produce byte-identical
fold CSVs and stats CSVs; `generated_at` is the only field allowed to differ.

---

## 5. HTML report (`_report.py` + `_templates/report.html.j2`)

One self-contained file. `plotly.io.to_html(fig, include_plotlyjs="inline", full_html=False)`
for the first figure and `include_plotlyjs=False` for the rest, so the ~3 MB `plotly.js`
bundle is embedded exactly once.

Sections, in order — each renders from a table in §4, so a missing table degrades to a
"not computed (`--no-signal-stats`)" note rather than a crash:

1. **Header** — dataset name, version, license, source link, generation timestamp,
   ECGBench version.
2. **Overview** — KPI tiles (records, patients, leads, duration, rates, excluded records)
   + the `dataset_summary` table.
3. **Metadata completeness** — horizontal bar of `%` missing per column; full table below.
4. **Demographics** — age histogram (implausible values in a visually distinct bin),
   sex pie/bar, age×sex stacked histogram.
5. **Labels** — sorted bar of label frequency (log-scale toggle), co-occurrence heatmap,
   label-cardinality bar.
6. **Splits & folds** — grouped bar of records per fold, fold×label heatmap of deviation
   from prior (diverging colour scale centred at 0), and a prominent
   **PASS/FAIL patient-leakage banner**.
7. **Signal quality** — per-lead amplitude box plots, `baseline_drift` and
   `powerline_ratio` histograms, clipping/flat/NaN summary table.
8. **Signal previews** — 12-lead traces for a seeded random sample (default 3 records),
   one subplot per lead, shared x-axis. Add one deliberately-invalid record when
   validation found any, so the report shows what a rejected record looks like.
9. **Validation** — `quality_checks` table + per-fold breakdown.
10. **Provenance** — the full §4.4 block, including file hashes, in a collapsed `<details>`.
11. **Dataset extras** — anything the script passed via `extra_sections`.

Styling notes: a small inline `<style>` block, no external fonts or CSS (a strict-CSP or
offline viewer must render it identically); a sticky table-of-contents sidebar; tables
wrapped in `overflow-x: auto`; light/dark via `@media (prefers-color-scheme: dark)`.
Before writing any chart code, load the `dataviz` skill for the palette and mark specs —
one consistent categorical palette across all figures, and colour must never be the only
signal (PASS/FAIL gets a text label too).

**Verify in tests:** `report.html` contains no `src="http` / `href="http` for assets, and
`plotly.js` appears exactly once.

---

## 6. Implementation phases

### Phase 1 — Scaffolding
- [ ] Resolve **R1** (tick M1a / M1b / M1c above) before writing code.
- [ ] `pyproject.toml`: add `analysis = ["plotly>=5.18", "jinja2>=3.1"]`; add it to `all`;
      confirm `dev` picks it up via `ecgbench[all]`.
- [ ] If M1a: add `scripts/**` to `[tool.hatch.build.targets.sdist].include`.
- [ ] Create `scripts/` with `_common.py` skeleton: `build_parser()`, `run_analysis()`
      stages 1–4 delegating to the existing engines, `AnalysisResult`, `print_summary()`.
- [ ] Lazy-import `plotly`/`jinja2` inside `_report.py` only, and raise a clear
      "install `ecgbench[analysis]`" error — matching how `croissant.py` handles
      `mlcroissant`.
- [ ] Verify parity: `python scripts/analyse_ptbxl.py --data-path ...` produces an
      `original/`+`clean/` tree byte-identical to `ecgbench splits`. **This is the gate
      for everything after it.**

### Phase 2 — Statistics core
- [ ] `_stats_metadata.py` — §4.1. Generic label parsing over all four `label_format` values.
- [ ] `_stats_splits.py` — §4.2. Leakage check exits non-zero.
- [ ] `_stats_signal.py` — §4.3. Parallel, sampled, seeded, `numpy`-only feature code.
- [ ] CSV + `analysis.json` writers in `_common.py`; stable column order and row sort for
      byte-reproducibility.

### Phase 3 — HTML report
- [ ] Load the `dataviz` skill; fix the palette and mark specs first.
- [ ] `_report.py` — one `make_*_figure()` per §5 section, all returning `go.Figure`.
- [ ] `_templates/report.html.j2` — TOC, sections, inline CSS, dark mode.
- [ ] Single-file check: inline `plotly.js` once, no external asset URLs.

### Phase 4 — Per-dataset scripts
- [ ] `analyse_ptbxl.py` — SCP superclass + subclass tables (reuse `SCP_TO_SUPERCLASS`
      from `splitting/strategies/ptbxl.py`; do not re-declare the mapping),
      `device`/`site`/`nurse` distributions, `recording_date` coverage, and the
      `age == 300` sentinel called out explicitly.
- [ ] `analyse_chapman_shaoxing.py` — condition-code distribution, rhythm breakdown,
      and the `ECGData/` path prefix quirk (see `strategies/chapman.py`).
- [ ] `_template_analyse.py` — commented copy-me file, referenced from
      `ADD_DATASET_TODO.md`.

### Phase 5 — CLI migration (D1)
- [ ] Add `ecgbench analyse` to `cli/_main.py` mirroring the script flags, so the
      installed CLI and the scripts stay reachable from one place.
- [ ] `cli/splits.py`: keep `run_splits()` working, emit a `DeprecationWarning` +
      stderr pointer to `python scripts/analyse_<slug>.py` / `ecgbench analyse`.
- [ ] Update `README.md`, `CLAUDE.md` ("Common Commands", "Adding a New Dataset"),
      `ADD_DATASET_TODO.md` Phase 4, and `ECGBench_architecture/ARCHITECTURE.md`.
- [ ] Leave `cli/upload.py` and `dataset.py` **untouched** — the layout invariants hold.

### Phase 6 — Tests (`pytest`, no network, no real ECG data)
- [ ] `tests/test_analysis_stats.py` — each stats function against `conftest`'s mock
      metadata DataFrames and synthetic signal arrays (`synthetic_signal_bad_nan`,
      `_flat`, `_truncated`, `_amplitude_outlier`). Assert the known-bad arrays land in
      the right quality buckets.
- [ ] `tests/test_analysis_splits.py` — leakage detector catches an injected leak and
      passes on a clean `tmp_splits_dir`; deviation-from-prior maths on a hand-built case.
- [ ] `tests/test_analysis_report.py` — `pytest.importorskip("plotly")`; every section
      present; no external asset URLs; `plotly.js` inlined once.
- [ ] `tests/test_analysis_scripts.py` — run each script end-to-end via `subprocess` over
      the synthetic WFDB dataset with `--no-signal-stats --skip-croissant`; assert exit 0
      and the expected file tree.
- [ ] **Determinism test** — run twice, assert identical SHA-256 for every fold CSV and
      stats CSV (excluding `generated_at`).
- [ ] **Parity test** — `run_splits()` and `run_analysis()` produce identical
      `original/`/`clean/` trees.
- [ ] `ruff check ecgbench/ scripts/ && black ecgbench/ scripts/` — extend both to
      `scripts/` (there is no CI lint job, so this has to be a local habit).

### Phase 7 — Docs & website
- [ ] `--emit-catalogue-sections` prints paste-ready YAML for `docs/_datasets/<slug>.md`:
      a `table` section from `label_distribution`, and a `links` section pointing at the
      published report.
- [ ] Decide how the report reaches the website. `_includes/sections/plot.html` expects a
      **static image** (`image:` → `<img>`), so a Plotly `report.html` cannot use it.
      Cheapest path: commit the report under `docs/assets/reports/<slug>.html` and link it
      from a `links` section. Static PNG export would need `kaleido` — do not add it now.
- [ ] Set `status: needs_review` in the catalogue front matter until numbers are checked,
      then `completed`.

---

## 7. Per-dataset rollout checklist

Once Phases 1–6 land, adding analysis for dataset N is:

- [ ] `cp scripts/_template_analyse.py scripts/analyse_<config-slug>.py`; set `SLUG`.
- [ ] Run with `--no-signal-stats` first — fastest way to surface metadata surprises.
- [ ] Read `metadata_completeness.csv`: any column >50% missing that the config depends on
      is a config bug, not a data quirk.
- [ ] Read `patient_leakage.csv` — **must be empty**.
- [ ] Read `stratification_quality.csv` — investigate any label with max deviation >5 pp.
- [ ] Add dataset-specific tables via `extra_tables()` (label taxonomies, device/site
      fields, sentinel-value handling).
- [ ] Full run with `--html`; read the report top to bottom before publishing anything.
- [ ] Cross-check `records` / `patients` against the catalogue front matter in
      `docs/_datasets/<catalogue-slug>.md`; a mismatch means one of the two is wrong.
- [ ] Add the script to `tests/test_analysis_scripts.py`'s parametrised list.

---

## 8. Gotchas

- **Two slug namespaces.** Scripts, configs, and `@register()` use the **underscored**
  config slug (`ptbxl`); catalogue Markdown files use the **dashed** slug (`ptb-xl`).
  Nothing maps one to the other. Scripts are named for the config slug.
- **`GenericSplitter` fallback is silent.** A dataset with no registered strategy still
  "works" — with possibly wrong stratification. Log the resolved splitter class in the
  report's provenance section so the fallback is never invisible.
- **Sentinel values are not outliers.** PTB-XL's `age == 300` means ">89". Report such
  values in their own column; never mix them into age percentiles.
- **`conftest.py` builds `DatasetConfig` in Python, not from YAML.** Test fixtures can
  drift from the shipped configs, so a green test suite does not prove a real config is
  correct. Smoke-run against real data before declaring a dataset done.
- **Sampling silently changes numbers.** Any signal statistic computed under
  `--signal-sample N` must be labelled with the sample size in both the CSV header
  comment and the report. A number that looks dataset-wide but is not is worse than a
  missing number.
- **`has_predefined_splits` gates predefined folds.** A fully-populated
  `predefined_splits:` block with the flag left `false` is ignored and folds get
  regenerated. Assert the flag's effective value in provenance.
- **`plotly.js` size.** Embed once. Naively calling `to_html(..., include_plotlyjs="inline")`
  per figure produces a 40 MB file.
- **Python-only changes deploy nothing.** No CI test/lint job exists; the Pages and HF
  Space workflows fire only on `docs/**` changes or `v*` tags. Run `pytest`, `ruff`, and
  `black` locally before pushing.
