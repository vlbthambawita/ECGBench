# ECGBench metadata layer — execution plan for Claude Code

Companion to `ecgbench_metadata_layer_plan.pdf` (design and technology survey, 19 Sep 2026).
That report says *what* and *why*; this file says *how*, in the order Claude Code should do it.
Each phase is one PR-sized unit of work with its own kickoff prompt (Appendix B), verification
commands, and a "done when" list. Phases are sequential; do not start a phase before the
previous one's "done when" holds on `main`.

Repository: `/work/vajira/DL2026/ECGBench` (branch `main`, baseline commit `6ea2c80`).

## 0. Ground rules for every phase

Read before writing anything: `CLAUDE.md`, `ecgbench/catalogue.py`, `ecgbench/config.py`,
`ecgbench/cli/_main.py`, `ecgbench/cli/croissant.py` (the subcommand contract),
`ecgbench/__init__.py` (lazy-import policy), `tests/conftest.py`, `tests/test_catalogue.py`.

- **Conventions.** Python 3.10+ typing (`str | None`), `pathlib.Path` everywhere, dataclasses not
  dicts, line length 100, `ruff check ecgbench/ tests/` must be clean. `black` is *not* enforced
  (36 files already fail it) — do not reformat unrelated files.
- **Dependencies.** Phases 0–5 add **no** runtime dependency. `sqlite3`, `json`, `hashlib`,
  `dataclasses` are standard library. DuckDB appears only in Phase 6 behind an extra.
- **Import policy.** Anything importable from `ecgbench.metadata` must be cheap: no pandas at
  module top level in `model.py`, `identity.py`, `store.py`. New public symbols that need a
  heavy dependency go into `_LAZY_IMPORTS` **and** `__all__` in `ecgbench/__init__.py`.
- **CLI contract.** Every new subcommand module has a public `run_X(...)` (kwargs only, no
  argparse), a private `_cli_run(args)`, and `add_subparser(subparsers)` ending in
  `p.set_defaults(func=_cli_run)`; register in `_main.py`, re-export in `cli/__init__.py`, and
  add the subcommand name to the parametrised `test_subcommand_help_parses` in `tests/test_cli.py`.
- **Docs.** `mkdocs build --strict` parses docstrings with griffe: Google style, no bare
  `x["a"]["b"]` in prose, consistent indentation in `Returns:` blocks. CLI docs live in
  `README.md` between `<!-- --8<-- [start:cli] -->` and `[end:cli]`; never delete those markers.
  Add one page `docs-src/reference/api/metadata.md` containing `::: ecgbench.metadata` and a nav
  entry in `mkdocs.yml`.
- **Do not touch** `ecgbench/splitting/strategies/` or the `@register` registry (being retired),
  do not recreate a repo-root `scripts/` directory, do not write into `docs/` from a build step
  (it is the wheel's `ecgbench/_datasets` source), do not bundle or publish any per-record data.
- **Sources stay the truth.** Markdown front matter and YAML configs are edited by humans; the
  index is derived. Never hand-edit `ecgbench/data/metadata.sqlite` or `metadata.json`.
- **Verify before claiming done.** Run `ruff check ecgbench/ tests/ && pytest -q` and paste the
  summary line into the PR description. Commits end with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## 0b. Facts established while planning (corrections to the report)

- There are **51** configs (`list_available_configs()`), not 53: the directory also holds
  `_template.yaml` and a stray `vajira.code-workspace`.
- `catalogue.get_config()`'s hyphen/underscore normalisation resolves only **5 of 51** configs
  (`ptbxl`, `brugada_huca`, `echonext`, `mimic_iv_ecg`, `mimic_iv_ecg_demo`). Phase 0 is
  therefore mandatory, not cosmetic.
- The mapping is **one-to-one** for all 51 configs (Appendix A). The CLAUDE.md sentence saying
  two Chapman catalogue entries are served by one config is out of date:
  `chapman-shaoxing-arrhythmia.md` describes the PhysioNet `ecg-arrhythmia` release, i.e. the
  `ecg_arrhythmia` config, and `chapman-shaoxing-ecg-database-10-646-patients.md` describes the
  figshare release, i.e. `chapman_shaoxing`. Fix that sentence in Phase 0.
- 13 catalogue entries have no config **by design** (derived layers such as `ptb-xl-plus`,
  `mimic-iv-ecg-ext-icd`, `symile-mimic`, `eye-tracking-…`, `vitaldb-arrhythmia-database`; not
  yet implemented ones; withdrawn `kurias-ecg`). The layer must represent them as
  `implementation_state = "catalogue_only"`, never fail on them.
- 47 datasets have a custom loader in `labels/__init__.py:_custom_loaders()`; 3 are declarative
  (`brugada_huca`, `ecg_arrhythmia`, and one more — confirm with a grep for `labels:` blocks that
  have `columns:`); 1 declares labels unavailable (`mimic_iv_ecg_demo`).

---

## Phase 0 — Identity mapping and decision record (0.5 day)

**Goal.** Every config slug is declared, not guessed, from its catalogue entry.

**Steps.**
1. Add `config_slug: "<slug>"` to the front matter of the 51 files listed in Appendix A,
   directly under `slug:`. Liquid ignores unknown keys; `catalogue.py` keeps them in `raw`.
2. In `ecgbench/catalogue.py`: add `config_slug: str | None = None` to `CatalogueEntry`, populate
   it in `_entry_from_meta`, and rewrite `get_config()` to use it (fall back to the old
   normalisation only when `config_slug` is absent, and log a warning naming the file).
3. In `tests/test_catalogue.py` add `TestConfigSlugMapping`:
   - every `config_slug` names an existing YAML (`list_available_configs()`);
   - every config is claimed by exactly one catalogue entry (one-to-one; a `dict` of
     config → [catalogue slugs] must have all lengths 1);
   - `get_config("mit-bih-arrhythmia-database").slug == "mitdb"` and
     `get_config("chapman-shaoxing-arrhythmia").slug == "ecg_arrhythmia"`.
4. Update the CLAUDE.md paragraph about the two Chapman entries (see 0b) and the Phase 1
   checklist in `ADD_DATASET_TODO.md` (add `config_slug` to the identity fields bullet).
5. Do **not** change `status:` values.

**Verify.** `pytest tests/test_catalogue.py -q`; `python -c "import ecgbench; print(sum(1 for e in ecgbench.list_datasets() if e.config_slug))"` prints `51`.

**Done when.** Tests above pass; `ruff` clean; CLAUDE.md corrected; no other files changed.

---

## Phase 1 — Unified model and pure-Python store (3 days)

**Goal.** One typed record per dataset, merged from catalogue + config with provenance, exported
as `ecgbench/data/metadata.json`, queryable from Python and from `ecgbench list / info / related`.
No SQLite yet.

**Files to create.**
```
ecgbench/metadata/__init__.py
ecgbench/metadata/model.py
ecgbench/metadata/identity.py
ecgbench/metadata/build.py
ecgbench/metadata/store.py
ecgbench/data/metadata.json            # generated, committed (Phase 2 adds the build hook)
ecgbench/data/metadata.schema.json     # hand-written JSON Schema, committed
ecgbench/cli/catalog.py                # list / info / related
tests/test_metadata.py
docs-src/reference/api/metadata.md
```

**model.py** — frozen dataclasses, no heavy imports:
```python
@dataclass(frozen=True)
class Provenance:            source: str; source_path: str; observed_at: str | None = None
@dataclass(frozen=True)
class Fact:                  key: str; value: object; provenance: Provenance
@dataclass(frozen=True)
class SignalMeta:            format: str; leads: int; lead_names: tuple[str, ...] | None;
                             alternate_lead_names: dict[int, tuple[str, ...]] | None;
                             record_lead_layouts: tuple[tuple[str, ...], ...] | None;
                             sampling_rates: tuple[int, ...]; default_sampling_rate: int;
                             duration_seconds: float; units: str; unit_scale: float;
                             zero_padded_identifiers: bool
@dataclass(frozen=True)
class AccessMeta:            access: str; license_text: str | None; license_url: str | None;
                             url: str; download_url: str | None; publish_fold_csvs: bool;
                             no_publish_reason: str
@dataclass(frozen=True)
class SplitMeta:             n_folds: int; predefined_column: str | None;
                             has_patient_id: bool; record_id_column: str | None
@dataclass(frozen=True)
class RelationMeta:          target: str; relation: str; shares_records: bool | None;
                             verified: bool; note: str; derived: bool
@dataclass(frozen=True)
class DatasetMeta:
    dataset_id: str                       # config slug if any, else catalogue slug
    aliases: tuple[str, ...]              # catalogue slug, config slug, display name
    name: str; category: str; status: str
    implementation_state: str             # catalogue_only | config | config_labels | published
    version: str | None
    description: str; paper_title: str | None; paper_doi: str | None; citation: str
    origin_institution: str; origin_country: str | None; search_keywords: str
    records: int | None; patients: int | None
    records_display: str; patients_display: str
    signal: SignalMeta | None; access: AccessMeta; split: SplitMeta | None
    relations: tuple[RelationMeta, ...]
    facts: tuple[Fact, ...]               # every value with its provenance, incl. duplicates
    prose: str                            # concatenated section text, for search only
```
Rules: `implementation_state` is derived — `config_labels` when a config exists and either a
custom loader or a `labels.columns` block exists; `published` additionally when
`publish_fold_csvs` is true (Hub presence is not checked — no network). `records`/`patients`
are parsed from the display strings with a tolerant parser (`"18,869"` → 18869; `"~1,000"`,
`"n/a"`, ranges → `None`); the build logs every unparsed value once.

**identity.py** — `resolve(key) -> str` mapping any alias (either slug, display name,
case-insensitive) to `dataset_id`; raise `UnknownDatasetError` listing close matches
(`difflib.get_close_matches`).

**build.py** — `build_model() -> tuple[DatasetMeta, ...]` from `catalogue._load()` and
`load_config()`; `to_json(model) -> str` (sorted keys, `ensure_ascii=False`, indent 1);
`content_digest(model) -> str` (SHA-256 of `to_json`); `write_json(path)`. Precedence when the
same key has several facts: `manifest > validation_report > config > catalogue`. Validation
errors are collected and raised together (same pattern as `catalogue._load`).

**store.py** — `MetadataStore` with `open_store() -> MetadataStore` (loads bundled JSON;
Phase 2 swaps the backend), `all()`, `get(key)`, `search(query=None, *, leads=None, fs=None,
signal_format=None, access=None, license=None, category=None, state=None, min_records=None,
max_records=None, has_labels=None, has_patient_id=None, published=None) -> list[DatasetMeta]`,
`related(key) -> list[RelationMeta]`. Phase 1 free-text search is case-insensitive substring
over name, keywords, description, institution, country, prose — i.e. a superset of the current
`catalogue.search()`.

**Public API.** `ecgbench/__init__.py`: eager `from .metadata import search as search_metadata,
get as get_metadata, open_store` is acceptable only if importing `ecgbench.metadata` costs no
heavy import — measure with `python -X importtime -c "import ecgbench" 2>&1 | tail -3` before
and after; otherwise route through `_LAZY_IMPORTS`. Keep `catalogue.search()`, `get_dataset()`,
`list_datasets()` working unchanged (they may delegate to the store; behaviour must be a
superset). Add a deprecation note in their docstrings, not a warning yet.

**CLI (`cli/catalog.py`).**
```
ecgbench list   [--state STATE] [--category C] [--format table|json|csv]
ecgbench info   <id-or-alias> [--verbose] [--format table|json]
ecgbench related <id-or-alias> [--format table|json]
```
Table output: plain aligned text via `str.ljust`, no third-party formatter. `--json` must be
valid JSON on stdout with nothing else on stdout (logging goes to stderr — check
`logging.basicConfig` in `_main.py` writes to stderr, which is the default).

**Tests (`tests/test_metadata.py`).**
- every catalogue entry yields exactly one `DatasetMeta`; count == 64;
- both slugs and the display name resolve for all 51 configured datasets;
- `implementation_state` distribution matches expectations (`catalogue_only` == 13);
- a synthetic pair (catalogue says 100 records, a fact from `config` says 101) resolves by
  precedence and keeps both facts;
- `to_json` is stable: building twice gives identical digests;
- committed `metadata.json` digest equals a fresh build (this is the drift guard until the
  Phase 2 hook exists);
- CLI: `main(["list", "--format", "json"])` parses as JSON; `main(["info", "ptb-xl"])` and
  `main(["info", "ptbxl"])` print the same `dataset_id`; unknown id exits non-zero and names a
  close match.

**Docs.** README CLI table gains three rows and a short subsection per command inside the
snippet markers; `docs-src/reference/api/metadata.md`; `mkdocs.yml` nav entry.

**Done when.** Tests pass, `mkdocs build --strict` passes locally, `ruff` clean,
`ecgbench info mit-bih-arrhythmia-database` prints the MIT-BIH record.

---

## Phase 2 — SQLite FTS5 index, build hook, `search` (3 days)

**Goal.** Ranked full-text search plus structured filters over a read-only SQLite file shipped in
the wheel; deterministic rebuild checked by tests and by `ecgbench metadata build --check`.

**Steps.**
1. `build.py`: add `write_sqlite(model, path)` creating the schema in §4.6 of the report
   (`dataset`, `alias`, `fact`, `field` (empty until Phase 3), `relation`, `artefact` (empty
   until Phase 4), `dataset_fts` (FTS5, `tokenize='porter unicode61'`), `meta` with
   `schema_version`, `content_digest`, `built_at`, `ecgbench_version`). Wrap FTS5 creation in
   a probe: `PRAGMA compile_options` contains `ENABLE_FTS5`; if not, write everything except the
   virtual table and set `meta.fts = "none"`.
2. `store.py`: SQLite backend. Open with `mode=ro` URI; copy the bundled file to a temp
   directory first if the package location is read-only *and* SQLite needs a journal (it does
   not for `mode=ro`, but test it in a `tmp_path` chmod 0o555 fixture). Ranked query:
   `SELECT dataset_id, bm25(dataset_fts, 10, 5, 3, 2, 1, 1) ... WHERE dataset_fts MATCH ?`
   joined to `dataset` with the structured predicates. If `meta.fts == "none"` or the runtime
   SQLite lacks FTS5, fall back to the Phase 1 substring path and `warnings.warn` once.
   Pass FTS5 syntax through verbatim; convert `sqlite3.OperationalError` into
   `MetadataQueryError` quoting SQLite's message.
3. Editable-install staleness: `open_store()` compares `meta.content_digest` with a digest of
   the source files' mtimes+sizes cached next to the db (`metadata.sources.json`); when stale
   and the sources are present (repo checkout), rebuild into `ecgbench/data/` and log once.
   In an installed wheel the sources are absent and this branch is skipped.
4. Hatch build hook: `hatch_build.py` at repo root, registered as
   `[tool.hatch.build.hooks.custom]`, calling `ecgbench.metadata.build.build_all()` before the
   wheel/sdist is assembled. Add `pyyaml` to `[build-system].requires`. Confirm the generated
   files are inside `packages = ["ecgbench"]` so no extra force-include is needed. Verify with
   `python -m build --wheel` and `unzip -l dist/*.whl | grep metadata`.
5. `cli/metadata.py`: `ecgbench metadata build [--check] [--output DIR]`. `--check` exits 1 and
   prints a diff summary (added/removed/changed dataset ids) when the committed JSON digest
   differs from a fresh build.
6. `cli/catalog.py`: add `ecgbench search [QUERY] [filters…] [--limit N] [--format …]` using
   `MetadataStore.search`; document FTS5 syntax examples in README.
7. Decide whether `metadata.sqlite` is committed. Recommendation: **commit `metadata.json`,
   generate `metadata.sqlite` at build/first use, add it to `.gitignore`.** Reason: SQLite bytes
   are not deterministic, so a committed binary would churn on every rebuild.

**Tests.**
- rebuild twice → same `content_digest`; different `sqlite` bytes are *not* compared;
- FTS: `search("ptb")` ranks `ptbxl` first; `search("atrial fib*")` includes `afdb` and
  `ltafdb`; `search("holter NOT paediatric")` excludes `picsdb`;
- structured: `search(leads=12, signal_format="wfdb", access="open")` returns only matching
  configs (cross-check against `load_config` in the test);
- fallback: monkeypatch the probe to report no FTS5 and assert the substring path returns a
  superset of the FTS hits for a plain word;
- `main(["metadata", "build", "--check"])` returns 0 on a clean tree and 1 after a
  `monkeypatch`ed catalogue change;
- read-only location test (chmod 0o555 tmp dir) still opens.

**Done when.** `python -m build` produces a wheel containing `ecgbench/data/metadata.json` and
`.sqlite`; `pip install dist/*.whl` into a fresh venv, then `ecgbench search "atrial"` works
offline; tests and ruff clean.

---

## Phase 3 — Field inventory (4–6 days, split into per-dataset PRs)

**Goal.** Each dataset's label/metadata columns are declared data: name, type, description,
unit, vocabulary, nullable; surfaced by `ecgbench fields <id>` and indexed for search.

**Steps.**
1. `model.py`: add `FieldMeta(name, type, description, unit=None, vocabulary=None,
   nullable=True, example=None, source="labels")` where `type` is a Frictionless Table Schema
   type (`string`, `integer`, `number`, `boolean`, `array`, `object`, `date`, `datetime`) plus
   the item-type suffix convention `array[string]`.
2. `ecgbench/labels/_fields.py`: `Field` dataclass (same shape) and `fields_for(config) ->
   tuple[Field, ...]` that returns the custom module's `FIELDS` when present, else builds them
   from `config.labels.columns` (type `string`, description from an optional new YAML block
   `labels.fields: {col: {type, description, unit, vocabulary}}` — add it to `LabelConfig`
   and `_parse_labels` in `config.py` and to `_template.yaml`).
3. Add `FIELDS` to the 47 custom label modules. Work from the existing docstrings, which
   already list columns; start with `ptbxl`, `mitdb`, `afdb`, `challenge2020/2021`,
   `mimic_iv_ecg`, then the rest alphabetically, ~8 modules per PR.
4. Consistency test, parametrised over every dataset that has a conftest fixture
   (`tmp_ptbxl_label_data`, `tmp_labels_data`, …): `set(f.name for f in FIELDS) ==
   set(load_labels(...).columns)`. For modules without a fixture, add a minimal synthetic
   fixture in the same PR — do not skip; the point is that declarations cannot rot.
5. Build: populate the `field` table and append field names + descriptions to the
   `dataset_fts.field_text` column, so `ecgbench search "recorder"` finds `mitdb`.
6. `cli/catalog.py`: `ecgbench fields <id> [--format table|json|frictionless]`; the
   `frictionless` format emits a Table Schema JSON (`{"fields": [...], "primaryKey": ...}`).
7. `ADD_DATASET_TODO.md` Phase 2b gains a checklist item: declare `FIELDS` (or `labels.fields`)
   and run the consistency test.

**Done when.** All 50 label-bearing datasets have declared fields; the consistency test covers
all of them; `ecgbench fields ptbxl` lists `superclasses` with the NORM/MI/STTC/CD/HYP vocabulary.

---

## Phase 4 — Artefact snapshots (1.5 days)

**Goal.** The trustworthy, recomputed numbers (fold digests, record counts original/clean,
per-check failure counts, ECGBench version and date) reach `info --verbose` with provenance.

**Steps.**
1. Define `ecgbench/data/snapshots/<slug>.json` (schema in `metadata.schema.json`): `dataset`,
   `dataset_version`, `ecgbench_version`, `created`, `records {original, clean}`,
   `fold_digest {original, clean}`, `n_folds`, `random_state`, `quality_checks: [{check,
   records_failed}]`, `inputs: {file: sha256}`. This is a *subset* of `manifest.json` plus the
   summary block of `validation_report.json`; it never contains record ids.
2. `ecgbench/manifest.py` (or a new `snapshot.py`): `write_snapshot(output_dir, dest)` that
   derives the file from an existing `output/<slug>/` tree. Expose as
   `ecgbench metadata snapshot --dataset <slug> --output-dir output/<slug>/`. Seed the 4 shipped
   manifests (`ecgbench/data/manifests/*.json`) into snapshots as well.
3. Build: load every snapshot into `artefact` rows and into `facts` with
   `source="manifest"` / `"validation_report"`, which by precedence override catalogue counts.
4. `info --verbose` prints every fact grouped by key with source and date; `info` (plain)
   prints the winning value and a `†` marker when sources disagree.
5. Coordinate with the per-dataset-scripts refactor (memory note `per-dataset-scripts-replace-
   splits`): the scripts' last step should call `write_snapshot`. Do not block on that refactor;
   the CLI command above is sufficient for now.

**Done when.** Snapshots exist for all datasets with an `output/<slug>/` tree on this machine
(50), tests cover precedence and the id-free guarantee (assert no snapshot key looks like a
record list), `ecgbench info ptbxl --verbose` shows 49 excluded records from the validation
snapshot next to the catalogue's 21,799.

---

## Phase 5 — Exports: Croissant 1.1 collection, schema.org, website JSON (2 days)

**Steps.**
1. `ecgbench/metadata/export.py`: `to_croissant_collection(model) -> dict` (JSON-LD, one
   `sc:Dataset` per dataset with `distribution` omitted for unpublished ones, PROV-O
   `wasDerivedFrom` from snapshot inputs, ODRL/DUO usage terms from `publish_fold_csvs`;
   emit 1.1 constructs behind `include_1_1=True`), `to_schema_org(dataset) -> dict`,
   `to_json(model)`.
2. `ecgbench metadata export --croissant PATH | --schema-org DIR | --json PATH`. Validation
   with `mlcroissant` is optional (`--validate`, `pytest.importorskip` in tests, same pattern
   as `test_croissant.py`).
3. Website: `docs/_layouts/dataset.html` embeds the schema.org JSON-LD. Since Jekyll cannot
   import Python, generate `docs/_data/metadata.json` **by the export command, committed**,
   and read it in Liquid (`site.data.metadata[page.slug]`). Add a test that the committed copy
   matches a fresh export (same pattern as the Phase 1 drift guard). This is the one write into
   `docs/`, and it is a committed data file, not a build-time side effect.
4. Optional: switch `docs/index.html`'s search to the same JSON (keywords + field names), keeping
   the current substring behaviour as a floor.

**Done when.** `mlcroissant` validates the collection file (in `.[dev]`), Google's Rich
Results test accepts one dataset page's JSON-LD (manual check, paste result in PR), drift test
passes.

---

## Phase 6 — Record-level extra (2 days)

**Steps.**
1. `pyproject.toml`: `analytics = ["duckdb>=1.1.0"]`, include in `all`.
2. `ecgbench/metadata/records.py`: `load_records(dataset, data_path=None, version="clean",
   split=None, fold_numbers=None) -> pd.DataFrame` joining `load_labels()` with fold CSVs
   (local tree or Hub via the existing `ECGDataset._load_from_hf` logic — factor that method's
   fetch into a reusable function rather than duplicating it; respect `SplitsNotPublishedError`).
3. `ecgbench records <id> --data-path P [--version clean] [--split train] [--sql "…"]
   [--format table|json|csv|parquet]`. `--sql` requires duckdb: register the DataFrame as view
   `records` and run the query read-only. Without duckdb, `--sql` errors with the install hint.
4. Hub read path: with duckdb, `--hub` reads `hf://datasets/vlbthambawita/ECGBench/<slug>/
   <version>/folds.csv` directly for published datasets (no labels — identifiers only).
5. Tests use the existing `tmp_wfdb_signal_dataset` fixtures; duckdb tests
   `pytest.importorskip("duckdb")`.

**Done when.** `ecgbench records ptbxl --data-path … --sql "select sex, count(*) from records
where fold = 9 group by 1"` reproduces the fold-9 sex counts on a real PTB-XL copy (manual
smoke run; paste output in PR).

---

## Phase 7 — MCP server (optional, 1 day)

`mcp = ["mcp>=1.0"]` extra; `ecgbench mcp` serving tools `search_datasets`, `get_dataset`,
`list_fields`, `related_datasets` over stdio, each a thin wrapper on `MetadataStore`. Manual
check: add to Claude Code's MCP config and ask it to list 2-lead Holter datasets.

---

## Appendix A — `config_slug` mapping to add in Phase 0 (51 rows, one-to-one)

| Catalogue file (`docs/_datasets/`) | `config_slug` |
|---|---|
| apnea-ecg-database.md | apnea_ecg |
| bidmc-congestive-heart-failure-database.md | chfdb |
| brno-university-of-technology-ecg-quality-database-but-qdb.md | butqdb |
| brugada-huca.md | brugada_huca |
| chapman-shaoxing-arrhythmia.md | ecg_arrhythmia |
| chapman-shaoxing-ecg-database-10-646-patients.md | chapman_shaoxing |
| cipa-ecg-validation-study.md | ecgcipa |
| code-15-pct-telehealth-network-of-minas-gerais-15-pct-subset.md | code15 |
| code-test-827-record-hold-out-test-set.md | code_test |
| cpsc-2018-china-physiological-signal-challenge-2018.md | cpsc_2018 |
| ecg-capable-smartwatches-dataset.md | ecg_capable_smartwatches |
| ecg-effects-of-dofetilide-moxifloxacin-and-combinations-ecgdmmld.md | ecgdmmld |
| ecg-effects-of-ranolazine-dofetilide-verapamil-quinidine-ecgrdvq.md | ecgrdvq |
| ecg-id-database.md | ecgiddb |
| echonext.md | echonext |
| edgar-experimental-data-and-geometric-analysis-repository.md | edgar |
| european-st-t-database-edb.md | edb |
| ikem-dataset-institute-for-clinical-and-experimental-medicine-prague.md | ikem |
| leipzig-heart-center-ecg-database.md | leipzig_heart_center_ecg |
| lobachevsky-university-ecg-database-ludb.md | ludb |
| long-term-af-database-ltafdb.md | ltafdb |
| long-term-st-database-ltstdb.md | ltstdb |
| medalcare-xl-synthetic-12-lead-ecgs-from-simulations.md | medalcare_xl |
| mhd-effect-on-12-lead-ecgs-in-mri-scanners.md | mhd_effect_ecg_mri |
| mimic-iv-ecg.md | mimic_iv_ecg |
| mimic-iv-ecg-demo.md | mimic_iv_ecg_demo |
| mit-bih-arrhythmia-database.md | mitdb |
| mit-bih-atrial-fibrillation-database.md | afdb |
| mit-bih-normal-sinus-rhythm-database.md | nsrdb |
| mit-bih-st-change-database.md | stdb |
| mit-bih-supraventricular-arrhythmia-database.md | svdb |
| ningbo-first-hospital-ecg-database-idiopathic-ventricular-arrhythmia.md | ningbo_iva |
| norwegian-endurance-athlete-ecg-database.md | norwegian_athlete_ecg |
| physionet-cinc-challenge-2017-af-classification.md | challenge2017 |
| physionet-cinc-challenge-2020.md | challenge2020 |
| physionet-cinc-challenge-2021.md | challenge2021 |
| post-ictal-heart-rate-oscillations-in-partial-epilepsy.md | szdb |
| preterm-infant-cardio-respiratory-signals-database-picsdb.md | picsdb |
| ptb-diagnostic-ecg-database.md | ptbdb |
| ptb-xl.md | ptbxl |
| qt-database-qtdb.md | qtdb |
| sami-trop-chagas-cardiomyopathy-cohort.md | sami_trop |
| shandong-provincial-hospital-ecg-database-sphdb.md | sph |
| shdb-af-saitama-holter-database-atrial-fibrillation.md | shdb_af |
| st-petersburg-incart-12-lead-arrhythmia-database.md | incartdb |
| st-vincent-s-ucd-sleep-apnea-database-ucddb.md | ucddb |
| staff-iii-database.md | staffiii |
| sudden-cardiac-death-holter-database.md | sddb |
| toliet-thigh-based-ecg-toilet-seat.md | tollet |
| wilson-central-terminal-ecg-database.md | wctecgdb |
| zzu-pecg-zhengzhou-university-pediatric-ecg-database.md | zzu_pecg |

Verify each row by comparing the front matter `source_url` with the config `url` before adding it
(they agree for every row above except where the catalogue points at a landing page; `tollet`'s
catalogue slug is misspelt `toliet` — keep the filename, do not rename it, the site links to it).

Catalogue-only entries (no `config_slug`, expected `implementation_state = catalogue_only`):
`ptb-xl-plus`, `mimic-iv-ecg-ext-icd`, `symile-mimic`, `eye-tracking-dataset-for-12-lead-ecg-interpretation`,
`vitaldb-arrhythmia-database`, `kurias-ecg`, `code-full-dataset-2-3m-records`,
`harvard-emory-ecg-database-heedb`, `nightingale-bwh-emergency-dept-ecg-dataset`,
`nightingale-ntuh-cardiac-arrest-ecg-dataset`, `gu-ecg-gazi-university-ptca-induced-ischaemia`,
`icentia11k-single-lead-continuous-ecg`, `mimic-iii-waveform-database-matched-subset`.

## Appendix B — Kickoff prompts (one per Claude Code session)

Each prompt assumes the session starts in `/work/vajira/DL2026/ECGBench` on an up-to-date `main`.

**Phase 0.**
> Read `/work/vajira/DL2026/ECGBench_reports/ecgbench_metadata_layer_execution_plan.md`,
> sections 0, 0b and Phase 0, and CLAUDE.md. Execute Phase 0 exactly: add `config_slug` to the
> 51 catalogue files per Appendix A (verify each against the config `url` first), extend
> `CatalogueEntry` and `get_config`, add `TestConfigSlugMapping`, correct the Chapman sentence in
> CLAUDE.md and the Phase 1 checklist in ADD_DATASET_TODO.md. Run ruff and pytest. Create a
> branch `metadata-layer/phase-0`, commit, and open a PR describing what changed and the test
> summary. Do not start Phase 1.

**Phase 1.**
> Read the execution plan sections 0 and Phase 1, the design report §4 (PDF alongside it), and
> the files listed under "Read before writing anything". Implement `ecgbench/metadata/`
> (`model.py`, `identity.py`, `build.py`, `store.py`), generate and commit
> `ecgbench/data/metadata.json` and `metadata.schema.json`, add `ecgbench list/info/related`
> following the CLI contract, keep `catalogue.search/get_dataset/list_datasets` behaviour a
> superset of today, add `tests/test_metadata.py`, README CLI rows inside the snippet markers,
> and the mkdocs API page. Measure `import ecgbench` time before and after. Run ruff, pytest,
> and `mkdocs build --strict`. Branch `metadata-layer/phase-1`, PR.

**Phase 2.**
> Read the execution plan Phase 2 and the report §4.6–4.8. Add the SQLite FTS5 backend with the
> FTS5 probe and substring fallback, deterministic content digest, staleness rebuild for
> editable installs, the hatchling custom build hook, `ecgbench metadata build --check`, and
> `ecgbench search`. Decide per step 7 (commit JSON, gitignore sqlite). Prove the wheel contains
> the index by building it and installing into a fresh venv. Branch `metadata-layer/phase-2`.

**Phase 3 (repeat per batch).**
> Read the execution plan Phase 3. Add `FieldMeta`/`Field`, `labels/_fields.py`, the
> `labels.fields` YAML block, `ecgbench fields`, and `FIELDS` declarations for these modules:
> <list of ~8>. For each, derive the declaration from the module docstring and the columns the
> loader returns; add a synthetic fixture where none exists; the consistency test must cover
> every module in this batch. Branch `metadata-layer/phase-3-<n>`.

**Phase 4.**
> Read the execution plan Phase 4 and the memory notes on the splits retirement. Define the
> snapshot schema, implement `write_snapshot` and `ecgbench metadata snapshot`, generate
> snapshots for every `output/<slug>/` present locally plus the 4 shipped manifests, feed them
> into the build with provenance, and extend `info --verbose`. Assert snapshots hold no record
> identifiers. Branch `metadata-layer/phase-4`.

**Phase 5.**
> Read the execution plan Phase 5 and `ecgbench/croissant.py`. Implement `metadata/export.py`
> and `ecgbench metadata export`, the committed `docs/_data/metadata.json` with its drift test,
> and the schema.org JSON-LD block in `docs/_layouts/dataset.html`. Validate with mlcroissant
> where installed. Branch `metadata-layer/phase-5`.

**Phase 6.**
> Read the execution plan Phase 6 and `ECGDataset._load_from_hf` in `ecgbench/dataset.py`.
> Add the `analytics` extra, `metadata/records.py` (factoring the Hub fetch out of
> `_load_from_hf` without changing its behaviour), and `ecgbench records` with `--sql` via
> DuckDB. Tests use existing fixtures and `importorskip`. Branch `metadata-layer/phase-6`.

## Appendix C — `metadata.json` shape (one entry)

```json
{
  "schema_version": 1,
  "content_digest": "sha256:…",
  "built_at": "2026-09-19T00:00:00+00:00",
  "datasets": [
    {
      "dataset_id": "mitdb",
      "aliases": ["mit-bih-arrhythmia-database", "mitdb", "MIT-BIH Arrhythmia Database"],
      "name": "MIT-BIH Arrhythmia Database",
      "category": "two-lead",
      "status": "not_started",
      "implementation_state": "published",
      "version": "1.0.0",
      "records": 48, "records_display": "48",
      "patients": 47, "patients_display": "47",
      "signal": {"format": "wfdb", "leads": 2, "lead_names": ["MLII", "V1"],
                 "record_lead_layouts": [["MLII","V1"],["V5","MLII"], "…"],
                 "sampling_rates": [360], "default_sampling_rate": 360,
                 "duration_seconds": 1805.0, "units": "mV", "unit_scale": 1.0,
                 "zero_padded_identifiers": false},
      "access": {"access": "open", "license_text": "ODC-By 1.0", "license_url": "…",
                 "url": "https://physionet.org/content/mitdb/1.0.0/", "download_url": "…",
                 "publish_fold_csvs": true, "no_publish_reason": ""},
      "split": {"n_folds": 10, "predefined_column": null, "has_patient_id": true,
                "record_id_column": "record_name"},
      "fields": [{"name": "recorder", "type": "string",
                  "description": "Del Mar Avionics recorder id", "nullable": true}],
      "relations": [{"target": "mit-bih-supraventricular-arrhythmia-database",
                     "relation": "sibling_release", "shares_records": false,
                     "verified": false, "note": "…", "derived": false}],
      "facts": [{"key": "records", "value": 48, "source": "catalogue",
                 "source_path": "docs/_datasets/mit-bih-arrhythmia-database.md"},
                {"key": "records", "value": 48, "source": "manifest",
                 "source_path": "ecgbench/data/snapshots/mitdb.json",
                 "observed_at": "2026-08-01T10:00:00+00:00"}]
    }
  ]
}
```
