# Finding datasets, and using ECGBench from an agent

Everything ECGBench knows about a dataset — the catalogue page, the YAML
config, the label columns a loader returns, the record counts a split run
recomputed — is merged into one record per dataset and kept searchable. This
page is the tour of that layer: how to ask it questions from the shell and
from Python, how to get at the records behind a dataset, and how to hand the
whole thing to an agent over the Model Context Protocol.

None of it needs a data path or a network connection. The layer is compiled
into the package and reads only the bundled `metadata.json` and the SQLite
full-text index beside it; `import ecgbench` loads it in a few milliseconds.

## One record per dataset, any name for it

```bash
ecgbench info ptbxl
ecgbench info ptb-xl          # the same record
ecgbench info "PTB-XL"        # and again
```

Every dataset has one id — the config slug where a config exists, such as
`ptbxl` or `mitdb`, else the catalogue slug — and a set of aliases: the dashed
catalogue slug, the display name, the config's own name. All of them resolve,
case-insensitively, and an unknown key fails with the closest matches named.

The record says which of four **implementation states** a dataset is in,
derived from what actually exists rather than declared:

| State | Meaning | Count |
|---|---|---|
| `catalogue_only` | surveyed, no config; nothing to validate or load | 13 |
| `config` | a config, but labels are unavailable | 1 |
| `config_labels` | labels, but the fold CSVs are withheld — see [distribution policy](distribution-policy.md) | 4 |
| `published` | labels and fold CSVs on the Hub | 46 |

`ecgbench info --verbose` also shows where each value came from. When the
catalogue and the config disagree the config wins, except for the display
name; when a split run has been snapshotted, its recomputed record count
outranks both, and the catalogue's figure is shown beside it.

## Searching

```bash
ecgbench search holter --leads 2
ecgbench search '"atrial fib*"' --access open --labels
ecgbench search paediatric NOT simulated
ecgbench list --state published --format json
```

The query is [SQLite FTS5 syntax](https://www.sqlite.org/fts5.html) verbatim,
ranked by `bm25()` over the name, keywords, description, institution, prose
and the declared field names, with a small push-down for datasets that are
catalogue-only so a derived layer's shorter page does not outrank its parent.
Two consequences of "verbatim": a hyphenated term must be quoted (`"ptb-xl"`,
because bare `ptb-xl` is a column reference to FTS5), and `NOT`, `OR` and
prefix `*` work as in FTS5. The structured filters — `--leads`, `--fs`,
`--signal-format`, `--access`, `--license`, `--category`, `--state`,
`--min-records`, `--max-records`, `--labels`, `--patient-id`, `--published` —
narrow the candidates before ranking, and work without a query at all.

Field names are indexed too, so `ecgbench search recorder` finds the MIT-BIH
databases whose header exposes the Holter recorder model.

From Python the same calls are one import away:

```python
import ecgbench

meta = ecgbench.get_metadata("mit-bih-arrhythmia-database")   # DatasetMeta
meta.signal.leads, meta.access.license_text, meta.implementation_state

for m in ecgbench.search_metadata("holter", leads=2, access="open"):
    print(m.dataset_id, m.records_display)
```

## What a loader returns, and what leaks

```bash
ecgbench fields ptbxl                      # superclasses  array[string]  NORM, MI, STTC, CD, HYP …
ecgbench fields mitdb --format frictionless
ecgbench related ptb-xl
```

`fields` lists the columns `load_labels()` returns for a dataset as declared
data — Frictionless type, unit, closed vocabulary, and a description that
spells out the sentinels (PTB-XL's `age` of 300 meaning over 89). A test pins
every declaration to the loader's real output, so the list is not a guess.

`related` prints the relationships declared in the catalogue, both directions,
with the one flag that matters for a benchmark: `shares_records`. When it is
true the two datasets hold the same recordings — QTDB's `sel1*` records are
excerpts of MIT-BIH Arrhythmia, PTB-XL is a subset of Challenge 2020 — and
training on one while evaluating on the other contaminates the test set.

## The records behind a dataset

The layer describes datasets; `ecgbench records` is the one command that
returns rows. It joins a dataset's fold table — the record ids, folds and
default split ECGBench published — with the labels read from your local copy
of the source dataset, and optionally runs SQL over the result.

```bash
ecgbench records ptbxl --data-path /data/ptb-xl --split val --format csv
ecgbench records ptbxl --data-path /data/ptb-xl \
    --sql "select sex, count(*) as n from records where fold = 9 group by 1"
ecgbench records mimic_iv_ecg --splits-dir output/mimic_iv_ecg --no-labels
ecgbench records ptbxl --hub --sql "select fold, count(*) from records group by 1"
```

The fold CSVs come from the Hub by default. For a dataset whose fold CSVs are
withheld, `--splits-dir` points at the `output/<slug>/` tree that
`ecgbench splits` wrote; the Hub path refuses such a dataset before any
download, quoting the command that regenerates the identical partition.
`--sql` needs the `analytics` extra (DuckDB) and registers the frame as the
view `records` on a connection with file-system access disabled, so a query can
read the view and nothing else. `--hub` streams a published `folds.csv` in
place through DuckDB's `httpfs`, identifiers only, with no labels and no
download.

```python
df = ecgbench.load_records("ptbxl", data_path="/data/ptb-xl", split="val")
ecgbench.query_records(df, "select sex, count(*) as n from records group by 1")
```

## Handing it to an agent

The same store is available to any MCP client as five tools:

```bash
pip install ecgbench[mcp]
claude mcp add ecgbench -- ecgbench mcp
```

| Tool | Returns |
|---|---|
| `search_datasets` | ranked summaries; the FTS5 query plus every filter above |
| `list_datasets` | every dataset, optionally by state or category |
| `get_dataset` | the full record for any id, slug or display name |
| `list_fields` | the declared label columns |
| `related_datasets` | the leakage edges, both directions |

Asked for "two-lead Holter datasets", an agent calls `search_datasets` with
`query="holter"` and `leads=2` and gets the ten long-term MIT-BIH-family sets
back, best match first. A bad query or an unknown dataset comes back as a tool
error carrying the same message the CLI prints, close matches included, so the
agent can correct itself. The server reads the bundled index only — no
records, no signals, no network — so it is safe to register for any project
without thinking about data paths or credentials. Any other client registers
the command `ecgbench` with arguments `["mcp"]` over stdio.

## Exports for the web and for Croissant

```bash
ecgbench metadata export --croissant ecgbench-croissant.jsonld --validate
ecgbench metadata export --schema-org /tmp/schema-org/
```

The same model is rendered as a `schema.org/Dataset` block per dataset, which
every catalogue page on this site embeds as JSON-LD, and as one Croissant 1.1
`DataCatalog` whose members name the published fold CSVs with their SHA-256 and
carry an ODRL prohibition on distribution where the fold CSVs are withheld.
Two builds of one model are byte-identical.

## Keeping it fresh

The sources are human-edited: the catalogue front matter in
`docs/_datasets/*.md`, the YAML configs, the `FIELDS` tuple in each label
module, and the id-free snapshots a split run leaves behind. Everything else is
derived and never hand-edited.

```bash
ecgbench metadata snapshot --dataset <slug>   # after a split run
ecgbench metadata build                       # metadata.json, metadata.sqlite, the website copy
ecgbench metadata build --check               # exit 1 on drift, naming the dataset ids
```

In a source checkout the index also refreshes itself whenever a source file
changes, so `ecgbench search` never runs against a stale index there. The
per-dataset steps are in the [adding-a-dataset checklist](adding-a-dataset.md);
the module-by-module reference is under [API → Metadata](../reference/api/metadata.md).
