# Metadata

`ecgbench.metadata` — the catalogue and the configs merged into one typed record
per dataset, `DatasetMeta`, with every sourced value kept as a `Fact` carrying
its provenance. Any alias resolves: `ptb-xl`, `ptbxl` and `PTB-XL` are the same
record, and `mit-bih-arrhythmia-database` is `mitdb`.

The merged model is compiled by `build` from `docs/_datasets/*.md` and
`ecgbench/data/configs/*.yaml`, exported to `ecgbench/data/metadata.json`
(committed, and pinned to a fresh build by a test), and read back by
`MetadataStore`. Importing the package costs nothing beyond the standard
library; the JSON is parsed on first use.

```python
from ecgbench import metadata

meta = metadata.get("mit-bih-arrhythmia-database")   # same as get("mitdb")
meta.signal.leads, meta.access.license_text, meta.implementation_state

for m in metadata.search("holter", leads=2, access="open"):
    print(m.dataset_id, m.records_display)

metadata.related("ptbxl")   # leakage edges, both directions
```

The same views are available on the command line as `ecgbench list`,
`ecgbench info` and `ecgbench related` — see [CLI](../cli.md).

`ecgbench.metadata.mcp_server` exposes the same store to agents as five
Model Context Protocol tools (`ecgbench mcp`); the tool functions are plain
Python on `ToolFunctions`, the SDK is needed only to serve them.

`ecgbench.metadata.records` is the one module here that returns *records*
rather than dataset descriptions: `load_records` joins a dataset's fold table
with its labels, `query_records` runs DuckDB SQL over the result. It needs
pandas and is not imported with the package; reach it as
`ecgbench.load_records` / `ecgbench.query_records` or `ecgbench records`.

## Model

::: ecgbench.metadata.model

## Store

::: ecgbench.metadata.store

## Identity

::: ecgbench.metadata.identity

## Build

::: ecgbench.metadata.build

## Artefact snapshots

::: ecgbench.metadata.snapshot

## Exports

::: ecgbench.metadata.export

## Records

::: ecgbench.metadata.records

## MCP server

::: ecgbench.metadata.mcp_server
