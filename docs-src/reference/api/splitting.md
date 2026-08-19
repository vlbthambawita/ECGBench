# Splitting

Deterministic 10-fold splits with patient-level grouping. Folds are 1-indexed,
and the default split mapping is derived from the fold count rather than fixed:
`train=1..n-2`, `val=n-1`, `test=n`.

## Engine

::: ecgbench.splitting.engine

## Base classes

::: ecgbench.splitting.base

## Registry

Splitters register themselves with `@register("<config-slug>")`. A strategy
module must also be imported in `ecgbench/splitting/strategies/__init__.py`,
otherwise the decorator never runs and lookup silently falls back to
`GenericSplitter`.

::: ecgbench.splitting.registry

## Export

Fold CSVs carry minimal columns only — record ID, patient ID, signal paths,
`fold`, `default_split`, plus `is_valid`/`quality_issues` in `original/`. Full
metadata stays in the dataset's own CSV.

::: ecgbench.splitting.export
