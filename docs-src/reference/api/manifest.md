# Manifest

A `manifest.json` is written next to every generated split, holding the seed,
the fold count, a SHA-256 per input file, record counts, and a `fold_digest` —
a hash over the whole record-to-fold mapping in canonical order, so two runs
agree if and only if they produced the same partition.

That is what makes the [recipe distribution](../../guides/distribution-policy.md)
of a restricted dataset's splits verifiable: `verify_splits()` compares your
local run against the reference manifest shipped in
`ecgbench/data/manifests/<slug>.json` and names the differing input file on
mismatch.

::: ecgbench.manifest
