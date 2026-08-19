# Distribution policy

Not every dataset's fold CSVs are published to the public HuggingFace repository.
Where they are not, ECGBench distributes the split as a *recipe* instead, and
`ecgbench/manifest.py` is what makes that reproducible partition trustworthy.

The rule is enforced in both directions rather than left to whoever runs the
command: `cli/upload.py` raises `PermissionError` before any network call, and
`ECGDataset._load_from_hf` raises `SplitsNotPublishedError` — quoting the config's
`no_publish_reason`, which must therefore contain the regeneration command —
instead of letting the user hit a bare 404.

--8<-- "README.md:restricted"

## Adding another restricted dataset

See [Phase 7 of the dataset checklist](adding-a-dataset.md), which tabulates
what may and may not be published for a credentialed or restricted source.
