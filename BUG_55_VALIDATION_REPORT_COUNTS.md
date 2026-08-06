# Bug #55 — `validation_report.json` miscounts quality-check failures

**Issue:** https://github.com/vlbthambawita/ECGBench/issues/55
**Status:** open, not fixed. This document is the analysis and the fix procedure; no code
has been changed.
**Affected artefact:** `output/<slug>/validation_report.json` → `quality_checks[]`
(and its upstream, `ValidationResult.summary`).
**Blast radius:** every dataset ever validated. 14 of the 23 reports currently in
`output/` carry wrong numbers; 9 of them also carry a fabricated check name.

---

## 1. Summary

`ValidationResult.summary` is documented as `check_name -> failed_count`
(`ecgbench/validation/engine.py:43`) and is consumed as such by
`generate_report()`. It is neither:

1. **It counts issues, not records.** Lead-level checks emit one issue string *per
   affected lead*, so a single record with 7 bad leads adds 7 to the count. For
   `cpsc_2018`, `amplitude_outlier` is reported as `records_failed: 696` when only
   **99 records** actually failed that check (696 is the lead count).
2. **It derives the check name by string surgery that mangles two names.**
   `issue.split(":")[0].split("_lead_")[0]` turns `missing_lead_3` into `"missing"` and
   `truncated:4500_vs_5000` into `"truncated"` — neither is a registered check. Since
   `_CHECK_DESCRIPTIONS` in `report.py` keys on the real names, these entries render
   with `"description": ""`, e.g. `{"check": "missing", "description": "",
   "records_failed": 32}` in `output/cpsc_2018/validation_report.json`.
3. **`records_failed` and `total_issues` are always equal**, which makes the report
   look internally consistent while both fields hold the same wrong number. They are
   computed two different ways from the same mangled key, and both ways land on the
   issue count.

The `excluded_records` list, the `original`/`clean` totals, `is_valid`,
`quality_issues`, `folds.csv`, `manifest.json` and the Croissant hashes are all
**correct** — the bug is confined to the aggregation step.

---

## 2. Where the bug is

### 2a. `ecgbench/validation/engine.py:315-320` — the aggregation

```python
# Compute summary
summary: dict[str, int] = {}
for v in validations:
    for issue in v.issues:
        check_name = issue.split(":")[0].split("_lead_")[0]
        summary[check_name] = summary.get(check_name, 0) + 1
```

Two defects in five lines: the inner loop is per *issue* with no per-record
de-duplication, and `check_name` is reverse-engineered from the issue string instead of
being carried alongside it.

### 2b. `ecgbench/validation/report.py:46-59` — the consumer

```python
quality_checks = []
for check_name, failed_count in sorted(result.summary.items()):
    total_issues = sum(
        len([i for i in v.issues if i.startswith(check_name)])
        for v in result.record_validations
    )
    quality_checks.append({
        "check": check_name,
        "description": _CHECK_DESCRIPTIONS.get(check_name, ""),
        "records_failed": failed_count,   # actually an issue count
        "total_issues": total_issues,     # the same issue count, recomputed
    })
```

`check_name` here is already the mangled key, so the `startswith` filter re-selects
exactly the issues that produced `failed_count`. Hence `records_failed == total_issues`
in every report on disk. (`"flat_line"` would also swallow any `flat_line_error:*`
issues, but no dataset has produced one.)

### 2c. Why the names mangle — the issue-string grammar (`validation/checks.py`)

The check functions use four incompatible naming conventions. There is no single place
that maps an issue string back to its check:

| Check (registry key) | Issue string emitted | One per | `split(":")[0].split("_lead_")[0]` gives |
|---|---|---|---|
| `missing_leads` | `missing_lead_{i}` | lead | `missing` ❌ |
| `flat_line` | `flat_line_lead_{i}` | lead | `flat_line` ✅ |
| `amplitude_outlier` | `amplitude_outlier:lead_{i}_min_…_max_…` | lead | `amplitude_outlier` ✅ |
| `truncated_signal` | `truncated:{actual}_vs_{expected}` | record | `truncated` ❌ |
| `nan_values` | `nan_values:{n}_NaN_samples` | record | `nan_values` ✅ |
| `corrupt_header` (engine) | `corrupt_header:{exc}` | record | `corrupt_header` ✅ |
| `load_error` (engine fallback) | `load_error:{exc}` | record | `load_error` ✅ |
| any check raising | `{check_name}_error:{exc}` | record | `{check_name}_error` ⚠️ |

`truncated_signal` has not yet fired in any dataset in `output/`, so the `"truncated"`
symptom is latent rather than observed — but it is the same defect and will surface on
the first dataset with short records.

---

## 3. Evidence

Recomputed from each report's own `excluded_records` list (which stores the full,
unmangled issue strings per record). "reported" is the current `records_failed` value;
"true records" is the number of distinct records carrying that check's issues; "issues"
is the raw issue count.

| dataset | check | reported | true records | issues |
|---|---|---:|---:|---:|
| challenge2020 | amplitude_outlier | 1305 | **250** | 1305 |
| challenge2020 | missing_leads (as `missing`) | 85 | **36** | 85 |
| challenge2021 | amplitude_outlier | 2591 | **1021** | 2591 |
| challenge2021 | missing_leads (as `missing`) | 7144 | **1531** | 7144 |
| chapman_shaoxing | amplitude_outlier | 38 | **22** | 38 |
| chapman_shaoxing | missing_leads (as `missing`) | 49 | **17** | 49 |
| cpsc_2018 | amplitude_outlier | 696 | **99** | 696 |
| cpsc_2018 | missing_leads (as `missing`) | 32 | **18** | 32 |
| ecg_arrhythmia | amplitude_outlier | 1286 | **771** | 1286 |
| ecg_arrhythmia | missing_leads (as `missing`) | 7059 | **1495** | 7059 |
| ecgcipa | amplitude_outlier | 5 | **2** | 5 |
| ecgdmmld | amplitude_outlier | 4 | **2** | 4 |
| echonext | flat_line | 6 | **4** | 6 |
| mimic_iv_ecg | amplitude_outlier | 1588 | **989** | 1588 |
| mimic_iv_ecg | flat_line | 15 | **9** | 15 |
| mimic_iv_ecg | missing_leads (as `missing`) | 8858 | **1951** | 8858 |
| mimic_iv_ecg_demo | missing_leads (as `missing`) | 13 | **2** | 13 |
| ptbdb | flat_line | 3 | **1** | 3 |
| ptbxl | amplitude_outlier | 68 | **47** | 68 |
| ptbxl | missing_leads (as `missing`) | 1 | 1 | 1 |
| sph | amplitude_outlier | 1080 | **323** | 1080 |
| staffiii | amplitude_outlier | 139 | **40** | 139 |
| staffiii | missing_leads (as `missing`) | 1 | 1 | 1 |

Correct in every report (record-level checks, one issue per record): `nan_values`,
`corrupt_header`. Reports with no failures at all and therefore nothing to fix:
`brugada_huca`, `ludb`, `mhd_effect_ecg_mri`, `ningbo_iva`, `norwegian_athlete_ecg`,
`wctecgdb`. Also already correct: `incartdb`, `ecgrdvq`, `leipzig_heart_center_ecg`.

**The counts are not merely inflated — they are unbounded relative to the dataset.**
`ecg_arrhythmia` reports `missing: 7059` for a dataset where only **2243 records** were
excluded in total. Any reader treating `records_failed` as a record count sees a check
that failed more records than exist in the exclusion set.

### What is *not* affected

- `excluded_records[]` — accurate, and complete: `is_valid = len(all_issues) == 0`
  (`engine.py:184`), so every record with any issue is in this list. It is a lossless
  record of all issues, which is what makes the backfill in Step 5 possible.
- `original.total_records`, `clean.total_records`, `clean.removed` — accurate.
- `folds.csv`, per-fold CSVs, the `is_valid` / `quality_issues` columns — untouched by
  this bug.
- `manifest.json` — `fold_digest` hashes `id,fold` pairs only (`manifest.py:61-80`).
- `croissant.json` — hashes only CSVs under `clean/` and `original/`;
  `validation_report.json` sits at the dataset root, outside both. **A fix that changes
  only the report therefore invalidates no hash and requires no Croissant regeneration.**

---

## 4. Fix design (and the alternative that was rejected)

**Chosen approach — fix the aggregation, leave the issue strings alone.**
Add one canonical issue-string → check-name resolver, count records and issues
separately, and report both. Only `validation_report.json` changes.

**Rejected alternative — normalise the issue strings at source** (make every check emit
`{check_name}:{detail}`, e.g. `missing_leads:lead_3`). Cleaner in the abstract, but it
rewrites the `quality_issues` column in every `original/folds.csv`, which changes those
files' SHA-256, which invalidates every `croissant.json`, which forces a full
re-upload of all published datasets. Not worth it for a cosmetic gain. If it is ever
done, do it as its own change with a re-upload plan — not folded into this fix.

---

## 5. Step-by-step fix

### Step 1 — Write the failing tests first

`report.py` has **no test coverage at all** today (`tests/test_export.py:73` only asserts
the file exists). Add to `tests/test_validation.py`:

- `test_summary_counts_records_not_leads` — build a `RecordValidation` with
  `issues=["missing_lead_0", "missing_lead_1", "missing_lead_2"]` and assert the
  summary is `{"missing_leads": 1}`, not `{"missing": 3}`.
- `test_summary_uses_registry_check_names` — assert every key of `summary` is in
  `CHECK_REGISTRY` ∪ `{"corrupt_header", "load_error"}`, so no future check can
  reintroduce a fabricated name.
- `test_truncated_signal_keeps_its_registry_name` — one record with
  `issues=["truncated:4500_vs_5000"]` → `{"truncated_signal": 1}`. Guards the latent
  half of the bug.

Add a new `tests/test_report.py`:

- `test_records_failed_differs_from_total_issues` — two records, one with 3
  `amplitude_outlier` lead issues and one with 1, assert
  `records_failed == 2` and `total_issues == 4`.
- `test_every_check_has_a_description` — assert no `quality_checks[i]["description"]`
  is empty for the standard checks. This is the direct regression test for the
  `"missing"` symptom.
- `test_report_survives_a_skip_validation_stub` — `ValidationResult` with
  `record_validations=[]`, `summary={}` (the `cli/splits.py:60-73` stub) must produce
  `quality_checks: []` and not raise.

These will fail against `main`. Note that `tests/test_export.py:52` already builds
`summary={"missing_leads": 1, "nan_values": 1}` — canonical names the current engine can
never produce. After the fix, that fixture becomes truthful; before it, it is drift that
hides the bug from the suite.

### Step 2 — Add a canonical resolver to `ecgbench/validation/checks.py`

One place that knows how issue strings map to check names, next to the functions that
emit them:

```python
# Issue-string prefixes that do not match their registry key. Every check that
# emits a per-lead issue must be listed here, or its issues aggregate under a
# fabricated check name (see issue #55).
_ISSUE_PREFIX_TO_CHECK: tuple[tuple[str, str], ...] = (
    ("missing_lead_", "missing_leads"),
    ("flat_line_lead_", "flat_line"),
    ("truncated:", "truncated_signal"),
)


def check_name_for_issue(issue: str) -> str:
    """Map an issue string back to the check that produced it."""
    for prefix, name in _ISSUE_PREFIX_TO_CHECK:
        if issue.startswith(prefix):
            return name
    return issue.split(":", 1)[0]
```

The fallback covers `amplitude_outlier:…`, `nan_values:…`, `corrupt_header:…`,
`load_error:…` and the synthetic `{check}_error:…` strings unchanged.

### Step 3 — Fix the aggregation in `ecgbench/validation/engine.py`

Replace lines 315-320 with a per-record de-duplicating pass that keeps both numbers:

```python
from ecgbench.validation.checks import CHECK_REGISTRY, check_name_for_issue  # at top

# Compute summaries: records failed per check, and raw issue counts per check.
# Lead-level checks emit one issue per lead, so these differ (issue #55).
summary: dict[str, int] = {}
issue_summary: dict[str, int] = {}
for v in validations:
    seen: set[str] = set()
    for issue in v.issues:
        name = check_name_for_issue(issue)
        issue_summary[name] = issue_summary.get(name, 0) + 1
        seen.add(name)
    for name in seen:
        summary[name] = summary.get(name, 0) + 1
```

Then extend the dataclass — **appended last, with a default**, so the hand-built stub in
`cli/splits.py:65-73` and the fixture in `tests/test_export.py:47-55` keep constructing
positionally without edits:

```python
@dataclass
class ValidationResult:
    ...
    excluded_records: int
    issue_summary: dict[str, int] = field(default_factory=dict)  # check -> raw issue count
```

Fix the `summary` comment on line 43 to say `check_name -> records failed` and pass
`issue_summary=issue_summary` in the constructor call at line 336.

### Step 4 — Fix `ecgbench/validation/report.py`

```python
quality_checks = []
for check_name in sorted(result.summary):
    quality_checks.append({
        "check": check_name,
        "description": _CHECK_DESCRIPTIONS.get(check_name, ""),
        "records_failed": result.summary[check_name],
        # Lead-level checks produce several issues per record; fall back to the
        # record count for results built without issue_summary.
        "total_issues": result.issue_summary.get(check_name, result.summary[check_name]),
    })
```

No change to `_CHECK_DESCRIPTIONS` is needed — `missing_leads` and `truncated_signal`
are already in it, and the empty-description symptom disappears the moment the keys stop
being mangled. The old `startswith` scan over `record_validations` goes away entirely,
which also makes report generation O(records) instead of O(records × checks).

### Step 5 — Backfill the 23 reports already on disk without re-validating

Re-running validation for all datasets means re-reading ~1 M records (≈800 k for
`mimic_iv_ecg` alone) and needs every raw dataset mounted. It is unnecessary:
`excluded_records[]` in each existing report already contains every issue string of
every failing record, so `quality_checks` can be recomputed exactly.

Run as a **one-off** — do not commit it. Repo-root `scripts/` was deliberately removed
(commit `4952df3`); put this in the scratchpad or pipe it via a heredoc:

```python
import json
from pathlib import Path

from ecgbench.validation.checks import check_name_for_issue
from ecgbench.validation.report import _CHECK_DESCRIPTIONS

for path in sorted(Path("output").glob("*/validation_report.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, int] = {}
    issues: dict[str, int] = {}
    for rec in report["excluded_records"]:
        seen = set()
        for issue in rec["issues"]:
            name = check_name_for_issue(issue)
            issues[name] = issues.get(name, 0) + 1
            seen.add(name)
        for name in seen:
            records[name] = records.get(name, 0) + 1
    report["quality_checks"] = [
        {
            "check": name,
            "description": _CHECK_DESCRIPTIONS.get(name, ""),
            "records_failed": records[name],
            "total_issues": issues[name],
        }
        for name in sorted(records)
    ]
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path, report["quality_checks"])
```

Deliberate choices in the backfill:

- It rewrites **only** `quality_checks`. `validated_at` and `ecgbench_version` keep the
  values from the run that actually read the signals — the backfill did not re-validate
  anything and must not claim to have.
- `save_report` writes with `indent=2, ensure_ascii=False` and **no trailing newline**;
  the snippet matches that byte-for-byte, so a later real re-run produces no spurious
  diff.
- Reports whose `excluded_records` is empty end up with `quality_checks: []`, which is
  what they already have.

### Step 6 — Verify the backfill

For each report, these invariants must now hold (none of them hold today):

- `max(q["records_failed"] for q in quality_checks) <= clean["removed"]`
- `sum(q["records_failed"]) >= clean["removed"]` (equality iff no record failed two
  checks) and `<= clean["removed"] * len(quality_checks)`
- `len(excluded_records) == clean["removed"]`
- `q["total_issues"] >= q["records_failed"]` for every check
- every `q["description"]` is non-empty for a check in `_CHECK_DESCRIPTIONS`

Spot-check `output/cpsc_2018/validation_report.json`: it must read
`amplitude_outlier: records_failed 99 / total_issues 696` and
`missing_leads: records_failed 18 / total_issues 32` — the numbers named in issue #55.

Then re-run the pipeline end to end on one small dataset with real signals
(`ludb`, `incartdb` or `norwegian_athlete_ecg`) to confirm the engine path produces the
same shape as the backfill:

```bash
ecgbench splits --dataset incartdb --data-path /path/to/incartdb/ --output-dir /tmp/bug55-check
```

`validate_dataset()` has no end-to-end test coverage (the `ProcessPoolExecutor` path
never runs in the suite), so this manual smoke run is the only thing that exercises the
changed aggregation against real records.

### Step 7 — Re-upload the corrected reports

`cli/upload.py:98` ships `validation_report.json` alongside the fold CSVs, so the wrong
numbers are already on the Hub for every published dataset:

```bash
ecgbench upload --data-dir output/ --datasets <slug> --dry-run   # confirm the file list
ecgbench upload --data-dir output/ --datasets <slug>
```

`mimic_iv_ecg` has `publish_fold_csvs: false` and will raise `PermissionError` before any
network call — correct, leave it alone; its report is corrected locally only.

### Step 8 — Documentation touch-ups

- `ECGBench_architecture/ARCHITECTURE.md:241` — the `ValidationResult` node lists
  `summary`; no change needed unless `issue_summary` should appear there too.
- `Backups/DATASET_ANALYSIS_PLAN.md:264` — states `quality_checks` comes from
  `ValidationResult.summary` as "records_failed". That becomes true only after this fix;
  worth a note that it was previously an issue count.
- `ADD_DATASET_TODO.md:265` — the spot-check advice ("high `truncated_signal` counts
  usually mean `expected_samples` is wrong") is more reliable once counts are per-record;
  consider adding "`records_failed` can never exceed `clean.removed`" as the sanity rule.
- `README.md:751,766` — no change needed.

---

## 6. Gotchas for whoever implements this

- **A record that fails to load short-circuits.** `_validate_single_record` returns
  immediately on a load exception (`engine.py:144-151`), so `corrupt_header` /
  `load_error` records carry exactly one issue and never co-occur with other checks. Do
  not "fix" a per-record de-dup that appears to do nothing for them.
- **`{check}_error:{exc}` is a synthetic key**, not a registered check. It will get an
  empty description under `_CHECK_DESCRIPTIONS` — that is honest (it means the check
  itself crashed) but the strict test in Step 1 must allow it. Consider adding a generic
  description via an `endswith("_error")` fallback.
- **`records_failed` still does not sum to `clean.removed`** after the fix, because a
  record can fail several checks. That is correct behaviour, not a residual bug; say so
  in the report docstring so it is not "fixed" again later.
- **The `--skip-validation` stub** (`cli/splits.py:60-73`) passes `summary={}` and
  `record_validations=[]`; the new code must yield `quality_checks: []` rather than
  raising on a missing `issue_summary`. The `.get(..., default)` in Step 4 covers it,
  but keep the test.
- **Do not change the issue strings** as part of this fix — see §4 for why.
