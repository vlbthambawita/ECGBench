# Website Content via Markdown — Plan (v3)

Goal: **one Markdown file per dataset** is the single source of truth for both the catalogue table row and the dataset's detail page. No CSVs. Python code and website render from the same `.md` files.

**v3 scope change vs v2:**
- CSV files are deleted (`docs/ecg_datasets.csv` download link and `ecgbench/data/ecg_datasets.csv` Python source).
- `docs/_datasets/<slug>.md` front matter owns **all** catalogue row fields.
- `ecgbench/catalogue.py` parses the `.md` front matter directly at import time.
- The only way to consume the catalogue as tabular data is via the Python package: `from ecgbench import list_datasets`.

---

## 1. The one file

Each dataset = one file at `docs/_datasets/<slug>.md`. Filename stem is the slug.

Front matter owns the row fields (what the catalogue table shows). Body or `sections:` owns the detail page.

```markdown
---
# ── Identity ──
name: PTB-XL
category: 12-lead-physionet          # table this row belongs to
order: 1                             # row order within the category
status: completed                    # key into _data/statuses.yml

# ── Catalogue row fields ──
url: https://physionet.org/content/ptb-xl/1.0.3/
url_label: physionet.org
format: "12-lead · 10 s · 500 Hz (also 100 Hz)"
patients: "18,869"
records: "21,799"
access: open                         # open | credentialed | restricted
license: "CC BY 4.0"
origin_institution: "Physikalisch-Technische Bundesanstalt"
origin_country: "Germany"
leads: 12
paper_title: "PTB-XL: A Large Publicly Available ECG Dataset"
paper_doi: "https://doi.org/10.1038/s41597-020-0495-6"
search_keywords: "ptb-xl germany ptb physikalisch-technische bundesanstalt"

# ── Detail page (optional) ──
hero_image: /assets/datasets/ptbxl/hero.png
sections:
  - type: description
    title: "Overview"
    body: |
      PTB-XL is a 12-lead ECG dataset with 21,799 records from 18,869 patients…

  - type: plot
    title: "Class distribution"
    image: /assets/datasets/ptbxl/class_dist.png
    caption: "Superclass frequencies across the clean split."

  - type: table
    title: "Label taxonomy"
    headers: [Superclass, # records, Share]
    rows:
      - [NORM, 9514, "43.6%"]
      - [MI,   5486, "25.2%"]
      - [STTC, 5250, "24.1%"]

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset
      ds = ECGDataset("ptbxl", split="train", fold=1)
      x, y = ds[0]
    # or: file: snippets/ptbxl/load.py

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/ptb-xl/1.0.3/" }
      - { label: "Paper",          url: "https://doi.org/10.1038/s41597-020-0495-6" }
---

<!-- Free-form Markdown fallback: used when `sections:` is absent. -->
```

Rule: if `sections:` is present, it drives the page. Otherwise the Markdown body is rendered raw.

---

## 2. Supporting YAML (config, not data)

Three small config files, edited rarely:

| File                          | Purpose                                               |
| ----------------------------- | ----------------------------------------------------- |
| `docs/_data/tables.yml`       | Section order, titles, column sets per table          |
| `docs/_data/columns.yml`      | Column definitions (label, which field, cell template) |
| `docs/_data/statuses.yml`     | Status keys → label + icon + color                    |

### `_data/statuses.yml` — configurable status icons

```yaml
not_started:
  label: "Not started"
  icon: "○"
  icon_type: glyph               # glyph | svg | emoji
  color: "#8892a4"
  background: "rgba(136,146,164,0.12)"

implementing:
  label: "Implementing"
  icon: "◐"
  icon_type: glyph
  color: "#f59e0b"
  background: "rgba(245,158,11,0.15)"

completed:
  label: "Completed"
  icon: "✓"
  icon_type: glyph
  color: "#22c55e"
  background: "rgba(34,197,94,0.15)"

needs_review:
  label: "Needs review"
  icon: "⚠"
  icon_type: glyph
  color: "#ef4444"
  background: "rgba(239,68,68,0.15)"
```

Change an icon → edit one field. Add a new status → add a block + use the key in a dataset `.md`.

### `_data/tables.yml`

```yaml
- id: 12-lead-physionet
  title: "12-Lead ECG Datasets"
  columns: [num, dataset, format, patients, records, access, status, origin, paper]

- id: 12-lead-other
  title: "12-Lead ECG Datasets (Other Repositories)"
  columns: [num, dataset, format, patients, records, access, status, origin, paper]

- id: two-lead
  title: "2-Lead ECG Datasets"
  columns: [num, dataset, format, records, access, status, origin, paper]

# 1-lead, 3-lead, bspm …
```

### `_data/columns.yml`

```yaml
num:      { label: "#",        cell: _auto_index }
dataset:  { label: "Dataset",  cell: dataset }
format:   { label: "Format",   field: format }
patients: { label: "Patients", field: patients, class: count }
records:  { label: "Records",  field: records,  class: count }
access:   { label: "Access",   cell: access }
status:   { label: "Status",   cell: status }
origin:   { label: "Origin",   cell: origin }
paper:    { label: "Paper",    cell: paper }
```

Each `cell:` value maps to `_includes/cells/<name>.html`.

---

## 3. Catalogue table rendering

`index.html`'s 6 hand-coded tables collapse into one Liquid loop:

```liquid
{% for table in site.data.tables %}
  {% assign rows = site.datasets
       | where: "category", table.id
       | sort: "order" %}

  <div class="section-header">
    <h2>{{ table.title }}</h2>
    <span class="pill">{{ rows | size }} datasets</span>
  </div>

  <div class="table-wrap">
    <table id="{{ table.id }}-table">
      <thead><tr>
        {% for col_id in table.columns %}
          <th>{{ site.data.columns[col_id].label }}</th>
        {% endfor %}
      </tr></thead>
      <tbody>
        {% for row in rows %}
          {% assign status = site.data.statuses[row.status] %}
          <tr data-access="{{ row.access }}"
              data-leads="{{ row.leads }}"
              data-text="{{ row.search_keywords }}"
              data-status="{{ row.status }}">
            {% for col_id in table.columns %}
              {% include cells/{{ col_id }}.html row=row status=status %}
            {% endfor %}
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
{% endfor %}
```

- Row counts auto-computed — no more manual pill bumps.
- Existing search JS keeps working (we still emit `data-access` / `data-leads` / `data-text`).
- `data-status` added for future status filters.

---

## 4. Per-dataset page

One layout `docs/_layouts/dataset.html` renders:

1. **Hero** — name, access pill, status pill, hero image, link-out buttons (paper, dataset url), all pulled from the same front matter.
2. **Quick-facts table** — auto-built from the row fields (patients, records, format, license, origin, leads).
3. **Body** — either iterate `sections:` via `{% include sections/<type>.html %}`, or render the Markdown body.
4. **Footer** — "Edit this page on GitHub" link pointing at the `.md` source.

### Section types (v1 set)

One Liquid partial each, in `_includes/sections/`:

| Type         | Renders                                                          |
| ------------ | ---------------------------------------------------------------- |
| `description`| Prose (Markdown allowed in `body:`)                              |
| `plot`       | Static `<img>` (PNG / SVG) + caption                             |
| `table`      | Inline `headers:` + `rows:`, or reference `_data/<name>.yml`     |
| `code`       | Syntax-highlighted code block; inline `body:` or `file:` include |
| `notebook`   | Labeled link-out (Colab / Binder / Kaggle)                       |
| `links`      | Bulleted list of labeled URLs                                    |

Adding a new type later = one new partial + one entry in the dispatcher.

### Code examples

- **Inline** in `body: |` — simplest, good for short snippets.
- **External** via `file: snippets/<slug>/<name>.py` loaded with `{% include_relative %}` — good when the same snippet is runnable from the repo. `docs/_snippets/` is underscore-prefixed, so Jekyll won't publish it as pages; files are read at build via Liquid includes.

---

## 5. Python catalogue reads `.md` front matter

`ecgbench/catalogue.py` rewrites to parse YAML front matter instead of CSV. PyYAML is already a dep.

```python
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


@dataclass(frozen=True)
class CatalogueEntry:
    slug: str
    name: str
    category: str
    status: str
    url: str
    format: str
    patients: str
    records: str
    access: str
    license: str | None
    origin_institution: str
    origin_country: str
    leads: int
    paper_title: str
    paper_doi: str


def _datasets_dir() -> Path:
    # Installed wheel: bundled copy
    wheel = Path(__file__).parent / "_datasets"
    if wheel.is_dir():
        return wheel
    # Editable / source checkout
    repo = Path(__file__).resolve().parent.parent / "docs" / "_datasets"
    if repo.is_dir():
        return repo
    raise RuntimeError("ECGBench dataset definitions not found")


@cache
def list_datasets() -> list[CatalogueEntry]:
    entries = []
    for path in sorted(_datasets_dir().glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = _FRONT_MATTER.match(text)
        if not match:
            continue
        meta = yaml.safe_load(match.group(1)) or {}
        entries.append(CatalogueEntry(slug=path.stem, **_pick_row_fields(meta)))
    return entries


@cache
def get_dataset(slug: str) -> CatalogueEntry:
    for e in list_datasets():
        if e.slug == slug:
            return e
    raise KeyError(slug)
```

Usage stays the same for consumers:

```python
from ecgbench import list_datasets, get_dataset
for d in list_datasets():
    print(d.slug, d.status, d.records)
```

Parsing 64 YAML files at import time is ~milliseconds; `@cache` makes it a one-shot cost.

---

## 6. Packaging — how the wheel finds the `.md` files

Source tree puts them at `docs/_datasets/*.md` (required by Jekyll's collection convention). The wheel needs them too, so `pip install ecgbench` works.

Use hatch's `force-include` to copy them into the wheel under the package:

```toml
# pyproject.toml
[tool.hatch.build.targets.wheel]
packages = ["ecgbench"]

[tool.hatch.build.targets.wheel.force-include]
"docs/_datasets" = "ecgbench/_datasets"
```

`catalogue.py`'s `_datasets_dir()` (see §5) tries `ecgbench/_datasets/` first (wheel) then falls back to `docs/_datasets/` relative to the repo root (editable install / checkout). One path works for both.

Also include `_data/statuses.yml` etc. if Python ever needs them — not required for v1.

---

## 7. Editing flows

| Task                                     | File to edit                                                    |
| ---------------------------------------- | --------------------------------------------------------------- |
| Add a new dataset                        | Create `docs/_datasets/<slug>.md`                               |
| Change a dataset's status                | Edit `status:` in that file                                     |
| Edit any row field                       | Edit that field in the dataset's `.md`                          |
| Rename a status / change icon / color    | `docs/_data/statuses.yml`                                       |
| Add a new status value                   | Add block to `statuses.yml`; use key in datasets                |
| Reorder / rename tables                  | `docs/_data/tables.yml`                                         |
| Add / remove / reorder columns           | `docs/_data/tables.yml` + `columns.yml`                         |
| Write or expand a detail page            | Add / edit `sections:` in the dataset's `.md`                   |
| Add a plot                               | Drop image in `/docs/assets/datasets/<slug>/`; add `plot` section |
| Add a code example                       | Inline `code` section, or file under `docs/_snippets/<slug>/`   |
| Change page layout globally              | `_layouts/dataset.html` / `_includes/sections/*.html` (HTML)    |

End-to-end "add a dataset":
1. Create `docs/_datasets/myslug.md` with the front matter block.
2. Commit. Catalogue table, detail page, and Python `list_datasets()` all pick it up.

---

## 8. Migration

One-shot `scripts/html_to_datasets.py` (throwaway):
1. Parse `docs/index.html` with BeautifulSoup.
2. For each `<tr>`, extract cells + `data-*` attrs.
3. Derive `slug` from dataset name (slugify, lowercase, hyphenated).
4. Derive `category` from which `<tbody>` the row lived in.
5. Stamp `status: not_started` on all rows (user flips completed/in-progress ones by hand).
6. Write `docs/_datasets/<slug>.md` with a minimal front matter block and an empty `sections: []` (or omitted, letting the page stay bare until filled in).
7. Print row counts per category vs. the current "pill" counts — any mismatch = parse bug.

Then:
- Delete both CSVs (`docs/ecg_datasets.csv`, `ecgbench/data/ecg_datasets.csv`).
- Delete the inline `<tbody>` blocks from `docs/index.html`; replace with the Liquid loop.
- Verify `jekyll build` output matches the old page (screenshot diff).
- Update `ecgbench/catalogue.py` + tests.
- Remove the CSV download link from `index.html` (the green button near the top).

---

## 9. Risks / watch-outs

- **Jekyll collections on GitHub Pages** — `collections`, `_data/`, `_layouts/`, `_includes/` are all standard Jekyll; no plugins needed. Safe.
- **Editable install path** — `_datasets_dir()` uses `Path(__file__).resolve().parent.parent / "docs" / "_datasets"`, which assumes the package is at `<repo>/ecgbench/`. Breaks if someone moves the package. Acceptable.
- **Numeric fields as strings** — YAML leaves `"18,869"` as a string (good, that's the display form). If we later need numeric comparison, add `patients_int:` alongside.
- **Slug stability** — once published, `/datasets/<slug>/` is a URL. Don't rename; add a redirect if you must.
- **Missing front matter fields** — catalogue.py must be defensive (missing `license`, `leads`, etc.). Use `.get()` + `None`.
- **`search_keywords` duplication** — overlaps with other fields. Optional: auto-generate if missing by concatenating name + origin + country + free-form kwords.
- **Bundled wheel size** — 64 small `.md` files is negligible.

---

## 10. Phased todo

### Phase 0 — decisions (user) ✅
- [x] CSV removed, `.md` single source of truth
- [x] Python reads front matter at import time
- [x] Status vocabulary: `not_started`, `implementing`, `completed`, `needs_review`
- [x] URL shape: `/datasets/<slug>/`, with Jekyll's `baseurl` (`/ECGBench`) prepended automatically (use `{{ '/datasets/' | relative_url }}` in templates — never hard-code `/ECGBench/…`)
- [x] Plots: static images only (PNG / SVG). Interactive embeds out of scope for now.
- [x] Keep Style B — plain Markdown body renders when `sections:` is absent.

### Phase 1 — scaffolding
- [ ] Add `docs/_config.yml` entries: `collections.datasets` (output true), permalink `/datasets/:name/`
- [ ] Add `docs/_data/statuses.yml` + `tables.yml` + `columns.yml`
- [ ] Add `docs/_includes/cells/*.html` (num, dataset, format, patients, records, access, status, origin, paper)
- [ ] Add `docs/_includes/sections/*.html` (description, plot, table, code, notebook, links)
- [ ] Add `docs/_layouts/dataset.html`
- [ ] Add hatch `force-include` rule to `pyproject.toml`

### Phase 2 — migration
- [ ] Write + run `scripts/html_to_datasets.py`; get 64 `.md` files
- [ ] Manually review a handful for correctness
- [ ] Delete both CSVs
- [ ] Rewrite `ecgbench/catalogue.py` to read front matter; update `CatalogueEntry` dataclass + tests
- [ ] Update `ecgbench/__init__.py` exports (keep `list_datasets`, add `get_dataset`)

### Phase 3 — catalogue table swap
- [ ] Replace 6 hand-coded tables in `docs/index.html` with the Liquid loop
- [ ] Remove the CSV download button
- [ ] Verify pixel parity vs. current live page
- [ ] Verify search / filter / pill counts still work
- [ ] Set a few datasets to `implementing` / `completed`, confirm icons render

### Phase 4 — detail pages live
- [ ] Confirm all 64 stub pages render without error (sections optional)
- [ ] Fill **PTB-XL** as the reference page (description + plot + table + code + links)
- [ ] Link the `Dataset` cell on the catalogue table to `/datasets/<slug>/`

### Phase 5 — polish (optional)
- [ ] Status-based filter UI (dropdown or pill row) on the catalogue
- [ ] "Edit on GitHub" link in page footer
- [ ] README section: "How to add a dataset"
- [ ] Rename this plan file to `website_plan_todo.md`

---

## 11. Decisions (resolved)

1. **Status vocabulary:** `not_started`, `implementing`, `completed`, `needs_review`. Add more later by appending to `_data/statuses.yml`.
2. **URL shape:** `/datasets/<slug>/`. The GitHub Pages base path (`/ECGBench`) is prepended automatically by Jekyll via `baseurl` in `_config.yml` — templates must use `relative_url` / `absolute_url` filters and never hard-code the prefix.
3. **Plots:** static PNG / SVG only. `plot` section renders `<img>` + caption. Interactive embeds deferred.
4. **Authoring styles:** both Style A (`sections:` list) and Style B (plain Markdown body) supported. Page layout uses `sections:` if present, otherwise renders the body.
