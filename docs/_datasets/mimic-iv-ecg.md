---
slug: "mimic-iv-ecg"
name: "MIMIC-IV-ECG"
category: "12-lead-physionet"
order: 4
status: "not_started"
source_url: "https://physionet.org/content/mimic-iv-ecg/1.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz"
patients: "~160,000"
records: "~800,000"
access: "credentialed"
license: "PhysioNet DUA"
origin_institution: "Beth Israel Deaconess Medical Center"
origin_country: "USA — Boston, MA"
leads: 12
paper_title: "Gow et al."
paper_doi: "https://doi.org/10.13026/4nqg-sb35"
search_keywords: "mimic iv ecg usa boston beth israel mit"

related:
  - slug: "mimic-iv-ecg-demo"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      658 of the demo's 659 records were matched here on (subject_id, minute), and
      all 92 demo subjects appear. But study_ids are renumbered into a disjoint
      range and timestamps are truncated to the minute in this release, so the two
      cannot be joined on study_id — a study_id comparison shows 0% overlap and is
      misleading.
  - slug: "mimic-iv-ecg-ext-icd"
    relation: "has_derivative"
    shares_records: true
    verified: true
    note: >
      Adds ED and hospital ICD-10 diagnoses. Covers all 800,035 studies and all
      161,352 subjects exactly, and joins cleanly on study_id.
---
