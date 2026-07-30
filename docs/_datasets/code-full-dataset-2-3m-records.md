---
slug: "code-full-dataset-2-3m-records"
name: "CODE (Full Dataset, ~2.3M records)"
category: "12-lead-other"
order: 7
status: "not_started"
source_url: "https://figshare.scilifelab.se/articles/dataset/CODE_dataset/15169716"
url_label: "scilifelab.se"
format: "12-lead · 400 Hz · HDF5"
patients: "~1,676,384"
records: "~2,322,513"
access: "restricted"
license: "DUA required"
origin_institution: "Telehealth Network of Minas Gerais (TNMG)"
origin_country: "Brazil"
leads: 12
paper_title: "Ribeiro et al., Nature Communications, 2020"
paper_doi: "https://doi.org/10.1038/s41467-020-15432-4"
search_keywords: "code full dataset scilifelab figshare brazil tnmg 2 million hdf5 dua"

related:
  - slug: "code-15-pct-telehealth-network-of-minas-gerais-15-pct-subset"
    relation: "contains"
    shares_records: true
    verified: false
    note: >
      The open 15% sample is drawn from this cohort. Documented; not checked
      against the files.
  - slug: "code-test-827-record-hold-out-test-set"
    relation: "contains"
    shares_records: true
    verified: false
    note: >
      The 827-record annotated hold-out set comes from this cohort. A model
      trained on CODE-full or the 15% subset must not be evaluated on it without
      confirming those records were excluded.
---
