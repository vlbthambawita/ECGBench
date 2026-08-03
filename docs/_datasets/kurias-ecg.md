---
slug: "kurias-ecg"
name: "KURIAS-ECG"
category: "12-lead-physionet"
order: 11
status: "unavailable"
source_url: "https://physionet.org/content/kurias-ecg/1.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz · SNOMED CT + OMOP-CDM"
patients: "13,862"
records: "20,000"
access: "restricted"
license: "Pending audit"
origin_institution: "Korea University Anam Hospital"
origin_country: "South Korea — Seoul"
leads: 12
paper_title: "Dataset DOI"
paper_doi: "https://doi.org/10.13026/kga0-0270"
search_keywords: "kurias ecg south korea seoul anam snomed omop unavailable withdrawn downloads disabled"

sections:
  - type: description
    title: "Downloads are disabled — this dataset cannot currently be obtained"
    body: |
      **The authors have asked for downloads of KURIAS-ECG to be disabled until
      further notice**, following an internal audit at the Korea University Medical
      Center. The PhysioNet project page may still resolve, but the data are not
      obtainable from it.

      This is why the status above reads *Unavailable* rather than *Not started*:
      the dataset is not waiting on work from ECGBench. Nothing here can make it
      loadable while the source is withdrawn, so it has no config, no fold
      assignment and nothing on the HuggingFace Hub — and will not until the authors
      restore access.

      The entry is kept in the catalogue deliberately rather than deleted. The
      figures above describe the release as it was published, so a paper citing
      KURIAS-ECG can still be placed against it, and anyone holding a copy from
      before the suspension knows what they have.

      Check the PhysioNet project page for the current position; ECGBench does not
      track it automatically.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page (downloads disabled)", url: "https://physionet.org/content/kurias-ecg/1.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/kga0-0270" }
---
