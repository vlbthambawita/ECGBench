---
slug: "zzu-pecg-zhengzhou-university-pediatric-ecg-database"
name: "ZZU pECG (Zhengzhou University Pediatric ECG Database)"
category: "12-lead-other"
order: 15
status: "completed"
source_url: "https://doi.org/10.6084/m9.figshare.27078763"
url_label: "figshare.com"
format: "12-lead + 9-lead · 5-120 s · 500 Hz · WFDB"
patients: "11,643 children"
records: "14,190"
access: "open"
license: "CC BY 4.0"
origin_institution: "First Affiliated Hospital of Zhengzhou University"
origin_country: "China"
leads: 12
paper_title: "Tan et al., Scientific Data, 2025"
paper_doi: "https://doi.org/10.1038/s41597-025-05225-z"
search_keywords: "zzu pecg zhengzhou pediatric paediatric children ecg figshare wfdb china kawasaki myocarditis congenital heart disease cardiomyopathy icd-10 aha signal quality sqi nine lead"

sections:
  - type: description
    title: "Overview"
    body: |
      14,190 ECGs from 11,643 hospitalised children at the First Affiliated
      Hospital of Zhengzhou University, recorded 2018-2024 at 500 Hz. Ages run
      from **1 day to 15.0 years** (median 8.6), which makes this the only
      genuinely paediatric 12-lead resource in the catalogue.

      **Paediatric ECGs are not small adult ECGs.** Right-axis dominance and
      T-wave inversion in the right precordial leads are normal in an infant and
      abnormal in an adolescent, so age is not an optional covariate here and an
      adult-trained model should not be assumed to transfer. Age ships in
      **days**, and 546 records are under a year old — rounding to whole years
      collapses the entire neonatal and infant range.

      **Not every record has 12 leads, and this is the trap on this page.**
      12,334 records store the standard 12; the other **1,856 store only 9**,
      dropping V2, V4 and V6. The 9-lead layout is *not* a prefix of the 12-lead
      one — stored position 7 is V2 in one and **V3** in the other — so
      index-based lead selection silently crosses those two leads for 13% of the
      release. ECGBench declares both layouts, so
      `ECGDataset(leads=["V2"])` **raises** on a reduced record instead of
      handing back V3, and `n_leads` is exposed for filtering. This is the only
      dataset in the catalogue that needs that mechanism.

      **Dropping the 9-lead records is not a neutral filter.** They carry a
      studied cardiovascular diagnosis at more than twice the rate of the 12-lead
      records (933 of 1,856, 50.3%, against 2,783 of 12,334, 22.6%), so excluding
      them preferentially discards sick children.

      **Labels are unusually rich for an open release.** Every record carries one
      or more ECG findings coded in **both** the AHA and Chinese vocabularies with
      human-readable descriptions, a list of ICD-10 discharge diagnoses, and three
      **per-lead signal-quality indices** (pSQI, basSQI, bSQI) computed by the
      depositors.

      **Record length varies by a factor of 24**, from 5 s to 120 s across 67
      distinct lengths. So `expected_samples` is deliberately empty in the config,
      and any fixed `window=` must fit the 5 s shortest record — which is also
      what makes a `DataLoader` batch possible at all.

  - type: table
    title: "Disease groups (ICD-10 discharge diagnoses)"
    headers: ["Group", "Records", "% of 14,190", "Note"]
    rows:
      - ["Congenital heart disease", "2,787", "19.6%", "9 ICD-10 codes, incl. VSD and ASD"]
      - ["Myocarditis", "635", "4.5%", "4 codes, incl. fulminant and viral"]
      - ["Kawasaki disease", "194", "1.4%", "M30.3"]
      - ["Cardiomyopathy", "147", "1.0%", "4 codes, incl. dilated and hypertrophic"]
      - ["**≥1 of the four**", "**3,716**", "**26.2%**", "47 records carry more than one; 2,597 patients"]
      - ["none of the four", "10,474", "73.8%", "may still carry other ICD-10 codes"]

  - type: description
    title: "About those counts"
    body: |
      All five published files match the md5s figshare lists —
      `AttributesDictionary.csv` `ffbbb7ebd8ad4425b3a859739eb65eb1`,
      `ECGCode.csv` `7612c5af0e01e052eb85792c5e362f12`, `DiseaseCode.csv`
      `0bff597c23397c0d05a937fe09807ad4`, and both halves of the split zip.

      **The structural figures agree with the data descriptor exactly**: 14,190
      records, 11,643 children, 12,334 twelve-lead, 1,856 nine-lead, 500 Hz,
      5-120 s.

      **The diagnosis count does not.** The descriptor states that 3,516 records
      were "diagnosed with cardiovascular diseases"; recomputing gives **3,716**
      records over 2,597 patients. The derivation is stated exactly so the
      difference is checkable: split the `ICD-10 code` cell on `;`, strip the
      surrounding quotes, and count a record if any element is one of the 19 real
      ICD-10 codes in `DiseaseCode.csv`. Two details matter — the codes carry
      study prefixes (`(FO) Q21.1`, `(OSD) Q21.1`, `(F) I40.0`) which are part of
      the key and are matched literally, and that file's 20th row is the
      placeholder `See attribute dictionary file` rather than a code, so it is
      ignored. The 200-record gap is unexplained by the release.

      **The two code columns are parallel and complementary, and neither is
      sufficient alone.** `AHA_code` and `CHN_code` list the same findings in the
      same order, and each gives its own vocabulary's code *where one exists* and
      the plain-English **description** where it does not: `ECGCode.csv` has no
      AHA code for 14 of its 105 findings (Osborn wave, left ventricular high
      voltage, prolonged QTc, abnormal Q wave, …) and no CHN code for 29. So
      6,473 of the 26,797 `AHA_code` entries are prose, and reading the column as
      a code vocabulary invents 15 phantom "codes". ECGBench normalises both and
      exposes `ecg_findings` as the canonical description regardless of which
      vocabulary names the finding.

      **Modifiers are glued onto codes, differently in each vocabulary.** AHA
      writes `L145+Modifier362` and bare qualifiers like `Suggests208`; CHN writes
      `L121+Depression`, `F55+Frequent`, `D21+Occasional`, plus composites such as
      `J(111+112+113)`. `aha_base_codes` and `chn_base_codes` strip the modifier —
      and note the composite must *not* be split at its internal `+`, or the base
      comes out as the truncated `J(111+112`.

      **ICD-10 codes are admission diagnoses, not readings of the tracing.** A
      record can carry a diagnosis whose ECG signature is absent from that
      particular recording. The ECG-finding columns are the tracing-level labels;
      the disease groups are the patient-level ones. They are different
      quantities and the table above is the latter.

      **Amplitudes have a hard rail at about 26.6 mV** that 11.8% of records
      touch, with *sustained* rather than spike excursions (a median 2.1% of
      samples above 10 mV, up to 41%). These are genuine saturation, and the
      release's own baseline quality index independently agrees: railed records
      have a median basSQI of 0.961 against 0.983 for the rest (r = −0.36 against
      peak amplitude). The validation range is ±20 mV rather than ±10 because
      exclusion rates barely move across that span (±10: 18.8%, ±15: 16.4%, ±20:
      14.8%, ±25: 11.8%) — almost every offender is railed rather than merely
      large — and genuine paediatric voltages are high: "left ventricular high
      voltage" is the **second commonest finding** in the release at 2,789
      records.

  - type: table
    title: "ECGBench splits"
    headers: ["Version", "Records", "Excluded", "Why"]
    rows:
      - ["`original`", "14,190", "—", "all records, with `is_valid` and `quality_issues`"]
      - ["`clean`", "12,190", "2,000 (14.1%)", "all `amplitude_outlier`: a lead beyond ±20 mV, i.e. a saturated segment"]

  - type: code
    title: "Loading with ECGBench"
    language: "python"
    body: |
      from ecgbench import ECGDataset

      ds = ECGDataset(
          "zzu_pecg",
          split="train",
          data_path="/path/to/ZZU_pECG/",
          labels=True,
          # Sized to the SHORTEST record (2,500 samples = 5 s). Anything wider
          # raises WindowOutOfRangeError naming the record and its true length.
          window=(0, 2500),
      )
      print(len(ds))                      # 9738
      s = ds[0]
      print(tuple(s["signal"].shape))     # (9, 2500)   <- a NINE-lead record
      print(s["record_id"])               # P00001_E01
      print(s["labels"]["n_leads"])       # 9
      print(s["labels"]["age_days"])      # 572   (1.57 years)
      print(s["labels"]["ecg_findings"])  # Left ventricular high voltage;T-wave abnormality
      print(s["labels"]["disease_groups"])  # Congenital heart disease

      # Leads present in BOTH layouts resolve for every record...
      safe = ECGDataset("zzu_pecg", split="train", data_path="/path/to/ZZU_pECG/",
                        window=(0, 2500), leads=["I", "II", "V1", "V5"])
      print(tuple(safe[0]["signal"].shape))    # (4, 2500)

      # ...and one absent from the 9-lead layout raises rather than returning V3:
      #   ValueError: Lead 'V2' is not in 'zzu_pecg'. Available: [... 'V1', 'V3', 'V5']
      # Filter on n_leads if you need all twelve — but see the note above about
      # what dropping the reduced-lead records does to the case mix.

      # Batching needs leads= as well as window=: a batch mixing 9- and 12-lead
      # records cannot be stacked, and raises
      #   RuntimeError: stack expects each tensor to be equal size
      from torch.utils.data import DataLoader
      from ecgbench import ecg_collate_fn
      batch = next(iter(DataLoader(safe, batch_size=8, collate_fn=ecg_collate_fn)))
      print(tuple(batch["signal"].shape))      # (8, 4, 2500)

  - type: links
    title: "Links"
    items:
      - label: "figshare record"
        url: "https://doi.org/10.6084/m9.figshare.27078763"
      - label: "Paper (Scientific Data)"
        url: "https://doi.org/10.1038/s41597-025-05225-z"
      - label: "Example script"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_zzu_pecg.py"
---
