# ECGBench — Architecture & Main Flow

This document explains how ECGBench works end to end, using Mermaid diagrams.
Every box names a real function, class, or dataclass so the diagrams double as a
map into the source. Diagrams progress from the big picture down to each
subsystem's internal call flow.

> **How to read these:** rounded boxes are *functions*, rectangles are *modules /
> data*, diamonds are *decisions*, and `((...))` are *external libraries*.
> Function names match the source exactly (e.g. `validate_dataset`,
> `split_dataset`, `export_splits`).

---

## 1. System Overview — the five subsystems

ECGBench is config-driven: one YAML file per dataset feeds every subsystem.
There are two distinct user journeys — **building** a benchmark (`ecgbench
splits`) and **consuming** one (`ECGDataset` in a training loop) — plus a third
that touches no waveform at all: **querying** what exists through the metadata
layer (§11), which the CLI, the Python API, the website and an MCP server for
agents all read.

```mermaid
flowchart TB
    YAML["YAML config<br/>ecgbench/data/configs/&lt;slug&gt;.yaml"]
    MD["Catalogue front-matter<br/>docs/_datasets/&lt;slug&gt;.md"]

    subgraph CFG["Config system — config.py"]
        load_config["load_config()"]
        DatasetConfig["DatasetConfig<br/>(+ nested dataclasses)"]
    end

    subgraph CAT["Catalogue — catalogue.py"]
        list_datasets["list_datasets / search /<br/>get_dataset / categories /<br/>to_dataframe"]
    end

    subgraph BUILD["BUILD pipeline (ecgbench splits → run_splits)"]
        direction TB
        download["download.py<br/>resolve_data_path()"]
        validation["validation/<br/>validate_dataset()"]
        splitting["splitting/<br/>split_dataset()"]
        export["splitting/export.py<br/>export_splits()"]
        croissant["croissant.py<br/>save_croissant()"]
        upload["cli/upload.py<br/>run_upload()"]
    end

    subgraph CONSUME["CONSUME (training time)"]
        ECGDataset["dataset.py<br/>ECGDataset + ecg_collate_fn"]
    end

    subgraph META["QUERY — metadata/ (§11)"]
        build_model["build.py<br/>build_model()"]
        store["store.py<br/>MetadataStore (JSON + SQLite FTS5)"]
        views["cli/catalog.py · export.py ·<br/>records.py · mcp_server.py"]
    end

    HF[("HuggingFace Hub<br/>vlbthambawita/ECGBench")]
    OUT["output/&lt;slug&gt;/<br/>{original,clean}/ + reports"]
    SNAP["data/snapshots/&lt;slug&gt;.json<br/>(id-free counts + digests)"]

    YAML --> load_config --> DatasetConfig
    MD --> list_datasets

    DatasetConfig --> download --> validation --> splitting --> export --> croissant
    export --> OUT
    croissant --> OUT
    OUT --> upload --> HF

    DatasetConfig --> ECGDataset
    HF -. "metadata_source='hf'" .-> ECGDataset

    YAML --> build_model
    MD --> build_model
    OUT -- "ecgbench metadata snapshot" --> SNAP --> build_model
    build_model --> store --> views
    OUT -. "metadata_source='local'" .-> ECGDataset
    ECGDataset --> Batch["batch dict<br/>signal + metadata tensors"]
```

---

## 2. The main BUILD flow — `run_splits()`

`run_splits()` (in `cli/splits.py`, also exposed as `ecgbench.run_splits` and the
`ecgbench splits` CLI command) is the orchestrator that chains every build
subsystem together. This is the single most important flow in the library.

```mermaid
flowchart TD
    start([ecgbench splits --dataset X]) --> cli["_cli_run() → run_splits()"]

    cli --> lc["load_config(dataset)<br/>→ DatasetConfig"]
    lc --> rdp["resolve_data_path(data_path, config)<br/>→ local/auto-downloaded root"]
    rdp --> gs["get_splitter(dataset)<br/>→ DatasetSplitter instance"]

    gs --> lm["splitter.load_metadata(path, config)<br/>→ metadata DataFrame"]
    lm --> gsl["splitter.get_stratification_labels(df, config)<br/>→ label Series"]

    gsl --> skipval{skip_validation?}
    skipval -- no --> vd["validate_dataset(path, config,<br/>sampling_rate, max_workers)<br/>→ ValidationResult"]
    skipval -- yes --> stub["build stub ValidationResult<br/>(all records is_valid=True)"]

    vd --> sd
    stub --> sd

    sd["split_dataset(df, labels, config, n_folds)<br/>→ SplitResult (1-indexed folds)"]
    sd --> es["export_splits(split_result, val_result,<br/>output_dir, config)<br/>→ stats dict"]

    es --> skipcr{skip_croissant?}
    skipcr -- no --> cr["save_croissant(config, version_dir,<br/>path, version) ×{clean, original}"]
    skipcr -- yes --> done
    cr --> done

    done(["return {dataset, dataset_name,<br/>output_dir, original, clean, excluded}"])
    done --> ps["_print_summary(result)"]
```

**Key sequence (who calls whom, with the data passed):**

```mermaid
sequenceDiagram
    autonumber
    actor U as User / CLI
    participant R as run_splits
    participant C as config.load_config
    participant D as download.resolve_data_path
    participant S as splitting.get_splitter
    participant V as validation.validate_dataset
    participant SP as splitting.split_dataset
    participant E as export.export_splits
    participant CR as croissant.save_croissant

    U->>R: dataset, data_path, n_folds, max_workers
    R->>C: load_config(slug)
    C-->>R: DatasetConfig
    R->>D: resolve_data_path(path, config)
    D-->>R: dataset root Path
    R->>S: get_splitter(slug)
    S-->>R: DatasetSplitter
    R->>S: load_metadata() + get_stratification_labels()
    S-->>R: df, labels
    R->>V: validate_dataset(path, config, ...)
    V-->>R: ValidationResult (original_df, clean_df, summary)
    R->>SP: split_dataset(df, labels, config, n_folds)
    SP-->>R: SplitResult (folds, default split mapping)
    R->>E: export_splits(split_result, val_result, out, config)
    E-->>R: stats {original, clean, excluded}
    R->>CR: save_croissant(config, version_dir) ×2
    CR-->>R: croissant.json paths
    R-->>U: summary dict
```

---

## 3. Config system — `load_config()`

A YAML file is parsed by a set of `_parse_*` helpers into one typed
`DatasetConfig` (with nested dataclasses). Required fields are validated before
construction.

```mermaid
flowchart TD
    A["load_config(dataset_slug)"] --> B{"configs/&lt;slug&gt;.yaml<br/>exists?"}
    B -- no --> E1["raise FileNotFoundError<br/>(lists list_available_configs())"]
    B -- yes --> C["yaml.safe_load()"]
    C --> D{"required fields present?<br/>name, slug, version, url,<br/>metadata_csv, record_id_column,<br/>label_column"}
    D -- missing --> E2["raise ValueError"]
    D -- ok --> P

    subgraph P["parse helpers → nested dataclasses"]
        direction LR
        pc["_parse_creators → CreatorInfo[]"]
        ps["_parse_stratification → StratificationConfig"]
        pv["_parse_validation → ValidationConfig"]
        pp["_parse_predefined_splits → PredefinedSplitConfig"]
        pcr["_parse_croissant → CroissantConfig"]
        psp["_parse_signal_path_columns → dict[int,str]"]
    end

    P --> R["DatasetConfig<br/>(typed, passed to ALL other modules)"]
```

**Config dataclass model** (`config.py`):

```mermaid
classDiagram
    class DatasetConfig {
        +str name, slug, version, url
        +str download_url, license, citation, doi
        +str signal_format
        +int leads, default_sampling_rate
        +list~int~ sampling_rates
        +str metadata_csv, record_id_column
        +str patient_id_column
        +dict~int,str~ signal_path_columns
        +str label_column, label_format
        +bool has_predefined_splits
    }
    class CreatorInfo {
        +str type, name, url
    }
    class StratificationConfig {
        +str method
        +str mapping_source, superclass_column
    }
    class ValidationConfig {
        +int expected_leads
        +dict~int,int~ expected_samples
        +list~str~ checks
        +tuple amplitude_range_mv
    }
    class PredefinedSplitConfig {
        +str column
        +dict~str,list~ fold_mapping
    }
    class CroissantConfig {
        +list~str~ keywords
        +str rai_data_collection
        +str rai_data_biases
        +str rai_personal_sensitive_info
    }
    DatasetConfig "1" --> "*" CreatorInfo
    DatasetConfig --> StratificationConfig
    DatasetConfig --> ValidationConfig
    DatasetConfig --> PredefinedSplitConfig
    DatasetConfig --> CroissantConfig
```

---

## 4. Validation engine — `validate_dataset()`

Reads the metadata CSV, loads each signal file, and runs the configured quality
checks **in parallel** via `ProcessPoolExecutor`. Produces both an `original_df`
(all records + flags) and a `clean_df` (valid only).

```mermaid
flowchart TD
    VD["validate_dataset(data_path, config,<br/>sampling_rate, max_workers)"] --> rate["resolve rate &<br/>signal_path_columns[rate]"]
    rate --> read["pd.read_csv(metadata_csv)"]
    read --> build["build (record_id, record_path) list<br/>(strip wfdb suffix)"]
    build --> cd["_config_to_dict(config)<br/>(picklable for subprocesses)"]

    cd --> mode{max_workers &gt; 1?}
    mode -- yes --> pool["ProcessPoolExecutor<br/>submit _validate_single_record per record"]
    mode -- no --> seq["sequential loop"]
    pool -. "on failure" .-> seq

    pool --> collect["collect RecordValidation[]"]
    seq --> collect

    collect --> mapcols["map is_valid + quality_issues<br/>onto DataFrame by record_id"]
    mapcols --> summ["build summary: check → failed_count"]
    summ --> dfs["original_df = all + flags<br/>clean_df = df[is_valid], flags dropped"]
    dfs --> VR(["ValidationResult<br/>(original_df, clean_df, record_validations,<br/>summary, total, valid, excluded)"])
```

**Per-record check dispatch** — `_validate_single_record()` → `CHECK_REGISTRY`:

```mermaid
flowchart TD
    VSR["_validate_single_record(record_id, path, ...)"] --> load["_load_signal(path, format)<br/>((wfdb.rdrecord))"]
    load -- raises --> ch["record corrupt_header / load_error<br/>→ is_valid = False"]
    load -- "(leads, samples) ndarray" --> loop["for each check_name in config.validation.checks"]

    loop --> reg{"CHECK_REGISTRY[name]"}
    reg --> c1["check_missing_leads<br/>(all-NaN / all-zero lead)"]
    reg --> c2["check_nan_values<br/>(any NaN sample)"]
    reg --> c3["check_truncated_signal<br/>(samples &lt; expected_samples[rate])"]
    reg --> c4["check_flat_line<br/>(lead var &lt; 1e-6)"]
    reg --> c5["check_amplitude_outlier<br/>(outside amplitude_range_mv)"]
    note["corrupt_header handled in engine,<br/>not in registry"]

    c1 --> agg["collect issue strings"]
    c2 --> agg
    c3 --> agg
    c4 --> agg
    c5 --> agg
    agg --> RV(["RecordValidation<br/>(record_id, is_valid=len(issues)==0, issues)"])
```

The report side: `generate_report()` / `save_report()` (in `validation/report.py`)
turn a `ValidationResult` into `validation_report.json` with per-check stats and
the full excluded-records list.

---

## 5. Splitting framework — `split_dataset()` + strategies

Two pieces: a **strategy** (how to load metadata & derive stratification labels,
dataset-specific) and the **engine** (how to assign folds, universal). The
registry picks the strategy; `GenericSplitter` is the config-only fallback.

```mermaid
flowchart TD
    GS["get_splitter(slug)"] --> reg{"slug in _REGISTRY?"}
    reg -- ptbxl --> PT["PTBXLSplitter"]
    reg -- chapman_shaoxing --> CH["ChapmanSplitter"]
    reg -- "else" --> GEN["GenericSplitter (fallback)"]

    PT --> meth
    CH --> meth
    GEN --> meth
    meth["load_metadata() → df<br/>get_stratification_labels() → labels"]

    meth --> SD["split_dataset(df, labels, config, n_folds)"]
    SD --> dec{routing}
    dec -- "has_predefined_splits" --> P1["_split_predefined<br/>(read fold column, e.g. PTB-XL strat_fold)"]
    dec -- "patient_id_column set" --> P2["_split_grouped<br/>((StratifiedGroupKFold)) — no patient leakage"]
    dec -- "otherwise" --> P3["_split_simple<br/>((StratifiedKFold))"]

    P1 --> SR
    P2 --> SR
    P3 --> SR
    SR(["SplitResult<br/>folds{1..N}, default_train/val/test_folds,<br/>stratify_column, group_column, split_metadata"])
```

**Strategy class hierarchy** (`splitting/base.py` + `strategies/`):

```mermaid
classDiagram
    class DatasetSplitter {
        <<abstract>>
        +load_metadata(path, config)* DataFrame
        +get_stratification_labels(df, config)* Series
    }
    class PTBXLSplitter {
        +load_metadata() renames filename_lr/hr
        +get_stratification_labels() SCP→superclass
    }
    class ChapmanSplitter {
        +load_metadata() prepends ECGData/
        +get_stratification_labels() Rhythm column
    }
    class GenericSplitter {
        +load_metadata() plain read_csv
        +get_stratification_labels() label_column as-is
    }
    DatasetSplitter <|-- PTBXLSplitter
    DatasetSplitter <|-- ChapmanSplitter
    DatasetSplitter <|-- GenericSplitter

    class SplitResult {
        +dict~int,DataFrame~ folds
        +list default_train_folds
        +list default_val_folds
        +list default_test_folds
        +train() property
        +val() property
        +test() property
        +get_fold(n)
        +get_kfold_split(val_fold, test_fold)
    }
```

PTB-XL's stratification detail — `get_stratification_labels()` maps SCP codes to
diagnostic superclasses (`NORM, MI, STTC, HYP, CD`):

```mermaid
flowchart LR
    raw["scp_codes string<br/>'{IMI: 100.0, ...}'"] --> parse["_parse_scp_codes()<br/>((ast.literal_eval))"]
    parse --> map["_get_superclass()<br/>sum scores via SCP_TO_SUPERCLASS"]
    map --> out["dominant superclass<br/>(or 'OTHER')"]
```

---

## 6. Export — `export_splits()`

Writes **minimal-column** fold CSVs for both `original/` (with quality flags) and
`clean/` (valid only) versions, plus the validation report. Full metadata stays
in the source CSV; users join on `record_id`.

```mermaid
flowchart TD
    ES["export_splits(split_result, validation_result, output_dir, config)"] --> bsc["_build_split_column()<br/>fold → train/val/test"]
    bsc --> master["concat all folds → master_df<br/>add 'fold' + 'default_split' columns"]
    master --> merge["merge is_valid + quality_issues<br/>from validation_result.original_df"]
    merge --> sort["sort by record_id (deterministic)"]

    sort --> orig["ORIGINAL version"]
    sort --> clean["CLEAN version"]

    subgraph orig_b["original/"]
        orig --> oc["_minimal_columns(include_quality=True)"]
        oc --> osel["_select_columns()"]
        osel --> ofolds["write folds.csv"]
        osel --> osplit["_write_split_csvs()<br/>train/ val/ test/ fold_N.csv"]
    end

    subgraph clean_b["clean/"]
        clean --> filt["keep rows is_valid==True"]
        filt --> cc["_minimal_columns(include_quality=False)"]
        cc --> csel["_select_columns()"]
        csel --> cfolds["write folds.csv"]
        csel --> csplit["_write_split_csvs()"]
    end

    osplit --> rep["save_report() → validation_report.json"]
    csplit --> rep
    rep --> stats(["stats dict<br/>original/clean per-split counts + excluded"])
```

**Resulting on-disk layout:**

```
output/<slug>/
├── original/
│   ├── folds.csv
│   ├── train/fold_1.csv … fold_8.csv
│   ├── val/fold_9.csv
│   ├── test/fold_10.csv
│   └── croissant.json
├── clean/
│   ├── folds.csv
│   ├── train/ val/ test/ …
│   └── croissant.json
└── validation_report.json
```

---

## 7. Croissant metadata — `save_croissant()`

Generates MLCommons Croissant 1.1 JSON-LD describing every CSV (with SHA-256
hashes) and the train/val/test record sets. Uses `mlcroissant` if available, with
a hand-built JSON-LD fallback.

```mermaid
flowchart TD
    SC["save_croissant(config, splits_dir, output_path, version)"] --> GC["generate_croissant()"]
    GC --> disc["_discover_csv_files()<br/>rglob('*.csv')"]
    disc --> fo["FileObject per CSV<br/>+ _sha256(file)"]
    fo --> rs["RecordSet per split<br/>(train/val/test):<br/>infer Fields via _infer_field_type"]
    rs --> cre["map config.creators →<br/>((mlc.Organization / mlc.Person))"]
    cre --> build{"build mlc.Metadata()<br/>succeeds?"}
    build -- yes --> tj["metadata.to_json()"]
    build -- no --> man["_build_manual_jsonld()<br/>(fallback dict)"]
    tj --> write["json.dump → croissant.json"]
    man --> write
    write --> done([Path to croissant.json])

    VC["validate_croissant(path)<br/>→ (is_valid, errors)"]
```

---

## 8. Consuming a benchmark — `ECGDataset` at training time

The consumer side. `ECGDataset` reads fold CSVs (from HuggingFace Hub by default,
or local disk) to know *which* records belong to a split, then lazily loads each
signal file on `__getitem__`. `ecg_collate_fn` batches heterogeneous samples.

Two selection axes, independent of each other:

- **Which records.** `fold_numbers` narrows a split to specific folds. Because each
  fold was exported under exactly one split, `split=None` + `fold_numbers` is the
  only way to select folds across split boundaries (custom cross-validation); it
  routes through `folds.csv` and `_filter_master()` rather than the per-split files.
- **Which samples.** `window=(start, length)` is pushed into the reader —
  `sampfrom`/`sampto` for wfdb, `skiprows`/`max_rows` for csv — so the discarded
  samples are never decoded. Together with `leads=`, `units=` and `transform=` these
  are read-time adapters: they shape the returned tensor and never affect the source
  files, the fold CSVs, or validation. Note `validation/engine.py` keeps its own
  window-less copy of `_load_signal`, because validation must always see whole
  records.

```mermaid
flowchart TD
    init["ECGDataset(dataset, split, version, data_path,<br/>fold_numbers, window, leads, units, metadata_source)"] --> cfg["load_config() if slug<br/>→ self.config"]
    cfg --> rdp["resolve_data_path()<br/>→ self.data_path (signals)"]
    rdp --> lmeta["_load_metadata(fold_numbers)"]

    lmeta --> src{metadata_source}
    src -- "hf (default)" --> hf["_load_from_hf()<br/>((hf_hub_download)) fold CSVs<br/>from vlbthambawita/ECGBench"]
    src -- local --> loc["_load_from_local()<br/>_read_fold_csvs() from disk"]
    hf --> fil["_filter_master()<br/>by default_split and/or fold<br/>(split=None ⇒ fold only)"]
    loc --> fil
    fil --> mdf["self.metadata_df"]

    mdf --> getitem["__getitem__(idx)"]
    getitem --> sig["_load_signal(path, format, scale, window)<br/>((wfdb.rdrecord sampfrom/sampto))<br/>→ (leads, window)"]
    sig --> t["torch tensor<br/>→ leads → units → transform"]
    t --> dict["sample dict:<br/>signal, record_id, split, fold,<br/>+ other metadata (_parse_dict_string)"]

    dict --> collate["ecg_collate_fn(batch)<br/>stack tensors, keep dicts/str as lists"]
    collate --> batch(["batch dict → DataLoader"])
```

**DataLoader sequence:**

```mermaid
sequenceDiagram
    autonumber
    participant DL as torch DataLoader
    participant DS as ECGDataset
    participant W as wfdb
    participant CF as ecg_collate_fn

    Note over DS: __init__ already loaded metadata_df<br/>(HF Hub or local fold CSVs)
    loop batch_size times
        DL->>DS: __getitem__(idx)
        DS->>W: rdrecord(signal_path, sampfrom, sampto)<br/>(window=None ⇒ whole record)
        W-->>DS: p_signal (leads × window)
        DS-->>DL: {signal, record_id, split, fold, ...}
    end
    DL->>CF: ecg_collate_fn(samples)
    CF-->>DL: {signal: (B,leads,samples), record_id: [...], ...}
```

---

## 9. Download resolution — `resolve_data_path()`

The single entry point all modules use to locate signal files. Returns a local
path, the cache, or triggers an auto-download.

```mermaid
flowchart TD
    R["resolve_data_path(data_path, config, auto_download)"] --> given{data_path given?}
    given -- "yes and exists" --> ret1([return Path])
    given -- "yes but missing" --> err1["raise FileNotFoundError"]
    given -- no --> cache{"~/.ecgbench/datasets/&lt;slug&gt;/<br/>exists?"}
    cache -- yes --> find["_find_metadata_csv()<br/>(walk 1 level deep)"]
    find -- found --> ret2([return dataset root])
    cache -- no --> dl{"auto_download and<br/>config.download_url?"}
    dl -- yes --> DD["download_dataset(config)"]
    dl -- no --> err2["raise FileNotFoundError<br/>(manual download hint)"]

    DD --> arch["urllib download archive<br/>(_get_archive_type)"]
    arch --> http{HTTP status}
    http -- 403 --> perr["PermissionError<br/>(PhysioNet credentials)"]
    http -- ok --> extract["extract zip/tar (+ tqdm progress)"]
    extract --> find2["_find_metadata_csv()"]
    find2 --> ret3([return dataset root])
```

---

## 10. Catalogue & public API surface

The **catalogue** is independent of the heavy pipeline — pure metadata, always
importable. The package uses **lazy imports** (`__getattr__` in `__init__.py`) so
`import ecgbench` never pulls in torch / wfdb / mlcroissant.

```mermaid
flowchart LR
    subgraph eager["Eager (lightweight, always available)"]
        cat["catalogue: list_datasets, search,<br/>get_dataset, categories, to_dataframe"]
        cfg["config: load_config,<br/>list_available_configs, DatasetConfig"]
    end

    subgraph lazy["Lazy via __getattr__ (heavy deps)"]
        ds["dataset: ECGDataset, ecg_collate_fn,<br/>WindowOutOfRangeError"]
        val["validation: validate_dataset, ValidationResult"]
        spl["splitting: split_dataset, SplitResult,<br/>get_splitter, export_splits"]
        crs["croissant: generate/save/validate_croissant"]
        dld["download: download_dataset, resolve_data_path"]
        pipe["pipelines: run_splits, run_croissant, run_upload"]
    end

    src["docs/_datasets/*.md<br/>(or ecgbench/_datasets in wheel)"] --> cat
    yaml["data/configs/*.yaml"] --> cfg
```

**CLI dispatch** (`cli/_main.py` → `ecgbench` console script):

```mermaid
flowchart LR
    main["main(argv)"] --> bp["_build_parser()"]
    bp --> sub{"&lt;command&gt;"}
    sub -- splits --> sp["run_splits()<br/>§2 full pipeline"]
    sub -- croissant --> cr["run_croissant()<br/>standalone §7"]
    sub -- upload --> up["run_upload()<br/>→ HuggingFace Hub"]
    sub -- "list / search / info /<br/>fields / related" --> cat["cli/catalog.py<br/>run_list … run_related<br/>§11 views over MetadataStore"]
    sub -- "metadata build /<br/>snapshot / export" --> mb["cli/metadata.py<br/>run_metadata_build / _snapshot / _export"]
    sub -- records --> rec["cli/records.py<br/>run_records() §11.4"]
    sub -- mcp --> mcp["cli/mcp.py<br/>run_mcp() §11.5"]
```

`run_upload()` walks `output/<slug>/{original,clean}/**.csv` plus
`validation_report.json` / `croissant.json`, resolves an HF token
(arg → `HF_TOKEN` → `HUGGINGFACE_HUB_TOKEN` → `.env`), and pushes each file via
`HfApi.upload_file` (or lists them under `--dry-run`).

---

## 11. Metadata layer — `MetadataStore`, exports, records, MCP

The metadata layer is the **query** journey: one typed record per dataset,
`DatasetMeta`, merged from the catalogue front matter and the YAML config,
searchable by any alias, and read by every surface that answers "what exists?".
It is standard-library only (`import ecgbench` loads it in ~14 ms, and the
JSON is parsed on first use); pandas appears only in `records.py`, which is
imported on demand.

### 11.1 Build — `build_model()` → `metadata.json` + `metadata.sqlite`

```mermaid
flowchart TB
    MD["docs/_datasets/&lt;slug&gt;.md<br/>front matter (config_slug:, status, …)"]
    YAML["data/configs/&lt;slug&gt;.yaml"]
    FIELDS["labels/&lt;slug&gt;.py FIELDS<br/>or labels.fields: in YAML"]
    SNAP["data/snapshots/&lt;slug&gt;.json"]

    subgraph BUILD["metadata/build.py"]
        bm["build_model()"]
        dfs["declared_fields_from_source()<br/>(reads FIELDS with ast — no pandas)"]
        merge["merge per dataset:<br/>SOURCE_PRECEDENCE config &gt; catalogue,<br/>except name; every value also a Fact"]
        state["implementation_state derived:<br/>catalogue_only / config / config_labels / published"]
        tj["to_json() + content_digest()"]
        ws["write_sqlite()<br/>dataset · alias · fact · relation · field · artefact<br/>+ dataset_fts (FTS5 porter unicode61)"]
    end

    subgraph SNAPB["metadata/snapshot.py"]
        bs["build_snapshot()<br/>manifest.json + validation_report.json<br/>→ counts, fold_digest, SHA-256s"]
        cif["check_id_free()<br/>refuses anything that looks like a record list"]
    end

    OUT["output/&lt;slug&gt;/"] --> bs --> cif --> SNAP
    MD --> bm
    YAML --> bm
    FIELDS --> dfs --> bm
    SNAP --> bm
    bm --> merge --> state --> tj
    tj --> JSON["ecgbench/data/metadata.json<br/>(committed)"]
    tj --> ws --> SQL["ecgbench/data/metadata.sqlite<br/>(generated, gitignored,<br/>shipped via hatch artifacts)"]
    tj --> WEB["docs/_data/metadata.json<br/>(committed website copy)"]
```

The sqlite file is rebuilt in three places: by `hatch_build.py` into every
wheel, by `open_store()` when it is missing or its digest differs from the
JSON's, and in a source checkout whenever the mtime+size fingerprint of the
sources changes. `ecgbench metadata build --check` and
`tests/test_metadata.py` fail on drift; the snapshot step is
`ecgbench metadata snapshot --dataset <slug>` after a split run.

### 11.2 Read — `MetadataStore` and the views over it

```mermaid
flowchart LR
    JSON["metadata.json"] --> os["open_store()"]
    SQL["metadata.sqlite"] -. attach_index .-> os
    os --> store["MetadataStore"]

    store --> get["get(key) / resolve(key)<br/>AliasIndex: ptbxl = ptb-xl = PTB-XL"]
    store --> search["search(query, **filters)<br/>bm25() × BM25_WEIGHTS + STATE_PENALTY;<br/>substring fallback without FTS5"]
    store --> related["related(key)<br/>leakage edges, both directions"]

    subgraph VIEWS["Views"]
        cli["ecgbench list / search / info /<br/>fields / related  (cli/catalog.py)"]
        py["ecgbench.get_metadata /<br/>search_metadata / related_metadata"]
        exp["metadata/export.py<br/>to_schema_org · to_croissant(_collection) · to_website"]
        rec["metadata/records.py<br/>load_records · query_records  §11.4"]
        mcp["metadata/mcp_server.py<br/>ToolFunctions  §11.5"]
    end

    get --> cli & py & exp & rec & mcp
    search --> cli & py & mcp
    related --> cli & py & mcp
```

### 11.3 Exports — `metadata/export.py`

Three deterministic views over the same model, written by
`ecgbench metadata export`: a `schema.org/Dataset` block per dataset (embedded
as JSON-LD in `docs/_layouts/dataset.html` from the committed website copy), a
Croissant 1.1 `DataCatalog` whose `distribution` names the published fold CSVs
with the snapshot's SHA-256 and whose withheld datasets carry an ODRL
prohibition instead, and the website JSON keyed by catalogue slug.
`validate_croissant` runs `mlcroissant` over every member — as
`mlc.Dataset`, since `mlc.Metadata` alone never reports an error.

### 11.4 Records — `metadata/records.py` (`ecgbench records`)

The one per-record surface. It also owns the fold-table readers `ECGDataset`
delegates to, so the two cannot disagree about the Hub layout or the
identifier dtypes.

```mermaid
flowchart TB
    key["dataset id / alias / DatasetConfig"] --> rc["resolve_config()<br/>(catalogue-only → RecordsUnavailableError)"]
    rc --> lft["load_fold_table(source=…)"]
    lft -- hf --> hub["fetch_hub_fold_table()<br/>SplitsNotPublishedError before any download"]
    lft -- local --> loc["load_local_fold_table(splits_dir)<br/>output/&lt;slug&gt;/ or the data dir"]
    hub --> rfc["read_fold_csv()<br/>identifier_dtypes() keeps afdb's 00735"]
    loc --> rfc
    rfc --> fft["filter_fold_table(split, fold_numbers)"]
    fft --> jl["join_labels()<br/>left join on record id; fold columns win"]
    ll["labels.load_labels()"] --> jl
    jl --> df["pandas DataFrame"]
    df --> qr["query_records(df, sql)<br/>DuckDB view `records`,<br/>external access off, config locked"]
    df --> fmt["cli/records.py<br/>table · json · csv · parquet"]
    ECG["ECGDataset._load_from_hf /<br/>_load_from_local"] -. delegate .-> hub & loc
```

### 11.5 MCP server — `metadata/mcp_server.py` (`ecgbench mcp`)

```mermaid
flowchart LR
    agent(("MCP client<br/>e.g. Claude Code")) <-- "stdio JSON-RPC" --> srv["build_mcp_server()<br/>MCPServer (mcp 2.x) / FastMCP (1.x)"]
    srv --> tf["ToolFunctions(store)"]
    tf --> t1["search_datasets"] & t2["list_datasets"] & t3["get_dataset"] & t4["list_fields"] & t5["related_datasets"]
    t1 & t2 & t3 & t4 & t5 --> store["MetadataStore"]
    tf -. "errors as the SDK's ToolError,<br/>message intact" .-> agent
```

The tools are plain methods returning JSON-ready dicts, registered on the SDK
server by name, so they run and are tested without the SDK; the schema comes
from their type hints and docstrings. The server reads the bundled index and
nothing else — no records, signals or network.

---

## Function index (quick reference)

| Subsystem | Module | Key functions / classes |
|---|---|---|
| Config | `config.py` | `load_config`, `list_available_configs`, `DatasetConfig`, `CreatorInfo`, `StratificationConfig`, `ValidationConfig`, `PredefinedSplitConfig`, `CroissantConfig` |
| Catalogue | `catalogue.py` | `list_datasets`, `search`, `get_dataset`, `categories`, `to_dataframe`, `CatalogueEntry` |
| Download | `download.py` | `resolve_data_path`, `download_dataset`, `_find_metadata_csv`, `_get_archive_type` |
| Validation | `validation/engine.py` | `validate_dataset`, `_validate_single_record`, `_load_signal`, `_config_to_dict`, `ValidationResult`, `RecordValidation` |
| Validation | `validation/checks.py` | `check_missing_leads`, `check_nan_values`, `check_truncated_signal`, `check_flat_line`, `check_amplitude_outlier`, `CHECK_REGISTRY` |
| Validation | `validation/report.py` | `generate_report`, `save_report` |
| Splitting | `splitting/engine.py` | `split_dataset`, `_split_predefined`, `_split_grouped`, `_split_simple` |
| Splitting | `splitting/base.py` | `DatasetSplitter` (ABC), `SplitResult` |
| Splitting | `splitting/registry.py` | `register`, `get_splitter` |
| Splitting | `strategies/` | `PTBXLSplitter`, `ChapmanSplitter`, `GenericSplitter`, `_get_superclass`, `_parse_scp_codes` |
| Splitting | `splitting/export.py` | `export_splits`, `_minimal_columns`, `_select_columns`, `_build_split_column`, `_write_split_csvs` |
| Croissant | `croissant.py` | `generate_croissant`, `save_croissant`, `validate_croissant`, `_build_manual_jsonld`, `_discover_csv_files`, `_sha256`, `_infer_field_type` |
| Dataset | `dataset.py` | `ECGDataset` (`_load_metadata`, `_load_from_hf`, `_load_from_local` — both delegate to `metadata/records.py`, `__getitem__`), `ecg_collate_fn`, `_load_signal`, `_resolve_window`, `_resolve_leads`, `_resolve_units`, `WindowOutOfRangeError`, `_parse_dict_string` |
| Metadata | `metadata/model.py` | `DatasetMeta`, `Fact`, `FieldMeta`, `RelationMeta`, `ArtefactMeta`, `SOURCE_PRECEDENCE`, `IMPLEMENTATION_STATES` |
| Metadata | `metadata/build.py` | `build_model`, `build_all`, `to_json`, `write_sqlite`, `content_digest`, `diff_exports`, `declared_fields_from_source`, `is_source_checkout` |
| Metadata | `metadata/store.py` | `MetadataStore` (`get`, `resolve`, `search`, `search_ranked`, `related`, `all`), `open_store`, `SearchHit`, `MetadataQueryError`, `BM25_WEIGHTS`, `STATE_PENALTY` |
| Metadata | `metadata/identity.py` | `AliasIndex`, `resolve`, `UnknownDatasetError` |
| Metadata | `metadata/snapshot.py` | `build_snapshot`, `write_snapshot`, `load_snapshots`, `check_id_free` |
| Metadata | `metadata/export.py` | `to_schema_org`, `to_croissant`, `to_croissant_collection`, `to_website`, `write_website_json`, `validate_croissant` |
| Metadata | `metadata/records.py` | `load_records`, `query_records`, `query_hub`, `load_fold_table`, `fetch_hub_fold_table`, `load_local_fold_table`, `read_fold_csv`, `filter_fold_table`, `join_labels`, `SplitsNotPublishedError`, `RecordsUnavailableError`, `RecordsQueryError` |
| Metadata | `metadata/mcp_server.py` | `ToolFunctions` (`search_datasets`, `list_datasets`, `get_dataset`, `list_fields`, `related_datasets`), `build_mcp_server`, `run_server` |
| Pipelines / CLI | `cli/` | `run_splits`, `run_croissant`, `run_upload`, `run_list`, `run_search`, `run_info`, `run_fields`, `run_related`, `run_metadata_build`, `run_metadata_check`, `run_metadata_snapshot`, `run_metadata_export`, `run_records`, `run_mcp`, `main`, `_build_parser` |
| Public API | `__init__.py` | eager catalogue+config+metadata (all standard-library); lazy `__getattr__` for everything else |
