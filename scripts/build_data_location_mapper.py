import csv, json, os
from datetime import date
from pathlib import Path
import ecgbench
from ecgbench.metadata import open_store

REQ = Path("/work/vajira/DL2026/data")
SEARCH = Path("/global/D1/homes/vajira/data/SEARCH")
REPO = Path("/work/vajira/DL2026/ECGBench")
MARKERS = {"ecgbench_metadata.csv", "ptbxl_database.csv", "record_list.csv"}
SKIP_PREFIXES = ("p0", "p1", "patient", "records", "WFDB", ".cache", "ecgbench_extracted", "files_")

def walk_markers(tree):
    for dirpath, dirnames, filenames in os.walk(tree):
        if len(Path(dirpath).relative_to(tree).parts) > 7:
            dirnames[:] = []
            continue
        for f in filenames:
            if f in MARKERS:
                yield Path(dirpath) / f
        dirnames[:] = [d for d in dirnames if not d.startswith(SKIP_PREFIXES)]

def rows(p):
    with open(p, newline="") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1

def manifest_count(slug):
    m = REPO / "output" / slug / "manifest.json"
    if m.exists():
        return json.load(open(m))["records"]["original"], "manifest"
    r = REPO / "output" / slug / "validation_report.json"
    if r.exists():
        d = json.load(open(r))
        orig = d.get("original")
        if isinstance(orig, dict):
            for k in ("n_records", "total_records", "count", "records"):
                if isinstance(orig.get(k), int):
                    return orig[k], "validation_report"
    return None, None

NAME_TABLE = {
    "cpsc_2018": "cpsc_2018",
    "chapman-shaoxing ecg database_figshare": "chapman_shaoxing",
    "ikem_dataset_v1.0.0": "ikem",
    "medalcare-xl": "medalcare_xl",
    "ningbo_first_hospital_ecg_database": "ningbo_iva",
    "sami-trop": "sami_trop",
    "shandong_provincial_hospital_ecg_database": "sph",
    "zzu_pecg": "zzu_pecg",
    "edgar": "edgar",
    "wilson-central-terminal-ecg-database-1.0.1": "wctecgdb",
    "norwegian-endurance-athlete-ecg-database-1.0.0": "norwegian_athlete_ecg",
    "influence-of-the-mhd-effect-on-12-lead-and-3-lead-ecgs-recorded-in-1t-to-7t-mri-scanners-1.0.0": "mhd_effect_ecg_mri",
}
PHYSIONET_SPECIAL = {
    "ptb_xl": "ptbxl", "challenge_2017": "challenge2017", "challenge_2020": "challenge2020",
    "challenge_2021": "challenge2021",
    "mimic_iv_ecg_diagnostic_electrocardiogram_matched_subset_1.0": "mimic_iv_ecg",
}

def guess_slug(root: Path, config_slugs):
    name = root.name.lower()
    if name in NAME_TABLE:
        return NAME_TABLE[name]
    if root.parent.parent.name == "files" or root.parent.name == "files":   # files/<name>[/<version>]
        pn = root.parent.name if root.parent.parent.name == "files" else root.name
        cand = pn.lower().replace("-", "_")
        cand = PHYSIONET_SPECIAL.get(cand, cand)
        if cand in config_slugs:
            return cand
    parts = [p.lower() for p in root.parts]
    if "code-test-827" in parts:
        return "code_test"
    if any("ode-15-pct" in p for p in parts):
        return "code15"
    return None

config_slugs = set(ecgbench.list_available_configs())
found: dict[str, list[dict]] = {}
unmapped = []
for tree in (REQ, SEARCH):
    for mk in sorted(walk_markers(tree)):
        slug = guess_slug(mk.parent, config_slugs)
        if slug is None:
            unmapped.append(str(mk)); continue
        found.setdefault(slug, []).append({"path": str(mk.parent), "marker": mk.name, "marker_rows": rows(mk)})

mit = SEARCH / "two_leads/physionet.org/files/mitdb/1.0.0"
if mit.is_dir():
    n = len([f for f in os.listdir(mit) if f.endswith(".hea")])
    found["mitdb"] = [{"path": str(mit), "marker": "*.hea", "marker_rows": n}]

SECONDARY = {   # same release present again without an ECGBench marker file
    "ecgcipa": [REQ / "cipa-ecg-validation-study-1.0.0"],
    "mhd_effect_ecg_mri": [SEARCH / "physionet.org/files/mhd-effect-ecg-mri/1.0.0"],
    "norwegian_athlete_ecg": [SEARCH / "physionet.org/files/norwegian-athlete-ecg/1.0.0"],
    "wctecgdb": [SEARCH / "physionet.org/files/wctecgdb/1.0.1"],
}

def tree_of(p: str):
    if p.startswith(str(REQ)): return "requested_root"
    if p.startswith(str(SEARCH)): return "search_tree"
    return None

store = open_store()
catalogue = {e.slug: e for e in ecgbench.list_datasets()}
by_config = {e.config_slug: e for e in catalogue.values() if e.config_slug}

datasets = {}
for meta in sorted(store.all(), key=lambda m: m.dataset_id):
    cfg = meta.dataset_id if meta.has_config else None
    cat = by_config[cfg].slug if cfg else meta.dataset_id
    entry = {
        "name": meta.name,
        "catalogue_slug": cat,
        "config_slug": cfg,
        "implementation_state": meta.implementation_state,
        "data_path": None,
        "tree": None,
        "in_requested_root": False,
        "verification": None,
        "other_copies": [],
    }
    if cfg:
        expected, src = manifest_count(cfg)
        cands = found.get(cfg, [])
        # prefer a verified root; prefer the requested root on ties
        def score(c):
            ok = expected is not None and c["marker_rows"] == expected
            return (ok, c["path"].startswith(str(REQ)))
        cands = sorted(cands, key=score, reverse=True)
        if cands:
            best = cands[0]
            entry["data_path"] = best["path"]
            entry["tree"] = tree_of(best["path"])
            entry["in_requested_root"] = best["path"].startswith(str(REQ))
            entry["verification"] = {
                "marker_file": best["marker"],
                "marker_rows": best["marker_rows"],
                "expected_records": expected,
                "expected_source": src,
                "status": ("verified" if expected == best["marker_rows"]
                           else "unverified" if expected is None else "MISMATCH"),
            }
            for c in cands[1:]:
                entry["other_copies"].append({"path": c["path"], "tree": tree_of(c["path"]),
                                              "marker_rows": c["marker_rows"], "has_marker": True})
        for p in SECONDARY.get(cfg, []):
            if p.is_dir():
                entry["other_copies"].append({"path": str(p), "tree": tree_of(str(p)), "has_marker": False})
    datasets[meta.dataset_id] = entry

# Datasets present on disk that ECGBench has no config for
UNCONFIGURED = {
    "HEEDB": (REQ / "HEEDB", "harvard-emory-ecg-database-heedb"),
    "icentia11k-continuous-ecg": (REQ / "one_lead_ecg/physionet.org/files/icentia11k-continuous-ecg/1.0",
                                  "icentia11k-single-lead-continuous-ecg"),
    "vitaldb-arrhythmia": (REQ / "one_lead_ecg/physionet.org/files/vitaldb-arrhythmia/1.0.0",
                           "vitaldb-arrhythmia-database"),
    "mimic3wdb-matched": (REQ / "three_leads_ecgs/physionet.org/files/mimic3wdb-matched/1.0",
                          "mimic-iii-waveform-database-matched-subset"),
}
on_disk_only = {}
for k, (p, cat_slug) in UNCONFIGURED.items():
    if p.exists():
        assert cat_slug in catalogue, cat_slug
        on_disk_only[k] = {"data_path": str(p), "tree": tree_of(str(p)), "catalogue_slug": cat_slug,
                           "note": "present on disk; catalogue entry exists but no ECGBench config"}
        if cat_slug in datasets and datasets[cat_slug]["data_path"] is None:
            datasets[cat_slug]["data_path"] = str(p)
            datasets[cat_slug]["tree"] = tree_of(str(p))
            datasets[cat_slug]["in_requested_root"] = str(p).startswith(str(REQ))
            datasets[cat_slug]["verification"] = {"status": "no_config", "marker_file": None}

n_cfg = sum(1 for d in datasets.values() if d["config_slug"])
n_loc = sum(1 for d in datasets.values() if d["data_path"])
n_req = sum(1 for d in datasets.values() if d["in_requested_root"])
n_ver = sum(1 for d in datasets.values() if d["verification"] and d["verification"]["status"] == "verified")
out = {
    "_about": {
        "generated": date.today().isoformat(),
        "purpose": "Maps every ECGBench dataset id to the directory to pass as --data-path / data_path.",
        "requested_root": str(REQ),
        "search_tree": str(SEARCH),
        "how_located": "A dataset root is the directory holding the file named in the config's metadata_csv "
                       "(ecgbench_metadata.csv, ptbxl_database.csv or record_list.csv); mitdb has no such file and "
                       "is the WFDB directory itself. 'verified' means that file's row count equals the record count "
                       "in output/<slug>/manifest.json (or the validation report).",
        "key": "metadata dataset_id: the config slug where a config exists, else the catalogue slug",
        "counts": {"datasets": len(datasets), "with_config": n_cfg, "located": n_loc,
                   "in_requested_root": n_req, "verified": n_ver,
                   "catalogue_only": len(datasets) - n_cfg},
    },
    "datasets": datasets,
    "on_disk_without_config": on_disk_only,
    "unmapped_marker_files": unmapped,
}
(REPO / "data_location_mapper.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["_about"]["counts"], indent=1))
for k, d in datasets.items():
    if d["config_slug"]:
        v = d["verification"]
        v = v or {}
        print(f"{k:26s} {str(d['tree'] or '-'):15s} {str(v.get('status','NOT FOUND')):10s} {str(v.get('marker_rows','')):>7} {str(v.get('expected_records','')):>7}  {d['data_path']}")
print("unmapped:", unmapped)
