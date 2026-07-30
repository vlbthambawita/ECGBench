#!/usr/bin/env python3
"""
Download the Chapman-Shaoxing ECG Database from figshare.

This is the original 10,646-patient release (figshare collection 4560497 v2), not
the 45,152-record PhysioNet `ecg-arrhythmia` merge — see the `ecg_arrhythmia`
config for that one.

ECGBench's own auto-download (`download_url` in the config) handles a single
archive; this collection is six separate files, and its metadata ships as .xlsx
which `pandas.read_csv` cannot open. So acquisition gets a script:

  1. fetch all six files, verifying figshare's md5 for each
  2. extract ECGData.zip and ECGDataDenoised.zip
  3. convert the four .xlsx files to .csv beside them, so the pipeline can read
     the metadata without an Excel dependency at runtime

Files are skipped when already present with the right md5, so re-running is cheap
and an interrupted download can be resumed by just running it again.

Requires `openpyxl` for step 3:  pip install openpyxl

Usage:
  python examples/download_chapman_figshare.py --dest /path/to/chapman-figshare/
  python examples/download_chapman_figshare.py --dest ... --skip-denoised
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

COLLECTION_URL = "https://figshare.com/collections/ChapmanECG/4560497/2"

# figshare file ids, resolved from the API on 2026-07-30. md5 and size come from
# figshare itself, so a mismatch means a corrupt or truncated download.
FILES = [
    # (name, download_url, md5, size_bytes, is_denoised)
    ("Diagnostics.xlsx", "https://ndownloader.figshare.com/files/15653771",
     "f9795256", 1_040_000, False),
    ("AttributesDictionary.xlsx", "https://ndownloader.figshare.com/files/15653762",
     "7c12a705", 12_000, False),
    ("RhythmNames.xlsx", "https://ndownloader.figshare.com/files/15651296",
     "a7af9552", 10_000, False),
    ("ConditionNames.xlsx", "https://ndownloader.figshare.com/files/15651293",
     "a37bbf63", 10_000, False),
    ("ECGData.zip", "https://ndownloader.figshare.com/files/15651326",
     "2bf32d64", 754_600_000, False),
    ("ECGDataDenoised.zip", "https://ndownloader.figshare.com/files/15652862",
     "049caea0", 2_008_700_000, True),
]

CHUNK = 1 << 20  # 1 MiB


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity check against figshare, not security
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_expected_md5s() -> dict[str, str]:
    """Ask figshare for the current md5 of every file in the collection.

    The hardcoded values above are prefixes only, so the authoritative digests are
    fetched at run time. Falls back to prefix matching if the API is unreachable.
    """
    import json

    out: dict[str, str] = {}
    try:
        req = urllib.request.Request(
            "https://api.figshare.com/v2/collections/4560497/articles?page_size=50",
            headers={"User-Agent": "ecgbench-download"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            articles = json.load(response)
        for article in articles:
            req = urllib.request.Request(
                f"https://api.figshare.com/v2/articles/{article['id']}",
                headers={"User-Agent": "ecgbench-download"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                for f in json.load(response)["files"]:
                    out[f["name"]] = f["computed_md5"]
    except Exception as e:  # noqa: BLE001 - offline is not fatal, just less strict
        print(f"  ! could not reach the figshare API ({e}); using md5 prefixes", flush=True)
    return out


def _download(url: str, dest: Path, expected_size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ecgbench-download"})
    with urllib.request.urlopen(req, timeout=120) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or expected_size)
        done = 0
        step = max(total // 20, CHUNK)
        next_report = step
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            if done >= next_report:
                pct = 100 * done / total if total else 0
                print(f"    {done / 1e6:8.0f} MB / {total / 1e6:.0f} MB ({pct:.0f}%)",
                      flush=True)
                next_report += step
    tmp.replace(dest)


def _verify(path: Path, expected: str) -> bool:
    """True if the file's md5 matches, tolerating a prefix as the expectation."""
    actual = _md5(path)
    return actual == expected if len(expected) == 32 else actual.startswith(expected)


def _extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        top = {Path(m).parts[0] for m in members if m.strip("/")}
        # Both archives already contain their own top-level directory; extracting
        # into dest therefore lands at dest/ECGData/... as the config expects.
        print(f"    {len(members)} entries, top level: {sorted(top)}", flush=True)
        zf.extractall(dest)


def _xlsx_to_csv(xlsx: Path) -> Path:
    """Write <name>.csv beside <name>.xlsx.

    The pipeline reads the metadata with pandas.read_csv, and validate_dataset
    re-reads it from disk, so a real CSV has to exist — converting once here beats
    requiring openpyxl at load time.
    """
    import pandas as pd

    csv = xlsx.with_suffix(".csv")
    frame = pd.read_excel(xlsx)
    frame.to_csv(csv, index=False)
    return csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Chapman-Shaoxing ECG Database from figshare"
    )
    parser.add_argument("--dest", required=True, help="Directory to download into")
    parser.add_argument("--skip-denoised", action="store_true",
                        help="Skip ECGDataDenoised.zip (2.0 GB of the 2.8 GB total)")
    parser.add_argument("--keep-archives", action="store_true",
                        help="Keep the .zip files after extracting")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Source: {COLLECTION_URL}")
    print(f"Dest:   {dest}\n")

    wanted = [f for f in FILES if not (args.skip_denoised and f[4])]
    print("Resolving checksums from the figshare API...")
    live_md5 = _fetch_expected_md5s()

    for name, url, md5_prefix, size, _ in wanted:
        target = dest / name
        expected = live_md5.get(name, md5_prefix)

        if target.exists() and _verify(target, expected):
            print(f"[skip] {name} (already present, md5 ok)")
        else:
            print(f"[get ] {name}  ({size / 1e6:.0f} MB)")
            _download(url, target, size)
            if not _verify(target, expected):
                print(f"  ! md5 mismatch for {name}: got {_md5(target)}, "
                      f"expected {expected}", file=sys.stderr)
                return 1
            print("  md5 ok")

    print("\nExtracting archives...")
    for name, _, _, _, _ in wanted:
        if not name.endswith(".zip"):
            continue
        archive = dest / name
        marker = dest / Path(name).stem
        if marker.is_dir() and any(marker.iterdir()):
            print(f"[skip] {name} (already extracted to {marker.name}/)")
            continue
        print(f"[open] {name}")
        _extract(archive, dest)

    print("\nConverting .xlsx metadata to .csv...")
    for name, _, _, _, _ in wanted:
        if not name.endswith(".xlsx"):
            continue
        try:
            csv = _xlsx_to_csv(dest / name)
        except ImportError:
            print("  ! openpyxl is required to read .xlsx — pip install openpyxl",
                  file=sys.stderr)
            return 1
        print(f"[conv] {name} -> {csv.name}")

    if not args.keep_archives:
        for name, _, _, _, _ in wanted:
            if name.endswith(".zip") and (dest / name).exists():
                (dest / name).unlink()
                print(f"[rm  ] {name}")

    print("\nDone. Contents:")
    for child in sorted(dest.iterdir()):
        if child.is_dir():
            n = sum(1 for _ in child.rglob("*") if _.is_file())
            print(f"  {child.name}/  ({n} files)")
        else:
            print(f"  {child.name}  ({child.stat().st_size / 1e6:.1f} MB)")
    used = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"\nDisk used: {used / 1e9:.2f} GB")
    print(f"\nNext:\n  ecgbench splits --dataset chapman_shaoxing --data-path '{dest}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
