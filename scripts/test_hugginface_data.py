
#!/usr/bin/env python3
"""
Utility script to validate that a Hugging Face dataset file matches the local ECGBench CSV.

By default it downloads `ptbxl/train/fold_1.csv` from `deepsynthbody/ECGBench`
and compares it against the reference file created during preprocessing.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def compute_md5(file_path: Path) -> str:
    """Compute the MD5 checksum for a file."""
    md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compare_csvs(local_csv: Path, hf_csv: Path) -> None:
    """Print summary stats and equality checks for two CSV files."""
    local_df = pd.read_csv(local_csv)
    hf_df = pd.read_csv(hf_csv)

    print(f"Local rows: {len(local_df)}, columns: {len(local_df.columns)}")
    print(f"HuggingFace rows: {len(hf_df)}, columns: {len(hf_df.columns)}")

    if list(local_df.columns) != list(hf_df.columns):
        missing = set(local_df.columns) - set(hf_df.columns)
        extra = set(hf_df.columns) - set(local_df.columns)
        raise ValueError(
            "Column mismatch detected.\n"
            f"  Missing columns in HF file: {sorted(missing)}\n"
            f"  Extra columns in HF file: {sorted(extra)}"
        )

    if len(local_df) != len(hf_df):
        raise ValueError("Row count mismatch between local and HF CSV files.")

    if not local_df.equals(hf_df):
        sample_diff = (
            (local_df != hf_df)
            .any(axis=1)
            .reset_index()
            .rename(columns={0: "diff"})
            .loc[lambda df: df["diff"], "index"]
            .tolist()[:5]
        )
        raise ValueError(
            "CSV contents differ between local and HF files. "
            f"Example differing indices: {sample_diff}"
        )

    print("✓ CSV files are identical.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate ptbxl fold CSV hosted on Hugging Face against local ECGBench CSV."
    )
    parser.add_argument(
        "--hf-repo-id",
        default="deepsynthbody/ECGBench",
        help="Hugging Face dataset repository ID (default: deepsynthbody/ECGBench)",
    )
    parser.add_argument(
        "--hf-file",
        default="ptbxl/train/fold_1.csv",
        help="Path to CSV inside the Hugging Face repo (default: ptbxl/train/fold_1.csv)",
    )
    parser.add_argument(
        "--local-file",
        default="/global/D1/homes/vajira/data/SEARCH/ECGBench/ptbxl/train/fold_1.csv",
        help="Path to the local reference CSV (default: ECGBench ptbxl/train/fold_1.csv)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for Hugging Face downloads.",
    )

    args = parser.parse_args()

    local_csv = Path(args.local_file).expanduser()
    if not local_csv.exists():
        raise FileNotFoundError(f"Local CSV not found: {local_csv}")

    print(f"Downloading {args.hf_file} from {args.hf_repo_id}…")
    hf_file_path = hf_hub_download(
        repo_id=args.hf_repo_id,
        filename=args.hf_file,
        repo_type="dataset",
        cache_dir=args.cache_dir,
    )
    hf_csv = Path(hf_file_path)
    print(f"✓ Downloaded to {hf_csv}")

    print("\nComparing MD5 checksums…")
    local_md5 = compute_md5(local_csv)
    hf_md5 = compute_md5(hf_csv)
    print(f"  Local MD5: {local_md5}")
    print(f"  HF MD5   : {hf_md5}")

    if local_md5 != hf_md5:
        print("MD5 mismatch detected. Performing row-by-row comparison for diagnostics.")
    else:
        print("✓ MD5 checksums match.")

    print("\nComparing CSV contents…")
    compare_csvs(local_csv, hf_csv)


if __name__ == "__main__":
    main()


