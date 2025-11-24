#!/usr/bin/env python3
"""
Script to upload ECGBench dataset folders to Hugging Face Hub.
Uploads folder structure to deepsynthbody/ECGBench dataset repository.
Supports multiple datasets (e.g., ptbxl) with train/val/test splits.
"""

import argparse
from pathlib import Path
from huggingface_hub import create_repo, upload_folder
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def upload_dataset_folder(
    local_path: Path,
    dataset_name: str,
    hf_repo_id: str,
    hf_token: str = None,
    create_repo_if_not_exists: bool = True,
):
    """
    Upload a dataset folder to Hugging Face Hub.
    
    Args:
        local_path: Local path to the dataset folder (e.g., ptbxl)
        dataset_name: Name of the dataset (e.g., 'ptbxl')
        hf_repo_id: Hugging Face repository ID (e.g., 'deepsynthbody/ECGBench')
        hf_token: Hugging Face authentication token (defaults to HF_TOKEN env var)
        create_repo_if_not_exists: Whether to create the repo if it doesn't exist
    """
    if not local_path.exists():
        raise ValueError(f"Local path does not exist: {local_path}")
    
    if not local_path.is_dir():
        raise ValueError(f"Local path is not a directory: {local_path}")
    
    # Get token from argument, .env file, or environment variable
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token is None:
            # Debug: Check what env vars are available
            debug_msg = "Hugging Face token not found. "
            debug_msg += f"Checked HF_TOKEN and HUGGINGFACE_HUB_TOKEN environment variables. "
            debug_msg += f"Current working directory: {os.getcwd()}. "
            debug_msg += "Set HF_TOKEN in .env file (in project root), HF_TOKEN/HUGGINGFACE_HUB_TOKEN environment variable, "
            debug_msg += "or pass --hf-token argument."
            raise ValueError(debug_msg)
    
    # Create repository if it doesn't exist
    if create_repo_if_not_exists:
        try:
            create_repo(
                repo_id=hf_repo_id,
                repo_type="dataset",
                token=hf_token,
                exist_ok=True,
            )
            print(f"Repository {hf_repo_id} is ready (created or already exists)")
        except Exception as e:
            print(f"Note: Repository creation check returned: {e}")
    
    # Upload folder to Hugging Face
    # The folder structure will be preserved in the repo
    remote_folder_path = dataset_name  # e.g., 'ptbxl' in the repo
    
    print(f"Uploading {local_path} to {hf_repo_id}/{remote_folder_path}...")
    
    try:
        upload_folder(
            folder_path=str(local_path),
            repo_id=hf_repo_id,
            repo_type="dataset",
            token=hf_token,
            path_in_repo=remote_folder_path,  # This preserves the dataset name in the repo
            ignore_patterns=[".git*", "__pycache__", "*.pyc", ".DS_Store"],
        )
        print(f"✓ Successfully uploaded {dataset_name} to {hf_repo_id}/{remote_folder_path}")
    except Exception as e:
        print(f"✗ Error uploading {dataset_name}: {e}")
        raise


def main():
    # Load .env file if python-dotenv is available
    env_loaded = False
    if load_dotenv is not None:
        # Try to load .env from the project root (parent of scripts directory)
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent
        env_file = project_root / ".env"
        
        if env_file.exists():
            load_dotenv(env_file, override=True)
            env_loaded = True
            # Verify token was loaded
            token_from_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if token_from_env:
                print(f"✓ Loaded .env file from: {env_file} (token found)")
            else:
                print(f"⚠ Loaded .env file from: {env_file} but HF_TOKEN not found in file")
        else:
            # Also try loading from current directory
            loaded = load_dotenv(override=True)
            if loaded:
                env_loaded = True
                token_from_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                if token_from_env:
                    print(f"✓ Loaded .env file from current directory (token found)")
                else:
                    print(f"⚠ Loaded .env file from current directory but HF_TOKEN not found")
            else:
                print(f"⚠ No .env file found. Looked in: {env_file} and current directory")
    else:
        print("⚠ python-dotenv not installed. Install it with: pip install python-dotenv")
    
    parser = argparse.ArgumentParser(
        description="Upload ECGBench dataset folders to Hugging Face Hub"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/global/D1/homes/vajira/data/SEARCH/ECGBench",
        help="Base directory containing dataset folders (default: /global/D1/homes/vajira/data/SEARCH/ECGBench)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ptbxl"],
        help="List of dataset names to upload (default: ['ptbxl'])",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="deepsynthbody/ECGBench",
        help="Hugging Face repository ID (default: deepsynthbody/ECGBench)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face authentication token (or set HF_TOKEN in .env file or env var)",
    )
    parser.add_argument(
        "--skip-repo-creation",
        action="store_true",
        help="Skip repository creation (use if repo already exists)",
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    print(f"Data directory: {data_dir}")
    print(f"Hugging Face repository: {args.hf_repo_id}")
    print(f"Datasets to upload: {args.datasets}")
    print("-" * 60)
    
    # Upload each dataset
    for dataset_name in args.datasets:
        dataset_path = data_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"⚠ Warning: Dataset folder '{dataset_path}' does not exist. Skipping...")
            continue
        
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"  Local path: {dataset_path}")
        
        try:
            upload_dataset_folder(
                local_path=dataset_path,
                dataset_name=dataset_name,
                hf_repo_id=args.hf_repo_id,
                hf_token=args.hf_token,
                create_repo_if_not_exists=not args.skip_repo_creation,
            )
        except Exception as e:
            print(f"✗ Failed to upload {dataset_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Upload process completed!")
    print(f"You can view your dataset at: https://huggingface.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    main()

