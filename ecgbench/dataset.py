"""
PyTorch Dataset class for loading ECG data from PhysioNet datasets.
"""

import ast
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import wfdb
except ImportError:
    raise ImportError(
        "wfdb is required to load ECG data. Install it with: pip install wfdb"
    )


def ecg_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for ECG dataset batches.
    
    Handles dictionaries (like scp_codes) by keeping them as lists instead of
    trying to collate their keys, which would fail when different samples have
    different dictionary keys.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched dictionary with collated tensors and lists of other types
    """
    from torch.utils.data._utils.collate import default_collate
    
    if not batch:
        return {}
    
    # Get all keys from the first sample
    all_keys = set(batch[0].keys())
    
    # Separate items that can be collated normally from those that need special handling
    collatable = {}
    non_collatable = {}
    
    # For each key, check all samples to determine how to handle it
    for key in all_keys:
        values = [sample[key] for sample in batch]
        
        # Check if all values are of a type that needs special handling
        if all(isinstance(v, dict) for v in values):
            # All are dicts - keep as list of dicts
            non_collatable[key] = values
        elif all(isinstance(v, (str, type(None))) for v in values):
            # All are strings or None - keep as list
            non_collatable[key] = values
        else:
            # Try to collate normally (tensors, numbers, etc.)
            collatable[key] = values
    
    # Collate normal items
    result = default_collate(collatable)
    
    # Add non-collatable items as lists
    result.update(non_collatable)
    
    return result


class ECGDataset(Dataset):
    """
    PyTorch Dataset for loading ECG data from PhysioNet datasets.

    This dataset loads ECG signals from PhysioNet WFDB format files and
    associated metadata from ECGBench CSV files.

    Args:
        physionet_path: Path to the root directory of the PhysioNet dataset
                       (e.g., /path/to/physionet.org/files/ptb-xl/1.0.3/)
        dataset_name: Name of the dataset (e.g., 'ptbxl')
        split: Data split to load ('train', 'val', or 'test')
        fold_numbers: Fold number(s) to load. Can be a single int, list of ints, or None for all folds
        frequency: ECG sampling frequency to load ('100' or '500'). Default is '100'
        ecgbench_root: Root path to ECGBench folder containing metadata CSVs.
                      Default is None, which will try to find it relative to this file.

    Example:
        >>> dataset = ECGDataset(
        ...     physionet_path='/path/to/physionet.org/files/ptb-xl/1.0.3/',
        ...     dataset_name='ptbxl',
        ...     split='train',
        ...     fold_numbers=[1, 2],
        ...     frequency='100'
        ... )
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    """

    def __init__(
        self,
        physionet_path: Union[str, Path],
        dataset_name: str = "ptbxl",
        split: str = "train",
        fold_numbers: Optional[Union[int, List[int]]] = None,
        frequency: str = "100",
        ecgbench_root: Optional[Union[str, Path]] = None,
    ):
        self.physionet_path = Path(physionet_path)
        self.dataset_name = dataset_name
        self.split = split.lower()
        self.frequency = frequency

        # Validate split
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        # Validate frequency
        if self.frequency not in ["100", "500"]:
            raise ValueError(f"frequency must be '100' or '500', got '{frequency}'")

        # Determine ECGBench root path
        if ecgbench_root is None:
            # Try to find ECGBench root relative to this file
            current_file = Path(__file__).parent
            ecgbench_root = current_file.parent
        self.ecgbench_root = Path(ecgbench_root)

        # Load metadata
        self.metadata = self._load_metadata(fold_numbers)
        self.data_list = self._prepare_data_list()

    def _load_metadata(self, fold_numbers: Optional[Union[int, List[int]]]) -> pd.DataFrame:
        """
        Load metadata from ECGBench CSV files.

        Args:
            fold_numbers: Fold number(s) to load

        Returns:
            Combined DataFrame with metadata from specified folds
        """
        metadata_dir = self.ecgbench_root / "ecgbench" / "datasets" / self.dataset_name / self.split

        if not metadata_dir.exists():
            raise ValueError(
                f"Metadata directory not found: {metadata_dir}. "
                f"Make sure the ECGBench folder structure is correct."
            )

        # Determine which folds to load
        if fold_numbers is None:
            # Load all folds in the split directory
            fold_files = sorted(metadata_dir.glob("fold_*.csv"))
            if not fold_files:
                raise ValueError(f"No fold CSV files found in {metadata_dir}")
        else:
            # Convert single int to list
            if isinstance(fold_numbers, int):
                fold_numbers = [fold_numbers]

            # Load specified folds
            fold_files = []
            for fold_num in fold_numbers:
                fold_file = metadata_dir / f"fold_{fold_num}.csv"
                if not fold_file.exists():
                    raise ValueError(f"Fold file not found: {fold_file}")
                fold_files.append(fold_file)

        # Load and combine all fold CSVs
        dfs = []
        for fold_file in sorted(fold_files):
            df = pd.read_csv(fold_file)
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No metadata files loaded from {metadata_dir}")

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def _prepare_data_list(self) -> List[Dict[str, Any]]:
        """
        Prepare list of data items with file paths and metadata.

        Returns:
            List of dictionaries containing file paths and metadata for each ECG record
        """
        data_list = []

        # Determine filename column based on frequency
        filename_col = f"filename_{self.frequency}"

        if filename_col not in self.metadata.columns:
            raise ValueError(
                f"Frequency column '{filename_col}' not found in metadata. "
                f"Available columns: {list(self.metadata.columns)}"
            )

        for idx, row in self.metadata.iterrows():
            # Get filename path (without extension)
            filename_path = row[filename_col]

            # Construct full path to WFDB record
            record_path = self.physionet_path / filename_path

            # Store metadata (excluding filename columns to avoid redundancy)
            metadata_dict = {
                "ecg_id": row.get("ecg_id", None),
                "patient_id": row.get("patient_id", None),
                "record_path": str(record_path),
                "split": self.split,
                "frequency": self.frequency,
            }

            # Add all other columns as metadata
            for col in self.metadata.columns:
                if col not in ["filename_100", "filename_500"]:
                    value = row[col]
                    # Parse string representations of dicts (like scp_codes)
                    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                        try:
                            value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            pass
                    metadata_dict[col] = value

            data_list.append(metadata_dict)

        return data_list

    def _load_ecg_signal(self, record_path: str) -> np.ndarray:
        """
        Load ECG signal from WFDB format file.

        Args:
            record_path: Path to WFDB record (without extension)

        Returns:
            ECG signal as numpy array with shape (channels, samples)
        """
        try:
            # Load WFDB record
            record = wfdb.rdrecord(record_path)

            # Extract signal data
            signal = record.p_signal

            # Handle case where signal might be None or empty
            if signal is None:
                raise ValueError(f"ECG signal is None for record: {record_path}")

            # Transpose to (channels, samples) format
            signal = signal.T

            return signal

        except Exception as e:
            raise RuntimeError(f"Error loading ECG signal from {record_path}: {str(e)}")

    def __len__(self) -> int:
        """Return the number of ECG records in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single ECG record with metadata.

        Args:
            idx: Index of the record to retrieve

        Returns:
            Dictionary containing:
                - 'signal': ECG signal tensor with shape (channels, samples)
                - 'ecg_id': ECG ID
                - 'metadata': Dictionary with all metadata fields
        """
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_list)}")

        data_item = self.data_list[idx]
        record_path = data_item["record_path"]

        # Load ECG signal
        signal = self._load_ecg_signal(record_path)

        # Convert to torch tensor
        signal_tensor = torch.from_numpy(signal).float()

        # Prepare return dictionary
        result = {
            "signal": signal_tensor,
            "ecg_id": data_item.get("ecg_id"),
        }

        # Add all metadata fields
        for key, value in data_item.items():
            if key not in ["record_path", "ecg_id"]:  # Already handled above
                # Convert metadata to tensors where appropriate
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result[key] = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, dict):
                    # For dict-like metadata (e.g., scp_codes), keep as dict
                    result[key] = value
                else:
                    # Keep other types as-is
                    result[key] = value

        return result

