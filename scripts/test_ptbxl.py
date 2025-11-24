from ecgbench import ECGDataset, ecg_collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = ECGDataset(
    physionet_path='/global/D1/homes/vajira/data/SEARCH/physionet.org/files/ptb-xl/1.0.3/',
    dataset_name='ptbxl',
    split='train',
    fold_numbers=[1, 2, 3],  # or None for all folds
    frequency='100'  # or '500'
)

# Create DataLoader with custom collate function
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=ecg_collate_fn)

# Use in training loop
for batch in dataloader:
    signals = batch['signal']  # Shape: (batch_size, channels, samples)
    ecg_ids = batch['ecg_id']
    scp_codes = batch['scp_codes']  # Ground truth labels
    # ... other metadata fields available in batch