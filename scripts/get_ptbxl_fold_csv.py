#!/usr/bin/env python3
"""
Script to divide PTB-XL CSV file into train, val, and test sets based on strat_fold column.
Folds 1-8: train
Fold 9: val
Fold 10: test
Creates separate CSV files for each fold with filename_100 and filename_500 as first two columns.
"""

import pandas as pd
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Split PTB-XL CSV by fold')
    parser.add_argument('--input', '-i', 
                       default='/global/D1/homes/vajira/data/SEARCH/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv',
                       help='Input CSV file path')
    parser.add_argument('--output-dir', '-o',
                       default='../ecgbench/datasets/ptbxl',
                       help='Output directory for fold CSV files')
    args = parser.parse_args()
    
    # Read the CSV file
    print(f"Reading CSV file: {args.input}")
    df = pd.read_csv(args.input)
    
    # Check if strat_fold column exists
    if 'strat_fold' not in df.columns:
        raise ValueError("Column 'strat_fold' not found in CSV file")
    
    # Check if filename_lr and filename_hr columns exist
    if 'filename_lr' not in df.columns:
        raise ValueError("Column 'filename_lr' not found in CSV file")
    if 'filename_hr' not in df.columns:
        raise ValueError("Column 'filename_hr' not found in CSV file")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train, val, and test subdirectories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique fold values
    unique_folds = sorted(df['strat_fold'].unique())
    print(f"Found folds: {unique_folds}")
    
    # Rename filename_lr to filename_100 and filename_hr to filename_500
    df = df.rename(columns={'filename_lr': 'filename_100', 'filename_hr': 'filename_500'})
    
    # Get all column names
    cols = df.columns.tolist()
    
    # Move filename_100 and filename_500 to first two positions
    cols.remove('filename_100')
    cols.remove('filename_500')
    cols.insert(0, 'filename_100')
    cols.insert(1, 'filename_500')
    df = df[cols]
    
    # Process each fold
    for fold_num in unique_folds:
        fold_data = df[df['strat_fold'] == fold_num].copy()
        
        if len(fold_data) == 0:
            continue
            
        # Determine split type and output directory
        if fold_num <= 8:
            split_type = 'train'
            split_dir = train_dir
        elif fold_num == 9:
            split_type = 'val'
            split_dir = val_dir
        elif fold_num == 10:
            split_type = 'test'
            split_dir = test_dir
        else:
            split_type = 'unknown'
            split_dir = output_dir
        
        # Create output filename (fold_1.csv, fold_2.csv, etc. - 1-indexed)
        output_file = split_dir / f'fold_{fold_num}.csv'
        
        # Save to CSV
        fold_data.to_csv(output_file, index=False)
        print(f"Created {output_file} (fold {fold_num}, {split_type}, {len(fold_data)} records)")
    
    # Print summary
    print("\nSummary:")
    train_folds = [f for f in unique_folds if f <= 8]
    val_folds = [f for f in unique_folds if f == 9]
    test_folds = [f for f in unique_folds if f == 10]
    
    train_count = df[df['strat_fold'].isin(train_folds)].shape[0]
    val_count = df[df['strat_fold'].isin(val_folds)].shape[0]
    test_count = df[df['strat_fold'].isin(test_folds)].shape[0]
    
    print(f"Train (folds 1-8): {train_count} records")
    print(f"Val (fold 9): {val_count} records")
    print(f"Test (fold 10): {test_count} records")
    print(f"Total: {len(df)} records")


if __name__ == '__main__':
    main()

