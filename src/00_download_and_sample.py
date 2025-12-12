"""
Script to download and sample the LendingClub dataset.
Downloads from Kaggle (requires kaggle API setup) or user can manually download.
Creates manageable samples for rapid iteration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def download_from_kaggle():
    """
    Download LendingClub data from Kaggle.
    Requires: kaggle API credentials setup (~/.kaggle/kaggle.json)
    """
    try:
        import kaggle
        print("Downloading LendingClub data from Kaggle...")
        kaggle.api.dataset_download_files(
            'wordsforthewise/lending-club',
            path='../data/',
            unzip=True
        )
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Could not download from Kaggle: {e}")
        print("Please manually download 'accepted_2007_to_2018.csv' from:")
        print("https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        print("and place it in the data/ folder")
        return False

def sample_data(input_path, output_path, sample_size=200000, seed=SEED):
    """
    Create a stratified sample from the full dataset.
    
    Args:
        input_path: Path to full dataset
        output_path: Path to save sampled data
        sample_size: Number of rows to sample
        seed: Random seed for reproducibility
    """
    print(f"Loading data from {input_path}...")
    
    # Read data in chunks to handle large file
    chunk_size = 100000
    chunks = []
    total_rows = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"Loaded {total_rows:,} rows...")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")
    
    # Filter to relevant loan statuses (Fully Paid or Charged Off/Default)
    print("\nFiltering to completed loans (Fully Paid or Defaulted)...")
    relevant_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(relevant_statuses)]
    print(f"Rows after filtering: {len(df):,}")
    
    # Create binary target
    df['target'] = (df['loan_status'].isin(['Charged Off', 'Default'])).astype(int)
    print(f"\nClass distribution:")
    print(df['target'].value_counts())
    print(f"Default rate: {df['target'].mean():.2%}")
    
    # Stratified sampling
    if len(df) > sample_size:
        print(f"\nSampling {sample_size:,} rows (stratified)...")
        from sklearn.model_selection import train_test_split
        df_sample, _ = train_test_split(
            df, 
            train_size=sample_size, 
            stratify=df['target'],
            random_state=seed
        )
    else:
        print(f"\nDataset is smaller than sample size, using all {len(df):,} rows")
        df_sample = df
    
    print(f"\nSample class distribution:")
    print(df_sample['target'].value_counts())
    print(f"Sample default rate: {df_sample['target'].mean():.2%}")
    
    # Save sample
    print(f"\nSaving sampled data to {output_path}...")
    df_sample.to_csv(output_path, index=False)
    print(f"Saved {len(df_sample):,} rows")
    
    return df_sample

def main():
    parser = argparse.ArgumentParser(description='Download and sample LendingClub data')
    parser.add_argument('--download', action='store_true', help='Attempt to download from Kaggle')
    parser.add_argument('--sample-size', type=int, default=200000, help='Sample size (default: 200000)')
    parser.add_argument('--input', type=str, default='../data/accepted_2007_to_2018.csv', 
                        help='Path to full dataset')
    parser.add_argument('--output', type=str, default='../data/sampled_data.csv',
                        help='Path to save sampled data')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    Path('../data').mkdir(parents=True, exist_ok=True)
    
    # Download if requested
    if args.download:
        download_from_kaggle()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        print("Please download the dataset manually or use --download flag")
        return
    
    # Sample the data
    sample_data(
        input_path=args.input,
        output_path=args.output,
        sample_size=args.sample_size
    )
    
    print("\n✓ Done! Sampled data is ready for preprocessing.")

if __name__ == '__main__':
    main()
