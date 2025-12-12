"""
PyTorch Dataset and DataLoader for LendingClub loan data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class LoanDataset(Dataset):
    """PyTorch Dataset for loan default prediction."""
    
    def __init__(self, dataframe, feature_cols=None):
        """
        Args:
            dataframe: pandas DataFrame with features and target
            feature_cols: list of feature column names (exclude target, loan_amnt, int_rate)
        """
        self.df = dataframe.reset_index(drop=True)
        
        # Determine feature columns
        if feature_cols is None:
            # Exclude target and metadata columns
            exclude_cols = ['target', 'loan_amnt', 'int_rate']
            feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        
        self.feature_cols = feature_cols
        
        # Extract features and target
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.target = self.df['target'].values.astype(np.float32)
        
        # Keep metadata for RL conversion
        self.loan_amnt = self.df['loan_amnt'].values.astype(np.float32) if 'loan_amnt' in self.df.columns else None
        self.int_rate = self.df['int_rate'].values.astype(np.float32) if 'int_rate' in self.df.columns else None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: torch.Tensor of shape (num_features,)
            target: torch.Tensor scalar
        """
        features = torch.from_numpy(self.features[idx])
        target = torch.tensor(self.target[idx])
        
        return features, target
    
    def get_full_row(self, idx):
        """Get full row including metadata for RL conversion."""
        return {
            'features': self.features[idx],
            'target': self.target[idx],
            'loan_amnt': self.loan_amnt[idx] if self.loan_amnt is not None else None,
            'int_rate': self.int_rate[idx] if self.int_rate is not None else None,
        }

def create_dataloaders(train_df, val_df, test_df, batch_size=512, num_workers=0):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df, val_df, test_df: pandas DataFrames
        batch_size: batch size for training
        num_workers: number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader, feature_cols
    """
    # Determine feature columns from train set
    exclude_cols = ['target', 'loan_amnt', 'int_rate']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Create datasets
    train_dataset = LoanDataset(train_df, feature_cols)
    val_dataset = LoanDataset(val_df, feature_cols)
    test_dataset = LoanDataset(test_df, feature_cols)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset):,} samples, {len(test_loader)} batches")
    print(f"  Features: {len(feature_cols)}")
    
    return train_loader, val_loader, test_loader, feature_cols


class MLPClassifier(torch.nn.Module):
    """
    Multi-Layer Perceptron for binary classification.
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        """
        Args:
            input_dim: number of input features
            hidden_dims: list of hidden layer dimensions
            dropout: dropout probability
        """
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(torch.nn.Linear(prev_dim, 1))
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (batch_size, input_dim)
        
        Returns:
            logits: torch.Tensor of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict_proba(self, x):
        """
        Get probability predictions.
        
        Args:
            x: torch.Tensor of shape (batch_size, input_dim)
        
        Returns:
            probs: torch.Tensor of shape (batch_size,) with probabilities
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits).squeeze()
        return probs
