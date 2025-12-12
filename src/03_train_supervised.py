"""
Train supervised deep learning model for loan default prediction.
Uses PyTorch MLP with BCE loss, tracks AUC and F1 metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

from dataset import create_dataloaders, MLPClassifier

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc='Training')
    for features, targets in pbar:
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs.flatten())
        all_targets.extend(targets.cpu().numpy().flatten())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    auc = roc_auc_score(all_targets, all_preds)
    f1 = f1_score(all_targets, (all_preds > 0.5).astype(int))
    
    return avg_loss, auc, f1

def evaluate(model, data_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            logits = model(features)
            loss = criterion(logits, targets)
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    auc = roc_auc_score(all_targets, all_preds)
    f1 = f1_score(all_targets, (all_preds > 0.5).astype(int))
    
    return avg_loss, auc, f1, all_preds, all_targets

def plot_learning_curves(history, output_path):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # AUC
    axes[1].plot(history['train_auc'], label='Train')
    axes[1].plot(history['val_auc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('AUC Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1
    axes[2].plot(history['train_f1'], label='Train')
    axes[2].plot(history['val_f1'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score Curves')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {output_path}")

def train_model(train_loader, val_loader, input_dim, 
                hidden_dims=[256, 128, 64], dropout=0.3,
                lr=0.001, epochs=30, patience=5, device='cpu', 
                output_dir='../models'):
    """
    Train supervised model with early stopping.
    
    Args:
        train_loader, val_loader: DataLoaders
        input_dim: number of input features
        hidden_dims: list of hidden layer dimensions
        dropout: dropout probability
        lr: learning rate
        epochs: maximum number of epochs
        patience: early stopping patience
        device: 'cuda' or 'cpu'
        output_dir: directory to save model and metrics
    """
    print("\n" + "="*60)
    print("Training Supervised Model")
    print("="*60)
    
    # Create model
    model = MLPClassifier(input_dim, hidden_dims, dropout).to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {num_params:,}")
    
    # Loss and optimizer
    # Compute pos_weight from training set to mitigate class imbalance (if possible)
    try:
        pos_count = 0
        neg_count = 0
        for _, y in train_loader:
            # y may be tensor or numpy array
            y_arr = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
            pos_count += int(y_arr.sum())
            neg_count += int((1 - y_arr).sum())
        pos_weight_val = float(neg_count / (pos_count + 1e-8))
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_val:.4f} (pos={pos_count}, neg={neg_count})")
    except Exception as e:
        print("Could not compute pos_weight, falling back to unweighted BCE:", e)
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Create scheduler in a version-compatible way (some torch versions don't accept verbose)
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_auc': [], 'train_f1': [],
        'val_loss': [], 'val_auc': [], 'val_f1': []
    }
    
    print(f"\nTraining for {epochs} epochs (patience={patience})...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_auc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_auc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_auc)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_f1': val_f1,
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'dropout': dropout
            }, output_path / 'best_model.pth')
            print(f"✓ Saved best model (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"{'='*60}\n")
    
    # Save history
    output_path = Path(output_dir)
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot learning curves
    plot_learning_curves(history, output_path / 'learning_curves.png')
    
    # Determine best decision threshold on validation set (maximize F1)
    try:
        _, val_auc_final, val_f1_final, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        # compute F1 for each threshold from precision-recall thresholds
        prec, rec, thresholds = precision_recall_curve(val_targets, val_preds)
        if thresholds.size > 0:
            f1_scores = [f1_score(val_targets, (val_preds > t).astype(int)) for t in thresholds]
            best_idx = int(np.argmax(f1_scores))
            best_threshold = float(thresholds[best_idx])
            best_threshold_f1 = float(f1_scores[best_idx])
        else:
            best_threshold = 0.5
            best_threshold_f1 = float(val_f1_final)
        thresh_info = {
            'best_threshold': best_threshold,
            'best_threshold_f1': best_threshold_f1,
            'val_auc': float(val_auc_final)
        }
        with open(output_path / 'best_threshold.json', 'w') as f:
            json.dump(thresh_info, f, indent=2)
        print(f"Best threshold on val set: {best_threshold:.4f} (F1={best_threshold_f1:.4f}, AUC={val_auc_final:.4f})")
    except Exception as e:
        print("Could not compute/save best threshold:", e)
    
    return model, history, best_val_auc

def main():
    parser = argparse.ArgumentParser(description='Train supervised model')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='Directory with processed data')
    parser.add_argument('--output-dir', type=str, default='../models',
                        help='Output directory for model')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading preprocessed data...")
    data_dir = Path(args.data_dir)
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    
    # Create dataloaders
    train_loader, val_loader, test_loader, feature_cols = create_dataloaders(
        train_df, val_df, test_df, batch_size=args.batch_size, num_workers=0
    )
    
    input_dim = len(feature_cols)
    print(f"Input dimension: {input_dim}")
    
    # Train model
    model, history, best_val_auc = train_model(
        train_loader, val_loader,
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        output_dir=args.output_dir
    )
    
    print("✓ Training complete!")
    print(f"\nRun evaluation with: python 04_eval_supervised.py")

if __name__ == '__main__':
    main()
