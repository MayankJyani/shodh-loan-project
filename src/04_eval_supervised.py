"""
Evaluate trained supervised model on test set.
Produces AUC, F1, confusion matrix, ROC curve, and threshold analysis.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                              confusion_matrix, roc_curve, precision_recall_curve, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataloaders, MLPClassifier

# Reproducibility
SEED = 42
np.random.seed(SEED)

def evaluate_model(model, data_loader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_targets.extend(targets.numpy().flatten())
    
    return np.array(all_preds), np.array(all_targets)

def plot_roc_curve(y_true, y_pred_proba, output_path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {output_path}")

def plot_precision_recall_curve(y_true, y_pred_proba, output_path):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fully Paid', 'Default'],
                yticklabels=['Fully Paid', 'Default'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def threshold_analysis(y_true, y_pred_proba, output_path):
    """Analyze metrics across different thresholds."""
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = {'threshold': [], 'precision': [], 'recall': [], 'f1': []}
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if len(np.unique(y_pred)) > 1:  # Avoid division by zero
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            precision = recall = f1 = 0
        
        metrics['threshold'].append(threshold)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['threshold'], metrics['precision'], label='Precision', marker='o')
    plt.plot(metrics['threshold'], metrics['recall'], label='Recall', marker='s')
    plt.plot(metrics['threshold'], metrics['f1'], label='F1 Score', marker='^')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold analysis to {output_path}")
    
    # Find optimal threshold (maximum F1)
    optimal_idx = np.argmax(metrics['f1'])
    optimal_threshold = metrics['threshold'][optimal_idx]
    optimal_f1 = metrics['f1'][optimal_idx]
    
    return optimal_threshold, optimal_f1, pd.DataFrame(metrics)

def main():
    parser = argparse.ArgumentParser(description='Evaluate supervised model')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='Directory with processed data')
    parser.add_argument('--model-path', type=str, default='../models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='../models',
                        help='Output directory for evaluation results')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size')
    
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
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model = MLPClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout=checkpoint['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    y_pred_proba, y_true = evaluate_model(model, test_loader, device)
    
    # Calculate metrics with default threshold (0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"\nTest Set Metrics (threshold=0.5):")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fully Paid', 'Default']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Paid    Default")
    print(f"Actual Paid    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Default {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Threshold analysis
    output_dir = Path(args.output_dir)
    optimal_threshold, optimal_f1, threshold_df = threshold_analysis(
        y_true, y_pred_proba, output_dir / 'threshold_analysis.png'
    )
    
    print(f"\nOptimal threshold (max F1): {optimal_threshold:.3f} (F1={optimal_f1:.4f})")
    
    # Re-evaluate with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    f1_optimal = f1_score(y_true, y_pred_optimal)
    precision_optimal = precision_score(y_true, y_pred_optimal)
    recall_optimal = recall_score(y_true, y_pred_optimal)
    
    print(f"\nTest Set Metrics (threshold={optimal_threshold:.3f}):")
    print(f"  F1 Score:  {f1_optimal:.4f}")
    print(f"  Precision: {precision_optimal:.4f}")
    print(f"  Recall:    {recall_optimal:.4f}")
    
    # Save results
    results = {
        'test_auc': float(auc),
        'test_f1_default_threshold': float(f1),
        'test_precision_default_threshold': float(precision),
        'test_recall_default_threshold': float(recall),
        'optimal_threshold': float(optimal_threshold),
        'test_f1_optimal_threshold': float(f1_optimal),
        'test_precision_optimal_threshold': float(precision_optimal),
        'test_recall_optimal_threshold': float(recall_optimal),
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'test_results.json'}")
    
    # Generate plots
    plot_roc_curve(y_true, y_pred_proba, output_dir / 'roc_curve.png')
    plot_precision_recall_curve(y_true, y_pred_proba, output_dir / 'precision_recall_curve.png')
    plot_confusion_matrix(y_true, y_pred, output_dir / 'confusion_matrix.png')
    
    # Save threshold analysis data
    threshold_df.to_csv(output_dir / 'threshold_metrics.csv', index=False)
    
    # Save predictions for RL comparison
    predictions_df = pd.DataFrame({
        'true_label': y_true,
        'pred_proba': y_pred_proba,
        'pred_label_0.5': y_pred,
        'pred_label_optimal': y_pred_optimal
    })
    predictions_df.to_csv(output_dir / 'test_predictions.csv', index=False)
    print(f"Saved predictions to {output_dir / 'test_predictions.csv'}")
    
    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print("="*60)

if __name__ == '__main__':
    main()
