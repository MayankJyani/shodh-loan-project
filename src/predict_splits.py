import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataset import create_dataloaders, MLPClassifier

def load_model(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    # support different ckpt formats
    state = ckpt.get('model_state_dict', ckpt)
    input_dim = ckpt.get('input_dim', None)
    hidden_dims = ckpt.get('hidden_dims', [256,128,64])
    dropout = ckpt.get('dropout', 0.3)
    if input_dim is None:
        # try to infer from checkpoint or fallback
        input_dim = ckpt.get('input_size', None)
        if input_dim is None:
            raise RuntimeError("Could not infer input_dim from checkpoint; please include input_dim in ckpt")
    model = MLPClassifier(input_dim, hidden_dims, dropout)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def predict_loader(model, loader, device='cpu'):
    preds = []
    trues = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)
            trues.extend(y.numpy().flatten())
    preds = np.array(preds)
    trues = np.array(trues)
    return pd.DataFrame({
        'true_label': trues.astype(int),
        'pred_proba': preds,
        'pred_label_0.5': (preds >= 0.5).astype(int)
    })

def main():
    model_path = Path('../models/best_model.pth')
    data_dir = Path('../data/processed')
    out_dir = Path('../models')
    out_dir.mkdir(parents=True, exist_ok=True)

    # load splits
    train = pd.read_parquet(data_dir / 'train.parquet').reset_index(drop=True)
    val = pd.read_parquet(data_dir / 'val.parquet').reset_index(drop=True)
    test = pd.read_parquet(data_dir / 'test.parquet').reset_index(drop=True)

    # create dataloaders (reuse your project function)
    train_loader, val_loader, test_loader, feature_cols = create_dataloaders(train, val, test, batch_size=1024, num_workers=0)

    model = load_model(model_path, device='cpu')

    df_train = predict_loader(model, train_loader)
    df_val = predict_loader(model, val_loader)
    df_test = predict_loader(model, test_loader)

    # try to load best threshold if it exists
    try:
        import json
        th = json.load(open('../models/best_threshold.json'))['best_threshold']
    except Exception:
        th = 0.5
    for df in (df_train, df_val, df_test):
        df['pred_label_optimal'] = (df['pred_proba'] >= th).astype(int)

    df_train.to_csv(out_dir / 'train_predictions.csv', index=False)
    df_val.to_csv(out_dir / 'val_predictions.csv', index=False)
    df_test.to_csv(out_dir / 'test_predictions.csv', index=False)
    print("Saved train/val/test prediction CSVs to", out_dir)

if __name__ == '__main__':
    main()