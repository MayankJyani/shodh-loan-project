import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256,128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='../models/rl_q/q_agent_best.pth')
    p.add_argument('--data-dir', default='../data/rl')
    p.add_argument('--proc-dir', default='../data/processed')
    p.add_argument('--output-dir', default='../models/rl_q')
    p.add_argument('--use-gpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')
    ckpt = torch.load(args.model, map_location='cpu')
    input_dim = int(ckpt.get('input_dim'))
    hidden_dims = tuple(ckpt.get('hidden_dims', [256,128]))

    model = QNetwork(input_dim, hidden_dims).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    d = np.load(Path(args.data_dir) / 'test_rl.npz')
    obs = d['observations'].astype(np.float32)
    rewards = d['rewards'].astype(np.float32)
    # get test true labels from processed parquet (aligned by row order)
    test_df = pd.read_parquet(Path(args.proc_dir) / 'test.parquet').reset_index(drop=True)
    true_paid = (test_df['target'].astype(int) == 0).astype(int)  # 1 if paid, 0 if default

    with torch.no_grad():
        logits = model(torch.from_numpy(obs).to(device)).cpu().numpy()
    q0 = logits[:,0]; q1 = logits[:,1]
    actions = (q1 > q0).astype(int)  # 1 = approve

    avg_reward = float((rewards * (actions == 1)).mean())
    total_reward = float((rewards * (actions == 1)).sum())
    approve_rate = actions.mean()

    print(f"Policy test avg reward: {avg_reward:.6f}, total reward: {total_reward:.2f}, approve rate: {approve_rate:.4f}")

    # Classification w.r.t. paid outcome (approve -> predicts paid)
    y_true = true_paid.values if hasattr(true_paid, 'values') else np.array(true_paid)
    y_pred = actions
    print("Confusion matrix (rows=true paid (1)/default (0), cols=pred approve (1)/reject (0)):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report (approve as positive class):")
    print(classification_report(y_true, y_pred, digits=4))

    # Save predictions
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        'q0': q0, 'q1': q1, 'action': actions, 'reward': rewards, 'true_paid': y_true
    })
    df_out.to_csv(out / 'q_agent_test_predictions.csv', index=False)
    print("Saved predictions to", out / 'q_agent_test_predictions.csv')

if __name__ == '__main__':
    main()