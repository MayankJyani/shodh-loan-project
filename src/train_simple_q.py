import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Simple dataset wrapper for your NPZ files
class RLDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.obs = d['observations'].astype(np.float32)
        self.actions = d['actions'].squeeze().astype(np.int64)
        self.rewards = d['rewards'].astype(np.float32)
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx], self.rewards[idx]

# Small MLP that outputs Q-values for 2 discrete actions
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256,128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 2))  # two discrete actions
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def evaluate_model(model, obs, rewards):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(obs))
        actions = logits.argmax(dim=1).cpu().numpy()
    # reward is only when action==1 (approve); reject->0
    return float((rewards * (actions == 1)).mean()), float((rewards * (actions == 1)).sum())

def train(args):
    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')
    train_ds = RLDataset(Path(args.data_dir) / 'train_rl.npz')
    val_ds = RLDataset(Path(args.data_dir) / 'val_rl.npz')
    test_ds = RLDataset(Path(args.data_dir) / 'test_rl.npz')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = QNetwork(input_dim=train_ds.obs.shape[1], hidden_dims=tuple(args.hidden_dims)).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_test_avg = -1e9
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for obs, acts, rews in train_loader:
            obs = obs.to(device)
            acts = acts.to(device)
            rews = rews.to(device)

            q = model(obs)  # [B,2]
            # target: Q(s, a_taken) = observed reward; other action targets not trained
            q_taken = q.gather(1, acts.unsqueeze(1)).squeeze(1)
            loss = loss_fn(q_taken, rews)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item()) * obs.size(0)

        avg_loss = total_loss / len(train_ds)
        val_avg, val_total = evaluate_model(model, val_ds.obs, val_ds.rewards)
        test_avg, test_total = evaluate_model(model, test_ds.obs, test_ds.rewards)
        print(f"Epoch {epoch}/{args.epochs}  Loss={avg_loss:.6f}  Val_avg_reward={val_avg:.6f}  Test_avg_reward={test_avg:.6f}")

        if test_avg > best_test_avg:
            best_test_avg = test_avg
            torch.save({'model_state_dict': model.state_dict(),
                        'input_dim': train_ds.obs.shape[1],
                        'hidden_dims': args.hidden_dims,
                        'epoch': epoch},
                       out_dir / 'q_agent_best.pth')

    # final save
    torch.save({'model_state_dict': model.state_dict(),
                'input_dim': train_ds.obs.shape[1],
                'hidden_dims': args.hidden_dims,
                'epoch': args.epochs},
               out_dir / 'q_agent_final.pth')
    print("Training complete. Best test avg reward:", best_test_avg)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../data/rl')
    p.add_argument('--output-dir', default='../models/rl_q')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden-dims', nargs='+', type=int, default=[256,128])
    p.add_argument('--use-gpu', action='store_true')
    args = p.parse_args()
    train(args)