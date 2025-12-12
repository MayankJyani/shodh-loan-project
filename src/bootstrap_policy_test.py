import numpy as np
import pandas as pd
from pathlib import Path
import sys

def avg_reward_for_actions(rewards, actions):
    return (rewards * (actions == 1)).mean()

def bootstrap_diff(rewards, actions_a, actions_b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(rewards)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = avg_reward_for_actions(rewards[idx], actions_a[idx])
        rb = avg_reward_for_actions(rewards[idx], actions_b[idx])
        diffs.append(ra - rb)
    diffs = np.array(diffs)
    return diffs, np.percentile(diffs, [2.5, 97.5]), diffs.mean()

def main():
    base = Path(__file__).resolve().parent
    rl_npz = base.parent / 'data' / 'rl' / 'test_rl.npz'
    q_preds = base.parent / 'models' / 'rl_q' / 'q_agent_test_predictions.csv'
    sup_preds = base.parent / 'models' / 'test_predictions.csv'

    if not rl_npz.exists():
        print("Missing", rl_npz); sys.exit(1)
    rewards = np.load(rl_npz)['rewards'].astype(float)

    if not q_preds.exists():
        print("Missing RL preds:", q_preds); sys.exit(1)
    q_df = pd.read_csv(q_preds)
    q_actions = q_df['action'].astype(int).values

    if not sup_preds.exists():
        print("Supervised predictions not found:", sup_preds); sys.exit(1)
    mdf = pd.read_csv(sup_preds)
    if 'pred_label_optimal' in mdf.columns:
        sup_actions = mdf['pred_label_optimal'].astype(int).values
    elif 'pred_label_0.5' in mdf.columns:
        sup_actions = mdf['pred_label_0.5'].astype(int).values
    elif 'pred_proba' in mdf.columns:
        sup_actions = (mdf['pred_proba'] >= 0.5).astype(int).values
    else:
        print("No usable pred_label column in", sup_preds); sys.exit(1)

    actions_all = np.ones_like(q_actions)
    actions_none = np.zeros_like(q_actions)

    print("Point estimates:")
    print(" RL avg reward:", avg_reward_for_actions(rewards, q_actions))
    print(" Supervised avg reward:", avg_reward_for_actions(rewards, sup_actions))
    print(" Approve-all avg reward:", avg_reward_for_actions(rewards, actions_all))

    diffs, ci, mean_diff = bootstrap_diff(rewards, q_actions, sup_actions, n_boot=5000)
    print(f"\nRL - Supervised mean diff: {mean_diff:.6f}, 95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")

    out_dir = base.parent / 'models' / 'rl_q'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(diffs).to_csv(out_dir / 'bootstrap_rl_minus_sup.csv', index=False)
    print("Saved bootstrap samples to", out_dir / 'bootstrap_rl_minus_sup.csv')

if __name__ == "__main__":
    main()