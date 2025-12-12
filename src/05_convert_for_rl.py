"""
Convert preprocessed loan data into offline RL format.
Creates state-action-reward-next_state transitions for d3rlpy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import joblib

# Reproducibility
SEED = 42
np.random.seed(SEED)

def calculate_reward(action, loan_status, loan_amnt, int_rate, loss_given_default=1.0):
    """
    Single-value reward (kept for compatibility). Vectorized version used in create_rl_dataset.
    action: 0 (reject) or 1 (approve)
    loan_status: 0 paid, 1 default
    loan_amnt: principal
    int_rate: either percent (10.5) or decimal (0.105)
    loss_given_default: fraction of principal lost on default (default 1.0)
    """
    if pd.isna(loan_amnt) or pd.isna(int_rate):
        return 0.0
    lr = float(int_rate)
    # convert decimals to percent if needed
    if lr <= 1.0:
        lr = lr * 100.0
    if action == 0:
        return 0.0
    else:
        if int(loan_status) == 0:
            return float(loan_amnt * (lr / 100.0))
        else:
            return float(-loan_amnt * float(loss_given_default))

def create_rl_dataset(df, feature_cols, loss_given_default=1.0, actions_override=None):
    """Convert supervised dataframe to RL arrays."""
    print(f"Converting {len(df):,} samples to RL format...")
    # Required columns
    for col in ('target', 'loan_amnt', 'int_rate'):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe")

    # Observations (states)
    observations = df[feature_cols].astype(np.float32).values

    # Actions: use provided override (predictions) if available, else try historical column, else fallback to all 1
    if actions_override is not None:
        actions = np.asarray(actions_override).reshape(-1, 1).astype(np.float32)
    elif 'is_approved' in df.columns:
        actions = df['is_approved'].astype(int).values.reshape(-1, 1).astype(np.float32)
    else:
        actions = np.ones((len(df), 1), dtype=np.float32)

    # Loan amounts and interest rates (vectorized)
    loan_amnts = pd.to_numeric(df['loan_amnt'], errors='coerce').fillna(0.0).astype(float)
    int_rates = pd.to_numeric(df['int_rate'], errors='coerce').fillna(0.0).astype(float)

    # Convert int_rates from decimal to percent if necessary
    if np.nanmax(int_rates) <= 1.0:
        int_rates = int_rates * 100.0

    # Targets: 0 = paid, 1 = default
    targets = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int).values

    # Vectorized reward: +loan_amt * int_rate/100 for paid, -loan_amnt * loss_given_default for default
    rewards = np.where(
        targets == 0,
        loan_amnts.values * (int_rates.values / 100.0),
        -loan_amnts.values * float(loss_given_default)
    ).astype(np.float32)

    # Terminals (single-step episodes)
    terminals = np.ones(len(df), dtype=np.uint8)

    # Next observations (same because episodes are single-step)
    next_observations = observations.copy()

    # Print stats
    print("Reward statistics:")
    print(f"  Mean: {rewards.mean():.4f}, Std: {rewards.std():.4f}, Min: {rewards.min():.4f}, Max: {rewards.max():.4f}, Sum: {rewards.sum():.4f}")
    print("Reward distribution:")
    print(f"  Paid(+): {(rewards>0).sum():,}  Default(-): {(rewards<0).sum():,}  Zero: {(rewards==0).sum():,}")

    return observations, actions, rewards, next_observations, terminals

def convert_and_save(data_dir='../data/processed', output_dir='../data/rl', loss_given_default=1.0,
                     predictions_csv=None, action_col='pred_label_optimal'):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load splits (expected names)
    train_path = data_dir / 'train.parquet'
    val_path = data_dir / 'val.parquet'
    test_path = data_dir / 'test.parquet'
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run preprocessing to create processed splits.")

    train_df = pd.read_parquet(train_path).reset_index(drop=True)
    val_df = pd.read_parquet(val_path).reset_index(drop=True)
    test_df = pd.read_parquet(test_path).reset_index(drop=True)

    # optionally load predictions to get actions (ensure row order matches)
    preds_train = preds_val = preds_test = None
    # support either single CSV (old behavior) or directory with per-split files
    if predictions_csv:
        p = Path(predictions_csv)
        if p.is_dir():
            t = p / 'train_predictions.csv'
            v = p / 'val_predictions.csv'
            s = p / 'test_predictions.csv'
            if t.exists(): preds_train = pd.read_csv(t)
            if v.exists(): preds_val = pd.read_csv(v)
            if s.exists(): preds_test = pd.read_csv(s)
            print("Loaded predictions from directory:", p, " found:", [f.exists() for f in (t,v,s)])
        else:
            preds_all = pd.read_csv(p) if p.exists() else None
            if preds_all is not None:
                print(f"Loaded predictions from {p} (columns: {list(preds_all.columns)})")
            preds_train = preds_val = preds_test = preds_all

    # Exclude columns that should not be state features
    exclude_cols = {'target', 'loan_amnt', 'int_rate', 'id', 'index'}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Using {len(feature_cols)} feature columns for state. Example: {feature_cols[:6]}")

    # Helper to pick actions for a split (use preds only if length matches)
    def _actions_for_split(split_df, preds_df):
        if preds_df is not None and action_col in preds_df.columns and len(preds_df) == len(split_df):
            return preds_df[action_col].astype(int).values
        return None

    # Create RL arrays (pass actions_override when available)
    train_obs, train_act, train_rew, train_next, train_term = create_rl_dataset(
        train_df, feature_cols, loss_given_default, actions_override=_actions_for_split(train_df, preds_train)
    )
    val_obs, val_act, val_rew, val_next, val_term = create_rl_dataset(
        val_df, feature_cols, loss_given_default, actions_override=_actions_for_split(val_df, preds_val)
    )
    test_obs, test_act, test_rew, test_next, test_term = create_rl_dataset(
        test_df, feature_cols, loss_given_default, actions_override=_actions_for_split(test_df, preds_test)
    )

    # Save NPZ files for d3rlpy / offline RL
    np.savez_compressed(output_dir / 'train_rl.npz',
                        observations=train_obs,
                        actions=train_act,
                        rewards=train_rew,
                        next_observations=train_next,
                        terminals=train_term)
    np.savez_compressed(output_dir / 'val_rl.npz',
                        observations=val_obs,
                        actions=val_act,
                        rewards=val_rew,
                        next_observations=val_next,
                        terminals=val_term)
    np.savez_compressed(output_dir / 'test_rl.npz',
                        observations=test_obs,
                        actions=test_act,
                        rewards=test_rew,
                        next_observations=test_next,
                        terminals=test_term)

    # Save feature column list for later use
    pd.Series(feature_cols).to_csv(output_dir / 'feature_cols.csv', index=False)
    print(f"Saved RL datasets to {output_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../data/processed')
    p.add_argument('--output-dir', default='../data/rl')
    p.add_argument('--predictions', default=None, help='CSV with prediction/action column for splits')
    p.add_argument('--action-col', default='pred_label_optimal')
    p.add_argument('--loss-given-default', type=float, default=1.0)
    args = p.parse_args()
    convert_and_save(data_dir=args.data_dir,
                     output_dir=args.output_dir,
                     loss_given_default=args.loss_given_default,
                     predictions_csv=args.predictions,
                     action_col=args.action_col)
 
if __name__ == '__main__':
    main()
