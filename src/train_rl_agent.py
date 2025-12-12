import argparse
from pathlib import Path
import numpy as np
import json

# Try importing d3rlpy; fail with helpful message
try:
    from d3rlpy.algos import DiscreteCQL
    from d3rlpy.dataset import MDPDataset
except Exception as e:
    raise ImportError("d3rlpy is required. Install with: pip install d3rlpy") from e

def load_npz(path):
    d = np.load(path)
    obs = d['observations']
    acts = d['actions'].squeeze()
    rews = d['rewards']
    nxt = d.get('next_observations', None)
    terms = d.get('terminals', None)
    return obs, acts, rews, nxt, terms

def eval_policy_on_dataset(algo, obs, actions, rewards):
    # algo.predict returns discrete actions
    pred = algo.predict(obs)
    total = (pred * rewards).sum()
    avg = total / len(rewards)
    return float(total), float(avg)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../data/rl', help='Directory with train_rl.npz, val_rl.npz, test_rl.npz')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--output-dir', default='../models/rl_cql')
    p.add_argument('--use-gpu', action='store_true')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load datasets
    train_obs, train_act, train_rew, train_next, train_term = load_npz(data_dir / 'train_rl.npz')
    val_obs, val_act, val_rew, val_next, val_term = load_npz(data_dir / 'val_rl.npz')
    test_obs, test_act, test_rew, test_next, test_term = load_npz(data_dir / 'test_rl.npz')

    # Create MDPDataset in a version-compatible way (some d3rlpy versions don't accept next_observations kwarg)
    try:
        train_ds = MDPDataset(train_obs, train_act, train_rew, terminals=train_term)
        val_ds = MDPDataset(val_obs, val_act, val_rew, terminals=val_term)
    except TypeError:
        # Older d3rlpy variants expect positional terminals argument
        train_ds = MDPDataset(train_obs, train_act, train_rew, train_term)
        val_ds = MDPDataset(val_obs, val_act, val_rew, val_term)

    # create algorithm in a version-tolerant way
    import inspect
    algo = None
    ctor = DiscreteCQL
    try:
        # try the simplest call first
        algo = ctor()
    except Exception:
        # inspect constructor params and build kwargs for a best-effort call
        sig = inspect.signature(ctor.__init__)
        params = sig.parameters
        kwargs = {}
        if 'config' in params:
            kwargs['config'] = {}
        if 'device' in params:
            kwargs['device'] = 'cpu'
        if 'enable_ddp' in params:
            kwargs['enable_ddp'] = False
        if 'gpu' in params and args.use_gpu:
            kwargs['gpu'] = 0
        if 'batch_size' in params:
            kwargs['batch_size'] = args.batch_size
        # Filter out None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            algo = ctor(**kwargs)
        except Exception as e:
            raise RuntimeError(
                "Failed to construct DiscreteCQL with the installed d3rlpy version. "
                "Two options:\n"
                "  1) Install a compatible d3rlpy: pip install 'd3rlpy==0.103' (recommended)\n"
                "  2) Inspect your d3rlpy API and modify train_rl_agent.py to match its constructor.\n"
            ) from e

    best_avg = -1e12
    best_info = {}
    # Train in small epoch increments and evaluate
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} — training one epoch...")
        # train one epoch (fit with n_epochs=1 will continue from previous state)
        # pass batch_size here for compatibility with d3rlpy versions
        try:
            algo.fit(train_ds, n_epochs=1, verbose=0, batch_size=args.batch_size)
        except TypeError:
            # older/newer variants may not accept batch_size in fit signature
            algo.fit(train_ds, n_epochs=1, verbose=0)

        # evaluate on val and test by running policy to get reward
        val_total, val_avg = eval_policy_on_dataset(algo, val_obs, val_act, val_rew)
        test_total, test_avg = eval_policy_on_dataset(algo, test_obs, test_act, test_rew)
        print(f"  Val avg reward: {val_avg:.6f}  Test avg reward: {test_avg:.6f}")

        # save best by test avg reward
        if test_avg > best_avg:
            best_avg = test_avg
            best_info = {'epoch': epoch, 'test_avg': test_avg, 'val_avg': val_avg}
            # save model
            model_path = out_dir / f'cql_best_epoch_{epoch}'
            algo.save_model(str(model_path))
            # save info
            with open(out_dir / 'best_info.json', 'w') as f:
                json.dump(best_info, f, indent=2)
            print(f"  Saved best model (test_avg={test_avg:.6f}) to {model_path}")

    print("Training complete. Best info:", best_info)
    algo.save_model(str(out_dir / 'cql_final'))
    print("Final model saved to", out_dir / 'cql_final')

if __name__ == '__main__':
    main()