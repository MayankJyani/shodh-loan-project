"""
Train offline RL agent using Conservative Q-Learning (CQL) via d3rlpy.
"""

import numpy as np
from pathlib import Path
import argparse
import joblib
import json

import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig
from d3rlpy.metrics import TDErrorEvaluator, DiscountedSumOfAdvantageEvaluator

# Reproducibility
SEED = 42

def load_rl_data(data_path):
    """Load RL dataset from npz file."""
    data = np.load(data_path)
    return (
        data['observations'],
        data['actions'].astype(int).flatten(),  # d3rlpy expects 1D discrete actions
        data['rewards'],
        data['next_observations'],
        data['terminals']
    )

def create_mdp_dataset(observations, actions, rewards, next_observations, terminals):
    """Create d3rlpy MDPDataset."""
    # Create episodes (since all are terminal, each sample is one episode)
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_terminals=terminals
    )
    return dataset

def train_cql(train_dataset, val_dataset, n_epochs=50, batch_size=256, 
              learning_rate=3e-4, output_dir='../models/rl'):
    """
    Train CQL agent.
    
    Args:
        train_dataset: d3rlpy MDPDataset for training
        val_dataset: d3rlpy MDPDataset for validation
        n_epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
        output_dir: directory to save model
    """
    print("\n" + "="*60)
    print("Training Offline RL Agent (CQL)")
    print("="*60)
    
    # Get observation and action dimensions
    observation_shape = train_dataset.observations.shape[1:]
    n_actions = 2  # 0=deny, 1=approve
    
    print(f"\nDataset info:")
    print(f"  Observations: {len(train_dataset)} samples")
    print(f"  Observation shape: {observation_shape}")
    print(f"  Action space: {n_actions} actions")
    print(f"  Reward range: [{train_dataset.rewards.min():.2f}, {train_dataset.rewards.max():.2f}]")
    
    # Configure CQL algorithm
    print(f"\nConfiguring CQL...")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    cql = CQLConfig(
        actor_learning_rate=learning_rate,
        critic_learning_rate=learning_rate,
        batch_size=batch_size,
        alpha=1.0,  # Conservative regularization coefficient
    ).create(device='cpu')
    
    # Set up evaluators for validation
    td_error_evaluator = TDErrorEvaluator()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training
    print(f"\nStarting training...")
    
    try:
        # Fit the CQL algorithm
        results = cql.fit(
            train_dataset,
            n_steps=n_epochs * (len(train_dataset) // batch_size),
            n_steps_per_epoch=len(train_dataset) // batch_size,
            evaluators={
                'td_error': td_error_evaluator,
            },
            eval_episodes=val_dataset,
            save_interval=10,
            experiment_name=f"cql_loan_{SEED}",
            with_timestamp=False,
            logger_adapter=d3rlpy.logging.FileAdapterFactory(root_dir=str(output_path))
        )
        
        print("\n" + "="*60)
        print("✓ Training complete!")
        print("="*60)
        
        # Save final model
        cql.save(str(output_path / 'cql_model.d3'))
        print(f"\nSaved model to {output_path / 'cql_model.d3'}")
        
        return cql, results
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("This might be due to d3rlpy version compatibility.")
        print("Attempting simplified training...")
        
        # Simplified training without evaluators
        cql.fit(
            train_dataset,
            n_steps=n_epochs * (len(train_dataset) // batch_size),
            n_steps_per_epoch=len(train_dataset) // batch_size,
            save_interval=10,
            experiment_name=f"cql_loan_{SEED}",
            with_timestamp=False,
        )
        
        print("\n" + "="*60)
        print("✓ Training complete!")
        print("="*60)
        
        # Save final model
        cql.save(str(output_path / 'cql_model.d3'))
        print(f"\nSaved model to {output_path / 'cql_model.d3'}")
        
        return cql, None

def main():
    parser = argparse.ArgumentParser(description='Train offline RL agent')
    parser.add_argument('--data-dir', type=str, default='../data/rl',
                        help='Directory with RL data')
    parser.add_argument('--output-dir', type=str, default='../models/rl',
                        help='Output directory for RL model')
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Load RL data
    print("Loading RL datasets...")
    data_path = Path(args.data_dir)
    
    train_obs, train_act, train_rew, train_next_obs, train_term = load_rl_data(
        data_path / 'train_rl.npz'
    )
    val_obs, val_act, val_rew, val_next_obs, val_term = load_rl_data(
        data_path / 'val_rl.npz'
    )
    
    print(f"Train set: {len(train_obs):,} transitions")
    print(f"Val set: {len(val_obs):,} transitions")
    
    # Create MDP datasets
    print("\nCreating MDP datasets...")
    train_dataset = create_mdp_dataset(train_obs, train_act, train_rew, train_next_obs, train_term)
    val_dataset = create_mdp_dataset(val_obs, val_act, val_rew, val_next_obs, val_term)
    
    # Train CQL
    cql, results = train_cql(
        train_dataset,
        val_dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    print("\n✓ RL training complete!")
    print(f"\nRun evaluation with: python 07_eval_rl.py")

if __name__ == '__main__':
    main()
