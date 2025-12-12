"""
Evaluate offline RL agent and compare with supervised model.
Computes estimated policy value and finds decision differences.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import torch

import d3rlpy
from dataset import MLPClassifier

# Reproducibility
SEED = 42
np.random.seed(SEED)

def load_rl_data(data_path):
    """Load RL dataset."""
    data = np.load(data_path)
    return (
        data['observations'],
        data['actions'],
        data['rewards'],
        data['terminals']
    )

def calculate_policy_value(cql, observations, rewards, actions_taken):
    """
    Calculate estimated policy value.
    
    Policy value = average reward when following the learned policy.
    """
    # Get RL agent's actions
    rl_actions = cql.predict(observations)
    
    # Calculate rewards for RL policy
    # Since we only have data for action=1 (approve), we estimate:
    # - If RL says approve (1): use actual reward
    # - If RL says deny (0): reward = 0
    estimated_rewards = np.where(rl_actions == 1, rewards, 0)
    
    policy_value = estimated_rewards.mean()
    total_return = estimated_rewards.sum()
    
    # Also calculate the behavioral policy value (all approved)
    behavioral_value = rewards.mean()
    behavioral_total = rewards.sum()
    
    # Action distribution
    approve_rate = (rl_actions == 1).mean()
    deny_rate = (rl_actions == 0).mean()
    
    return {
        'policy_value': policy_value,
        'total_return': total_return,
        'behavioral_value': behavioral_value,
        'behavioral_total': behavioral_total,
        'improvement': policy_value - behavioral_value,
        'improvement_pct': 100 * (policy_value - behavioral_value) / abs(behavioral_value) if behavioral_value != 0 else 0,
        'approve_rate': approve_rate,
        'deny_rate': deny_rate,
        'n_samples': len(observations)
    }

def load_supervised_model(model_path, device):
    """Load trained supervised model."""
    checkpoint = torch.load(model_path, map_location=device)
    model = MLPClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout=checkpoint['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_supervised_decisions(supervised_model, observations, threshold=0.5, device='cpu'):
    """Get supervised model decisions."""
    # Convert to tensor
    obs_tensor = torch.from_numpy(observations.astype(np.float32)).to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = supervised_model(obs_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Predict action: if prob(default) < threshold, approve (1), else deny (0)
    # Lower default probability = safer = approve
    decisions = (probs < threshold).astype(int)
    
    return decisions, probs

def compare_policies(rl_actions, sup_actions, observations, rewards, probs, test_df):
    """
    Find interesting cases where RL and supervised models disagree.
    """
    # Find disagreements
    disagreements = (rl_actions != sup_actions)
    
    print(f"\nPolicy Comparison:")
    print(f"  Agreement rate: {(~disagreements).mean():.2%}")
    print(f"  Disagreement rate: {disagreements.mean():.2%}")
    
    # Cases where supervised denies but RL approves
    sup_deny_rl_approve = (sup_actions == 0) & (rl_actions == 1)
    print(f"\n  Supervised DENIES, RL APPROVES: {sup_deny_rl_approve.sum():,} cases")
    
    # Cases where supervised approves but RL denies
    sup_approve_rl_deny = (sup_actions == 1) & (rl_actions == 0)
    print(f"  Supervised APPROVES, RL DENIES: {sup_approve_rl_deny.sum():,} cases")
    
    # Analyze these cases
    comparisons = []
    
    # Find high-risk applicants that RL approves but supervised denies
    if sup_deny_rl_approve.sum() > 0:
        # Get indices of these cases
        indices = np.where(sup_deny_rl_approve)[0]
        
        # Sort by default probability (highest risk first)
        sorted_indices = indices[np.argsort(-probs[indices])]
        
        # Take top 10
        for i in sorted_indices[:10]:
            comparisons.append({
                'case_type': 'high_risk_rl_approves',
                'index': int(i),
                'default_prob': float(probs[i]),
                'actual_reward': float(rewards[i]),
                'supervised_decision': 'DENY',
                'rl_decision': 'APPROVE',
                'loan_amnt': float(test_df['loan_amnt'].iloc[i]),
                'int_rate': float(test_df['int_rate'].iloc[i]),
                'actual_outcome': 'DEFAULT' if test_df['target'].iloc[i] == 1 else 'PAID'
            })
    
    # Find low-risk applicants that RL denies but supervised approves
    if sup_approve_rl_deny.sum() > 0:
        indices = np.where(sup_approve_rl_deny)[0]
        
        # Sort by default probability (lowest risk first)
        sorted_indices = indices[np.argsort(probs[indices])]
        
        # Take top 10
        for i in sorted_indices[:10]:
            comparisons.append({
                'case_type': 'low_risk_rl_denies',
                'index': int(i),
                'default_prob': float(probs[i]),
                'actual_reward': float(rewards[i]),
                'supervised_decision': 'APPROVE',
                'rl_decision': 'DENY',
                'loan_amnt': float(test_df['loan_amnt'].iloc[i]),
                'int_rate': float(test_df['int_rate'].iloc[i]),
                'actual_outcome': 'DEFAULT' if test_df['target'].iloc[i] == 1 else 'PAID'
            })
    
    return comparisons

def main():
    parser = argparse.ArgumentParser(description='Evaluate RL agent and compare policies')
    parser.add_argument('--rl-model-path', type=str, default='../models/rl/cql_model.d3',
                        help='Path to trained RL model')
    parser.add_argument('--supervised-model-path', type=str, default='../models/best_model.pth',
                        help='Path to trained supervised model')
    parser.add_argument('--rl-data-dir', type=str, default='../data/rl',
                        help='Directory with RL test data')
    parser.add_argument('--processed-data-dir', type=str, default='../data/processed',
                        help='Directory with processed test data')
    parser.add_argument('--output-dir', type=str, default='../models/rl',
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Supervised model threshold')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Evaluating RL Agent and Comparing Policies")
    print("="*60)
    
    # Load RL model
    print(f"\nLoading RL model from {args.rl_model_path}...")
    try:
        cql = d3rlpy.load_learnable(args.rl_model_path, device='cpu')
        print("✓ RL model loaded")
    except Exception as e:
        print(f"Error loading RL model: {e}")
        print("Make sure you've trained the RL model first with 06_train_offline_rl.py")
        return
    
    # Load supervised model
    print(f"\nLoading supervised model from {args.supervised_model_path}...")
    supervised_model = load_supervised_model(args.supervised_model_path, device)
    print("✓ Supervised model loaded")
    
    # Load test data
    print(f"\nLoading test data...")
    rl_data_path = Path(args.rl_data_dir)
    test_obs, test_act, test_rew, test_term = load_rl_data(rl_data_path / 'test_rl.npz')
    
    processed_data_path = Path(args.processed_data_dir)
    test_df = pd.read_parquet(processed_data_path / 'test.parquet')
    
    print(f"Test set: {len(test_obs):,} samples")
    
    # Evaluate RL policy
    print("\n" + "-"*60)
    print("RL Agent Policy Evaluation")
    print("-"*60)
    
    rl_policy_metrics = calculate_policy_value(cql, test_obs, test_rew, test_act)
    
    print(f"\nRL Policy Value:")
    print(f"  Estimated value (avg reward): ${rl_policy_metrics['policy_value']:,.2f}")
    print(f"  Total return: ${rl_policy_metrics['total_return']:,.2f}")
    print(f"\nBehavioral Policy Value (baseline - approve all):")
    print(f"  Average reward: ${rl_policy_metrics['behavioral_value']:,.2f}")
    print(f"  Total return: ${rl_policy_metrics['behavioral_total']:,.2f}")
    print(f"\nImprovement:")
    print(f"  Absolute: ${rl_policy_metrics['improvement']:,.2f} per loan")
    print(f"  Relative: {rl_policy_metrics['improvement_pct']:.2f}%")
    print(f"\nRL Policy Actions:")
    print(f"  Approve rate: {rl_policy_metrics['approve_rate']:.2%}")
    print(f"  Deny rate: {rl_policy_metrics['deny_rate']:.2%}")
    
    # Get RL decisions
    rl_actions = cql.predict(test_obs)
    
    # Get supervised decisions
    print("\n" + "-"*60)
    print("Supervised Model Policy Evaluation")
    print("-"*60)
    
    sup_actions, sup_probs = get_supervised_decisions(
        supervised_model, test_obs, threshold=args.threshold, device=device
    )
    
    # Calculate supervised policy value
    sup_estimated_rewards = np.where(sup_actions == 1, test_rew, 0)
    sup_policy_value = sup_estimated_rewards.mean()
    sup_total_return = sup_estimated_rewards.sum()
    sup_approve_rate = (sup_actions == 1).mean()
    
    print(f"\nSupervised Policy Value (threshold={args.threshold}):")
    print(f"  Estimated value (avg reward): ${sup_policy_value:,.2f}")
    print(f"  Total return: ${sup_total_return:,.2f}")
    print(f"  Approve rate: {sup_approve_rate:.2%}")
    
    # Compare policies
    print("\n" + "-"*60)
    print("Policy Comparison")
    print("-"*60)
    
    comparisons = compare_policies(rl_actions, sup_actions, test_obs, test_rew, sup_probs, test_df)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'rl_policy': rl_policy_metrics,
        'supervised_policy': {
            'policy_value': float(sup_policy_value),
            'total_return': float(sup_total_return),
            'approve_rate': float(sup_approve_rate),
            'threshold': args.threshold
        },
        'comparison': {
            'agreement_rate': float((rl_actions == sup_actions).mean()),
            'disagreement_rate': float((rl_actions != sup_actions).mean()),
        }
    }
    
    with open(output_path / 'rl_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_path / 'rl_evaluation.json'}")
    
    # Save comparisons
    if comparisons:
        comparisons_df = pd.DataFrame(comparisons)
        comparisons_df.to_csv(output_path / 'policy_comparisons.csv', index=False)
        print(f"✓ Saved policy comparisons to {output_path / 'policy_comparisons.csv'}")
        
        # Print sample comparisons
        print(f"\n" + "-"*60)
        print("Sample Decision Differences")
        print("-"*60)
        
        high_risk_rl_approves = [c for c in comparisons if c['case_type'] == 'high_risk_rl_approves']
        if high_risk_rl_approves:
            print(f"\nHigh-Risk Applicants RL APPROVES (Supervised DENIES):")
            for i, case in enumerate(high_risk_rl_approves[:3], 1):
                print(f"\n  Case {i}:")
                print(f"    Default Probability: {case['default_prob']:.2%}")
                print(f"    Loan Amount: ${case['loan_amnt']:,.0f}")
                print(f"    Interest Rate: {case['int_rate']:.1f}%")
                print(f"    Actual Outcome: {case['actual_outcome']}")
                print(f"    Actual Reward: ${case['actual_reward']:,.2f}")
    
    print("\n" + "="*60)
    print("✓ RL evaluation complete!")
    print("="*60)

if __name__ == '__main__':
    main()
