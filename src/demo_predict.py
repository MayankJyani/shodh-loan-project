"""
Demo script to predict loan approval for a single applicant.
Shows predictions from both supervised and RL models.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib

import d3rlpy
from dataset import MLPClassifier

def create_sample_applicant(loan_amnt=10000, int_rate=12.5, annual_inc=50000, 
                            dti=15.0, fico_score=700, term_months=36):
    """Create a sample applicant with specified features."""
    return {
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'installment': (loan_amnt / term_months) * (1 + int_rate / 1200),
        'annual_inc': annual_inc,
        'dti': dti,
        'fico_range_low': fico_score - 5,
        'fico_range_high': fico_score + 5,
        'fico_score': fico_score,
        'open_acc': 10,
        'pub_rec': 0,
        'revol_bal': 5000,
        'revol_util': 50.0,
        'credit_utilization': 0.5,
        'total_acc': 15,
        'mort_acc': 0,
        'pub_rec_bankruptcies': 0,
        'delinq_2yrs': 0,
        'inq_last_6mths': 1,
        'income_to_loan': annual_inc / (loan_amnt + 1),
        'installment_to_income': (loan_amnt / term_months) * 12 / (annual_inc + 1),
        'emp_length_years': 5,
        'term_months': term_months,
        'term': 1,  # Encoded
        'grade': 2,  # Encoded (B)
        'sub_grade': 8,  # Encoded
        'emp_length': 3,  # Encoded
        'home_ownership': 1,  # Encoded (RENT)
        'verification_status': 1,  # Encoded
        'purpose': 5,  # Encoded (debt_consolidation)
        'addr_state': 5,  # Encoded (CA)
    }

def preprocess_applicant(applicant_dict, feature_info):
    """Convert applicant dict to preprocessed feature vector."""
    # Get feature columns in correct order
    feature_cols = [c for c in feature_info['all_features']]
    
    # Create DataFrame
    applicant_df = pd.DataFrame([applicant_dict])
    
    # Select features in correct order
    features = applicant_df[feature_cols].values.astype(np.float32)
    
    return features

def main():
    parser = argparse.ArgumentParser(description='Predict loan approval for single applicant')
    parser.add_argument('--loan-amnt', type=float, default=10000, help='Loan amount')
    parser.add_argument('--int-rate', type=float, default=12.5, help='Interest rate (%)')
    parser.add_argument('--annual-inc', type=float, default=50000, help='Annual income')
    parser.add_argument('--dti', type=float, default=15.0, help='Debt-to-income ratio')
    parser.add_argument('--fico-score', type=float, default=700, help='FICO credit score')
    parser.add_argument('--term-months', type=int, default=36, help='Loan term (months)')
    parser.add_argument('--supervised-model', type=str, default='../models/best_model.pth',
                        help='Path to supervised model')
    parser.add_argument('--rl-model', type=str, default='../models/rl/cql_model.d3',
                        help='Path to RL model')
    parser.add_argument('--feature-info', type=str, default='../data/processed/feature_info.pkl',
                        help='Path to feature info')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Loan Approval Prediction Demo")
    print("="*60)
    
    # Create applicant
    applicant = create_sample_applicant(
        loan_amnt=args.loan_amnt,
        int_rate=args.int_rate,
        annual_inc=args.annual_inc,
        dti=args.dti,
        fico_score=args.fico_score,
        term_months=args.term_months
    )
    
    print("\nApplicant Profile:")
    print(f"  Loan Amount: ${applicant['loan_amnt']:,.0f}")
    print(f"  Interest Rate: {applicant['int_rate']:.1f}%")
    print(f"  Annual Income: ${applicant['annual_inc']:,.0f}")
    print(f"  Debt-to-Income Ratio: {applicant['dti']:.1f}%")
    print(f"  FICO Score: {applicant['fico_score']:.0f}")
    print(f"  Loan Term: {applicant['term_months']} months")
    
    # Load feature info
    try:
        feature_info = joblib.load(args.feature_info)
    except FileNotFoundError:
        print(f"\nError: Feature info not found at {args.feature_info}")
        print("Please run the preprocessing script first (01_preprocess.py)")
        return
    
    # Preprocess applicant
    features = preprocess_applicant(applicant, feature_info)
    
    # Load supervised model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.supervised_model, map_location=device)
        supervised_model = MLPClassifier(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            dropout=checkpoint['dropout']
        ).to(device)
        supervised_model.load_state_dict(checkpoint['model_state_dict'])
        supervised_model.eval()
        
        # Predict with supervised model
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).to(device)
            logits = supervised_model(features_tensor)
            default_prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
        
        supervised_decision = "APPROVE" if default_prob < 0.5 else "DENY"
        
        print("\n" + "-"*60)
        print("Supervised Model (Deep Learning)")
        print("-"*60)
        print(f"  Default Probability: {default_prob:.2%}")
        print(f"  Decision (threshold=0.5): {supervised_decision}")
        
        if default_prob < 0.3:
            risk_level = "LOW RISK"
        elif default_prob < 0.5:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "HIGH RISK"
        print(f"  Risk Level: {risk_level}")
        
    except FileNotFoundError:
        print(f"\nSupervised model not found at {args.supervised_model}")
        print("Please train the supervised model first (03_train_supervised.py)")
    
    # Load RL model
    try:
        cql = d3rlpy.load_learnable(args.rl_model, device='cpu')
        
        # Predict with RL model
        rl_action = cql.predict(features)[0]
        rl_decision = "APPROVE" if rl_action == 1 else "DENY"
        
        # Estimate Q-values
        q_values = cql.predict_value(features, np.array([[0], [1]]))
        q_deny = q_values[0]
        q_approve = q_values[1]
        
        print("\n" + "-"*60)
        print("RL Agent (Conservative Q-Learning)")
        print("-"*60)
        print(f"  Decision: {rl_decision}")
        print(f"  Q-value (Deny): ${q_deny:.2f}")
        print(f"  Q-value (Approve): ${q_approve:.2f}")
        
        # Expected reward if approved
        if rl_action == 1:
            expected_profit = applicant['loan_amnt'] * (applicant['int_rate'] / 100)
            expected_loss = applicant['loan_amnt']
            break_even_default_rate = (applicant['int_rate'] / 100) / (1 + applicant['int_rate'] / 100)
            
            print(f"\n  If Approved:")
            print(f"    Max Profit (if paid): ${expected_profit:,.2f}")
            print(f"    Max Loss (if default): -${expected_loss:,.2f}")
            print(f"    Break-even default rate: {break_even_default_rate:.1%}")
        
    except FileNotFoundError:
        print(f"\nRL model not found at {args.rl_model}")
        print("Please train the RL model first (06_train_offline_rl.py)")
    except Exception as e:
        print(f"\nError loading RL model: {e}")
    
    # Comparison
    try:
        if supervised_decision != rl_decision:
            print("\n" + "-"*60)
            print("⚠️  Models DISAGREE!")
            print("-"*60)
            print(f"  Supervised says: {supervised_decision}")
            print(f"  RL Agent says: {rl_decision}")
            print(f"\n  Why might they disagree?")
            
            if rl_decision == "APPROVE" and supervised_decision == "DENY":
                print(f"    - RL considers risk-reward tradeoff")
                print(f"    - High interest rate ({applicant['int_rate']:.1f}%) may justify risk")
                print(f"    - Expected value calculation vs. binary classification")
            else:
                print(f"    - RL may be more conservative with low-reward loans")
                print(f"    - Considers opportunity cost of capital")
        else:
            print("\n✓ Both models agree: " + supervised_decision)
    except:
        pass
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
