"""
Preprocessing pipeline for LendingClub loan data.
Handles feature selection, missing values, encoding, and scaling.
Saves preprocessed data and fitted transformers for reproducibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Selected features based on domain knowledge and predictive power
NUMERICAL_FEATURES = [
    'loan_amnt',           # Loan amount
    'int_rate',            # Interest rate
    'installment',         # Monthly payment
    'annual_inc',          # Annual income
    'dti',                 # Debt-to-income ratio
    'fico_range_low',      # FICO score (low)
    'fico_range_high',     # FICO score (high)
    'open_acc',            # Number of open credit lines
    'pub_rec',             # Number of derogatory public records
    'revol_bal',           # Total credit revolving balance
    'revol_util',          # Revolving line utilization rate
    'total_acc',           # Total number of credit lines
    'mort_acc',            # Number of mortgage accounts
    'pub_rec_bankruptcies',# Number of public record bankruptcies
    'delinq_2yrs',         # Delinquencies in past 2 years
    'inq_last_6mths',      # Credit inquiries in last 6 months
]

CATEGORICAL_FEATURES = [
    'term',                # Loan term (36/60 months)
    'grade',               # LC assigned loan grade
    'sub_grade',           # LC assigned loan subgrade
    'emp_length',          # Employment length
    'home_ownership',      # Home ownership status
    'verification_status', # Income verification status
    'purpose',             # Loan purpose
    'addr_state',          # State
]

def create_fico_score(df):
    """Create average FICO score from range."""
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    return df

def engineer_features(df):
    """
    Create additional engineered features.
    """
    print("Engineering features...")
    
    # Create FICO score average
    df = create_fico_score(df)
    
    # Income to loan ratio
    df['income_to_loan'] = df['annual_inc'] / (df['loan_amnt'] + 1)
    
    # Installment to income ratio
    df['installment_to_income'] = (df['installment'] * 12) / (df['annual_inc'] + 1)
    
    # Credit utilization features
    df['credit_utilization'] = df['revol_util'] / 100.0  # Convert to decimal
    
    # Employment length numeric (extract years)
    if 'emp_length' in df.columns:
        df['emp_length_years'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
        df['emp_length_years'].fillna(0, inplace=True)  # "< 1 year" or "n/a" -> 0
    
    # Extract term as numeric (36 or 60 months)
    if 'term' in df.columns:
        df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float)
    
    return df

def select_features(df):
    """
    Select relevant features for modeling.
    """
    print("Selecting features...")
    
    # Check which features are available
    available_num = [f for f in NUMERICAL_FEATURES if f in df.columns]
    available_cat = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    
    # Add engineered features
    engineered = ['fico_score', 'income_to_loan', 'installment_to_income', 
                  'credit_utilization', 'emp_length_years', 'term_months']
    available_eng = [f for f in engineered if f in df.columns]
    
    feature_cols = available_num + available_cat + available_eng
    
    # Ensure target column exists
    if 'target' not in df.columns:
        if 'loan_status' in df.columns:
            df['target'] = (df['loan_status'].isin(['Charged Off', 'Default'])).astype(int)
        else:
            raise ValueError("Cannot create target variable: 'loan_status' column not found")
    
    # Also keep loan_amnt and int_rate for RL reward calculation
    metadata_cols = ['loan_amnt', 'int_rate', 'target']
    keep_cols = list(set(feature_cols + metadata_cols))
    
    available_cols = [c for c in keep_cols if c in df.columns]
    df_selected = df[available_cols].copy()
    
    print(f"Selected {len(feature_cols)} features")
    print(f"Numerical: {len(available_num)}, Categorical: {len(available_cat)}, Engineered: {len(available_eng)}")
    
    return df_selected, available_num + available_eng, available_cat

def handle_missing_values(df, num_features, cat_features, strategy='median'):
    """
    Handle missing values with imputation.
    """
    print("Handling missing values...")
    
    # Show missing value statistics
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({'count': missing, 'percentage': missing_pct})
    missing_df = missing_df[missing_df['count'] > 0].sort_values('count', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\nMissing values found in {len(missing_df)} columns:")
        print(missing_df.head(10))
    
    # Impute numerical features
    num_imputer = SimpleImputer(strategy=strategy)
    num_cols = [c for c in num_features if c in df.columns]
    if num_cols:
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Impute categorical features
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_cols = [c for c in cat_features if c in df.columns]
    if cat_cols:
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    return df, num_imputer, cat_imputer

def encode_categorical(df, cat_features):
    """
    Encode categorical features using label encoding.
    For high-cardinality features, consider target encoding in future.
    """
    print("Encoding categorical features...")
    
    encoders = {}
    for col in cat_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders

def scale_features(df, num_features):
    """
    Scale numerical features to zero mean and unit variance.
    """
    print("Scaling numerical features...")
    
    scaler = StandardScaler()
    num_cols = [c for c in num_features if c in df.columns and c not in ['target']]
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df, scaler

def preprocess_data(input_path, output_dir, test_size=0.2, val_size=0.1):
    """
    Full preprocessing pipeline.
    
    Args:
        input_path: Path to sampled data
        output_dir: Directory to save processed data
        test_size: Proportion of test set
        val_size: Proportion of validation set (from training data)
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Engineer features
    df = engineer_features(df)
    
    # Select features
    df, num_features, cat_features = select_features(df)
    
    # Handle missing values
    df, num_imputer, cat_imputer = handle_missing_values(df, num_features, cat_features)
    
    # Encode categorical features
    df, encoders = encode_categorical(df, cat_features)
    
    # Split data (stratified by target)
    print(f"\nSplitting data: train={1-test_size:.0%}, test={test_size:.0%}...")
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['target'], random_state=SEED
    )
    
    print(f"Splitting train into train/val: val={val_size:.0%}...")
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df['target'], random_state=SEED
    )
    
    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df):.1%})")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df):.1%})")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df):.1%})")
    
    # Scale numerical features (fit on train only!)
    train_df, scaler = scale_features(train_df, num_features)
    
    # Apply same scaling to val and test
    num_cols = [c for c in num_features if c in val_df.columns and c not in ['target']]
    if num_cols:
        val_df[num_cols] = scaler.transform(val_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    print(f"\nSaving processed data to {output_dir}...")
    train_df.to_parquet(output_path / 'train.parquet', index=False)
    val_df.to_parquet(output_path / 'val.parquet', index=False)
    test_df.to_parquet(output_path / 'test.parquet', index=False)
    
    # Save preprocessing artifacts
    preprocessing = {
        'num_imputer': num_imputer,
        'cat_imputer': cat_imputer,
        'encoders': encoders,
        'scaler': scaler,
        'num_features': num_features,
        'cat_features': cat_features,
        'feature_names': [c for c in df.columns if c not in ['target', 'loan_amnt', 'int_rate']]
    }
    
    joblib.dump(preprocessing, output_path / 'preprocessing.pkl')
    print("Saved preprocessing pipeline")
    
    # Save feature information
    feature_info = {
        'all_features': [c for c in df.columns if c not in ['target']],
        'num_features': num_features,
        'cat_features': cat_features,
        'target': 'target'
    }
    joblib.dump(feature_info, output_path / 'feature_info.pkl')
    
    print("\n✓ Preprocessing complete!")
    print(f"  Train default rate: {train_df['target'].mean():.2%}")
    print(f"  Val default rate: {val_df['target'].mean():.2%}")
    print(f"  Test default rate: {test_df['target'].mean():.2%}")
    
    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(description='Preprocess LendingClub data')
    parser.add_argument('--input', type=str, default='../data/sampled_data.csv',
                        help='Path to sampled data')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Validation set proportion of train (default: 0.1)')
    
    args = parser.parse_args()
    
    preprocess_data(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        val_size=args.val_size
    )

if __name__ == '__main__':
    main()
