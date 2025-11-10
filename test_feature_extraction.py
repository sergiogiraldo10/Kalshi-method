"""
Test feature extraction on one season to check target variable balance
"""
from src.feature_engineering import MomentumFeatureExtractor
import pandas as pd
import os

season = '2020-21'
file_path = os.path.join('data/raw', f'pbp_{season.replace("-", "_")}.csv')

print(f"Loading {season}...")
pbp_df = pd.read_csv(file_path)

print(f"Extracting features...")
extractor = MomentumFeatureExtractor()
features_df = extractor.extract_features(pbp_df, include_outcomes=True)

print(f"\nTotal features: {len(features_df):,}")
if 'run_extends' in features_df.columns:
    print(f"Positive class (run extends): {features_df['run_extends'].sum():,} ({features_df['run_extends'].mean()*100:.1f}%)")
    print(f"Negative class (run stops): {(1-features_df['run_extends']).sum():,} ({(1-features_df['run_extends'].mean())*100:.1f}%)")
else:
    print("No 'run_extends' column found!")

