"""
Test V2 feature extraction on one season
"""
from src.feature_engineering_v2 import MomentumFeatureExtractorV2
import pandas as pd

season = '2020-21'
file_path = f'data/raw/pbp_{season.replace("-", "_")}.csv'

print(f"Loading {season}...")
pbp_df = pd.read_csv(file_path)

print(f"Extracting V2 features with fixed run detection + NLP...")
extractor = MomentumFeatureExtractorV2()
features_df = extractor.extract_features(pbp_df, include_outcomes=True)

print(f"\n{'='*60}")
print("RESULTS:")
print(f"{'='*60}")

print(f"\nTotal features: {len(features_df):,}")

if 'run_extends' in features_df.columns:
    print(f"\nTarget Variable:")
    print(f"  Positive (run extends): {features_df['run_extends'].sum():,} ({features_df['run_extends'].mean()*100:.1f}%)")
    print(f"  Negative (run stops): {(1-features_df['run_extends']).sum():,} ({(1-features_df['run_extends'].mean())*100:.1f}%)")

print(f"\nRun Detection Stats:")
print(f"  Micro-runs detected: {features_df['is_micro_run'].sum():,} ({features_df['is_micro_run'].mean()*100:.1f}%)")
print(f"  Significant runs: {features_df['is_significant_run'].sum():,} ({features_df['is_significant_run'].mean()*100:.1f}%)")

print(f"\n  Run score distribution:")
print(features_df[features_df['is_micro_run'] == 1]['run_score'].describe())

print(f"\nNLP Features (sample from micro-runs):")
nlp_cols = ['team_steals_2min', 'team_blocks_2min', 'team_threes_2min', 
            'opponent_misses_2min', 'opponent_turnovers_2min', 'defensive_pressure']
micro_runs = features_df[features_df['is_micro_run'] == 1]
for col in nlp_cols:
    if col in micro_runs.columns:
        print(f"  {col:<30} mean: {micro_runs[col].mean():.2f}, max: {micro_runs[col].max():.0f}")

print(f"\n{'='*60}")
print("✅ V2 Features look good!" if features_df['run_extends'].mean() > 0.1 else "❌ Issue detected")
print(f"{'='*60}")

