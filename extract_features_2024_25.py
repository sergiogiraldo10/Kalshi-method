"""
Extract features for 2024-25 season
"""

import sys
sys.path.append('src')

from feature_engineering_v2 import MomentumFeatureExtractorV2
import pandas as pd
from pathlib import Path

print("\n" + "="*70)
print("EXTRACTING FEATURES FOR 2024-25 SEASON")
print("="*70)

season = '2024_25'
pbp_file = Path(f'data/raw/pbp_{season}.csv')

if not pbp_file.exists():
    print(f"\n[!] PBP file not found: {pbp_file}")
    exit()

# Load PBP data
print(f"\nLoading {pbp_file}...")
pbp_df = pd.read_csv(pbp_file)
print(f"  [OK] Loaded {len(pbp_df):,} plays from {pbp_df['GAME_ID'].nunique()} games")

# Initialize feature extractor
print("\nInitializing feature extractor...")
extractor = MomentumFeatureExtractorV2()

# Extract features
print(f"\nExtracting features for {season}...")
print("(This will take ~5-10 minutes)")

features_df = extractor.extract_features(pbp_df, include_outcomes=True)

if len(features_df) > 0:
    # Save features
    output_file = Path(f'data/processed/features_v2_{season}.csv')
    print(f"\nSaving to {output_file}...")
    features_df.to_csv(output_file, index=False)
    print(f"  [OK] Saved {len(features_df):,} samples from {features_df['game_id'].nunique()} games")
else:
    print(f"\n[!] No features extracted for {season}")

print("="*70 + "\n")

