"""
Extract features for 2025-26 season
"""

import sys
sys.path.append('src')

from feature_engineering_v2 import MomentumFeatureExtractorV2
import pandas as pd
from pathlib import Path

print("\n" + "="*70)
print("EXTRACTING FEATURES FOR 2025-26 SEASON")
print("="*70)

season = '2025_26'
pbp_file = Path(f'data/raw/pbp_{season}.csv')

if not pbp_file.exists():
    print(f"\n[!] PBP file not found: {pbp_file}")
    print("    Run: python download_2025_26_season.py")
    exit()

# Load PBP
print(f"\nLoading {pbp_file}...")
pbp_df = pd.read_csv(pbp_file)
print(f"  [OK] Loaded {len(pbp_df):,} plays from {pbp_df['GAME_ID'].nunique()} games")

# Extract features
print("\nExtracting features (5-10 minutes)...")
extractor = MomentumFeatureExtractorV2()
features_df = extractor.extract_features(pbp_df, include_outcomes=True)

if len(features_df) > 0:
    output_file = Path(f'data/processed/features_v2_{season}.csv')
    print(f"\nSaving to {output_file}...")
    features_df.to_csv(output_file, index=False)
    print(f"  [OK] Saved {len(features_df):,} samples")
else:
    print(f"\n[!] No features extracted")

print("\nNext step:")
print(f"  python add_team_features_2025_26.py")

print("="*70 + "\n")

