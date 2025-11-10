import pandas as pd
import glob
import os

# Find all partial files
pattern = 'data/raw/pbp_2020_21_partial_*.csv'
partial_files = glob.glob(pattern)

if partial_files:
    # Find the latest
    latest_file = max(partial_files, key=lambda x: int(x.split('_partial_')[1].split('.csv')[0]))
    print(f"Latest file: {latest_file}")
    
    # Load it
    df = pd.read_csv(latest_file)
    print(f"Total plays: {len(df):,}")
    print(f"Unique games: {df['GAME_ID'].nunique()}")
    print(f"Sample game IDs: {list(df['GAME_ID'].unique()[:5])}")
else:
    print("No partial files found")

