"""Check what's wrong with the count display"""
import pandas as pd
import glob

# Check latest partial file for 2023-24
pattern = 'data/raw/pbp_2023_24_partial_*.csv'
partial_files = glob.glob(pattern)
partial_files = [f for f in partial_files if 'BACKUP' not in f]

if partial_files:
    latest = max(partial_files, key=lambda x: int(x.split('_partial_')[1].split('.csv')[0]))
    checkpoint_num = int(latest.split('_partial_')[1].split('.csv')[0])
    
    df = pd.read_csv(latest)
    actual_games = df['GAME_ID'].nunique()
    
    print("="*60)
    print("COUNT DISCREPANCY CHECK")
    print("="*60)
    print(f"\nLatest checkpoint: {latest.split('/')[-1]}")
    print(f"Checkpoint number: {checkpoint_num}")
    print(f"Actual unique games in file: {actual_games}")
    print(f"\nDifference: {checkpoint_num - actual_games}")
    
    if checkpoint_num != actual_games:
        print(f"\n[ISSUE] Checkpoint says {checkpoint_num} but file has {actual_games} games")
        print("This means the checkpoint number includes failed downloads")
    else:
        print("\n[OK] Counts match!")
else:
    print("No partial files found")

