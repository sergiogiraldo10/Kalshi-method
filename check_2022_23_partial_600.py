import pandas as pd
import os

print("="*60)
print("2022-23 PARTIAL_600 FILE CHECK")
print("="*60)

file = 'data/raw/pbp_2022_23_partial_600.csv'
if os.path.exists(file):
    df = pd.read_csv(file)
    print(f"\nFile: {file}")
    print(f"  Unique games: {df['GAME_ID'].nunique()}")
    print(f"  Total plays: {len(df):,}")
    print(f"  Duplicate plays: {df.duplicated(subset=['GAME_ID', 'EVENTNUM']).sum():,}")
    
    print(f"\n  Checkpoint says: 650 games")
    print(f"  Actual games in file: {df['GAME_ID'].nunique()}")
    
    if df['GAME_ID'].nunique() == 650:
        print("\n  [OK] Checkpoint message is CORRECT")
        print("  This likely means the download resumed from a previous session")
        print("  with ~50 games already done, then downloaded 600 more attempts")
    else:
        print(f"\n  [ERROR] Mismatch! File has {df['GAME_ID'].nunique()} games, not 650")
else:
    print(f"\n[X] File not found: {file}")

print("\n" + "="*60)

