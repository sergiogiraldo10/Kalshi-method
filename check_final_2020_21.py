import pandas as pd
import os

print("="*60)
print("2020-21 SEASON STATUS CHECK")
print("="*60)

# Check if final file exists
final_file = 'data/raw/pbp_2020_21.csv'
if os.path.exists(final_file):
    df = pd.read_csv(final_file)
    print(f"\n[OK] Final file exists: {final_file}")
    print(f"  Total plays: {len(df):,}")
    print(f"  Unique games: {df['GAME_ID'].nunique()}")
    print(f"  Expected: ~1215 games")
    print(f"  Status: {'COMPLETE' if df['GAME_ID'].nunique() >= 1200 else 'INCOMPLETE'}")
    
    # Check for duplicates
    dupes = df.duplicated(subset=['GAME_ID', 'EVENTNUM']).sum()
    print(f"\n  Duplicate plays: {dupes:,}")
    if dupes > 0:
        print(f"  WARNING: File has {dupes:,} duplicate plays")
else:
    print(f"\n[X] Final file NOT found: {final_file}")
    print("  Season download is still in progress")

print("\n" + "="*60)

