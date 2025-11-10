"""
Check 2024-25 season for duplicates
"""

import pandas as pd
import os

print("\n" + "="*70)
print("CHECKING 2024-25 SEASON FOR DUPLICATES")
print("="*70)

file_path = 'data/raw/pbp_2024_25.csv'

if not os.path.exists(file_path):
    print(f"\n[!] File not found: {file_path}")
    exit()

print(f"\nLoading {file_path}...")
df = pd.read_csv(file_path)

print(f"\n  Total rows: {len(df):,}")
print(f"  Unique games: {df['GAME_ID'].nunique()}")
print(f"  Total plays: {len(df):,}")

# Check for duplicates
duplicates = df[df.duplicated(subset=['GAME_ID', 'EVENTNUM'], keep=False)]

if len(duplicates) > 0:
    print(f"\n[!] FOUND {len(duplicates):,} DUPLICATE PLAYS")
    print(f"    Affecting {duplicates['GAME_ID'].nunique()} games")
    
    # Remove duplicates
    print("\n  Removing duplicates...")
    df_clean = df.drop_duplicates(subset=['GAME_ID', 'EVENTNUM'], keep='first')
    
    print(f"  Before: {len(df):,} rows")
    print(f"  After:  {len(df_clean):,} rows")
    print(f"  Removed: {len(df) - len(df_clean):,} rows")
    
    # Create backup
    backup_path = file_path.replace('.csv', '_BACKUP.csv')
    print(f"\n  Creating backup: {backup_path}")
    os.rename(file_path, backup_path)
    
    # Save cleaned data
    print(f"  Saving cleaned data: {file_path}")
    df_clean.to_csv(file_path, index=False)
    
    print("\n[OK] Duplicates removed!")
else:
    print(f"\n[OK] NO DUPLICATES FOUND - Data is clean!")

print("\nFinal stats:")
print(f"  Games: {df['GAME_ID'].nunique() if len(duplicates) == 0 else df_clean['GAME_ID'].nunique()}")
print(f"  Plays: {len(df) if len(duplicates) == 0 else len(df_clean):,}")

print("="*70 + "\n")

