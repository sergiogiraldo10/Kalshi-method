"""
Deduplicate 2023-24 season file
"""
import pandas as pd

print("="*60)
print("DEDUPLICATING 2023-24 SEASON")
print("="*60)

# Load the file
file_path = 'data/raw/pbp_2023_24.csv'
print(f"\nLoading {file_path}...")
df = pd.read_csv(file_path)

print(f"Before deduplication:")
print(f"  Total rows: {len(df):,}")
print(f"  Unique games: {df['GAME_ID'].nunique()}")

# Check for duplicates
duplicates = df.duplicated(subset=['GAME_ID', 'EVENTNUM'])
print(f"  Duplicate rows: {duplicates.sum():,}")

# Remove duplicates
df_clean = df.drop_duplicates(subset=['GAME_ID', 'EVENTNUM'], keep='first')

print(f"\nAfter deduplication:")
print(f"  Total rows: {len(df_clean):,}")
print(f"  Unique games: {df_clean['GAME_ID'].nunique()}")
print(f"  Removed: {len(df) - len(df_clean):,} duplicate rows")

# Backup original
backup_file = 'data/raw/pbp_2023_24_BACKUP.csv'
print(f"\nCreating backup: {backup_file}")
df.to_csv(backup_file, index=False)

# Save cleaned version
print(f"Saving cleaned data: {file_path}")
df_clean.to_csv(file_path, index=False)

print("\n" + "="*60)
print("[OK] File cleaned successfully!")
print("="*60)

