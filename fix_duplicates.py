import pandas as pd
import os

print("="*60)
print("FIXING DUPLICATE DATA IN PARTIAL_900")
print("="*60)

# Load the file
df = pd.read_csv('data/raw/pbp_2020_21_partial_900.csv')

print(f"\nBefore deduplication:")
print(f"  Total rows: {len(df):,}")
print(f"  Unique games: {df['GAME_ID'].nunique()}")

# Remove duplicate plays (keep first occurrence)
df_clean = df.drop_duplicates(subset=['GAME_ID', 'EVENTNUM'], keep='first')

print(f"\nAfter deduplication:")
print(f"  Total rows: {len(df_clean):,}")
print(f"  Unique games: {df_clean['GAME_ID'].nunique()}")
print(f"  Removed: {len(df) - len(df_clean):,} duplicate rows")

# Backup original
backup_file = 'data/raw/pbp_2020_21_partial_900_BACKUP.csv'
print(f"\nCreating backup: {backup_file}")
df.to_csv(backup_file, index=False)

# Save cleaned version
output_file = 'data/raw/pbp_2020_21_partial_900.csv'
print(f"Saving cleaned data: {output_file}")
df_clean.to_csv(output_file, index=False)

print("\n" + "="*60)
print("[OK] File cleaned successfully!")
print("="*60)

