import pandas as pd

# Check the 750 checkpoint
df = pd.read_csv('data/raw/pbp_2021_22_partial_750.csv')

print("="*60)
print("CHECKPOINT 750 ANALYSIS")
print("="*60)
print(f"\nFile: pbp_2021_22_partial_750.csv")
print(f"Unique games: {df['GAME_ID'].nunique()}")
print(f"Total plays: {len(df):,}")
print(f"\nSample game IDs: {list(df['GAME_ID'].unique()[:5])}")

# Check if there are duplicates
duplicates = df.duplicated(subset=['GAME_ID', 'EVENTNUM']).sum()
print(f"\nDuplicate plays: {duplicates:,}")

print("\n" + "="*60)
print("VERDICT:")
if df['GAME_ID'].nunique() == 900:
    print("The file DOES have 900 unique games!")
    print("This means it's including old data from previous partial files.")
    print("\nExplanation:")
    print("- Checkpoint 750 = 750 download ATTEMPTS in current session")
    print("- But the file contains 900 total UNIQUE games")
    print("- This includes games from previous sessions")
else:
    print(f"The file has {df['GAME_ID'].nunique()} games, not 900")
    print("The display message is incorrect")
print("="*60)

