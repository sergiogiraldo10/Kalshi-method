import pandas as pd

print("="*60)
print("CHECKING FOR DUPLICATE GAMES IN PARTIAL_900")
print("="*60)

df = pd.read_csv('data/raw/pbp_2020_21_partial_900.csv')

print(f"\nTotal rows: {len(df):,}")
print(f"Unique games: {df['GAME_ID'].nunique()}")

# Check for duplicates
game_counts = df['GAME_ID'].value_counts()
duplicates = game_counts[game_counts > 1]

if len(duplicates) > 0:
    print(f"\n[WARNING] Found {len(duplicates)} games that appear multiple times!")
    print("\nDuplicate games:")
    print("-" * 40)
    for game_id, count in duplicates.head(20).items():
        plays_for_game = df[df['GAME_ID'] == game_id]
        print(f"  Game {game_id}: appears {count} times ({len(plays_for_game)} total plays)")
    
    if len(duplicates) > 20:
        print(f"  ... and {len(duplicates) - 20} more duplicates")
    
    # Calculate wasted space
    total_plays = len(df)
    unique_plays = df.drop_duplicates(subset=['GAME_ID', 'EVENTNUM']).shape[0]
    duplicate_plays = total_plays - unique_plays
    
    print(f"\nDuplicate play rows: {duplicate_plays:,}")
    print(f"Percentage duplicate: {(duplicate_plays/total_plays)*100:.1f}%")
else:
    print("\n[OK] No duplicate games found! Each game appears exactly once.")

# Check for duplicate plays within games
print("\n" + "="*60)
print("CHECKING FOR DUPLICATE PLAYS (within each game)")
print("="*60)

duplicate_plays = df[df.duplicated(subset=['GAME_ID', 'EVENTNUM'], keep=False)]

if len(duplicate_plays) > 0:
    print(f"\n[WARNING] Found {len(duplicate_plays):,} duplicate play records!")
    print(f"Percentage duplicate: {(len(duplicate_plays)/len(df))*100:.1f}%")
    
    # Show sample
    sample_game = duplicate_plays['GAME_ID'].iloc[0]
    sample = df[df['GAME_ID'] == sample_game].head(10)
    print(f"\nSample duplicate plays from game {sample_game}:")
    print(sample[['GAME_ID', 'EVENTNUM', 'PERIOD', 'PCTIMESTRING', 'HOMEDESCRIPTION']].to_string())
else:
    print("\n[OK] No duplicate plays found!")

print("\n" + "="*60)

