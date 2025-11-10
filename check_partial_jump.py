import pandas as pd

print("="*60)
print("CHECKING PARTIAL FILE PROGRESSION")
print("="*60)

# Check 600 and 650
df600 = pd.read_csv('data/raw/pbp_2020_21_partial_600.csv')
df650 = pd.read_csv('data/raw/pbp_2020_21_partial_650.csv')

print("\nPartial 600:")
print(f"  Total plays: {len(df600):,}")
print(f"  Unique games: {df600['GAME_ID'].nunique()}")

print("\nPartial 650:")
print(f"  Total plays: {len(df650):,}")
print(f"  Unique games: {df650['GAME_ID'].nunique()}")

new_plays = len(df650) - len(df600)
new_games = df650['GAME_ID'].nunique() - df600['GAME_ID'].nunique()

print(f"\nDifference (650 - 600):")
print(f"  New plays: {new_plays:,}")
print(f"  New games: {new_games}")
if new_games > 0:
    print(f"  Avg plays/game: {new_plays / new_games:.0f}")

# Find which games are in 650 but not in 600
games_600 = set(df600['GAME_ID'].unique())
games_650 = set(df650['GAME_ID'].unique())
new_games_ids = games_650 - games_600

print(f"\nNew games added in checkpoint 650:")
print(f"  Count: {len(new_games_ids)}")

if len(new_games_ids) > 0:
    print(f"\n  Sample new game IDs: {sorted(list(new_games_ids))[:10]}")
    
    # Check play counts for new games
    new_game_data = df650[df650['GAME_ID'].isin(new_games_ids)]
    plays_per_game = new_game_data.groupby('GAME_ID').size()
    print(f"\n  Play counts for new games:")
    print(f"    Min: {plays_per_game.min()}")
    print(f"    Max: {plays_per_game.max()}")
    print(f"    Mean: {plays_per_game.mean():.0f}")
    print(f"    Median: {plays_per_game.median():.0f}")

print("\n" + "="*60)

