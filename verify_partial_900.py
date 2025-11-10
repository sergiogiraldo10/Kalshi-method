import pandas as pd

df = pd.read_csv('data/raw/pbp_2020_21_partial_900.csv')

print("="*60)
print("PARTIAL_900 FILE ANALYSIS")
print("="*60)
print(f"\nTotal rows (plays): {len(df):,}")
print(f"Unique games: {df['GAME_ID'].nunique()}")

games = df.groupby('GAME_ID').size()
print(f"\nTotal games in file: {len(games)}")

print("\nFirst 10 games:")
for i, (gid, count) in enumerate(games.head(10).items()):
    print(f"  Game {i+1}: {gid} ({count} plays)")

print("\nLast 10 games:")
for i, (gid, count) in enumerate(games.tail(10).items()):
    print(f"  Game {len(games)-9+i}: {gid} ({count} plays)")

print("\n" + "="*60)
print(f"CHECKPOINT 900 = {len(games)} SUCCESSFUL GAMES")
print("="*60)

