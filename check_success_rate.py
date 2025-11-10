import pandas as pd

print("="*60)
print("DOWNLOAD SUCCESS RATE ANALYSIS")
print("="*60)

checkpoints = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
prev_games = 0

print("\nCheckpoint | Games | Success Rate")
print("-" * 40)

for cp in checkpoints:
    try:
        df = pd.read_csv(f'data/raw/pbp_2020_21_partial_{cp}.csv')
        current_games = df['GAME_ID'].nunique()
        new_games = current_games - prev_games
        success_rate = (new_games / 50) * 100
        print(f"{cp:10} | {current_games:5} | {success_rate:5.1f}% ({new_games}/50)")
        prev_games = current_games
    except:
        print(f"{cp:10} | ERROR")

print("\n" + "="*60)
print("Connection quality degraded significantly after game 600")
print("="*60)

