"""
Debug feature extraction to see what's happening
"""
import pandas as pd

# Load one game
df = pd.read_csv('data/raw/pbp_2020_21.csv')
game_id = df['GAME_ID'].iloc[0]
game_df = df[df['GAME_ID'] == game_id].copy()
game_df = game_df.sort_values('EVENTNUM')

print(f"Game ID: {game_id}")
print(f"Total plays: {len(game_df)}")

# Track score history
score_history = []

for idx, row in game_df.iterrows():
    if pd.isna(row['SCORE']):
        continue
    
    try:
        score_str = str(row['SCORE'])
        if ' - ' in score_str:
            score_parts = score_str.split(' - ')
        else:
            score_parts = score_str.split('-')
        
        if len(score_parts) != 2:
            continue
        away_score = int(score_parts[0])
        home_score = int(score_parts[1])
    except:
        continue
    
    try:
        period = int(row['PERIOD'])
        time_parts = str(row['PCTIMESTRING']).split(':')
        if len(time_parts) != 2:
            continue
        mins = int(time_parts[0])
        secs = int(time_parts[1])
        time_in_period = mins * 60 + secs
    except:
        continue
    
    if period <= 4:
        time_remaining = (4 - period) * 720 + time_in_period
    else:
        time_remaining = max(0, (5 - (period - 4)) * 300 + time_in_period)
    
    score_history.append({
        'home': home_score,
        'away': away_score,
        'time_remaining': time_remaining,
        'event_num': row['EVENTNUM']
    })

print(f"\nScore history length: {len(score_history)}")
print(f"First 5 scores: {score_history[:5]}")
print(f"Last 5 scores: {score_history[-5:]}")

# Check if we have enough lookforward
lookforward_plays = 20
samples_with_lookforward = len(score_history) - lookforward_plays
print(f"\nSamples with enough lookforward: {samples_with_lookforward}")

