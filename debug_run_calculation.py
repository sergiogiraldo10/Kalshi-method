"""
Debug what our run calculation is actually doing
"""
import pandas as pd

# Load one game
df = pd.read_csv('data/raw/pbp_2020_21.csv')
game_id = df['GAME_ID'].iloc[0]
game_df = df[df['GAME_ID'] == game_id].copy()
game_df = game_df.sort_values('EVENTNUM')

# Build score history
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
        
        away_score = int(score_parts[0])
        home_score = int(score_parts[1])
        
        period = int(row['PERIOD'])
        time_parts = str(row['PCTIMESTRING']).split(':')
        mins = int(time_parts[0])
        secs = int(time_parts[1])
        time_in_period = mins * 60 + secs
        
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
    except:
        continue

print("="*60)
print("DEBUG: Run Calculation")
print("="*60)

# Test _calculate_current_run logic
def calculate_current_run(score_history, current_time, lookback_seconds=120):
    if len(score_history) < 2:
        return 0, 0
    
    home_run = 0
    away_run = 0
    
    # Look at recent plays within lookback window
    for i in range(len(score_history) - 1, 0, -1):
        current = score_history[i]
        previous = score_history[i - 1]
        
        # Check if within time window
        if current_time - current['time_remaining'] > lookback_seconds:
            break
        
        home_points = current['home'] - previous['home']
        away_points = current['away'] - previous['away']
        
        if home_points > 0:
            home_run += home_points
        if away_points > 0:
            away_run += away_points
    
    return home_run, away_run

# Test at different points in the game
test_points = [20, 40, 60, 80, 100, 120]

for idx in test_points:
    if idx >= len(score_history):
        continue
    
    current_time = score_history[idx]['time_remaining']
    home_run, away_run = calculate_current_run(score_history[:idx+1], current_time)
    
    print(f"\nAt play {idx}:")
    print(f"  Score: {score_history[idx]['away']}-{score_history[idx]['home']}")
    print(f"  Time remaining: {current_time // 60}:{current_time % 60:02d}")
    print(f"  Home run (last 2 min): {home_run}")
    print(f"  Away run (last 2 min): {away_run}")
    print(f"  Max run: {max(home_run, away_run)}")
    
    # Show score progression in last 2 minutes
    window_start = current_time - 120
    recent = [s for s in score_history[:idx+1] if s['time_remaining'] >= window_start]
    if len(recent) >= 2:
        print(f"  Score progression (last 2 min): {recent[0]['away']}-{recent[0]['home']} -> {recent[-1]['away']}-{recent[-1]['home']}")

print("\n" + "="*60)
print("PROBLEM: We're summing ALL points scored in last 2 minutes,")
print("not detecting a 'run' where one team scores unanswered!")
print("="*60)

