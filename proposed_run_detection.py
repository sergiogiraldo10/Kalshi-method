"""
PROPOSED: Better Run Detection Algorithm

A "run" should detect when one team is on a hot streak (momentum)
"""

def detect_true_run(score_history, current_time, lookback_seconds=120):
    """
    Detect if there's a momentum run happening
    
    Returns: (team_with_run, run_score, opponent_score)
    Example: ('home', 10, 2) means home team on a 10-2 run
    """
    if len(score_history) < 3:
        return None, 0, 0
    
    # Find scores in the lookback window
    window_start_time = current_time - lookback_seconds
    window_scores = []
    
    for i in range(len(score_history) - 1, -1, -1):
        if score_history[i]['time_remaining'] >= window_start_time:
            window_scores.insert(0, score_history[i])
        else:
            break
    
    if len(window_scores) < 2:
        return None, 0, 0
    
    # Calculate points scored by each team in window
    start_score = window_scores[0]
    end_score = window_scores[-1]
    
    home_points = end_score['home'] - start_score['home']
    away_points = end_score['away'] - start_score['away']
    
    # Determine if there's a run (one team significantly outscoring the other)
    # Definition: Run exists if one team scores 4+ AND outscores opponent by 2:1 ratio
    
    if home_points >= 4 and home_points >= away_points * 2:
        return 'home', home_points, away_points
    elif away_points >= 4 and away_points >= home_points * 2:
        return 'away', away_points, home_points
    
    # Also detect strong runs where team scores a lot more (even if not 2:1)
    if home_points >= 8 and home_points >= away_points + 4:
        return 'home', home_points, away_points
    elif away_points >= 8 and away_points >= home_points + 4:
        return 'away', away_points, home_points
    
    return None, 0, 0


# Test on real data
import pandas as pd

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
            'time_remaining': time_remaining
        })
    except:
        continue

print("="*60)
print("PROPOSED RUN DETECTION - Finding Real Momentum Runs")
print("="*60)

runs_found = []

for idx in range(len(score_history)):
    current_time = score_history[idx]['time_remaining']
    team, team_score, opp_score = detect_true_run(score_history[:idx+1], current_time)
    
    if team:
        runs_found.append({
            'play_idx': idx,
            'team': team,
            'run': f"{team_score}-{opp_score}",
            'score': f"{score_history[idx]['away']}-{score_history[idx]['home']}",
            'time_min': current_time // 60
        })

print(f"\nFound {len(runs_found)} momentum runs in this game:\n")
for run in runs_found[:20]:  # Show first 20
    print(f"  {run['team'].upper()} on {run['run']} run | Score: {run['score']} | Time: {run['time_min']} min left")

print(f"\n... (showing first 20 of {len(runs_found)} total runs)")

print("\n" + "="*60)
print("This looks much more realistic!")
print("="*60)

