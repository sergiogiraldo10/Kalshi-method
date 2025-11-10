"""
FINAL FIX: Detect actual consecutive scoring runs
Example: 8-0 run means last 8 points were scored by one team
"""

def detect_consecutive_run(score_history, current_time, max_lookback_seconds=180):
    """
    Detect a consecutive scoring run (e.g., last 10 points scored 8-2)
    
    Returns: (team, team_points, opp_points) for the current run
    """
    if len(score_history) < 3:
        return None, 0, 0
    
    # Start from current and work backwards
    current_idx = len(score_history) - 1
    current_score = score_history[current_idx]
    
    # Track cumulative points going backwards
    home_points = 0
    away_points = 0
    
    # Go backwards through score history
    for i in range(current_idx, 0, -1):
        curr = score_history[i]
        prev = score_history[i-1]
        
        # Check if we've gone too far back in time
        time_diff = current_score['time_remaining'] - curr['time_remaining']
        if time_diff > max_lookback_seconds:
            break
        
        # Calculate points scored
        h_pts = curr['home'] - prev['home']
        a_pts = curr['away'] - prev['away']
        
        home_points += h_pts
        away_points += a_pts
        
        # Stop if we have enough data and one team has clear momentum
        if home_points + away_points >= 4:
            # Check if this qualifies as a "run"
            if home_points >= 4 and home_points >= away_points * 2:
                return 'home', home_points, away_points
            elif away_points >= 4 and away_points >= home_points * 2:
                return 'away', away_points, home_points
            elif home_points >= 8 and home_points >= away_points + 4:
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
print("FINAL RUN DETECTION - Consecutive Scoring")
print("="*60)

runs_found = []

for idx in range(10, len(score_history)):
    team, team_pts, opp_pts = detect_consecutive_run(score_history[:idx+1], score_history[idx]['time_remaining'])
    
    if team and team_pts >= 6:  # Only show significant runs
        runs_found.append({
            'team': team,
            'run': f"{team_pts}-{opp_pts}",
            'score': f"{score_history[idx]['away']}-{score_history[idx]['home']}",
            'time_min': score_history[idx]['time_remaining'] // 60
        })

print(f"\nFound {len(runs_found)} significant runs (6+ points) in this game:\n")
for i, run in enumerate(runs_found[:20]):
    print(f"  {i+1}. {run['team'].upper()} on {run['run']} run | Score: {run['score']} | {run['time_min']} min left")

print(f"\n... ({len(runs_found)} total)")

# Statistics
if runs_found:
    run_scores = [int(r['run'].split('-')[0]) for r in runs_found]
    print(f"\nRun score stats: min={min(run_scores)}, max={max(run_scores)}, mean={sum(run_scores)/len(run_scores):.1f}")

print("\n" + "="*60)

