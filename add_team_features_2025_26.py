"""
Add team features for 2025-26 season
(Copy of add_team_features_2024_25.py but for 2025-26)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

print("\n" + "="*70)
print("ADDING TEAM FEATURES FOR 2025-26 SEASON")
print("="*70)

season = '2025_26'
features_file = Path(f'data/processed/features_v2_{season}.csv')
pbp_file = Path(f'data/raw/pbp_{season}.csv')

if not features_file.exists():
    print(f"\n[!] Features file not found: {features_file}")
    print("    Run: python extract_features_2025_26.py")
    exit()

if not pbp_file.exists():
    print(f"\n[!] PBP file not found: {pbp_file}")
    exit()

# Load data
print(f"\nLoading features...")
features = pd.read_csv(features_file)
print(f"  [OK] {len(features):,} samples")

print(f"\nLoading PBP...")
pbp_df = pd.read_csv(pbp_file)
print(f"  [OK] {len(pbp_df):,} plays from {pbp_df['GAME_ID'].nunique()} games")

# Extract game info
print("\nExtracting game info...")
game_info_list = []

for game_id in pbp_df['GAME_ID'].unique():
    game_plays = pbp_df[pbp_df['GAME_ID'] == game_id].copy()
    
    # Get team IDs
    first_play = game_plays.iloc[0]
    home_team = str(first_play.get('PLAYER1_TEAM_ID', 'UNK')) if 'PLAYER1_TEAM_ID' in first_play else 'UNK'
    away_team = str(first_play.get('PLAYER2_TEAM_ID', 'UNK')) if 'PLAYER2_TEAM_ID' in first_play else 'UNK'
    
    if home_team == 'UNK' or away_team == 'UNK':
        home_team = f"HOME_{game_id}"
        away_team = f"AWAY_{game_id}"
    
    # Get final score
    last_play = game_plays.iloc[-1]
    score_str = str(last_play['SCORE'])
    
    if pd.isna(score_str) or score_str == 'nan':
        continue
    
    try:
        if ' - ' in score_str:
            away_score, home_score = map(int, score_str.split(' - '))
        else:
            away_score, home_score = map(int, score_str.split('-'))
    except:
        continue
    
    home_win = 1 if home_score > away_score else 0
    
    game_info_list.append({
        'game_id': float(game_id),
        'home_team': home_team,
        'away_team': away_team,
        'home_score': home_score,
        'away_score': away_score,
        'home_win': home_win
    })

games_df = pd.DataFrame(game_info_list)
print(f"  [OK] {len(games_df)} games")

# Sort chronologically
games_df = games_df.sort_values('game_id')

# Calculate season-to-date stats
print("\nCalculating team stats (no future data leakage)...")

team_stats = defaultdict(lambda: {
    'games': 0, 'wins': 0, 'points': 0, 'opp_points': 0, 'recent_wins': []
})

games_with_stats = []

for idx, game in games_df.iterrows():
    game_id = game['game_id']
    home_team = game['home_team']
    away_team = game['away_team']
    
    # Get stats BEFORE this game
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Calculate metrics
    home_win_pct = home_stats['wins'] / max(home_stats['games'], 1)
    away_win_pct = away_stats['wins'] / max(away_stats['games'], 1)
    home_ppg = home_stats['points'] / max(home_stats['games'], 1)
    away_ppg = away_stats['points'] / max(away_stats['games'], 1)
    home_opp_ppg = home_stats['opp_points'] / max(home_stats['games'], 1)
    away_opp_ppg = away_stats['opp_points'] / max(away_stats['games'], 1)
    home_form_5g = sum(home_stats['recent_wins'][-5:]) / max(len(home_stats['recent_wins'][-5:]), 1)
    away_form_5g = sum(away_stats['recent_wins'][-5:]) / max(len(away_stats['recent_wins'][-5:]), 1)
    
    games_with_stats.append({
        'game_id': game_id,
        'home_team': home_team,
        'away_team': away_team,
        'home_win_pct': home_win_pct,
        'away_win_pct': away_win_pct,
        'home_ppg': home_ppg,
        'away_ppg': away_ppg,
        'home_opp_ppg': home_opp_ppg,
        'away_opp_ppg': away_opp_ppg,
        'home_form_5g': home_form_5g,
        'away_form_5g': away_form_5g
    })
    
    # Update stats AFTER this game
    home_won = game['home_win'] == 1
    team_stats[home_team]['games'] += 1
    team_stats[home_team]['wins'] += home_won
    team_stats[home_team]['points'] += game['home_score']
    team_stats[home_team]['opp_points'] += game['away_score']
    team_stats[home_team]['recent_wins'].append(1 if home_won else 0)
    team_stats[away_team]['games'] += 1
    team_stats[away_team]['wins'] += (1 - home_won)
    team_stats[away_team]['points'] += game['away_score']
    team_stats[away_team]['opp_points'] += game['home_score']
    team_stats[away_team]['recent_wins'].append(0 if home_won else 1)

team_stats_df = pd.DataFrame(games_with_stats)
print(f"  [OK] {len(team_stats_df)} games")

# Merge
print("\nMerging...")
features_enhanced = features.merge(team_stats_df, on='game_id', how='left')
print(f"  [OK] {len(features_enhanced):,} samples")

# Add run team context
print("\nAdding run team context...")

if 'run_team' in features_enhanced.columns:
    run_team_mask = features_enhanced['run_team'] == 'home'
else:
    run_team_mask = pd.Series([True] * len(features_enhanced), index=features_enhanced.index)

# Home runs
features_enhanced.loc[run_team_mask, 'run_team_win_pct'] = features_enhanced.loc[run_team_mask, 'home_win_pct']
features_enhanced.loc[run_team_mask, 'run_team_ppg'] = features_enhanced.loc[run_team_mask, 'home_ppg']
features_enhanced.loc[run_team_mask, 'run_team_opp_ppg'] = features_enhanced.loc[run_team_mask, 'home_opp_ppg']
features_enhanced.loc[run_team_mask, 'run_team_form_5g'] = features_enhanced.loc[run_team_mask, 'home_form_5g']
features_enhanced.loc[run_team_mask, 'opp_team_win_pct'] = features_enhanced.loc[run_team_mask, 'away_win_pct']
features_enhanced.loc[run_team_mask, 'opp_team_ppg'] = features_enhanced.loc[run_team_mask, 'away_ppg']
features_enhanced.loc[run_team_mask, 'opp_team_opp_ppg'] = features_enhanced.loc[run_team_mask, 'away_opp_ppg']
features_enhanced.loc[run_team_mask, 'opp_team_form_5g'] = features_enhanced.loc[run_team_mask, 'away_form_5g']

# Away runs
features_enhanced.loc[~run_team_mask, 'run_team_win_pct'] = features_enhanced.loc[~run_team_mask, 'away_win_pct']
features_enhanced.loc[~run_team_mask, 'run_team_ppg'] = features_enhanced.loc[~run_team_mask, 'away_ppg']
features_enhanced.loc[~run_team_mask, 'run_team_opp_ppg'] = features_enhanced.loc[~run_team_mask, 'away_opp_ppg']
features_enhanced.loc[~run_team_mask, 'run_team_form_5g'] = features_enhanced.loc[~run_team_mask, 'away_form_5g']
features_enhanced.loc[~run_team_mask, 'opp_team_win_pct'] = features_enhanced.loc[~run_team_mask, 'home_win_pct']
features_enhanced.loc[~run_team_mask, 'opp_team_ppg'] = features_enhanced.loc[~run_team_mask, 'home_ppg']
features_enhanced.loc[~run_team_mask, 'opp_team_opp_ppg'] = features_enhanced.loc[~run_team_mask, 'home_opp_ppg']
features_enhanced.loc[~run_team_mask, 'opp_team_form_5g'] = features_enhanced.loc[~run_team_mask, 'home_form_5g']

# Derived features
features_enhanced['team_quality_diff'] = features_enhanced['run_team_win_pct'] - features_enhanced['opp_team_win_pct']
features_enhanced['team_offensive_advantage'] = features_enhanced['run_team_ppg'] - features_enhanced['opp_team_opp_ppg']
features_enhanced['team_defensive_advantage'] = features_enhanced['opp_team_ppg'] - features_enhanced['run_team_opp_ppg']
features_enhanced['team_form_advantage'] = features_enhanced['run_team_form_5g'] - features_enhanced['opp_team_form_5g']

# Save
output_file = Path(f'data/processed/features_v2_{season}_enhanced.csv')
print(f"\nSaving to {output_file}...")
features_enhanced.to_csv(output_file, index=False)
print(f"  [OK] Saved {len(features_enhanced):,} samples")

print("\nNext step:")
print("  python train_for_live_alerts.py")

print("="*70 + "\n")

