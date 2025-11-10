"""
Add Team Context Features to Training Data
Following ACTION PLAN Option 1 - Improve the Model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("\n" + "="*70)
print("ADDING TEAM CONTEXT FEATURES")
print("Following Action Plan to Profitability - Option 1")
print("="*70)

# Load play-by-play data
print("\nLoading 2023-24 season data...")
pbp_df = pd.read_csv('data/raw/pbp_2023_24.csv')
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')

print(f"  [OK] Loaded {len(pbp_df):,} plays")
print(f"  [OK] Loaded {len(features_df):,} feature samples")

# Extract team identifiers and game info
print("\nExtracting team and game information...")

# Get unique games and their teams
games_info = []
for game_id in pbp_df['GAME_ID'].unique():
    game_plays = pbp_df[pbp_df['GAME_ID'] == game_id]
    
    # Get teams from first play with description
    home_team = None
    away_team = None
    
    for idx, row in game_plays.iterrows():
        if pd.notna(row.get('HOMEDESCRIPTION', '')):
            home_team = str(row.get('PLAYER1_TEAM_ID', 0))
            break
    
    for idx, row in game_plays.iterrows():
        if pd.notna(row.get('VISITORDESCRIPTION', '')):
            away_team = str(row.get('PLAYER2_TEAM_ID', 0))
            break
    
    if home_team and away_team:
        games_info.append({
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team
        })

games_df = pd.DataFrame(games_info)
print(f"  [OK] Extracted info for {len(games_df)} games")

# Calculate team statistics up to each game
print("\nCalculating team statistics (win%, points per game)...")

def calculate_team_stats(games_df, pbp_df):
    """
    Calculate cumulative team statistics
    - Win percentage
    - Points per game
    - Opponent points per game
    - Recent form (last 5 games)
    """
    
    # Sort games by ID (chronological)
    games_df = games_df.sort_values('game_id').reset_index(drop=True)
    
    # Initialize team stats tracking
    team_stats = {}
    
    for idx, game in games_df.iterrows():
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Get final score for this game
        game_plays = pbp_df[pbp_df['GAME_ID'] == game_id]
        final_play = game_plays[game_plays['SCORE'].notna()].iloc[-1] if len(game_plays[game_plays['SCORE'].notna()]) > 0 else None
        
        if final_play is not None:
            try:
                score_str = str(final_play['SCORE'])
                if ' - ' in score_str:
                    away_score, home_score = map(int, score_str.split(' - '))
                else:
                    away_score, home_score = map(int, score_str.split('-'))
                
                home_won = 1 if home_score > away_score else 0
                away_won = 1 if away_score > home_score else 0
                
                # Initialize team stats if needed
                for team in [home_team, away_team]:
                    if team not in team_stats:
                        team_stats[team] = {
                            'games': [],
                            'wins': 0,
                            'losses': 0,
                            'points_for': [],
                            'points_against': []
                        }
                
                # Store current stats BEFORE updating (for this game)
                games_df.at[idx, 'home_win_pct'] = team_stats[home_team]['wins'] / max(1, len(team_stats[home_team]['games']))
                games_df.at[idx, 'away_win_pct'] = team_stats[away_team]['wins'] / max(1, len(team_stats[away_team]['games']))
                
                games_df.at[idx, 'home_ppg'] = np.mean(team_stats[home_team]['points_for']) if len(team_stats[home_team]['points_for']) > 0 else 0
                games_df.at[idx, 'away_ppg'] = np.mean(team_stats[away_team]['points_for']) if len(team_stats[away_team]['points_for']) > 0 else 0
                
                games_df.at[idx, 'home_opp_ppg'] = np.mean(team_stats[home_team]['points_against']) if len(team_stats[home_team]['points_against']) > 0 else 0
                games_df.at[idx, 'away_opp_ppg'] = np.mean(team_stats[away_team]['points_against']) if len(team_stats[away_team]['points_against']) > 0 else 0
                
                # Recent form (last 5 games)
                home_recent = team_stats[home_team]['games'][-5:] if len(team_stats[home_team]['games']) > 0 else []
                away_recent = team_stats[away_team]['games'][-5:] if len(team_stats[away_team]['games']) > 0 else []
                
                games_df.at[idx, 'home_form_5g'] = sum(home_recent) / max(1, len(home_recent))
                games_df.at[idx, 'away_form_5g'] = sum(away_recent) / max(1, len(away_recent))
                
                # UPDATE stats after recording
                team_stats[home_team]['games'].append(home_won)
                team_stats[home_team]['wins'] += home_won
                team_stats[home_team]['losses'] += (1 - home_won)
                team_stats[home_team]['points_for'].append(home_score)
                team_stats[home_team]['points_against'].append(away_score)
                
                team_stats[away_team]['games'].append(away_won)
                team_stats[away_team]['wins'] += away_won
                team_stats[away_team]['losses'] += (1 - away_won)
                team_stats[away_team]['points_for'].append(away_score)
                team_stats[away_team]['points_against'].append(home_score)
                
            except:
                continue
        
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(games_df)} games...")
    
    return games_df

games_with_stats = calculate_team_stats(games_df, pbp_df)
print("  [OK] Team statistics calculated")

# Merge team features with existing features
print("\nMerging team features with existing features...")

features_enhanced = features_df.merge(
    games_with_stats[['game_id', 'home_win_pct', 'away_win_pct', 
                      'home_ppg', 'away_ppg', 'home_opp_ppg', 'away_opp_ppg',
                      'home_form_5g', 'away_form_5g']],
    on='game_id',
    how='left'
)

# Determine which team has the run and add their stats (VECTORIZED for speed)
print("\nDetermining run team and adding contextual features...")

# Vectorized approach - much faster!
# Check if run_team column exists, default to 'home' if not
if 'run_team' in features_enhanced.columns:
    run_team_mask = features_enhanced['run_team'] == 'home'
else:
    # Default to home if column doesn't exist
    run_team_mask = pd.Series([True] * len(features_enhanced), index=features_enhanced.index)

# For home team runs
features_enhanced.loc[run_team_mask, 'run_team_win_pct'] = features_enhanced.loc[run_team_mask, 'home_win_pct']
features_enhanced.loc[run_team_mask, 'run_team_ppg'] = features_enhanced.loc[run_team_mask, 'home_ppg']
features_enhanced.loc[run_team_mask, 'run_team_opp_ppg'] = features_enhanced.loc[run_team_mask, 'home_opp_ppg']
features_enhanced.loc[run_team_mask, 'run_team_form_5g'] = features_enhanced.loc[run_team_mask, 'home_form_5g']
features_enhanced.loc[run_team_mask, 'opp_team_win_pct'] = features_enhanced.loc[run_team_mask, 'away_win_pct']
features_enhanced.loc[run_team_mask, 'opp_team_ppg'] = features_enhanced.loc[run_team_mask, 'away_ppg']
features_enhanced.loc[run_team_mask, 'opp_team_opp_ppg'] = features_enhanced.loc[run_team_mask, 'away_opp_ppg']
features_enhanced.loc[run_team_mask, 'opp_team_form_5g'] = features_enhanced.loc[run_team_mask, 'away_form_5g']

# For away team runs
features_enhanced.loc[~run_team_mask, 'run_team_win_pct'] = features_enhanced.loc[~run_team_mask, 'away_win_pct']
features_enhanced.loc[~run_team_mask, 'run_team_ppg'] = features_enhanced.loc[~run_team_mask, 'away_ppg']
features_enhanced.loc[~run_team_mask, 'run_team_opp_ppg'] = features_enhanced.loc[~run_team_mask, 'away_opp_ppg']
features_enhanced.loc[~run_team_mask, 'run_team_form_5g'] = features_enhanced.loc[~run_team_mask, 'away_form_5g']
features_enhanced.loc[~run_team_mask, 'opp_team_win_pct'] = features_enhanced.loc[~run_team_mask, 'home_win_pct']
features_enhanced.loc[~run_team_mask, 'opp_team_ppg'] = features_enhanced.loc[~run_team_mask, 'home_ppg']
features_enhanced.loc[~run_team_mask, 'opp_team_opp_ppg'] = features_enhanced.loc[~run_team_mask, 'home_opp_ppg']
features_enhanced.loc[~run_team_mask, 'opp_team_form_5g'] = features_enhanced.loc[~run_team_mask, 'home_form_5g']

# Derived features (vectorized)
features_enhanced['team_quality_diff'] = features_enhanced['run_team_win_pct'] - features_enhanced['opp_team_win_pct']
features_enhanced['team_offensive_advantage'] = features_enhanced['run_team_ppg'] - features_enhanced['opp_team_opp_ppg']
features_enhanced['team_defensive_advantage'] = features_enhanced['opp_team_ppg'] - features_enhanced['run_team_opp_ppg']
features_enhanced['team_form_advantage'] = features_enhanced['run_team_form_5g'] - features_enhanced['opp_team_form_5g']

# Fill any NaN values with 0.5 (neutral) for percentages and 100 for ppg
features_enhanced = features_enhanced.fillna({
    'run_team_win_pct': 0.5,
    'opp_team_win_pct': 0.5,
    'run_team_ppg': 100,
    'opp_team_ppg': 100,
    'run_team_opp_ppg': 100,
    'opp_team_opp_ppg': 100,
    'run_team_form_5g': 0.5,
    'opp_team_form_5g': 0.5,
    'team_quality_diff': 0,
    'team_offensive_advantage': 0,
    'team_defensive_advantage': 0,
    'team_form_advantage': 0
})

print("  [OK] Team features added")

# Show summary
print("\n" + "="*70)
print("NEW TEAM FEATURES ADDED")
print("="*70)

new_features = [
    'run_team_win_pct', 'opp_team_win_pct',
    'run_team_ppg', 'opp_team_ppg',
    'run_team_form_5g', 'opp_team_form_5g',
    'team_quality_diff', 'team_offensive_advantage',
    'team_defensive_advantage', 'team_form_advantage'
]

print("\nNew Features:")
for feat in new_features:
    mean_val = features_enhanced[feat].mean()
    std_val = features_enhanced[feat].std()
    print(f"  {feat:<30}: mean={mean_val:>6.3f}, std={std_val:>6.3f}")

# Save enhanced features
output_file = 'data/processed/features_v2_2023_24_enhanced.csv'
features_enhanced.to_csv(output_file, index=False)

print(f"\n[OK] Enhanced features saved to: {output_file}")
print(f"    Total features: {len(features_enhanced.columns)}")
print(f"    Total samples: {len(features_enhanced):,}")
print(f"    New team features: {len(new_features)}")

print("\n" + "="*70)
print("NEXT STEP: Retrain model with enhanced features")
print("="*70 + "\n")

