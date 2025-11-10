"""
Prepare enhanced features for multiple seasons
Add team context features to 2021-22 and 2022-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the add_team_features function
sys.path.append('.')

def calculate_team_stats(pbp_df):
    """Calculate season-to-date team statistics for each game"""
    print("\n  Calculating team statistics...")
    
    # Extract game-level info
    games_list = []
    
    for game_id in pbp_df['GAME_ID'].unique():
        game_plays = pbp_df[pbp_df['GAME_ID'] == game_id]
        
        # Get final score
        final_play = game_plays.iloc[-1]
        score_str = str(final_play['SCORE'])
        
        try:
            if ' - ' in score_str:
                away_final, home_final = map(int, score_str.split(' - '))
            else:
                away_final, home_final = map(int, score_str.split('-'))
        except:
            continue
        
        # Extract team names from player info
        home_team = None
        away_team = None
        
        for _, play in game_plays.head(50).iterrows():
            if not pd.isna(play.get('PLAYER1_TEAM_NICKNAME')):
                team_name = play['PLAYER1_TEAM_NICKNAME']
                if not pd.isna(play.get('HOMEDESCRIPTION')) and home_team is None:
                    home_team = team_name
                elif not pd.isna(play.get('VISITORDESCRIPTION')) and away_team is None:
                    away_team = team_name
            
            if home_team and away_team:
                break
        
        # Fallback: use first two unique teams
        if home_team is None or away_team is None:
            all_teams = []
            for _, play in game_plays.head(100).iterrows():
                if not pd.isna(play.get('PLAYER1_TEAM_NICKNAME')):
                    all_teams.append(play['PLAYER1_TEAM_NICKNAME'])
            
            unique_teams = list(set(all_teams))
            if len(unique_teams) >= 2:
                home_team = unique_teams[0] if home_team is None else home_team
                away_team = unique_teams[1] if away_team is None else away_team
        
        if home_team is None or away_team is None:
            continue
        
        games_list.append({
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_final,
            'away_score': away_final,
            'home_won': 1 if home_final > away_final else 0
        })
    
    games_df = pd.DataFrame(games_list)
    games_df = games_df.sort_values('game_id')
    
    print(f"    Found {len(games_df)} games")
    
    # Calculate season-to-date stats
    team_stats = {}
    
    for idx, game in games_df.iterrows():
        game_id = game['game_id']
        
        for team, is_home in [(game['home_team'], True), (game['away_team'], False)]:
            # Get all previous games for this team
            prev_games = games_df[
                ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
                (games_df['game_id'] < game_id)
            ]
            
            if len(prev_games) == 0:
                stats = {
                    'win_pct': 0.5,
                    'ppg': 100,
                    'opp_ppg': 100,
                    'form_5g': 0.5,
                    'games_played': 0
                }
            else:
                wins = 0
                total_points = 0
                total_opp_points = 0
                recent_results = []
                
                for _, pg in prev_games.iterrows():
                    if pg['home_team'] == team:
                        total_points += pg['home_score']
                        total_opp_points += pg['away_score']
                        result = 1 if pg['home_won'] == 1 else 0
                    else:
                        total_points += pg['away_score']
                        total_opp_points += pg['home_score']
                        result = 1 if pg['home_won'] == 0 else 0
                    
                    wins += result
                    recent_results.append(result)
                
                games_played = len(prev_games)
                win_pct = wins / games_played
                ppg = total_points / games_played
                opp_ppg = total_opp_points / games_played
                form_5g = np.mean(recent_results[-5:]) if len(recent_results) >= 5 else win_pct
                
                stats = {
                    'win_pct': win_pct,
                    'ppg': ppg,
                    'opp_ppg': opp_ppg,
                    'form_5g': form_5g,
                    'games_played': games_played
                }
            
            team_stats[(game_id, team)] = stats
    
    return team_stats, games_df


def add_team_features(features_path, pbp_path, output_path):
    """Add team context features to momentum features"""
    print(f"\nAdding team features to {features_path.name}...")
    
    # Load data
    features_df = pd.read_csv(features_path)
    pbp_df = pd.read_csv(pbp_path)
    
    print(f"  Features: {len(features_df):,} samples")
    print(f"  Play-by-play: {len(pbp_df):,} plays")
    
    # Calculate team stats
    team_stats, games_df = calculate_team_stats(pbp_df)
    
    # Add team names to features
    print("\n  Merging team information...")
    game_teams = games_df.set_index('game_id')[['home_team', 'away_team']].to_dict('index')
    
    features_df['home_team'] = features_df['game_id'].map(lambda x: game_teams.get(x, {}).get('home_team', None))
    features_df['away_team'] = features_df['game_id'].map(lambda x: game_teams.get(x, {}).get('away_team', None))
    
    # Add team stats
    print("\n  Adding team statistics...")
    
    def get_team_stats(row, team_key):
        key = (row['game_id'], row[team_key])
        stats = team_stats.get(key, {
            'win_pct': 0.5,
            'ppg': 100,
            'opp_ppg': 100,
            'form_5g': 0.5,
            'games_played': 0
        })
        return pd.Series(stats)
    
    # Get home and away team stats
    home_stats = features_df.apply(lambda row: get_team_stats(row, 'home_team'), axis=1)
    away_stats = features_df.apply(lambda row: get_team_stats(row, 'away_team'), axis=1)
    
    features_df['home_win_pct'] = home_stats['win_pct']
    features_df['home_ppg'] = home_stats['ppg']
    features_df['home_opp_ppg'] = home_stats['opp_ppg']
    features_df['home_form_5g'] = home_stats['form_5g']
    features_df['home_games_played'] = home_stats['games_played']
    
    features_df['away_win_pct'] = away_stats['win_pct']
    features_df['away_ppg'] = away_stats['ppg']
    features_df['away_opp_ppg'] = away_stats['opp_ppg']
    features_df['away_form_5g'] = away_stats['form_5g']
    features_df['away_games_played'] = away_stats['games_played']
    
    # Determine run team and add contextual features (VECTORIZED)
    print("\n  Adding run team context features...")
    
    run_team_mask = features_df['run_team'] == 'home'
    
    # For home team runs
    features_df.loc[run_team_mask, 'run_team_win_pct'] = features_df.loc[run_team_mask, 'home_win_pct']
    features_df.loc[run_team_mask, 'run_team_ppg'] = features_df.loc[run_team_mask, 'home_ppg']
    features_df.loc[run_team_mask, 'run_team_opp_ppg'] = features_df.loc[run_team_mask, 'home_opp_ppg']
    features_df.loc[run_team_mask, 'run_team_form_5g'] = features_df.loc[run_team_mask, 'home_form_5g']
    features_df.loc[run_team_mask, 'opp_team_win_pct'] = features_df.loc[run_team_mask, 'away_win_pct']
    features_df.loc[run_team_mask, 'opp_team_ppg'] = features_df.loc[run_team_mask, 'away_ppg']
    features_df.loc[run_team_mask, 'opp_team_opp_ppg'] = features_df.loc[run_team_mask, 'away_opp_ppg']
    features_df.loc[run_team_mask, 'opp_team_form_5g'] = features_df.loc[run_team_mask, 'away_form_5g']
    
    # For away team runs
    features_df.loc[~run_team_mask, 'run_team_win_pct'] = features_df.loc[~run_team_mask, 'away_win_pct']
    features_df.loc[~run_team_mask, 'run_team_ppg'] = features_df.loc[~run_team_mask, 'away_ppg']
    features_df.loc[~run_team_mask, 'run_team_opp_ppg'] = features_df.loc[~run_team_mask, 'away_opp_ppg']
    features_df.loc[~run_team_mask, 'run_team_form_5g'] = features_df.loc[~run_team_mask, 'away_form_5g']
    features_df.loc[~run_team_mask, 'opp_team_win_pct'] = features_df.loc[~run_team_mask, 'home_win_pct']
    features_df.loc[~run_team_mask, 'opp_team_ppg'] = features_df.loc[~run_team_mask, 'home_ppg']
    features_df.loc[~run_team_mask, 'opp_team_opp_ppg'] = features_df.loc[~run_team_mask, 'home_opp_ppg']
    features_df.loc[~run_team_mask, 'opp_team_form_5g'] = features_df.loc[~run_team_mask, 'home_form_5g']
    
    # Derived features
    print("\n  Creating derived features...")
    features_df['team_quality_diff'] = features_df['run_team_win_pct'] - features_df['opp_team_win_pct']
    features_df['team_offensive_advantage'] = features_df['run_team_ppg'] - features_df['opp_team_opp_ppg']
    features_df['team_defensive_advantage'] = features_df['opp_team_ppg'] - features_df['run_team_opp_ppg']
    features_df['team_form_advantage'] = features_df['run_team_form_5g'] - features_df['opp_team_form_5g']
    
    # Save enhanced features
    print(f"\n  Saving enhanced features to {output_path}...")
    features_df.to_csv(output_path, index=False)
    
    print(f"  [OK] {len(features_df):,} samples with {len(features_df.columns)} features")
    
    return features_df


def main():
    print("\n" + "="*70)
    print("PREPARING MULTI-SEASON FEATURES")
    print("Adding team context to 2021-22 and 2022-23")
    print("="*70)
    
    seasons_to_process = ['2021_22', '2022_23']
    
    for season in seasons_to_process:
        print(f"\n{'='*70}")
        print(f"Processing {season.replace('_', '-')}")
        print(f"{'='*70}")
        
        features_path = Path(f'data/processed/features_v2_{season}.csv')
        pbp_path = Path(f'data/raw/pbp_{season}.csv')
        output_path = Path(f'data/processed/features_v2_{season}_enhanced.csv')
        
        if not features_path.exists():
            print(f"  [!] Features not found: {features_path}")
            continue
        
        if not pbp_path.exists():
            print(f"  [!] Play-by-play not found: {pbp_path}")
            continue
        
        if output_path.exists():
            print(f"  [~] Enhanced features already exist, skipping")
            continue
        
        add_team_features(features_path, pbp_path, output_path)
    
    print("\n" + "="*70)
    print("PREPARATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

