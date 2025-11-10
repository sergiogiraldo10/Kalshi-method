"""
Feature Engineering Module
Extracts momentum and game-state features from play-by-play data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

class MomentumFeatureExtractor:
    def __init__(self):
        self.features_cache = {}
        
    def extract_features(self, pbp_df, include_outcomes=False):
        """
        Extract momentum features from play-by-play data
        
        Returns a dataframe where each row is a "game moment" with features
        """
        print(f"\nExtracting features from {len(pbp_df):,} plays...")
        
        all_features = []
        games_processed = 0
        
        for game_id in pbp_df['GAME_ID'].unique():
            game_df = pbp_df[pbp_df['GAME_ID'] == game_id].copy()
            game_df = game_df.sort_values('EVENTNUM')
            
            game_features = self._extract_game_features(game_df, game_id, include_outcomes)
            all_features.extend(game_features)
            
            games_processed += 1
            if games_processed % 100 == 0:
                print(f"  Processed {games_processed} games...")
        
        features_df = pd.DataFrame(all_features)
        print(f"\nExtracted {len(features_df):,} feature samples from {games_processed} games")
        
        return features_df
    
    def _extract_game_features(self, game_df, game_id, include_outcomes):
        """
        Extract features for a single game
        """
        # First pass: build complete score history for the game
        full_score_history = []
        
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
            
            full_score_history.append({
                'home': home_score,
                'away': away_score,
                'time_remaining': time_remaining,
                'event_num': row['EVENTNUM']
            })
        
        # Second pass: extract features using full history
        features_list = []
        event_history = []
        
        for current_idx in range(len(full_score_history)):
            score_entry = full_score_history[current_idx]
            
            # Only extract features if we have enough history
            if current_idx < 10:
                continue
            
            # Get score history up to this point
            score_history = full_score_history[:current_idx + 1]
            time_remaining = score_entry['time_remaining']
            home_score = score_entry['home']
            away_score = score_entry['away']
            
            # Extract momentum features
            momentum_features = self._calculate_momentum_features(
                score_history, event_history, time_remaining, 1  # period not used
            )
            
            if momentum_features is None:
                continue
            
            # Add game identifiers
            momentum_features['game_id'] = game_id
            momentum_features['event_num'] = score_entry['event_num']
            momentum_features['period'] = 1  # Will be corrected below
            momentum_features['time_remaining'] = time_remaining
            momentum_features['home_score'] = home_score
            momentum_features['away_score'] = away_score
            momentum_features['score_diff'] = home_score - away_score
            
            # If including outcomes, calculate if run extends using FULL history
            if include_outcomes:
                run_extends = self._check_run_extension(
                    full_score_history, current_idx
                )
                if run_extends is None:
                    continue  # Skip samples without enough future data
                momentum_features['run_extends'] = run_extends
            
            features_list.append(momentum_features)
        
        return features_list
    
    def _calculate_momentum_features(self, score_history, event_history, time_remaining, period):
        """
        Calculate momentum features at this point in the game
        """
        if len(score_history) < 10:
            return None
        
        features = {}
        
        # Get recent scoring runs
        recent_scores = score_history[-20:]  # Last 20 scoring plays
        
        # Calculate current run (using last 2 minutes of game time)
        current_time = time_remaining
        home_run, away_run = self._calculate_current_run(recent_scores, current_time)
        
        features['home_current_run'] = home_run
        features['away_current_run'] = away_run
        features['max_current_run'] = max(home_run, away_run)
        features['run_differential'] = home_run - away_run
        
        # Micro-run detection (4-0, 6-0, 8-0, etc.)
        features['is_micro_run'] = int(features['max_current_run'] >= 4)
        features['is_significant_run'] = int(features['max_current_run'] >= 6)
        
        # Which team has momentum
        features['home_has_momentum'] = int(home_run > away_run and home_run >= 4)
        features['away_has_momentum'] = int(away_run > home_run and away_run >= 4)
        
        # Scoring pace features
        features['points_last_2min'] = self._points_in_timeframe(recent_scores, current_time, 120)
        features['points_last_4min'] = self._points_in_timeframe(recent_scores, current_time, 240)
        
        # Scoring rate (points per minute)
        features['scoring_rate_last_2min'] = features['points_last_2min'] / 2.0 if features['points_last_2min'] > 0 else 0
        
        # Time since last opponent score
        features['time_since_opponent_score'] = self._time_since_last_opponent_score(
            recent_scores, features['home_has_momentum']
        )
        
        # Game state features
        features['period'] = period
        features['time_remaining_minutes'] = time_remaining / 60.0
        features['is_close_game'] = int(abs(score_history[-1]['home'] - score_history[-1]['away']) <= 5)
        features['is_clutch_time'] = int(time_remaining <= 300 and features['is_close_game'])  # Last 5 min + close
        
        # Volatility features (how much the score changes)
        features['score_volatility'] = self._calculate_score_volatility(recent_scores)
        
        return features
    
    def _calculate_current_run(self, score_history, current_time, lookback_seconds=120):
        """
        Calculate the current scoring run for each team
        Returns (home_run, away_run)
        """
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
    
    def _points_in_timeframe(self, score_history, current_time, timeframe_seconds):
        """
        Calculate total points scored in a timeframe
        """
        if len(score_history) < 2:
            return 0
        
        total_points = 0
        start_time = current_time - timeframe_seconds
        
        for i in range(len(score_history) - 1, 0, -1):
            current = score_history[i]
            
            if current['time_remaining'] < start_time:
                break
            
            if i > 0:
                previous = score_history[i - 1]
                points_scored = (current['home'] + current['away']) - (previous['home'] + previous['away'])
                total_points += points_scored
        
        return total_points
    
    def _time_since_last_opponent_score(self, score_history, home_has_momentum):
        """
        Calculate time since the opponent last scored
        """
        if len(score_history) < 2:
            return 0
        
        current_time = score_history[-1]['time_remaining']
        
        for i in range(len(score_history) - 1, 0, -1):
            current = score_history[i]
            previous = score_history[i - 1]
            
            if home_has_momentum:
                # Check when away team last scored
                if current['away'] > previous['away']:
                    return current_time - current['time_remaining']
            else:
                # Check when home team last scored
                if current['home'] > previous['home']:
                    return current_time - current['time_remaining']
        
        return 120  # Default max of 2 minutes
    
    def _calculate_score_volatility(self, score_history):
        """
        Calculate how volatile the scoring has been (std dev of score changes)
        """
        if len(score_history) < 3:
            return 0
        
        score_changes = []
        for i in range(1, len(score_history)):
            total_points = (score_history[i]['home'] + score_history[i]['away']) - \
                          (score_history[i-1]['home'] + score_history[i-1]['away'])
            score_changes.append(total_points)
        
        return np.std(score_changes) if score_changes else 0
    
    def _check_run_extension(self, score_history, current_idx, lookforward_plays=20):
        """
        Check if current micro-run extends to a super-run
        This is the TARGET VARIABLE for training
        
        Returns:
            1 if run extends (micro-run continues with strong momentum)
            0 if run stops or momentum shifts
        """
        if current_idx + lookforward_plays >= len(score_history):
            return None  # Not enough future data - skip this sample
        
        current_score = score_history[current_idx]
        
        # Check if there's a current micro-run
        current_time = current_score['time_remaining']
        home_run, away_run = self._calculate_current_run(
            score_history[:current_idx + 1], current_time
        )
        
        # Need at least a 4-0 run to check extension
        if home_run < 4 and away_run < 4:
            return None
        
        # Determine which team has momentum
        team_with_momentum = 'home' if home_run > away_run else 'away'
        current_run_size = max(home_run, away_run)
        
        # Look forward to see if run extends
        future_scores = score_history[current_idx:current_idx + lookforward_plays]
        
        momentum_team_points = 0
        opponent_points = 0
        
        for i in range(1, len(future_scores)):
            current = future_scores[i]
            previous = future_scores[i - 1]
            
            if team_with_momentum == 'home':
                momentum_team_points += (current['home'] - previous['home'])
                opponent_points += (current['away'] - previous['away'])
            else:
                momentum_team_points += (current['away'] - previous['away'])
                opponent_points += (current['home'] - previous['home'])
        
        # Check if run extended with strong momentum
        # Criteria:
        # 1. Momentum team scores at least 4 more points
        # 2. Momentum team outscores opponent by at least 2:1 ratio (or opponent scores <= 4)
        # 3. Total run becomes at least 8+ points
        
        total_run = current_run_size + momentum_team_points
        
        if momentum_team_points >= 4 and total_run >= 8:
            if opponent_points <= 4 or momentum_team_points >= opponent_points * 1.5:
                return 1  # Run extends
        
        return 0  # Run stops or weakens
    
    def save_features(self, features_df, season, output_dir='data/processed'):
        """
        Save extracted features to disk
        """
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'features_{season.replace("-", "_")}.csv')
        features_df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")


def extract_all_features(seasons, data_dir='data/raw', output_dir='data/processed', include_outcomes=True):
    """
    Extract features from all seasons
    """
    print("\n" + "="*60)
    print("Feature Engineering - Extracting Momentum Features")
    print("="*60 + "\n")
    
    extractor = MomentumFeatureExtractor()
    
    for season in seasons:
        print(f"\nProcessing season {season}...")
        
        file_path = os.path.join(data_dir, f'pbp_{season.replace("-", "_")}.csv')
        if not os.path.exists(file_path):
            print(f"  [WARNING] File not found: {file_path}")
            continue
        
        # Load play-by-play data
        pbp_df = pd.read_csv(file_path)
        
        # Extract features
        features_df = extractor.extract_features(pbp_df, include_outcomes=include_outcomes)
        
        # Save features
        extractor.save_features(features_df, season, output_dir)
        
        print(f"  Completed: {len(features_df):,} feature samples")
    
    print("\n" + "="*60)
    print("Feature extraction complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Extract features from all training and test seasons
    seasons = [
        '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22',  # Training
        '2022-23',  # Validation
        '2023-24',  # Test
    ]
    
    extract_all_features(seasons, include_outcomes=True)

