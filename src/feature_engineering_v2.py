"""
Feature Engineering Module V2 - IMPROVED
- Fixed run detection (actual momentum runs like 8-0, 10-2)
- Added NLP features from play descriptions
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

class MomentumFeatureExtractorV2:
    def __init__(self):
        self.features_cache = {}
        
    def extract_features(self, pbp_df, include_outcomes=False):
        """
        Extract momentum features from play-by-play data
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
        # First pass: build complete score history AND play history
        full_score_history = []
        full_play_history = []
        
        for idx, row in game_df.iterrows():
            # Track ALL plays for NLP features
            full_play_history.append({
                'event_num': row['EVENTNUM'],
                'event_type': row.get('EVENTMSGTYPE', 0),
                'home_desc': str(row.get('HOMEDESCRIPTION', '')),
                'away_desc': str(row.get('VISITORDESCRIPTION', '')),
                'neutral_desc': str(row.get('NEUTRALDESCRIPTION', ''))
            })
            
            # Track scores
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
                'event_num': row['EVENTNUM'],
                'period': period
            })
        
        # Second pass: extract features at each moment
        features_list = []
        
        for current_idx in range(len(full_score_history)):
            score_entry = full_score_history[current_idx]
            
            # Only extract features if we have enough history
            if current_idx < 10:
                continue
            
            # Get score history up to this point
            score_history = full_score_history[:current_idx + 1]
            
            # Get play history up to this point (for NLP features)
            play_idx = next((i for i, p in enumerate(full_play_history) 
                           if p['event_num'] == score_entry['event_num']), None)
            if play_idx is None:
                continue
            play_history = full_play_history[:play_idx + 1]
            
            time_remaining = score_entry['time_remaining']
            home_score = score_entry['home']
            away_score = score_entry['away']
            period = score_entry['period']
            
            # Extract momentum features with FIXED run detection
            momentum_features = self._calculate_momentum_features(
                score_history, play_history, time_remaining, period
            )
            
            if momentum_features is None:
                continue
            
            # Add game identifiers
            momentum_features['game_id'] = game_id
            momentum_features['event_num'] = score_entry['event_num']
            momentum_features['period'] = period
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
    
    def _calculate_momentum_features(self, score_history, play_history, time_remaining, period):
        """
        Calculate momentum features - FIXED VERSION
        """
        if len(score_history) < 10:
            return None
        
        features = {}
        
        # FIXED: Detect TRUE momentum runs
        run_team, run_score, opp_score = self._detect_true_run(
            score_history, time_remaining, max_lookback_seconds=180
        )
        
        # Run features
        features['run_team'] = 1 if run_team == 'home' else (-1 if run_team == 'away' else 0)
        features['run_score'] = run_score
        features['opp_score'] = opp_score
        features['run_differential'] = run_score - opp_score
        features['max_run_score'] = max(run_score, opp_score) if run_team else 0
        
        # Micro-run detection (4+ point advantage)
        features['is_micro_run'] = int(run_score >= 4 and run_score >= opp_score * 1.5)
        features['is_significant_run'] = int(run_score >= 6 and run_score >= opp_score * 2)
        
        # NLP Features from play descriptions
        nlp_features = self._extract_nlp_features(
            play_history, time_remaining, run_team
        )
        features.update(nlp_features)
        
        # Scoring pace features
        recent_scores = score_history[-min(20, len(score_history)):]
        features['points_last_2min'] = self._points_in_timeframe(recent_scores, time_remaining, 120)
        features['points_last_4min'] = self._points_in_timeframe(recent_scores, time_remaining, 240)
        features['scoring_rate_last_2min'] = features['points_last_2min'] / 2.0 if features['points_last_2min'] > 0 else 0
        
        # Game state features
        features['period'] = period
        features['time_remaining_minutes'] = time_remaining / 60.0
        features['is_close_game'] = int(abs(score_history[-1]['home'] - score_history[-1]['away']) <= 5)
        features['is_clutch_time'] = int(time_remaining <= 300 and features['is_close_game'])
        
        # Volatility
        features['score_volatility'] = self._calculate_score_volatility(recent_scores)
        
        return features
    
    def _detect_true_run(self, score_history, current_time, max_lookback_seconds=180):
        """
        FIXED: Detect consecutive scoring runs (e.g., last 10 points scored 8-2)
        
        Returns: (team_with_run, run_score, opponent_score)
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
    
    def _extract_nlp_features(self, play_history, current_time, run_team):
        """
        NEW: Extract features from play descriptions
        """
        features = {}
        
        # Get plays in last 2 minutes
        window_start = current_time - 120
        recent_plays = [p for p in play_history[-40:]]  # Last 40 plays max
        
        # Initialize counters
        home_misses = 0
        away_misses = 0
        home_steals = 0
        away_steals = 0
        home_blocks = 0
        away_blocks = 0
        home_turnovers = 0
        away_turnovers = 0
        home_threes = 0
        away_threes = 0
        home_fouls = 0
        away_fouls = 0
        
        for play in recent_plays:
            home_desc = play['home_desc']
            away_desc = play['away_desc']
            
            # Count misses (defensive stops)
            if 'MISS' in home_desc:
                home_misses += 1
            if 'MISS' in away_desc:
                away_misses += 1
            
            # Count steals
            if 'STEAL' in home_desc or 'STL' in home_desc:
                home_steals += 1
            if 'STEAL' in away_desc or 'STL' in away_desc:
                away_steals += 1
            
            # Count blocks
            if 'BLOCK' in home_desc or 'BLK' in home_desc:
                home_blocks += 1
            if 'BLOCK' in away_desc or 'BLK' in away_desc:
                away_blocks += 1
            
            # Count turnovers
            if 'Turnover' in home_desc:
                home_turnovers += 1
            if 'Turnover' in away_desc:
                away_turnovers += 1
            
            # Count 3-pointers
            if '3PT' in home_desc and 'MISS' not in home_desc:
                home_threes += 1
            if '3PT' in away_desc and 'MISS' not in away_desc:
                away_threes += 1
            
            # Count fouls
            if play['event_type'] == 6:  # Foul event type
                if home_desc and home_desc != 'nan':
                    home_fouls += 1
                if away_desc and away_desc != 'nan':
                    away_fouls += 1
        
        # Store features based on who has the run
        if run_team == 'home':
            features['team_steals_2min'] = home_steals
            features['team_blocks_2min'] = home_blocks
            features['team_threes_2min'] = home_threes
            features['opponent_misses_2min'] = away_misses
            features['opponent_turnovers_2min'] = away_turnovers
            features['opponent_fouls_2min'] = away_fouls
        elif run_team == 'away':
            features['team_steals_2min'] = away_steals
            features['team_blocks_2min'] = away_blocks
            features['team_threes_2min'] = away_threes
            features['opponent_misses_2min'] = home_misses
            features['opponent_turnovers_2min'] = home_turnovers
            features['opponent_fouls_2min'] = home_fouls
        else:
            # No clear run
            features['team_steals_2min'] = 0
            features['team_blocks_2min'] = 0
            features['team_threes_2min'] = 0
            features['opponent_misses_2min'] = 0
            features['opponent_turnovers_2min'] = 0
            features['opponent_fouls_2min'] = 0
        
        # Defensive pressure indicator
        features['defensive_pressure'] = features['team_steals_2min'] + features['team_blocks_2min']
        
        # Offensive efficiency indicator
        features['offensive_efficiency'] = features['team_threes_2min'] * 3  # 3-pointers are high value
        
        return features
    
    def _points_in_timeframe(self, score_history, current_time, timeframe_seconds):
        """Calculate total points scored in a timeframe"""
        if len(score_history) < 2:
            return 0
        
        start_time = current_time - timeframe_seconds
        relevant = [s for s in score_history if s['time_remaining'] >= start_time]
        
        if len(relevant) < 2:
            return 0
        
        total_points = (relevant[-1]['home'] + relevant[-1]['away']) - (relevant[0]['home'] + relevant[0]['away'])
        return total_points
    
    def _calculate_score_volatility(self, score_history):
        """Calculate scoring volatility"""
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
        Check if current micro-run extends - IMPROVED VERSION
        """
        if current_idx + lookforward_plays >= len(score_history):
            return None
        
        current_score = score_history[current_idx]
        current_time = current_score['time_remaining']
        
        # Check if there's a current micro-run
        run_team, run_score, opp_score = self._detect_true_run(
            score_history[:current_idx + 1], current_time
        )
        
        # Need at least a 4-point run to check extension
        if not run_team or run_score < 4:
            return None
        
        # Look forward to see if run extends
        future_scores = score_history[current_idx:current_idx + lookforward_plays]
        
        if len(future_scores) < 2:
            return None
        
        # Calculate how much each team scores going forward
        start = future_scores[0]
        end = future_scores[-1]
        
        if run_team == 'home':
            team_future_points = end['home'] - start['home']
            opp_future_points = end['away'] - start['away']
        else:
            team_future_points = end['away'] - start['away']
            opp_future_points = end['home'] - start['home']
        
        # Check if run extends with strong momentum
        total_run = run_score + team_future_points
        total_opp = opp_score + opp_future_points
        
        # Criteria for extension:
        # 1. Team scores 4+ more points
        # 2. Total run becomes 8+ points
        # 3. Team maintains momentum (outscores by 1.5:1 or opponent scores <=4)
        
        if team_future_points >= 4 and total_run >= 8:
            if total_opp <= 4 or total_run >= total_opp * 1.5:
                return 1  # Run extends
        
        return 0  # Run stops
    
    def save_features(self, features_df, season, output_dir='data/processed'):
        """Save extracted features"""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'features_v2_{season.replace("-", "_")}.csv')
        features_df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")


def extract_all_features_v2(seasons, data_dir='data/raw', output_dir='data/processed', include_outcomes=True):
    """
    Extract features V2 from all seasons
    """
    print("\n" + "="*60)
    print("Feature Engineering V2 - With Fixed Run Detection + NLP")
    print("="*60 + "\n")
    
    extractor = MomentumFeatureExtractorV2()
    
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
    print("Feature extraction V2 complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Extract features from all seasons
    seasons = [
        '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22',  # Training
        '2022-23',  # Validation
        '2023-24',  # Test
    ]
    
    extract_all_features_v2(seasons, include_outcomes=True)

