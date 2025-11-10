"""
Enhanced Feature Engineering V3 - Focus on Run Quality
Improvements:
1. Run quality scoring (purity, speed, intensity)
2. Better momentum detection
3. Enhanced NLP features
4. Team context integration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

class MomentumFeatureExtractorV3:
    def __init__(self):
        self.lookback_seconds = 180  # 3 minutes for momentum calculation
        self.lookforward_seconds = 300  # 5 minutes to check run extension
        
    def extract_features(self, pbp_data_path, include_outcomes=True):
        """
        Extract enhanced momentum features with quality scoring
        """
        print(f"\nExtracting features from {pbp_data_path}...")
        df = pd.read_csv(pbp_data_path)
        
        print(f"  Total plays: {len(df)}")
        print(f"  Unique games: {df['GAME_ID'].nunique()}")
        
        all_features = []
        games = df['GAME_ID'].unique()
        
        for i, game_id in enumerate(games):
            if (i + 1) % 100 == 0:
                print(f"  Processing game {i+1}/{len(games)}...")
            
            game_df = df[df['GAME_ID'] == game_id].copy()
            game_features = self._extract_game_features(game_df, game_id, include_outcomes)
            all_features.extend(game_features)
        
        features_df = pd.DataFrame(all_features)
        print(f"\n  Extracted {len(features_df)} feature samples")
        
        if include_outcomes and 'run_extends' in features_df.columns:
            pos_pct = features_df['run_extends'].mean() * 100
            print(f"  Positive samples: {pos_pct:.1f}%")
        
        return features_df
    
    def _extract_game_features(self, game_df, game_id, include_outcomes):
        """Extract features for a single game with quality scoring"""
        # Build full score history
        full_score_history = []
        play_history = []
        
        for idx, row in game_df.iterrows():
            # Parse score
            try:
                score_str = str(row['SCORE'])
                if pd.isna(score_str) or score_str == 'nan':
                    continue
                    
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
            
            # Calculate time remaining
            period = int(row['PERIOD']) if not pd.isna(row['PERIOD']) else 1
            time_str = str(row['PCTIMESTRING']) if not pd.isna(row['PCTIMESTRING']) else "12:00"
            
            try:
                if ':' in time_str:
                    mins, secs = map(int, time_str.split(':'))
                    time_in_period = mins * 60 + secs
                else:
                    time_in_period = 0
            except:
                time_in_period = 0
            
            # Calculate total time remaining (treating OT as 5-min periods)
            if period <= 4:
                periods_left = 4 - period
                time_remaining = periods_left * 720 + time_in_period
            else:
                time_remaining = time_in_period
            
            full_score_history.append({
                'time_remaining': time_remaining,
                'home': home_score,
                'away': away_score,
                'period': period,
                'event_num': row['EVENTNUM']
            })
            
            play_history.append({
                'time_remaining': time_remaining,
                'home_desc': str(row['HOMEDESCRIPTION']) if not pd.isna(row['HOMEDESCRIPTION']) else '',
                'away_desc': str(row['VISITORDESCRIPTION']) if not pd.isna(row['VISITORDESCRIPTION']) else '',
                'neutral_desc': str(row['NEUTRALDESCRIPTION']) if not pd.isna(row['NEUTRALDESCRIPTION']) else '',
                'event_type': row['EVENTMSGTYPE'] if 'EVENTMSGTYPE' in row else None
            })
        
        # Second pass: extract features
        features_list = []
        
        for current_idx in range(len(full_score_history)):
            score_entry = full_score_history[current_idx]
            time_remaining = score_entry['time_remaining']
            period = score_entry['period']
            
            # Get score history for lookback
            score_history = full_score_history[:current_idx+1]
            
            # Detect run with enhanced quality metrics
            run_info = self._detect_quality_run(score_history, play_history[:current_idx+1], time_remaining)
            
            if run_info is None:
                continue
            
            # Extract comprehensive features
            features = {
                'game_id': game_id,
                'event_num': score_entry['event_num'],
                'period': period,
                'time_remaining': time_remaining,
                'home_score': score_entry['home'],
                'away_score': score_entry['away'],
                'score_diff': score_entry['home'] - score_entry['away'] if run_info['run_team'] == 'home' else score_entry['away'] - score_entry['home'],
                
                # Run characteristics
                'run_team': run_info['run_team'],
                'run_score': run_info['run_score'],
                'opp_score': run_info['opp_score'],
                'run_net': run_info['run_score'] - run_info['opp_score'],
                'run_duration': run_info['duration'],
                
                # NEW: Run quality metrics
                'run_purity': run_info['purity'],  # 0-1, how "clean" the run is
                'run_speed': run_info['speed'],  # points per minute
                'run_intensity': run_info['intensity'],  # composite quality score
                
                # Momentum features
                'points_2min': run_info['points_2min'],
                'points_5min': run_info['points_5min'],
                'momentum_2min': run_info['momentum_2min'],
                'momentum_5min': run_info['momentum_5min'],
                
                # Defensive pressure
                'team_steals': run_info['team_steals'],
                'team_blocks': run_info['team_blocks'],
                'opp_turnovers': run_info['opp_turnovers'],
                'opp_fouls': run_info['opp_fouls'],
                'defensive_pressure': run_info['defensive_pressure'],
                
                # Offensive efficiency
                'team_threes': run_info['team_threes'],
                'team_assists': run_info['team_assists'],
                'offensive_efficiency': run_info['offensive_efficiency'],
                
                # Game context
                'is_early_game': 1 if period <= 2 else 0,
                'is_mid_game': 1 if period == 3 else 0,
                'is_late_game': 1 if period >= 4 else 0,
            }
            
            # Check if run extends (for training)
            if include_outcomes:
                run_extends = self._check_quality_run_extension(
                    full_score_history, current_idx, run_info
                )
                if run_extends is None:
                    continue
                features['run_extends'] = run_extends
            
            features_list.append(features)
        
        return features_list
    
    def _detect_quality_run(self, score_history, play_history, current_time):
        """
        Detect runs with quality metrics
        Returns dict with run info and quality scores
        """
        if len(score_history) < 3:
            return None
        
        current_idx = len(score_history) - 1
        current_score = score_history[current_idx]
        
        # Track scoring events
        home_points = 0
        away_points = 0
        scoring_times = []
        start_time = current_time
        
        # Look back through recent plays
        for i in range(current_idx, max(0, current_idx - 20), -1):
            curr = score_history[i]
            
            # Check time window (3 minutes)
            time_diff = start_time - curr['time_remaining']
            if time_diff > 180:
                break
            
            if i > 0:
                prev = score_history[i-1]
                h_pts = curr['home'] - prev['home']
                a_pts = curr['away'] - prev['away']
                
                if h_pts > 0 or a_pts > 0:
                    scoring_times.append(time_diff)
                
                home_points += h_pts
                away_points += a_pts
        
        # Determine if this is a meaningful run
        if home_points >= 6 and home_points >= away_points + 3:
            run_team = 'home'
            run_score = home_points
            opp_score = away_points
        elif away_points >= 6 and away_points >= home_points + 3:
            run_team = 'away'
            run_score = away_points
            opp_score = home_points
        else:
            return None
        
        # Calculate quality metrics
        duration = scoring_times[0] if scoring_times else 180
        
        # Purity: how clean is the run (6-0 = 1.0, 6-3 = 0.5)
        purity = (run_score - opp_score) / run_score if run_score > 0 else 0
        purity = max(0, min(1, purity))
        
        # Speed: points per minute
        speed = (run_score / (duration / 60)) if duration > 0 else 0
        
        # Intensity: composite score (weighted combination)
        intensity = (
            purity * 0.4 +  # Clean runs are better
            min(speed / 10, 1) * 0.3 +  # Fast runs are better (cap at 10 pts/min)
            min(run_score / 12, 1) * 0.3  # Larger runs are better (cap at 12 pts)
        )
        
        # Extract NLP features for this run
        nlp_features = self._extract_nlp_features(
            play_history, current_time, run_team, lookback=180
        )
        
        # Calculate momentum in different windows
        momentum_2min = self._calculate_momentum(score_history, current_time, 120)
        momentum_5min = self._calculate_momentum(score_history, current_time, 300)
        points_2min = self._points_in_window(score_history, current_time, 120, run_team)
        points_5min = self._points_in_window(score_history, current_time, 300, run_team)
        
        return {
            'run_team': run_team,
            'run_score': run_score,
            'opp_score': opp_score,
            'duration': duration,
            'purity': purity,
            'speed': speed,
            'intensity': intensity,
            'points_2min': points_2min,
            'points_5min': points_5min,
            'momentum_2min': momentum_2min,
            'momentum_5min': momentum_5min,
            **nlp_features
        }
    
    def _check_quality_run_extension(self, full_score_history, current_idx, run_info):
        """
        Check if a quality run extends further
        A run "extends" if:
        - Team continues to dominate (maintains 2:1 ratio or better)
        - Total run reaches 10+ points with 6+ net
        - OR reaches 12+ points with 4+ net
        """
        run_team = run_info['run_team']
        future_window = min(25, len(full_score_history) - current_idx - 1)
        
        if future_window < 5:
            return None
        
        current_score = full_score_history[current_idx]
        
        # Look forward to see if run extends
        max_team_total = run_info['run_score']
        max_opp_total = run_info['opp_score']
        
        for i in range(current_idx + 1, min(current_idx + future_window + 1, len(full_score_history))):
            prev = full_score_history[i-1]
            curr = full_score_history[i]
            
            if run_team == 'home':
                team_pts = curr['home'] - prev['home']
                opp_pts = curr['away'] - prev['away']
            else:
                team_pts = curr['away'] - prev['away']
                opp_pts = curr['home'] - prev['home']
            
            if team_pts > 0:
                max_team_total += team_pts
            if opp_pts > 0:
                max_opp_total += opp_pts
            
            # Check if run extended successfully
            net = max_team_total - max_opp_total
            
            if max_team_total >= 10 and net >= 6:
                return 1
            if max_team_total >= 12 and net >= 4:
                return 1
            
            # Check if run momentum broken
            if max_opp_total > max_team_total * 0.6:  # Opponent scored too much
                return 0
        
        # Didn't extend enough
        return 0
    
    def _extract_nlp_features(self, play_history, current_time, run_team, lookback=120):
        """Extract defensive and offensive features from play descriptions"""
        team_steals = 0
        team_blocks = 0
        team_threes = 0
        team_assists = 0
        opp_turnovers = 0
        opp_fouls = 0
        opp_misses = 0
        
        for play in reversed(play_history[-30:]):  # Last 30 plays
            time_diff = current_time - play['time_remaining']
            if time_diff > lookback:
                break
            
            home_desc = play['home_desc'].lower()
            away_desc = play['away_desc'].lower()
            
            if run_team == 'home':
                team_desc = home_desc
                opp_desc = away_desc
            else:
                team_desc = away_desc
                opp_desc = home_desc
            
            # Team positive actions
            if 'steal' in team_desc:
                team_steals += 1
            if 'block' in team_desc:
                team_blocks += 1
            if '3pt' in team_desc or 'three' in team_desc:
                team_threes += 1
            if 'assist' in team_desc:
                team_assists += 1
            
            # Opponent negative actions
            if 'turnover' in opp_desc or 'lost ball' in opp_desc:
                opp_turnovers += 1
            if 'foul' in opp_desc:
                opp_fouls += 1
            if 'miss' in opp_desc:
                opp_misses += 1
        
        # Calculate composite scores
        defensive_pressure = team_steals + team_blocks + opp_turnovers + (opp_fouls * 0.5)
        offensive_efficiency = team_threes + (team_assists * 0.5)
        
        return {
            'team_steals': team_steals,
            'team_blocks': team_blocks,
            'team_threes': team_threes,
            'team_assists': team_assists,
            'opp_turnovers': opp_turnovers,
            'opp_fouls': opp_fouls,
            'defensive_pressure': defensive_pressure,
            'offensive_efficiency': offensive_efficiency
        }
    
    def _calculate_momentum(self, score_history, current_time, window_seconds):
        """Calculate momentum as point differential in window"""
        if len(score_history) < 2:
            return 0
        
        home_pts = 0
        away_pts = 0
        
        for i in range(len(score_history) - 1, 0, -1):
            curr = score_history[i]
            time_diff = current_time - curr['time_remaining']
            
            if time_diff > window_seconds:
                break
            
            prev = score_history[i-1]
            home_pts += curr['home'] - prev['home']
            away_pts += curr['away'] - prev['away']
        
        return home_pts - away_pts
    
    def _points_in_window(self, score_history, current_time, window_seconds, team):
        """Calculate points scored by team in window"""
        if len(score_history) < 2:
            return 0
        
        points = 0
        
        for i in range(len(score_history) - 1, 0, -1):
            curr = score_history[i]
            time_diff = current_time - curr['time_remaining']
            
            if time_diff > window_seconds:
                break
            
            prev = score_history[i-1]
            if team == 'home':
                points += curr['home'] - prev['home']
            else:
                points += curr['away'] - prev['away']
        
        return points


def main():
    """Extract features for all available seasons"""
    extractor = MomentumFeatureExtractorV3()
    
    data_dir = Path('data/raw')
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process 2023-24 season (test set)
    print("\n" + "="*70)
    print("EXTRACTING FEATURES V3 - Enhanced Run Quality Detection")
    print("="*70)
    
    test_file = data_dir / 'pbp_2023_24.csv'
    if test_file.exists():
        print(f"\nProcessing test set: {test_file.name}")
        features = extractor.extract_features(test_file, include_outcomes=True)
        output_path = output_dir / 'features_v3_2023_24.csv'
        features.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Feature extraction complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

