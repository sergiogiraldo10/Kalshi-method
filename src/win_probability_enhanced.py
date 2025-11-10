"""
Enhanced Win Probability Estimation Module
Estimates win probability from game state using historical data with advanced features
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class EnhancedWinProbabilityModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = []
        
    def prepare_training_data(self, pbp_df):
        """
        Prepare training data with enhanced features
        
        Features:
        - Basic: score_diff, time_remaining, period
        - Momentum: points last 2min/5min, current run
        - Game context: timeouts, fouls, clutch time
        - Scoring pace: points per minute, lead changes
        """
        print("Preparing enhanced training data for win probability model...")
        
        game_outcomes = self._get_game_outcomes(pbp_df)
        training_data = []
        games_processed = 0
        
        for game_id in pbp_df['GAME_ID'].unique():
            game_df = pbp_df[pbp_df['GAME_ID'] == game_id].copy()
            game_df = game_df.sort_values('EVENTNUM')
            
            home_won = game_outcomes.get(game_id, None)
            if home_won is None:
                continue
            
            # Track game state over time
            score_history = []
            event_history = []
            home_timeouts = 7
            away_timeouts = 7
            home_fouls = 0
            away_fouls = 0
            
            for idx, row in game_df.iterrows():
                # Extract score
                if pd.isna(row['SCORE']):
                    continue
                
                try:
                    score_parts = str(row['SCORE']).split(' - ')
                    if len(score_parts) != 2:
                        continue
                    away_score = int(score_parts[0])
                    home_score = int(score_parts[1])
                except:
                    continue
                
                # Parse time
                period = row['PERIOD']
                try:
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
                
                # Track score history
                score_history.append({
                    'home': home_score,
                    'away': away_score,
                    'time_remaining': time_remaining
                })
                
                # Track events
                event_type = row['EVENTMSGTYPE']
                event_history.append({
                    'event_type': event_type,
                    'time_remaining': time_remaining,
                    'home_desc': str(row.get('HOMEDESCRIPTION', '')),
                    'away_desc': str(row.get('VISITORDESCRIPTION', ''))
                })
                
                # Update timeouts and fouls
                home_desc = str(row.get('HOMEDESCRIPTION', '')).lower()
                away_desc = str(row.get('VISITORDESCRIPTION', '')).lower()
                
                if 'timeout' in home_desc:
                    home_timeouts = max(0, home_timeouts - 1)
                if 'timeout' in away_desc:
                    away_timeouts = max(0, away_timeouts - 1)
                
                if event_type == 6:  # Foul
                    if pd.notna(row.get('HOMEDESCRIPTION')):
                        home_fouls += 1
                    if pd.notna(row.get('VISITORDESCRIPTION')):
                        away_fouls += 1
                
                # Only create features if we have enough history
                if len(score_history) < 10:
                    continue
                
                # Calculate enhanced features
                score_diff = home_score - away_score
                
                # 1. Momentum features (last 2 and 5 minutes)
                points_home_2min = self._points_in_window(score_history, time_remaining, 120, 'home')
                points_away_2min = self._points_in_window(score_history, time_remaining, 120, 'away')
                points_home_5min = self._points_in_window(score_history, time_remaining, 300, 'home')
                points_away_5min = self._points_in_window(score_history, time_remaining, 300, 'away')
                
                momentum_2min = points_home_2min - points_away_2min
                momentum_5min = points_home_5min - points_away_5min
                
                # 2. Current run
                current_run = self._calculate_current_run(score_history)
                
                # 3. Scoring pace
                recent_plays = [s for s in score_history if s['time_remaining'] >= time_remaining - 300]
                if len(recent_plays) > 1:
                    time_elapsed = recent_plays[0]['time_remaining'] - recent_plays[-1]['time_remaining']
                    if time_elapsed > 0:
                        points_per_minute = (recent_plays[-1]['home'] + recent_plays[-1]['away'] - 
                                           recent_plays[0]['home'] - recent_plays[0]['away']) / (time_elapsed / 60)
                    else:
                        points_per_minute = 0
                else:
                    points_per_minute = 0
                
                # 4. Clutch time indicator
                is_clutch = (time_remaining <= 300) and (abs(score_diff) <= 5)
                
                # 5. Lead changes
                lead_changes = self._count_lead_changes(score_history, time_remaining, 300)
                
                # Create feature vector
                features = {
                    # Basic features
                    'score_diff': score_diff,
                    'time_remaining': time_remaining,
                    'period': period,
                    
                    # Momentum features
                    'points_home_2min': points_home_2min,
                    'points_away_2min': points_away_2min,
                    'momentum_2min': momentum_2min,
                    'points_home_5min': points_home_5min,
                    'points_away_5min': points_away_5min,
                    'momentum_5min': momentum_5min,
                    'current_run': current_run,
                    
                    # Game context
                    'home_timeouts': home_timeouts,
                    'away_timeouts': away_timeouts,
                    'home_fouls': home_fouls,
                    'away_fouls': away_fouls,
                    'is_clutch': int(is_clutch),
                    
                    # Scoring pace
                    'points_per_minute': points_per_minute,
                    'lead_changes_5min': lead_changes,
                    
                    # Target
                    'home_won': int(home_won)
                }
                
                training_data.append(features)
            
            games_processed += 1
            if games_processed % 100 == 0:
                print(f"  Processed {games_processed} games...")
        
        df = pd.DataFrame(training_data)
        print(f"Created {len(df):,} training samples with enhanced features from {games_processed} games")
        return df
    
    def _get_game_outcomes(self, pbp_df):
        """Determine the outcome of each game"""
        outcomes = {}
        
        for game_id in pbp_df['GAME_ID'].unique():
            game_df = pbp_df[pbp_df['GAME_ID'] == game_id]
            final_plays = game_df[game_df['SCORE'].notna()].tail(10)
            
            for idx, row in final_plays.iterrows():
                try:
                    score_parts = str(row['SCORE']).split(' - ')
                    if len(score_parts) == 2:
                        away_score = int(score_parts[0])
                        home_score = int(score_parts[1])
                        outcomes[game_id] = home_score > away_score
                except:
                    continue
        
        return outcomes
    
    def _points_in_window(self, score_history, current_time, window_seconds, team):
        """Calculate points scored in time window"""
        if len(score_history) < 2:
            return 0
        
        window_start = current_time - window_seconds
        relevant = [s for s in score_history if s['time_remaining'] >= window_start]
        
        if len(relevant) < 2:
            return 0
        
        return relevant[-1][team] - relevant[0][team]
    
    def _calculate_current_run(self, score_history):
        """Calculate current run (consecutive points by one team)"""
        if len(score_history) < 3:
            return 0
        
        # Look at last few plays
        recent = score_history[-min(10, len(score_history)):]
        
        run = 0
        for i in range(len(recent) - 1, 0, -1):
            home_diff = recent[i]['home'] - recent[i-1]['home']
            away_diff = recent[i]['away'] - recent[i-1]['away']
            
            if home_diff > 0 and away_diff == 0:
                run += home_diff
            elif away_diff > 0 and home_diff == 0:
                run -= away_diff
            else:
                break
        
        return run
    
    def _count_lead_changes(self, score_history, current_time, window_seconds):
        """Count how many times the lead changed in time window"""
        window_start = current_time - window_seconds
        relevant = [s for s in score_history if s['time_remaining'] >= window_start]
        
        if len(relevant) < 2:
            return 0
        
        changes = 0
        for i in range(len(relevant) - 1):
            diff1 = relevant[i]['home'] - relevant[i]['away']
            diff2 = relevant[i+1]['home'] - relevant[i+1]['away']
            
            if (diff1 > 0 and diff2 < 0) or (diff1 < 0 and diff2 > 0):
                changes += 1
        
        return changes
    
    def train(self, training_df):
        """Train with enhanced features using XGBoost"""
        print("\nTraining enhanced win probability model with XGBoost...")
        
        # Select feature columns (exclude target)
        self.feature_cols = [col for col in training_df.columns if col != 'home_won']
        X = training_df[self.feature_cols].values
        y = training_df['home_won'].values
        
        print(f"Training with {len(self.feature_cols)} features")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X_scaled, y)
        
        train_accuracy = self.model.score(X_scaled, y)
        print(f"Training accuracy: {train_accuracy:.3f}")
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<25} {row['importance']:.4f}")
        
        self.is_trained = True
        return self.model
    
    def predict_win_probability(self, **features):
        """
        Predict home team win probability
        
        Args:
            **features: Keyword arguments for all required features
        
        Returns:
            Probability that home team wins (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create feature vector in correct order
        feature_vector = [features.get(col, 0) for col in self.feature_cols]
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0][1]
        return prob
    
    def save(self, filepath='models/win_probability_enhanced.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            print("Warning: Saving untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='models/win_probability_enhanced.pkl'):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def train_enhanced_win_probability_model(seasons_data_dir='data/raw'):
    """
    Train enhanced win probability model on historical data
    """
    print("\n" + "="*60)
    print("Training Enhanced Win Probability Model")
    print("="*60 + "\n")
    
    # Load training seasons (not test season)
    training_seasons = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    
    all_data = []
    for season in training_seasons:
        file_path = os.path.join(seasons_data_dir, f'pbp_{season.replace("-", "_")}.csv')
        if os.path.exists(file_path):
            print(f"Loading {season}...")
            df = pd.read_csv(file_path)
            all_data.append(df)
        else:
            print(f"[WARNING] File not found: {file_path}")
    
    if not all_data:
        print("No training data found!")
        return None
    
    # Combine all seasons
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal training data: {len(combined_df):,} plays from {combined_df['GAME_ID'].nunique()} games")
    
    # Initialize and train model
    wp_model = EnhancedWinProbabilityModel()
    training_df = wp_model.prepare_training_data(combined_df)
    wp_model.train(training_df)
    
    # Save model
    wp_model.save()
    
    # Test some predictions
    print("\nTesting predictions:")
    test_cases = [
        {
            'desc': "Tied, start of game",
            'score_diff': 0, 'time_remaining': 2880, 'period': 1,
            'momentum_2min': 0, 'momentum_5min': 0, 'current_run': 0,
            'points_home_2min': 0, 'points_away_2min': 0,
            'points_home_5min': 0, 'points_away_5min': 0,
            'home_timeouts': 7, 'away_timeouts': 7,
            'home_fouls': 0, 'away_fouls': 0,
            'is_clutch': 0, 'points_per_minute': 0, 'lead_changes_5min': 0
        },
        {
            'desc': "+10, half game remaining, on 6-0 run",
            'score_diff': 10, 'time_remaining': 1440, 'period': 2,
            'momentum_2min': 6, 'momentum_5min': 12, 'current_run': 6,
            'points_home_2min': 10, 'points_away_2min': 4,
            'points_home_5min': 18, 'points_away_5min': 6,
            'home_timeouts': 5, 'away_timeouts': 4,
            'home_fouls': 3, 'away_fouls': 5,
            'is_clutch': 0, 'points_per_minute': 22, 'lead_changes_5min': 1
        },
        {
            'desc': "-5, 5 min left, CLUTCH",
            'score_diff': -5, 'time_remaining': 300, 'period': 4,
            'momentum_2min': -4, 'momentum_5min': -6, 'current_run': -4,
            'points_home_2min': 4, 'points_away_2min': 8,
            'points_home_5min': 12, 'points_away_5min': 18,
            'home_timeouts': 2, 'away_timeouts': 1,
            'home_fouls': 4, 'away_fouls': 3,
            'is_clutch': 1, 'points_per_minute': 30, 'lead_changes_5min': 3
        },
        {
            'desc': "+3, 1 min left, CLUTCH",
            'score_diff': 3, 'time_remaining': 60, 'period': 4,
            'momentum_2min': 2, 'momentum_5min': 5, 'current_run': 2,
            'points_home_2min': 8, 'points_away_2min': 6,
            'points_home_5min': 15, 'points_away_5min': 10,
            'home_timeouts': 1, 'away_timeouts': 0,
            'home_fouls': 5, 'away_fouls': 4,
            'is_clutch': 1, 'points_per_minute': 25, 'lead_changes_5min': 2
        },
    ]
    
    for test_case in test_cases:
        desc = test_case.pop('desc')
        prob = wp_model.predict_win_probability(**test_case)
        print(f"  {desc:<40} -> Home win prob: {prob:.3f}")
    
    print("\nEnhanced win probability model training complete!")
    
    return wp_model


if __name__ == '__main__':
    model = train_enhanced_win_probability_model()

