"""
Win Probability Estimation Module
Estimates win probability from game state using historical data
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

class WinProbabilityModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_training_data(self, pbp_df):
        """
        Prepare training data from play-by-play dataframe
        
        For each play, we'll create features:
        - Score differential (home - away)
        - Time remaining (in seconds)
        - Quarter/Period
        - Home vs Away indicator
        
        Target: Did home team win? (1 = yes, 0 = no)
        """
        print("Preparing training data for win probability model...")
        
        # We need to determine game outcomes first
        game_outcomes = self._get_game_outcomes(pbp_df)
        
        training_data = []
        
        for game_id in pbp_df['GAME_ID'].unique():
            game_df = pbp_df[pbp_df['GAME_ID'] == game_id].copy()
            game_df = game_df.sort_values('EVENTNUM')
            
            # Get final outcome for this game
            home_won = game_outcomes.get(game_id, None)
            if home_won is None:
                continue
            
            # Process each play in the game
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
                
                # Calculate time remaining
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
                
                # Calculate total time remaining in game (assuming 12 min quarters)
                if period <= 4:
                    time_remaining = (4 - period) * 720 + time_in_period
                else:
                    # Overtime (5 min periods)
                    time_remaining = (period - period) * 300 + time_in_period
                
                # Create feature vector
                features = {
                    'score_diff': home_score - away_score,
                    'time_remaining': time_remaining,
                    'period': period,
                    'home_won': int(home_won)
                }
                
                training_data.append(features)
        
        df = pd.DataFrame(training_data)
        print(f"Created {len(df):,} training samples from {pbp_df['GAME_ID'].nunique()} games")
        
        return df
    
    def _get_game_outcomes(self, pbp_df):
        """
        Determine the outcome of each game (did home team win?)
        """
        outcomes = {}
        
        for game_id in pbp_df['GAME_ID'].unique():
            game_df = pbp_df[pbp_df['GAME_ID'] == game_id]
            
            # Get final score (last play with a score)
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
    
    def train(self, training_df):
        """
        Train the win probability model
        """
        print("\nTraining win probability model...")
        
        # Prepare features and target
        X = training_df[['score_diff', 'time_remaining', 'period']].values
        y = training_df['home_won'].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train logistic regression
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Evaluate training accuracy
        train_accuracy = self.model.score(X_scaled, y)
        print(f"Training accuracy: {train_accuracy:.3f}")
        
        self.is_trained = True
        
        return self.model
    
    def predict_win_probability(self, score_diff, time_remaining, period):
        """
        Predict home team win probability
        
        Args:
            score_diff: Home score - Away score
            time_remaining: Seconds remaining in game
            period: Current period/quarter
        
        Returns:
            Probability that home team wins (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare feature vector
        X = np.array([[score_diff, time_remaining, period]])
        X_scaled = self.scaler.transform(X)
        
        # Predict probability
        prob = self.model.predict_proba(X_scaled)[0][1]  # Probability of class 1 (home wins)
        
        return prob
    
    def predict_batch(self, score_diffs, time_remainings, periods):
        """
        Predict win probabilities for multiple game states
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = np.column_stack([score_diffs, time_remainings, periods])
        X_scaled = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        return probs
    
    def save(self, filepath='models/win_probability_model.pkl'):
        """
        Save the trained model
        """
        if not self.is_trained:
            print("Warning: Saving untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='models/win_probability_model.pkl'):
        """
        Load a trained model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def train_win_probability_model(seasons_data_dir='data/raw'):
    """
    Train win probability model on historical data
    """
    print("\n" + "="*60)
    print("Training Win Probability Model")
    print("="*60 + "\n")
    
    # Load all training seasons
    training_seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    
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
    wp_model = WinProbabilityModel()
    training_df = wp_model.prepare_training_data(combined_df)
    wp_model.train(training_df)
    
    # Save model
    wp_model.save()
    
    # Test some predictions
    print("\nTesting predictions:")
    test_cases = [
        (0, 2880, 1, "Tied, start of game"),
        (10, 1440, 2, "+10 with half game remaining"),
        (-5, 300, 4, "-5 with 5 min left"),
        (3, 60, 4, "+3 with 1 min left"),
        (-10, 1800, 3, "-10 at half time"),
    ]
    
    for score_diff, time_rem, period, description in test_cases:
        prob = wp_model.predict_win_probability(score_diff, time_rem, period)
        print(f"  {description:<35} -> Home win prob: {prob:.3f}")
    
    print("\nWin probability model training complete!")
    
    return wp_model


if __name__ == '__main__':
    model = train_win_probability_model()

