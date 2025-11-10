"""
Momentum Prediction Model V2
Trains on improved features with fixed run detection + NLP
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

class MomentumPredictionModelV2:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = []
        
    def prepare_data(self, features_df):
        """
        Prepare data for training
        Filter for micro-runs and prepare target variable
        """
        print(f"Preparing data from {len(features_df):,} samples...")
        
        # Filter for micro-runs (runs with 4+ point advantage)
        micro_run_df = features_df[features_df['is_micro_run'] == 1].copy()
        
        print(f"Found {len(micro_run_df):,} micro-run samples ({len(micro_run_df)/len(features_df)*100:.1f}%)")
        
        # Check if target variable exists
        if 'run_extends' not in micro_run_df.columns:
            print("[WARNING] No 'run_extends' target variable - cannot train")
            return None, None
        
        # Remove samples with missing target
        micro_run_df = micro_run_df[micro_run_df['run_extends'].notna()]
        
        # Feature columns (exclude identifiers and target)
        exclude_cols = ['game_id', 'event_num', 'run_extends', 'period', 'home_score', 'away_score']
        self.feature_cols = [col for col in micro_run_df.columns if col not in exclude_cols]
        
        X = micro_run_df[self.feature_cols].values
        y = micro_run_df['run_extends'].values
        
        # Check class balance
        positive_rate = y.mean()
        print(f"\nTarget variable stats:")
        print(f"  Positive class (run extends): {y.sum():,} ({positive_rate*100:.1f}%)")
        print(f"  Negative class (run stops): {(1-y).sum():,} ({(1-positive_rate)*100:.1f}%)")
        
        if positive_rate < 0.01 or positive_rate > 0.99:
            print("[WARNING] Severe class imbalance detected!")
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model to predict run extension
        """
        print(f"\nTraining momentum prediction model V2...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {len(self.feature_cols)}")
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Calculate scale_pos_weight for class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train XGBoost with class balancing
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            min_child_weight=3
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Training metrics
        train_preds = self.model.predict(X_train_scaled)
        train_probs = self.model.predict_proba(X_train_scaled)[:, 1]
        train_accuracy = (train_preds == y_train).mean()
        train_auc = roc_auc_score(y_train, train_probs)
        
        print(f"\nTraining metrics:")
        print(f"  Accuracy: {train_accuracy:.3f}")
        print(f"  AUC-ROC: {train_auc:.3f}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            print(f"\nValidation samples: {len(X_val):,}")
            X_val_scaled = self.scaler.transform(X_val)
            val_preds = self.model.predict(X_val_scaled)
            val_probs = self.model.predict_proba(X_val_scaled)[:, 1]
            val_accuracy = (val_preds == y_val).mean()
            val_auc = roc_auc_score(y_val, val_probs)
            
            print(f"\nValidation metrics:")
            print(f"  Accuracy: {val_accuracy:.3f}")
            print(f"  AUC-ROC: {val_auc:.3f}")
            
            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_preds, target_names=['Run Stops', 'Run Extends']))
            
            print("\nValidation Confusion Matrix:")
            cm = confusion_matrix(y_val, val_preds)
            print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
            print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        for idx, row in feature_importance.head(20).iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")
        
        self.is_trained = True
        return self.model
    
    def predict_probability(self, features_dict):
        """
        Predict probability that run will extend
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        feature_vector = [features_dict.get(col, 0) for col in self.feature_cols]
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0][1]
        return prob
    
    def save(self, filepath='models/momentum_model_v2.pkl'):
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
        print(f"\nModel saved to {filepath}")
    
    def load(self, filepath='models/momentum_model_v2.pkl'):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def train_momentum_model_v2(processed_dir='data/processed'):
    """
    Train momentum prediction model V2
    Training: 2015-2022
    Validation: 2022-23
    """
    print("\n" + "="*60)
    print("Training Momentum Prediction Model V2")
    print("With Fixed Run Detection + NLP Features")
    print("="*60 + "\n")
    
    # Load training seasons
    training_seasons = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']
    validation_season = '2022-23'
    
    print("Loading training data...")
    train_data = []
    for season in training_seasons:
        file_path = os.path.join(processed_dir, f'features_v2_{season.replace("-", "_")}.csv')
        if os.path.exists(file_path):
            print(f"  Loading {season}...")
            df = pd.read_csv(file_path)
            train_data.append(df)
        else:
            print(f"  [WARNING] File not found: {file_path}")
    
    if not train_data:
        print("No training data found!")
        return None
    
    # Combine training data
    train_df = pd.concat(train_data, ignore_index=True)
    print(f"\nTotal training samples: {len(train_df):,}")
    
    # Load validation data
    val_file = os.path.join(processed_dir, f'features_v2_{validation_season.replace("-", "_")}.csv')
    if os.path.exists(val_file):
        print(f"Loading validation data ({validation_season})...")
        val_df = pd.read_csv(val_file)
        print(f"Validation samples: {len(val_df):,}")
    else:
        print(f"[WARNING] Validation file not found: {val_file}")
        val_df = None
    
    # Initialize model
    model = MomentumPredictionModelV2()
    
    # Prepare training data
    X_train, y_train = model.prepare_data(train_df)
    
    if X_train is None:
        print("Failed to prepare training data")
        return None
    
    # Prepare validation data
    X_val, y_val = None, None
    if val_df is not None:
        X_val, y_val = model.prepare_data(val_df)
    
    # Train model
    model.train(X_train, y_train, X_val, y_val)
    
    # Save model
    model.save()
    
    print("\n" + "="*60)
    print("Momentum model V2 training complete!")
    print("="*60 + "\n")
    
    return model


if __name__ == '__main__':
    model = train_momentum_model_v2()

