"""
Model Training Module
Trains XGBoost model to predict momentum run extensions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os
import json

class MomentumModel:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_training_data(self, features_df):
        """
        Prepare features for model training
        """
        # Features to use (exclude identifiers and target)
        exclude_cols = ['game_id', 'event_num', 'run_extends', 'home_score', 'away_score']
        
        available_cols = [col for col in features_df.columns if col not in exclude_cols]
        self.feature_columns = available_cols
        
        X = features_df[self.feature_columns].values
        
        if 'run_extends' in features_df.columns:
            y = features_df['run_extends'].values
        else:
            y = None
        
        return X, y
    
    def train(self, train_df, val_df=None, params=None):
        """
        Train XGBoost model
        """
        print("\n" + "="*60)
        print("Training Momentum Prediction Model")
        print("="*60 + "\n")
        
        # Prepare training data
        X_train, y_train = self.prepare_training_data(train_df)
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Positive examples: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
        
        # Default hyperparameters
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 400,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum()  # Handle class imbalance
            }
        
        print(f"\nHyperparameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Train model
        print(f"\nTraining XGBoost model...")
        self.model = xgb.XGBClassifier(**params)
        
        if val_df is not None:
            X_val, y_val = self.prepare_training_data(val_df)
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=50
            )
        else:
            self.model.fit(X_train, y_train, verbose=50)
        
        self.is_trained = True
        
        # Evaluate on training set
        print(f"\nEvaluating on training set...")
        y_pred_train = self.model.predict(X_train)
        y_pred_proba_train = self.model.predict_proba(X_train)[:, 1]
        
        self._print_metrics(y_train, y_pred_train, y_pred_proba_train, "Training")
        
        # Evaluate on validation set if provided
        if val_df is not None:
            print(f"\nEvaluating on validation set...")
            y_pred_val = self.model.predict(X_val)
            y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]
            
            self._print_metrics(y_val, y_pred_val, y_pred_proba_val, "Validation")
        
        # Feature importance
        self._print_feature_importance(top_n=15)
        
        return self.model
    
    def _print_metrics(self, y_true, y_pred, y_pred_proba, dataset_name):
        """
        Print evaluation metrics
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.0
        
        print(f"\n{dataset_name} Metrics:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  ROC-AUC:   {auc:.3f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    
    def _print_feature_importance(self, top_n=15):
        """
        Print top N most important features
        """
        if self.model is None:
            return
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        for idx, row in feature_importance.head(top_n).iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")
    
    def predict(self, features_df):
        """
        Predict momentum extension probability
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X, _ = self.prepare_training_data(features_df)
        
        predictions = self.model.predict_proba(X)[:, 1]
        
        return predictions
    
    def save(self, filepath='models/momentum_model.pkl'):
        """
        Save trained model
        """
        if not self.is_trained:
            print("Warning: Saving untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load(self, filepath='models/momentum_model.pkl'):
        """
        Load trained model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def train_momentum_model():
    """
    Main training function
    """
    print("\n" + "#"*60)
    print("# NBA Momentum Model Training")
    print("#"*60 + "\n")
    
    # Load training data
    print("Loading training data...")
    training_seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22']
    
    train_data = []
    for season in training_seasons:
        file_path = f'data/processed/features_{season.replace("-", "_")}.csv'
        if os.path.exists(file_path):
            print(f"  Loading {season}...")
            df = pd.read_csv(file_path)
            train_data.append(df)
        else:
            print(f"  [WARNING] File not found: {file_path}")
    
    if not train_data:
        print("\n[ERROR] No training data found!")
        print("Please run feature_engineering.py first.")
        return None
    
    train_df = pd.concat(train_data, ignore_index=True)
    print(f"\nTraining data: {len(train_df):,} samples from {len(training_seasons)} seasons")
    
    # Load validation data
    print("\nLoading validation data...")
    val_file = 'data/processed/features_2022_23.csv'
    if os.path.exists(val_file):
        val_df = pd.read_csv(val_file)
        print(f"Validation data: {len(val_df):,} samples")
    else:
        print("[WARNING] Validation file not found, training without validation set")
        val_df = None
    
    # Initialize and train model
    model = MomentumModel()
    model.train(train_df, val_df)
    
    # Save model
    model.save()
    
    print("\n" + "#"*60)
    print("# Model training complete!")
    print("#"*60 + "\n")
    
    return model


if __name__ == '__main__':
    # Install xgboost if not already installed
    try:
        import xgboost as xgb
    except ImportError:
        print("Installing xgboost...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'xgboost', 'scikit-learn'])
        import xgboost as xgb
    
    model = train_momentum_model()

