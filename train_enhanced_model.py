"""
Train Enhanced Momentum Model with Team Features
Following Action Plan Option 1 - Improve the Model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

print("\n" + "="*70)
print("TRAINING ENHANCED MOMENTUM MODEL")
print("With Team Context Features")
print("="*70)

# Load enhanced features
print("\nLoading enhanced features...")
features_df = pd.read_csv('data/processed/features_v2_2023_24_enhanced.csv')
print(f"  [OK] Loaded {len(features_df):,} samples with {len(features_df.columns)} features")

# Split: First 33% for training, rest for testing
features_df = features_df.sort_values('game_id')
unique_games = features_df['game_id'].unique()
n_games = len(unique_games)

train_cutoff = int(n_games * 0.33)
train_games = set(unique_games[:train_cutoff])
test_games = set(unique_games[train_cutoff:])

train_df = features_df[features_df['game_id'].isin(train_games)].copy()
test_df = features_df[features_df['game_id'].isin(test_games)].copy()

print(f"\n  TRAIN: {len(train_games)} games ({len(train_df):,} samples)")
print(f"  TEST:  {len(test_games)} games ({len(test_df):,} samples)")

# Select features (exclude non-feature columns)
exclude_cols = ['game_id', 'event_num', 'run_extends', 'run_team', 'time_remaining']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

print(f"\n  Using {len(feature_cols)} features (including {len([c for c in feature_cols if 'team' in c])} team features)")

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df['run_extends'].values

print(f"\n  Training data:")
print(f"    Samples: {len(X_train):,}")
print(f"    Positive class: {y_train.mean()*100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train XGBoost
print("\n  Training XGBoost model...")
model = XGBClassifier(
    n_estimators=300,  # More trees for better performance
    max_depth=7,       # Slightly deeper
    learning_rate=0.05, # Lower learning rate
    scale_pos_weight=(1-y_train.mean())/y_train.mean(),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train_scaled, y_train)
print("  [OK] Model trained")

# Calibrate probabilities (Platt scaling)
print("\n  Calibrating probabilities...")
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated_model.fit(X_train_scaled, y_train)
print("  [OK] Probabilities calibrated")

# Evaluate on training set
train_preds = calibrated_model.predict_proba(X_train_scaled)[:, 1]
train_acc = (train_preds > 0.5).astype(int) == y_train
print(f"\n  Training accuracy: {train_acc.mean()*100:.1f}%")

# Evaluate on test set
X_test = test_df[feature_cols].values
X_test_scaled = scaler.transform(X_test)
test_preds = calibrated_model.predict_proba(X_test_scaled)[:, 1]
test_acc = (test_preds > 0.5).astype(int) == test_df['run_extends'].values
print(f"  Test accuracy: {test_acc.mean()*100:.1f}%")

# Check calibration
print("\n  Calibration check:")
pred_bins = pd.cut(test_preds, bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0], labels=['<40%', '40-50%', '50-60%', '60-70%', '70%+'])
actual_rates = test_df.groupby(pred_bins)['run_extends'].mean()
print(f"    Predicted <40%:  Actual {actual_rates.get('<40%', 0)*100:.1f}%")
print(f"    Predicted 40-50%: Actual {actual_rates.get('40-50%', 0)*100:.1f}%")
print(f"    Predicted 50-60%: Actual {actual_rates.get('50-60%', 0)*100:.1f}%")
print(f"    Predicted 60-70%: Actual {actual_rates.get('60-70%', 0)*100:.1f}%")
print(f"    Predicted 70%+:   Actual {actual_rates.get('70%+', 0)*100:.1f}%")

# Feature importance
print("\n  Top 15 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    marker = " [TEAM]" if 'team' in row['feature'] else ""
    print(f"    {row['feature']:<35}: {row['importance']:.4f}{marker}")

# Save model
model_dict = {
    'model': calibrated_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'train_games': len(train_games),
    'test_games': len(test_games)
}

output_file = 'models/momentum_model_enhanced.pkl'
joblib.dump(model_dict, output_file)
print(f"\n[OK] Enhanced model saved to: {output_file}")

print("\n" + "="*70)
print("NEXT STEP: Backtest enhanced model")
print("="*70 + "\n")

