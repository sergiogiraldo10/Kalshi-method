"""
Retrain Momentum Model with Enhanced Team Features
Following Action Plan Option 1 - Improve the Model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

print("\n" + "="*70)
print("RETRAINING MODEL WITH TEAM FEATURES")
print("Following Action Plan to Profitability")
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

# Get feature columns (exclude non-feature columns)
exclude_cols = ['game_id', 'run_extends', 'is_micro_run', 'run_score', 'opp_score', 
                'run_team', 'period', 'time_remaining', 'event_num',
                'home_win_pct', 'away_win_pct', 'home_ppg', 'away_ppg',
                'home_opp_ppg', 'away_opp_ppg', 'home_form_5g', 'away_form_5g']

feature_cols = [col for col in features_df.columns if col not in exclude_cols]

print(f"\n  Using {len(feature_cols)} features (including {len([c for c in feature_cols if 'team' in c])} team features)")

# Prepare training data
print("\nPreparing training data...")
X_train = train_df[feature_cols].values
y_train = train_df['run_extends'].values

print(f"  Training samples: {len(X_train):,}")
print(f"  Positive class rate: {y_train.mean()*100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train XGBoost model
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
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
print("\nCalibrating probabilities...")
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated_model.fit(X_train_scaled, y_train)
print("  [OK] Model calibrated")

# Evaluate on training set
train_preds = calibrated_model.predict_proba(X_train_scaled)[:, 1]
train_acc = ((train_preds > 0.5).astype(int) == y_train).mean()
print(f"\n  Training accuracy: {train_acc*100:.1f}%")
print(f"  Training avg prediction: {train_preds.mean()*100:.1f}%")
print(f"  Training actual rate: {y_train.mean()*100:.1f}%")

# Evaluate on test set
print("\nEvaluating on test set...")
X_test = test_df[feature_cols].values
X_test_scaled = scaler.transform(X_test)
test_preds = calibrated_model.predict_proba(X_test_scaled)[:, 1]
y_test = test_df['run_extends'].values

test_acc = ((test_preds > 0.5).astype(int) == y_test).mean()
print(f"  Test accuracy: {test_acc*100:.1f}%")
print(f"  Test avg prediction: {test_preds.mean()*100:.1f}%")
print(f"  Test actual rate: {y_test.mean()*100:.1f}%")

# Check calibration
print("\n" + "="*70)
print("CALIBRATION CHECK")
print("="*70)

# Bin predictions and check actual rates
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_indices = np.digitize(test_preds, bins) - 1

print("\n  Prediction vs Reality:")
print(f"  {'Bin':<15} {'Predicted':<15} {'Actual':<15} {'Error'}")
print("  " + "-"*60)

for i in range(len(bin_centers)):
    mask = bin_indices == i
    if mask.sum() > 0:
        pred_avg = test_preds[mask].mean()
        actual_rate = y_test[mask].mean()
        error = (pred_avg - actual_rate) * 100
        print(f"  {bin_centers[i]:.1f}-{bins[i+1]:.1f}        {pred_avg*100:>6.1f}%        {actual_rate*100:>6.1f}%        {error:>+6.1f}%")

# Feature importance
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Top 20)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top features:")
for idx, row in feature_importance.head(20).iterrows():
    marker = " [TEAM]" if 'team' in row['feature'] else ""
    print(f"    {row['feature']:<35} {row['importance']:>8.4f}{marker}")

# Save model
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

model_dict = {
    'model': calibrated_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'base_model': model  # Keep original for feature importance
}

output_file = 'models/momentum_model_enhanced.pkl'
joblib.dump(model_dict, output_file)
print(f"\n[OK] Enhanced model saved to: {output_file}")

print("\n" + "="*70)
print("NEXT STEP: Backtest with enhanced model")
print("="*70 + "\n")

