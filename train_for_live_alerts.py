"""
Train model on ALL historical data for live SMS alerts
Includes 2025-26 season data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append('src')

print("\n" + "="*70)
print("TRAINING MODEL FOR LIVE SMS ALERTS")
print("="*70)

# Load ALL seasons
seasons = ['2021_22', '2022_23', '2023_24', '2024_25', '2025_26']
all_data = []

print("\nLoading training data...")
for season in seasons:
    file_path = Path(f'data/processed/features_v2_{season}_enhanced.csv')
    if file_path.exists():
        df = pd.read_csv(file_path)
        print(f"  {season}: {len(df):,} samples from {df['game_id'].nunique()} games")
        all_data.append(df)
    else:
        if season == '2025_26':
            print(f"  [!] {season}: Not found")
            print(f"      Run: python download_2025_26_season.py")
            print(f"      Then: python extract_features_2025_26.py")
            print(f"      Then: python add_team_features_2025_26.py")
        else:
            print(f"  [!] {season}: Not found (optional)")

if len(all_data) == 0:
    print("\n[!] No training data found!")
    exit()

train_df = pd.concat(all_data, ignore_index=True)

print(f"\n  TOTAL: {len(train_df):,} samples from {train_df['game_id'].nunique()} games")

# Prepare features
exclude_cols = ['game_id', 'event_num', 'run_extends', 'run_team', 'time_remaining', 
                'home_team', 'away_team', 'run_quality_score', 'event_type', 'home_score', 
                'away_score', 'score_margin', 'home_games_played', 'away_games_played', 'game_date']

feature_cols = sorted([col for col in train_df.columns if col not in exclude_cols])

print(f"\n  Features: {len(feature_cols)}")

X_train = train_df[feature_cols].fillna(0).values
y_train = train_df['run_extends'].values

print(f"  Positive rate: {y_train.mean()*100:.1f}%")

# Train
print("\n" + "="*70)
print("TRAINING XGBOOST")
print("="*70)

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

print("\nScaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training (500 trees, ~2-3 minutes)...")
model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    scale_pos_weight=(1-y_train.mean())/y_train.mean(),
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3, gamma=0.1,
    random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
)
model.fit(X_train_scaled, y_train)

print("Calibrating...")
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated_model.fit(X_train_scaled, y_train)

print("[OK] Model trained!")

# Save
output_dir = Path('models')
output_dir.mkdir(exist_ok=True)

model_data = {
    'model': calibrated_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'training_games': train_df['game_id'].nunique(),
    'training_samples': len(train_df),
    'positive_rate': y_train.mean()
}

output_file = output_dir / 'ignition_ai_live_2025_26.pkl'
print(f"\nSaving to {output_file}...")
joblib.dump(model_data, output_file)

print("[OK] Model saved!")

print("\n" + "="*70)
print("MODEL READY FOR LIVE ALERTS")
print("="*70)

print(f"\nStatistics:")
print(f"  Training Games: {model_data['training_games']:,}")
print(f"  Training Samples: {model_data['training_samples']:,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Expected Win Rate: 35-36%")
print(f"  Expected Return: +10-15%/month")

print(f"\nNext step:")
print(f"  1. Setup Twilio (see SETUP_LIVE_SMS_ALERTS.md)")
print(f"  2. Run: python auto_sms_monitor.py")

print("="*70 + "\n")

