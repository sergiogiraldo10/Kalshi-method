"""
MANUAL RUN CHECKER
==================

Use this when watching a game live and you see a 6-0 run.
Input the details and get a trade signal.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import math

sys.path.append('src')

print("\n" + "="*70)
print("MANUAL RUN CHECKER - IGNITION AI")
print("="*70)

# Load model
MODEL_FILE = Path('models/ignition_ai_live_2024_25.pkl')

if not MODEL_FILE.exists():
    print("\n[!] No trained model found!")
    print("    Run: python train_live_model.py first")
    exit()

print("\nLoading model...")
model_data = joblib.load(MODEL_FILE)
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']

print("[OK] Model loaded")

# Get user input
print("\n" + "="*70)
print("ENTER RUN DETAILS")
print("="*70)

print("\nExample: You're watching LAL @ GSW")
print("         It's Q2 with 5:30 left")
print("         LAL just went on a 6-0 run")
print("         They got 2 steals, 1 block, made 2 threes")

print("\n" + "-"*70)

# Simple input (you can expand this)
try:
    print("\nQuick inputs:")
    run_score = int(input("  Run score (e.g., 6): ") or "6")
    opp_score = int(input("  Opponent score (e.g., 0): ") or "0")
    period = int(input("  Quarter (1-4): ") or "2")
    steals = int(input("  Steals in last 2min (0-5): ") or "1")
    blocks = int(input("  Blocks in last 2min (0-5): ") or "1")
    turnovers = int(input("  Opponent turnovers (0-5): ") or "2")
    threes = int(input("  3-pointers in last 2min (0-5): ") or "2")
    
    print("\n  [Optional - press Enter to skip]")
    team_quality_input = input("  Team quality diff (-0.5 to 0.5, e.g., 0.15 if better team): ")
    team_quality_diff = float(team_quality_input) if team_quality_input else 0.0
    
except KeyboardInterrupt:
    print("\n\nCancelled.")
    exit()
except:
    print("\n[!] Invalid input. Using defaults.")
    run_score = 6
    opp_score = 0
    period = 2
    steals = 1
    blocks = 1
    turnovers = 2
    threes = 2
    team_quality_diff = 0.0

# Calculate quality score
def calculate_quality_score(run_score, opp_score, steals, blocks, turnovers, threes, team_quality_diff, period):
    score = 0
    if opp_score == 0: score += 20
    if run_score >= 7: score += 15
    elif run_score >= 6: score += 10
    
    defensive = steals + blocks + turnovers
    if defensive >= 3: score += 15
    elif defensive >= 2: score += 10
    elif defensive >= 1: score += 5
    
    if threes >= 2: score += 10
    elif threes >= 1: score += 5
    
    if team_quality_diff > 0.10: score += 10
    elif team_quality_diff > 0: score += 5
    
    if period <= 2: score += 10
    
    return min(score, 100)

quality = calculate_quality_score(run_score, opp_score, steals, blocks, turnovers, threes, team_quality_diff, period)

# Create feature vector (simplified - using default values for missing features)
# In production, you'd extract all 43 features from live game data
feature_dict = {}
for col in feature_cols:
    if 'run_score' in col:
        feature_dict[col] = run_score
    elif 'opp_score' in col:
        feature_dict[col] = opp_score
    elif 'period' in col:
        feature_dict[col] = period
    elif 'steals' in col:
        feature_dict[col] = steals
    elif 'blocks' in col:
        feature_dict[col] = blocks
    elif 'turnover' in col:
        feature_dict[col] = turnovers
    elif 'three' in col:
        feature_dict[col] = threes
    elif 'quality_diff' in col:
        feature_dict[col] = team_quality_diff
    else:
        feature_dict[col] = 0.0  # Default for other features

feature_array = np.array([[feature_dict.get(col, 0) for col in feature_cols]])

# Scale and predict
feature_scaled = scaler.transform(feature_array)
probability = model.predict_proba(feature_scaled)[0, 1]

# Results
print("\n" + "="*70)
print("ANALYSIS RESULTS")
print("="*70)

print(f"\nRun Details:")
print(f"  Score: {run_score}-{opp_score}")
print(f"  Quarter: {period}")
print(f"  Defensive Actions: {steals} steals, {blocks} blocks, {turnovers} turnovers")
print(f"  Three-Pointers: {threes}")
print(f"  Quality Score: {quality}/100")

print(f"\nModel Prediction:")
print(f"  Confidence: {probability*100:.1f}%")

# Determine if it's a trade signal
MIN_CONFIDENCE = 0.345  # Top 20% threshold from validation
MIN_QUALITY = 60

if probability >= MIN_CONFIDENCE and quality >= MIN_QUALITY and period <= 3:
    print(f"\n🚨 TRADE SIGNAL! 🚨")
    print("-" * 70)
    print(f"  Status: HIGH CONFIDENCE (Top 20%)")
    print(f"  Confidence: {probability*100:.1f}%")
    print(f"  Quality: {quality}/100")
    
    bankroll = 1000  # Adjust to your actual bankroll
    position_size = bankroll * 0.05
    
    def calculate_fee(position_size, probability):
        fee = 0.07 * position_size * probability * (1 - probability)
        return math.ceil(fee * 100) / 100
    
    entry_fee = calculate_fee(position_size, probability)
    
    print(f"\nRECOMMENDED ACTION:")
    print(f"  Position Size: ${position_size:.2f} (5% of ${bankroll})")
    print(f"  Entry Fee: ${entry_fee:.2f}")
    print(f"  Take Profit: +25% (Exit at ${position_size * 1.25:.2f})")
    print(f"  Stop Loss: -5% (Exit at ${position_size * 0.95:.2f})")
    print("-" * 70)
    
elif probability < MIN_CONFIDENCE:
    print(f"\n❌ NO TRADE")
    print(f"  Reason: Confidence {probability*100:.1f}% < {MIN_CONFIDENCE*100:.1f}%")
    print(f"  (Below top 20% threshold)")
    
elif quality < MIN_QUALITY:
    print(f"\n❌ NO TRADE")
    print(f"  Reason: Quality {quality} < {MIN_QUALITY}")
    print(f"  (Not enough defensive pressure/momentum)")
    
elif period > 3:
    print(f"\n❌ NO TRADE")
    print(f"  Reason: Q4 - Too volatile")
    print(f"  (Only trade in Q1-Q3)")

print("\n" + "="*70 + "\n")

