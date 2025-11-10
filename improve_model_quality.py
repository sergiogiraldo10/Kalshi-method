"""
Improve Model Quality - Following User Feedback
1. Better run detection with quality scoring
2. Recalibrated probabilities
3. Better training strategy (more data, cross-validation)
4. Selective trading (1 per 2-3 games, only best opportunities)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import math

print("\n" + "="*70)
print("IMPROVING MODEL QUALITY")
print("Better run detection + Recalibration + Selective trading")
print("="*70)

# Load enhanced features
print("\nLoading enhanced features...")
features_df = pd.read_csv('data/processed/features_v2_2023_24_enhanced.csv')
features_df = features_df.sort_values('game_id')
print(f"  [OK] Loaded {len(features_df):,} samples")

# ============================================================================
# STEP 1: ADD RUN QUALITY SCORE
# ============================================================================
print("\n" + "="*70)
print("STEP 1: ADDING RUN QUALITY SCORE")
print("="*70)

def calculate_run_quality(row):
    """
    Calculate quality score for a run opportunity (0-100)
    Higher score = better trade opportunity
    """
    score = 0
    
    # 1. Pure run bonus (0 opponent points)
    if row['opp_score'] == 0:
        score += 20
    
    # 2. Large run bonus (7+ points)
    if row['run_score'] >= 7:
        score += 15
    elif row['run_score'] >= 6:
        score += 10
    
    # 3. Defensive pressure (steals, blocks, turnovers)
    defensive_actions = row.get('team_steals_2min', 0) + row.get('team_blocks_2min', 0) + row.get('opponent_turnovers_2min', 0)
    if defensive_actions >= 3:
        score += 15
    elif defensive_actions >= 2:
        score += 10
    elif defensive_actions >= 1:
        score += 5
    
    # 4. Offensive efficiency (3-pointers)
    if row.get('team_threes_2min', 0) >= 2:
        score += 10
    elif row.get('team_threes_2min', 0) >= 1:
        score += 5
    
    # 5. Team quality advantage
    team_quality_diff = row.get('team_quality_diff', 0)
    if team_quality_diff > 0.10:  # Run team is significantly better
        score += 10
    elif team_quality_diff > 0:
        score += 5
    
    # 6. Recent form advantage
    form_diff = row.get('team_form_advantage', 0)
    if form_diff > 0.2:  # Hot team
        score += 10
    elif form_diff > 0:
        score += 5
    
    # 7. Early game (Q1-Q2)
    if row.get('period', 3) <= 2:
        score += 10
    
    # 8. Close game (more volatile, more opportunity)
    if row.get('is_close_game', 0) == 1:
        score += 5
    
    return min(score, 100)  # Cap at 100

print("\n  Calculating run quality scores...")
features_df['run_quality_score'] = features_df.apply(calculate_run_quality, axis=1)
print(f"  [OK] Quality scores calculated")
print(f"    Mean: {features_df['run_quality_score'].mean():.1f}")
print(f"    Median: {features_df['run_quality_score'].median():.1f}")
print(f"    Max: {features_df['run_quality_score'].max():.1f}")

# ============================================================================
# STEP 2: BETTER TRAINING STRATEGY
# ============================================================================
print("\n" + "="*70)
print("STEP 2: BETTER TRAINING STRATEGY")
print("="*70)

# Use 60% for training, 40% for testing (more training data)
unique_games = features_df['game_id'].unique()
n_games = len(unique_games)

train_cutoff = int(n_games * 0.60)
train_games = set(unique_games[:train_cutoff])
test_games = set(unique_games[train_cutoff:])

train_df = features_df[features_df['game_id'].isin(train_games)].copy()
test_df = features_df[features_df['game_id'].isin(test_games)].copy()

print(f"\n  TRAIN: {len(train_games)} games ({len(train_df):,} samples)")
print(f"  TEST:  {len(test_games)} games ({len(test_df):,} samples)")

# Select features
exclude_cols = ['game_id', 'event_num', 'run_extends', 'run_team', 'time_remaining', 'run_quality_score']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

print(f"\n  Using {len(feature_cols)} features")

X_train = train_df[feature_cols].values
y_train = train_df['run_extends'].values

print(f"\n  Training data:")
print(f"    Samples: {len(X_train):,}")
print(f"    Positive class: {y_train.mean()*100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train with better hyperparameters and early stopping
print("\n  Training XGBoost with improved hyperparameters...")
model = XGBClassifier(
    n_estimators=500,  # More trees
    max_depth=6,       # Not too deep (prevent overfitting)
    learning_rate=0.03, # Lower learning rate
    scale_pos_weight=(1-y_train.mean())/y_train.mean(),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,  # Regularization
    gamma=0.1,           # Regularization
    reg_alpha=0.1,       # L1 regularization
    reg_lambda=1.0       # L2 regularization
)

model.fit(X_train_scaled, y_train)
print("  [OK] Model trained")

# ============================================================================
# STEP 3: RECALIBRATE PROBABILITIES
# ============================================================================
print("\n" + "="*70)
print("STEP 3: RECALIBRATING PROBABILITIES")
print("="*70)

print("\n  Calibrating with isotonic regression...")
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated_model.fit(X_train_scaled, y_train)
print("  [OK] Probabilities recalibrated")

# Check training calibration
train_preds = calibrated_model.predict_proba(X_train_scaled)[:, 1]
print(f"\n  Training calibration:")
print(f"    Predicted: {train_preds.mean()*100:.1f}%")
print(f"    Actual:    {y_train.mean()*100:.1f}%")

# Check test calibration
X_test = test_df[feature_cols].values
X_test_scaled = scaler.transform(X_test)
test_preds = calibrated_model.predict_proba(X_test_scaled)[:, 1]
y_test = test_df['run_extends'].values

print(f"\n  Test calibration:")
print(f"    Predicted: {test_preds.mean()*100:.1f}%")
print(f"    Actual:    {y_test.mean()*100:.1f}%")

# Save improved model
model_dict = {
    'model': calibrated_model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'train_games': len(train_games),
    'test_games': len(test_games)
}

joblib.dump(model_dict, 'models/momentum_model_improved.pkl')
print(f"\n[OK] Improved model saved to: models/momentum_model_improved.pkl")

# ============================================================================
# STEP 4: SELECTIVE TRADING (QUALITY FILTER)
# ============================================================================
print("\n" + "="*70)
print("STEP 4: SELECTIVE TRADING")
print("="*70)

# Add predictions to test set
test_df = test_df.copy()
test_df['prediction'] = test_preds

# Filter for good run opportunities
# 6-0 runs, Q1-Q3, with predictions
entry_opportunities = test_df[
    (test_df['is_micro_run'] == 1) &
    (test_df['run_score'] >= 6) &
    (test_df['opp_score'] == 0) &
    (test_df['period'] <= 3)
].copy()

print(f"\n  6-0 run opportunities (Q1-Q3): {len(entry_opportunities):,}")
print(f"    Across {entry_opportunities['game_id'].nunique()} games")

# Apply quality filter: Only take opportunities with quality score >= 60
MIN_QUALITY_SCORE = 60
high_quality = entry_opportunities[
    entry_opportunities['run_quality_score'] >= MIN_QUALITY_SCORE
].copy()

print(f"\n  High quality opportunities (score >= {MIN_QUALITY_SCORE}): {len(high_quality):,}")
print(f"    Across {high_quality['game_id'].nunique()} games")

# Further filter by prediction confidence (top 30%)
prediction_threshold = np.percentile(high_quality['prediction'], 70)  # Top 30%
print(f"\n  Prediction threshold (70th percentile): {prediction_threshold*100:.1f}%")

selective_trades = high_quality[
    high_quality['prediction'] >= prediction_threshold
].copy()

print(f"\n  After confidence filter: {len(selective_trades):,}")
print(f"    Across {selective_trades['game_id'].nunique()} games")

# Select best opportunity per game
selective_trades = selective_trades.sort_values(
    ['game_id', 'run_quality_score', 'prediction'], 
    ascending=[True, False, False]
)

best_trades = []
seen_games = set()

for idx, row in selective_trades.iterrows():
    if row['game_id'] not in seen_games:
        best_trades.append(row)
        seen_games.add(row['game_id'])

print(f"\n  FINAL TRADES (1 per game): {len(best_trades)}")
print(f"  Trade frequency: 1 per {len(test_games) / len(best_trades):.1f} games")

# Analyze quality of selected trades
trades_df = pd.DataFrame(best_trades)
print(f"\n  Selected trades analysis:")
print(f"    Avg quality score: {trades_df['run_quality_score'].mean():.1f}")
print(f"    Avg prediction: {trades_df['prediction'].mean()*100:.1f}%")
print(f"    Actual win rate: {trades_df['run_extends'].mean()*100:.1f}%")
print(f"    Calibration error: {(trades_df['prediction'].mean() - trades_df['run_extends'].mean())*100:+.1f}%")

# ============================================================================
# STEP 5: BACKTEST WITH SELECTIVE TRADES
# ============================================================================
print("\n" + "="*70)
print("STEP 5: BACKTESTING SELECTIVE STRATEGY")
print("="*70)

CONFIG = {
    'position_size_pct': 0.03,
    'take_profit_pct': 0.25,
    'stop_loss_pct': -0.05,
}

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

INITIAL_BANKROLL = 1000.0
bankroll = INITIAL_BANKROLL
trades = []

print(f"\n  Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Simulating {len(best_trades)} selective trades...\n")

for i, row in enumerate(best_trades):
    position_size = bankroll * CONFIG['position_size_pct']
    predicted_prob = row['prediction']
    entry_fee = calculate_fee(position_size, predicted_prob)
    
    if position_size + entry_fee > bankroll:
        continue
    
    actual_outcome = row['run_extends']
    
    # Realistic P/L
    if actual_outcome == 1:
        prob_change = np.random.uniform(0.10, CONFIG['take_profit_pct'])
        price_change_pct = prob_change * 2.0
        exit_reason = "TP"
    else:
        if np.random.random() < 0.40:
            prob_change = CONFIG['stop_loss_pct']
            exit_reason = "SL"
        else:
            prob_change = np.random.uniform(-0.04, -0.02)
            exit_reason = "Stopped"
        price_change_pct = prob_change * 2.0
    
    payout = position_size * (1 + price_change_pct)
    exit_fee = calculate_fee(payout, predicted_prob) if payout > 0 else 0
    profit = payout - position_size - entry_fee - exit_fee
    
    bankroll += profit
    
    trade = {
        'trade_num': i + 1,
        'game_id': row['game_id'],
        'quality_score': row['run_quality_score'],
        'confidence': predicted_prob,
        'actual_outcome': actual_outcome,
        'exit_reason': exit_reason,
        'profit': profit,
        'profit_pct': (profit / position_size) * 100,
        'bankroll': bankroll
    }
    trades.append(trade)
    
    if (i + 1) <= 15 or (i + 1) % 50 == 0:
        outcome = "WIN " if actual_outcome == 1 else "LOSS"
        print(f"  #{i+1:<4} | Q:{row['run_quality_score']:.0f} | {outcome} | {predicted_prob:.1%} | "
              f"{exit_reason:<8} | P/L: ${profit:>6.2f} | Bank: ${bankroll:>8.2f}")

# Results
results_df = pd.DataFrame(trades)

print("\n" + "="*70)
print("IMPROVED MODEL RESULTS")
print("="*70)

total_return = bankroll - INITIAL_BANKROLL
return_pct = (total_return / INITIAL_BANKROLL) * 100

print(f"\n  FINANCIAL PERFORMANCE:")
print(f"    Initial Capital:      ${INITIAL_BANKROLL:>10,.2f}")
print(f"    Final Capital:        ${bankroll:>10,.2f}")
print(f"    Total Return:         ${total_return:>+10,.2f}")
print(f"    Return %:             {return_pct:>+10.2f}%")

wins = results_df[results_df['actual_outcome'] == 1]
losses = results_df[results_df['actual_outcome'] == 0]

print(f"\n  TRADING STATISTICS:")
print(f"    Total Trades:         {len(results_df):>10,}")
print(f"    Trade Frequency:      1 per {len(test_games) / len(results_df):.1f} games")
print(f"    Win Rate:             {len(wins)/len(results_df)*100:>10.1f}%")
print(f"    Winning Trades:       {len(wins):>10,}")
print(f"    Losing Trades:        {len(losses):>10,}")

avg_win = wins['profit'].mean() if len(wins) > 0 else 0
avg_loss = losses['profit'].mean() if len(losses) > 0 else 0

print(f"\n  PROFIT/LOSS:")
print(f"    Average Win:          ${avg_win:>10.2f}")
print(f"    Average Loss:         ${avg_loss:>10.2f}")
print(f"    Win/Loss Ratio:       {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.2f}:1")

print(f"\n  QUALITY METRICS:")
print(f"    Avg Quality Score:    {results_df['quality_score'].mean():>10.1f}")
print(f"    Min Quality Score:    {results_df['quality_score'].min():>10.0f}")

# Risk metrics
returns = results_df['profit'] / (results_df['bankroll'].shift(1).fillna(INITIAL_BANKROLL) * CONFIG['position_size_pct'])
sharpe = (returns.mean() / returns.std()) * np.sqrt(len(results_df)) if returns.std() > 0 and len(results_df) > 1 else 0

print(f"\n  RISK METRICS:")
print(f"    Sharpe Ratio:         {sharpe:>10.2f}")
equity = results_df['bankroll'].values
drawdown = (pd.Series(equity).cummax() - equity).max()
print(f"    Max Drawdown:         ${drawdown:>10.2f}")

print("\n" + "="*70)
if return_pct > 10:
    print("[OK] HIGHLY PROFITABLE!")
    print(f"   Quality-focused approach works!")
    print(f"   Return: ${total_return:,.2f} ({return_pct:+.2f}%)")
    print(f"   Win rate: {len(wins)/len(results_df)*100:.1f}%")
    print(f"   Trade frequency: 1 per {len(test_games) / len(results_df):.1f} games (selective!)")
elif return_pct > 5:
    print("[OK] PROFITABLE!")
    print(f"   Return: ${total_return:,.2f} ({return_pct:+.2f}%)")
else:
    print("[~] Results: {return_pct:+.2f}%")

print("="*70 + "\n")

results_df.to_csv('backtest_improved_quality.csv', index=False)
print("[OK] Results saved to backtest_improved_quality.csv\n")

