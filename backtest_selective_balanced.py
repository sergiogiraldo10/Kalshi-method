"""
SELECTIVE BALANCED BACKTEST
Relaxed ultra-selective criteria to get 20-50 trades
Still highly selective, but achievable
"""

import pandas as pd
import numpy as np
import joblib
import math
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("\n" + "="*70)
print("IGNITION AI - SELECTIVE BALANCED STRATEGY")
print("="*70)

# Load data
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')
features_df = features_df.sort_values('game_id')
unique_games = features_df['game_id'].unique()
n_games = len(unique_games)

train_cutoff = int(n_games * 0.33)
train_games = set(unique_games[:train_cutoff])
test_games = set(unique_games[train_cutoff:])

train_df = features_df[features_df['game_id'].isin(train_games)].copy()
test_df = features_df[features_df['game_id'].isin(test_games)].copy()

print(f"\n  TRAIN: {len(train_games)} games | TEST: {len(test_games)} games")

# Train model
original_model = joblib.load('models/momentum_model_v2.pkl')
feature_cols = original_model['feature_cols']

X_train = train_df[feature_cols].values
y_train = train_df['run_extends'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(1-y_train.mean())/y_train.mean(),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

print("  [OK] Model trained")

# BALANCED SELECTIVE CRITERIA
print("\n" + "="*70)
print("SELECTIVE CRITERIA (BALANCED)")
print("="*70)

def is_high_quality_opportunity(row, prediction):
    """
    High-quality opportunities with multiple positive signals
    Relaxed from ultra-selective but still demanding
    """
    
    score_points = 0
    reasons = []
    
    # Pure runs get priority
    if row['opp_score'] == 0:
        score_points += 2
        reasons.append("Pure run")
        
        # Larger pure runs get bonus
        if row['run_score'] >= 7:
            score_points += 1
            reasons.append(f"{int(row['run_score'])}-0 run")
    
    # High model confidence
    if prediction >= 0.60:
        score_points += 1
        reasons.append(f"{prediction:.1%} confidence")
    if prediction >= 0.65:
        score_points += 1
        reasons.append("Very high confidence")
    
    # Defensive pressure
    def_pressure = row.get('defensive_pressure', 0)
    if def_pressure >= 2:
        score_points += 1
        reasons.append(f"{def_pressure} defense")
    if def_pressure >= 3:
        score_points += 1
        reasons.append("Strong defense")
    
    # Early game
    period = row.get('period', 1)
    if period <= 2:
        score_points += 1
        reasons.append(f"Q{period}")
    
    # Offensive efficiency
    off_eff = row.get('offensive_efficiency', 0)
    if off_eff >= 3:
        score_points += 1
        reasons.append("3-pointers")
    
    # Need at least 4 points to qualify
    if score_points >= 4:
        return True, score_points, reasons
    
    return False, score_points, reasons

print("\n  High-Quality Criteria (score-based system):")
print("    Points awarded for:")
print("      +2: Pure run (0 opponent points)")
print("      +1: Large run (7-0+)")
print("      +1: High confidence (60%+)")
print("      +1: Very high confidence (65%+)")
print("      +1: Defensive pressure (2+)")
print("      +1: Strong defense (3+)")
print("      +1: Early game (Q1-Q2)")
print("      +1: Offensive efficiency (3-pt makes)")
print("    NEED: 4+ points to qualify")

# Filter
print("\n" + "="*70)
print("FINDING HIGH-QUALITY OPPORTUNITIES")
print("="*70)

micro_runs = test_df[test_df['is_micro_run'] == 1].copy()
X_test = micro_runs[feature_cols].values
X_test_scaled = scaler.transform(X_test)
predictions = model.predict_proba(X_test_scaled)[:, 1]
micro_runs['prediction'] = predictions

high_quality = []
score_distribution = {}

for idx, row in micro_runs.iterrows():
    pred = row['prediction']
    is_hq, score, reasons = is_high_quality_opportunity(row, pred)
    
    if score not in score_distribution:
        score_distribution[score] = 0
    score_distribution[score] += 1
    
    if is_hq:
        row_dict = row.to_dict()
        row_dict['quality_score'] = score
        row_dict['quality_reasons'] = ', '.join(reasons)
        high_quality.append(row_dict)

print(f"\n  Total opportunities evaluated: {len(micro_runs):,}")
print(f"  High-quality found (4+ points): {len(high_quality)}")

print(f"\n  Quality score distribution:")
for score in sorted(score_distribution.keys(), reverse=True):
    count = score_distribution[score]
    marker = " <-- SELECTED" if score >= 4 else ""
    print(f"    {score} points: {count:,}{marker}")

if len(high_quality) == 0:
    print("\n  [!] NO OPPORTUNITIES FOUND - Criteria too strict")
    exit()

hq_df = pd.DataFrame(high_quality)

# Select best per game
hq_df = hq_df.sort_values(['game_id', 'quality_score', 'prediction'], ascending=[True, False, False])
trades_to_make = []
seen_games = set()

for idx, row in hq_df.iterrows():
    if row['game_id'] not in seen_games:
        trades_to_make.append(row)
        seen_games.add(row['game_id'])

print(f"\n  Final trades (1 per game): {len(trades_to_make)}")

# Analyze
trades_temp = pd.DataFrame(trades_to_make)
actual_win_rate = trades_temp['run_extends'].mean()
avg_confidence = trades_temp['prediction'].mean()
avg_quality = trades_temp['quality_score'].mean()

print(f"\n  TRADE QUALITY ANALYSIS:")
print(f"    Avg quality score:    {avg_quality:.1f} points")
print(f"    Model confidence:     {avg_confidence*100:.1f}%")
print(f"    Actual win rate:      {actual_win_rate*100:.1f}%")
print(f"    Calibration error:    {(avg_confidence - actual_win_rate)*100:+.1f}%")

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

# Simulate
print("\n" + "="*70)
print("SIMULATING HIGH-QUALITY TRADES")
print("="*70)

INITIAL_BANKROLL = 1000.0
POSITION_SIZE_PCT = 0.05

bankroll = INITIAL_BANKROLL
trades = []

for i, row in enumerate(trades_to_make):
    position_size = bankroll * POSITION_SIZE_PCT
    predicted_prob = row['prediction']
    entry_fee = calculate_fee(position_size, predicted_prob)
    
    if position_size + entry_fee > bankroll:
        continue
    
    actual_outcome = row['run_extends']
    run_pattern = f"{int(row['run_score'])}-{int(row['opp_score'])}"
    quality = int(row['quality_score'])
    
    # Realistic P/L
    if actual_outcome == 1:
        prob_change = np.random.uniform(0.05, 0.15)
        price_change_pct = prob_change * 2.0
        exit_reason = "Win"
    else:
        prob_change = np.random.uniform(-0.08, -0.03)
        price_change_pct = prob_change * 2.0
        exit_reason = "Loss"
    
    payout = position_size * (1 + price_change_pct)
    exit_fee = calculate_fee(payout, predicted_prob) if payout > 0 else 0
    profit = payout - position_size - entry_fee - exit_fee
    
    bankroll += profit
    
    trade = {
        'trade_num': i + 1,
        'game_id': row['game_id'],
        'run_pattern': run_pattern,
        'quality_score': quality,
        'confidence': predicted_prob,
        'actual_outcome': actual_outcome,
        'profit': profit,
        'bankroll': bankroll
    }
    trades.append(trade)
    
    outcome = "WIN " if actual_outcome == 1 else "LOSS"
    if (i + 1) <= 30 or (i + 1) % 10 == 0:
        print(f"  #{i+1:<3} | {outcome} | {run_pattern:<5} | Q:{quality} | "
              f"{predicted_prob:.1%} | P/L: ${profit:>6.2f} | Bank: ${bankroll:>8.2f}")

# Results
trades_df = pd.DataFrame(trades)

print("\n" + "="*70)
print("SELECTIVE BALANCED RESULTS")
print("="*70)

total_return = bankroll - INITIAL_BANKROLL
return_pct = (total_return / INITIAL_BANKROLL) * 100

print(f"\n  FINANCIAL PERFORMANCE:")
print(f"    Initial Capital:      ${INITIAL_BANKROLL:>10,.2f}")
print(f"    Final Capital:        ${bankroll:>10,.2f}")
print(f"    Total Return:         ${total_return:>+10,.2f}")
print(f"    Return %:             {return_pct:>+10.2f}%")

wins = trades_df[trades_df['actual_outcome'] == 1]
losses = trades_df[trades_df['actual_outcome'] == 0]

print(f"\n  TRADING STATISTICS:")
print(f"    Total Trades:         {len(trades_df):>10,}")
print(f"    Win Rate:             {len(wins)/len(trades_df)*100:>10.1f}%")

avg_win = wins['profit'].mean() if len(wins) > 0 else 0
avg_loss = losses['profit'].mean() if len(losses) > 0 else 0

print(f"\n  PROFIT/LOSS:")
print(f"    Average Win:          ${avg_win:>10.2f}")
print(f"    Average Loss:         ${avg_loss:>10.2f}")
print(f"    Win/Loss Ratio:       {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.2f}:1")

# By quality score
print(f"\n  PERFORMANCE BY QUALITY SCORE:")
for q in sorted(trades_df['quality_score'].unique(), reverse=True):
    subset = trades_df[trades_df['quality_score'] == q]
    win_rate = (subset['actual_outcome'] == 1).mean() * 100
    avg_profit = subset['profit'].mean()
    print(f"    {q} points: {len(subset):>3} trades | {win_rate:>5.1f}% win rate | ${avg_profit:>6.2f} avg P/L")

print("\n" + "="*70)

if return_pct > 0:
    print("[OK] PROFITABLE!")
    print(f"   Selective approach achieved profitability")
    print(f"   Return: ${total_return:,.2f} ({return_pct:+.2f}%)")
    print(f"   Win rate: {actual_win_rate*100:.1f}% with {len(trades_df)} trades")
elif return_pct > -5:
    print("[~] NEAR BREAKEVEN")
    print(f"   Return: {return_pct:+.2f}%")
    print(f"   Very close with {len(trades_df)} selective trades")
else:
    print("[X] UNPROFITABLE")
    print(f"   Lost: ${abs(total_return):,.2f} ({return_pct:.2f}%)")
    print(f"   Win rate too low: {actual_win_rate*100:.1f}%")

print("="*70 + "\n")

trades_df.to_csv('backtest_selective_balanced_results.csv', index=False)
print("[OK] Saved to backtest_selective_balanced_results.csv\n")

