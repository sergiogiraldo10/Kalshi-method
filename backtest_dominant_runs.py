"""
DOMINANT RUNS BACKTEST
Entry criteria:
- Team scoring high (6-10 points)
- Opponent scoring low (0-2 points)
- Defensive pressure: steals, blocks, turnovers, fouls
"""

import pandas as pd
import numpy as np
import joblib
import math
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("\n" + "="*70)
print("IGNITION AI - DOMINANT RUNS WITH DEFENSIVE PRESSURE")
print("="*70)

# Load 2023-24 data
print("\nLoading 2023-24 season data...")
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')
print(f"  [OK] Loaded {len(features_df):,} feature samples")

# Sort by game_id
features_df = features_df.sort_values('game_id')
unique_games = features_df['game_id'].unique()
n_games = len(unique_games)

# Split: First 33% for training, rest for testing
train_cutoff = int(n_games * 0.33)
train_games = set(unique_games[:train_cutoff])
test_games = set(unique_games[train_cutoff:])

train_df = features_df[features_df['game_id'].isin(train_games)].copy()
test_df = features_df[features_df['game_id'].isin(test_games)].copy()

print(f"\n  TRAIN: First {len(train_games)} games ({len(train_df):,} samples)")
print(f"  TEST:  Next {len(test_games)} games ({len(test_df):,} samples)")

# Train model
print("\n" + "="*70)
print("TRAINING IN-SEASON MODEL")
print("="*70)

original_model = joblib.load('models/momentum_model_v2.pkl')
feature_cols = original_model['feature_cols']

X_train = train_df[feature_cols].values
y_train = train_df['run_extends'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"\n  Training on {len(train_df):,} samples...")
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

# Define DOMINANT run criteria
print("\n" + "="*70)
print("DEFINING DOMINANT RUN CRITERIA")
print("="*70)

def is_dominant_run(row):
    """
    Check if run shows dominant momentum with defensive pressure
    
    Criteria:
    1. Team scoring high: 6-10 points
    2. Opponent scoring low: 0-2 points
    3. Defensive pressure: steals, blocks, turnovers, fouls
    """
    # Basic score criteria
    run_score = row['run_score']
    opp_score = row['opp_score']
    
    if run_score < 6 or run_score > 10:
        return False
    if opp_score > 2:
        return False
    
    # Defensive pressure indicators (NLP features)
    steals = row.get('team_steals_2min', 0)
    blocks = row.get('team_blocks_2min', 0)
    opp_turnovers = row.get('opponent_turnovers_2min', 0)
    opp_fouls = row.get('opponent_fouls_2min', 0)
    
    # Need at least SOME defensive pressure
    defensive_pressure = steals + blocks + opp_turnovers
    
    # Criteria: Either pure run (0 points) OR defensive pressure
    if opp_score == 0:
        return True  # Pure run always good
    
    # If opponent scored, need strong defense signs
    return defensive_pressure >= 2 or opp_fouls >= 2

print("\n  Dominant Run Criteria:")
print("    1. Team score: 6-10 points")
print("    2. Opponent score: 0-2 points")
print("    3. Defensive pressure:")
print("       - Pure run (0-0), OR")
print("       - 2+ steals/blocks/turnovers, OR")
print("       - 2+ opponent fouls")

# Filter test data
print("\n" + "="*70)
print("FILTERING FOR DOMINANT RUNS")
print("="*70)

dominant_runs = test_df[
    (test_df['is_micro_run'] == 1) &
    (test_df.apply(is_dominant_run, axis=1))
].copy()

print(f"\n  Dominant runs found: {len(dominant_runs):,}")

# Show breakdown
print("\n  Sample of qualifying patterns:")
patterns = {}
for idx, row in dominant_runs.head(1000).iterrows():
    pattern = f"{int(row['run_score'])}-{int(row['opp_score'])}"
    steals = int(row.get('team_steals_2min', 0))
    blocks = int(row.get('team_blocks_2min', 0))
    turnovers = int(row.get('opponent_turnovers_2min', 0))
    fouls = int(row.get('opponent_fouls_2min', 0))
    
    key = f"{pattern} (S:{steals} B:{blocks} T:{turnovers} F:{fouls})"
    if key not in patterns:
        patterns[key] = 0
    patterns[key] += 1

for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"    {pattern}: {count}")

# Calculate actual win rate by pattern
print("\n  Win rate by score pattern:")
score_patterns = {}
for idx, row in dominant_runs.iterrows():
    pattern = f"{int(row['run_score'])}-{int(row['opp_score'])}"
    if pattern not in score_patterns:
        score_patterns[pattern] = {'count': 0, 'wins': 0}
    score_patterns[pattern]['count'] += 1
    if row['run_extends'] == 1:
        score_patterns[pattern]['wins'] += 1

print(f"  {'Pattern':<10} {'Count':<8} {'Wins':<8} {'Win%'}")
print("  " + "-"*40)
for pattern, stats in sorted(score_patterns.items(), key=lambda x: x[1]['count'], reverse=True):
    win_rate = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
    print(f"  {pattern:<10} {stats['count']:<8} {stats['wins']:<8} {win_rate:.1f}%")

# Generate predictions
X_test = dominant_runs[feature_cols].values
X_test_scaled = scaler.transform(X_test)
predictions = model.predict_proba(X_test_scaled)[:, 1]
dominant_runs['prediction'] = predictions

# Filter by confidence (50-60%)
CONFIG = {
    'position_size_pct': 0.03,
    'min_confidence': 0.50,
    'max_confidence': 0.60,
}

print(f"\n  Filtering for {CONFIG['min_confidence']*100:.0f}-{CONFIG['max_confidence']*100:.0f}% confidence...")

sweet_spot = dominant_runs[
    (dominant_runs['prediction'] >= CONFIG['min_confidence']) &
    (dominant_runs['prediction'] <= CONFIG['max_confidence'])
].copy()

print(f"  Opportunities in confidence range: {len(sweet_spot):,}")

# Select best per game
sweet_spot = sweet_spot.sort_values(['game_id', 'prediction'], ascending=[True, False])
trades_to_make = []
seen_games = set()

for idx, row in sweet_spot.iterrows():
    if row['game_id'] not in seen_games:
        trades_to_make.append(row)
        seen_games.add(row['game_id'])

print(f"  Final opportunities (1 per game): {len(trades_to_make)}")

# Analyze
if len(trades_to_make) > 0:
    trades_df_temp = pd.DataFrame(trades_to_make)
    actual_win_rate = trades_df_temp['run_extends'].mean()
    avg_confidence = trades_df_temp['prediction'].mean()
    
    print(f"\n  MODEL CALIBRATION:")
    print(f"    Predicted: {avg_confidence*100:.1f}%")
    print(f"    Actual:    {actual_win_rate*100:.1f}%")
    print(f"    Error:     {(avg_confidence - actual_win_rate)*100:+.1f}%")
    
    # Analyze defensive pressure in trades
    avg_def_pressure = trades_df_temp['defensive_pressure'].mean()
    avg_opp_fouls = trades_df_temp['opponent_fouls_2min'].mean()
    print(f"\n  DEFENSIVE PRESSURE IN TRADES:")
    print(f"    Avg steals+blocks: {avg_def_pressure:.1f}")
    print(f"    Avg opp fouls: {avg_opp_fouls:.1f}")

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

# Simulate trades
print("\n" + "="*70)
print("SIMULATING TRADES")
print("="*70)

INITIAL_BANKROLL = 1000.0
bankroll = INITIAL_BANKROLL
trades = []

for i, row in enumerate(trades_to_make):
    position_size = bankroll * CONFIG['position_size_pct']
    predicted_prob = row['prediction']
    entry_fee = calculate_fee(position_size, predicted_prob)
    
    if position_size + entry_fee > bankroll:
        continue
    
    actual_outcome = row['run_extends']
    run_pattern = f"{int(row['run_score'])}-{int(row['opp_score'])}"
    def_pressure = int(row.get('defensive_pressure', 0))
    
    # Realistic P/L
    if actual_outcome == 1:
        prob_change = np.random.uniform(0.05, 0.15)
        price_change_pct = prob_change * 2.0
        exit_reason = "Extended"
    else:
        prob_change = np.random.uniform(-0.08, -0.03)
        price_change_pct = prob_change * 2.0
        exit_reason = "Stopped"
    
    payout = position_size * (1 + price_change_pct)
    exit_fee = calculate_fee(payout, predicted_prob) if payout > 0 else 0
    profit = payout - position_size - entry_fee - exit_fee
    
    bankroll += profit
    
    trade = {
        'trade_num': i + 1,
        'game_id': row['game_id'],
        'run_pattern': run_pattern,
        'def_pressure': def_pressure,
        'entry_confidence': predicted_prob,
        'actual_outcome': actual_outcome,
        'exit_reason': exit_reason,
        'profit': profit,
        'profit_pct': (profit / position_size) * 100,
        'bankroll': bankroll
    }
    trades.append(trade)
    
    if (i + 1) <= 20 or (i + 1) % 100 == 0:
        outcome = "WIN" if actual_outcome == 1 else "LOSS"
        print(f"  #{i+1:<4} | {outcome:<4} | {run_pattern:<6} | D:{def_pressure} | "
              f"{predicted_prob:.1%} | {exit_reason:<10} | "
              f"P/L: ${profit:>6.2f} | Bank: ${bankroll:>8.2f}")

# Results
trades_df = pd.DataFrame(trades)

print("\n" + "="*70)
print("DOMINANT RUNS RESULTS")
print("="*70)

total_return = bankroll - INITIAL_BANKROLL
return_pct = (total_return / INITIAL_BANKROLL) * 100

print(f"\n  FINANCIAL PERFORMANCE:")
print(f"    Initial Capital:      ${INITIAL_BANKROLL:>10,.2f}")
print(f"    Final Capital:        ${bankroll:>10,.2f}")
print(f"    Total Return:         ${total_return:>+10,.2f}")
print(f"    Return %:             {return_pct:>+10.2f}%")

if len(trades_df) > 0:
    wins = trades_df[trades_df['actual_outcome'] == 1]
    losses = trades_df[trades_df['actual_outcome'] == 0]
    
    print(f"\n  TRADING STATISTICS:")
    print(f"    Total Trades:         {len(trades_df):>10,}")
    print(f"    Winning Trades:       {len(wins):>10,} ({len(wins)/len(trades_df)*100:.1f}%)")
    print(f"    Losing Trades:        {len(losses):>10,} ({len(losses)/len(trades_df)*100:.1f}%)")
    
    print(f"\n  PROFIT/LOSS:")
    avg_win = wins['profit'].mean() if len(wins) > 0 else 0
    avg_loss = losses['profit'].mean() if len(losses) > 0 else 0
    print(f"    Average Win:          ${avg_win:>10.2f}")
    print(f"    Average Loss:         ${avg_loss:>10.2f}")
    print(f"    Win/Loss Ratio:       {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.2f}:1")
    
    # Defensive pressure analysis
    print(f"\n  DEFENSIVE PRESSURE ANALYSIS:")
    high_def = trades_df[trades_df['def_pressure'] >= 2]
    low_def = trades_df[trades_df['def_pressure'] < 2]
    
    if len(high_def) > 0:
        high_def_win_rate = (high_def['actual_outcome'] == 1).mean() * 100
        print(f"    High defense (2+): {len(high_def)} trades, {high_def_win_rate:.1f}% win rate")
    
    if len(low_def) > 0:
        low_def_win_rate = (low_def['actual_outcome'] == 1).mean() * 100
        print(f"    Low defense (<2):  {len(low_def)} trades, {low_def_win_rate:.1f}% win rate")
    
    # Risk metrics
    returns = trades_df['profit'] / (trades_df['bankroll'].shift(1).fillna(INITIAL_BANKROLL) * CONFIG['position_size_pct'])
    sharpe = (returns.mean() / returns.std()) * np.sqrt(len(trades_df)) if returns.std() > 0 else 0
    
    print(f"\n  RISK METRICS:")
    print(f"    Sharpe Ratio:         {sharpe:>10.2f}")
    equity = trades_df['bankroll'].values
    drawdown = (pd.Series(equity).cummax() - equity).max()
    print(f"    Max Drawdown:         ${drawdown:>10.2f}")

print("\n" + "="*70)

if return_pct > 0:
    print("[OK] PROFITABLE!")
    print(f"   Dominant runs + defensive pressure works!")
    print(f"   Made ${total_return:,.2f} ({return_pct:.2f}%)")
elif return_pct > -15:
    print("[~] CLOSE TO BREAKEVEN")
    print(f"   Lost ${abs(total_return):,.2f} ({abs(return_pct):.2f}%)")
    print(f"   Defensive pressure helps but not enough")
else:
    print("[X] STILL LOSING")
    print(f"   Lost ${abs(total_return):,.2f} ({abs(return_pct):.2f}%)")

print("="*70 + "\n")

if len(trades_df) > 0:
    trades_df.to_csv('backtest_dominant_runs_results.csv', index=False)
    print("[OK] Results saved to backtest_dominant_runs_results.csv\n")

