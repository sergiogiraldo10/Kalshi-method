"""
STRONG MOMENTUM BACKTEST
Test entries on solidified runs: 6-0, 8-2, 10-2, 10-3, 12-4
Criteria: Net score >= 6 AND ratio >= 3:1
"""

import pandas as pd
import numpy as np
import joblib
import math
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("\n" + "="*70)
print("IGNITION AI - STRONG MOMENTUM RUNS")
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
print(f"  Positive class rate: {y_train.mean()*100:.1f}%")

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

# Test different run patterns
print("\n" + "="*70)
print("TESTING DIFFERENT RUN PATTERNS")
print("="*70)

def is_strong_momentum(run_score, opp_score, min_net=6, min_ratio=3.0):
    """Check if run shows strong momentum"""
    net_score = run_score - opp_score
    if net_score < min_net:
        return False
    if opp_score == 0:
        return True  # Pure run always qualifies
    ratio = run_score / opp_score if opp_score > 0 else 999
    return ratio >= min_ratio

# Analyze run patterns in test data
print("\n  Analyzing run patterns in test set:")
micro_runs = test_df[test_df['is_micro_run'] == 1].copy()

run_patterns = {}
for idx, row in micro_runs.iterrows():
    run_score = int(row['run_score'])
    opp_score = int(row['opp_score'])
    pattern = f"{run_score}-{opp_score}"
    
    if pattern not in run_patterns:
        run_patterns[pattern] = {
            'count': 0,
            'extends': 0,
            'is_strong': is_strong_momentum(run_score, opp_score)
        }
    
    run_patterns[pattern]['count'] += 1
    if row['run_extends'] == 1:
        run_patterns[pattern]['extends'] += 1

# Show top patterns
print("\n  Top Run Patterns (sorted by frequency):")
print(f"  {'Pattern':<10} {'Count':<8} {'Extends':<10} {'Win%':<8} {'Strong?'}")
print("  " + "-"*60)

sorted_patterns = sorted(run_patterns.items(), key=lambda x: x[1]['count'], reverse=True)
for pattern, data in sorted_patterns[:15]:
    win_rate = (data['extends'] / data['count'] * 100) if data['count'] > 0 else 0
    strong = "[STRONG]" if data['is_strong'] else ""
    print(f"  {pattern:<10} {data['count']:<8} {data['extends']:<10} {win_rate:<7.1f}% {strong}")

# Filter for strong momentum runs
print("\n" + "="*70)
print("FILTERING FOR STRONG MOMENTUM RUNS")
print("="*70)

strong_momentum_runs = test_df[
    (test_df['is_micro_run'] == 1) &
    (test_df.apply(lambda row: is_strong_momentum(row['run_score'], row['opp_score']), axis=1))
].copy()

print(f"\n  Strong momentum runs (net >= 6, ratio >= 3:1): {len(strong_momentum_runs):,}")

# Show breakdown by pattern
strong_patterns = {}
for idx, row in strong_momentum_runs.iterrows():
    pattern = f"{int(row['run_score'])}-{int(row['opp_score'])}"
    if pattern not in strong_patterns:
        strong_patterns[pattern] = 0
    strong_patterns[pattern] += 1

print("\n  Breakdown:")
for pattern, count in sorted(strong_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {pattern}: {count}")

# Generate predictions
X_test = strong_momentum_runs[feature_cols].values
X_test_scaled = scaler.transform(X_test)
predictions = model.predict_proba(X_test_scaled)[:, 1]
strong_momentum_runs['prediction'] = predictions

# Filter by confidence (50-60% range, slightly wider for more trades)
CONFIG = {
    'position_size_pct': 0.03,
    'min_confidence': 0.50,
    'max_confidence': 0.60,  # Wider range
}

print(f"\n  Confidence range: {CONFIG['min_confidence']*100:.0f}-{CONFIG['max_confidence']*100:.0f}%")

sweet_spot = strong_momentum_runs[
    (strong_momentum_runs['prediction'] >= CONFIG['min_confidence']) &
    (strong_momentum_runs['prediction'] <= CONFIG['max_confidence'])
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
    
    # Realistic P/L based on outcome
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
        'entry_confidence': predicted_prob,
        'actual_outcome': actual_outcome,
        'price_change_pct': price_change_pct,
        'exit_reason': exit_reason,
        'position_size': position_size,
        'profit': profit,
        'profit_pct': (profit / position_size) * 100,
        'bankroll': bankroll
    }
    trades.append(trade)
    
    if (i + 1) <= 20 or (i + 1) % 100 == 0:
        outcome = "WIN" if actual_outcome == 1 else "LOSS"
        print(f"  #{i+1:<4} | {outcome:<4} | {run_pattern:<8} | "
              f"{predicted_prob:.1%} | {exit_reason:<10} | "
              f"P/L: ${profit:>6.2f} | Bank: ${bankroll:>8.2f}")

# Results
trades_df = pd.DataFrame(trades)

print("\n" + "="*70)
print("STRONG MOMENTUM RESULTS")
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
    
    # Performance by run pattern
    print(f"\n  PERFORMANCE BY RUN PATTERN:")
    pattern_stats = {}
    for _, trade in trades_df.iterrows():
        pattern = trade['run_pattern']
        if pattern not in pattern_stats:
            pattern_stats[pattern] = {'trades': 0, 'wins': 0, 'profit': 0}
        pattern_stats[pattern]['trades'] += 1
        if trade['actual_outcome'] == 1:
            pattern_stats[pattern]['wins'] += 1
        pattern_stats[pattern]['profit'] += trade['profit']
    
    print(f"    {'Pattern':<10} {'Trades':<8} {'Win%':<8} {'Profit'}")
    print(f"    {'-'*45}")
    for pattern, stats in sorted(pattern_stats.items(), key=lambda x: x[1]['trades'], reverse=True):
        win_pct = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        print(f"    {pattern:<10} {stats['trades']:<8} {win_pct:<7.1f}% ${stats['profit']:>8.2f}")
    
    # Risk metrics
    returns = trades_df['profit'] / trades_df['position_size']
    sharpe = (returns.mean() / returns.std()) * np.sqrt(len(trades_df)) if returns.std() > 0 else 0
    
    print(f"\n  RISK METRICS:")
    print(f"    Sharpe Ratio:         {sharpe:>10.2f}")
    equity = trades_df['bankroll'].values
    drawdown = (pd.Series(equity).cummax() - equity).max()
    print(f"    Max Drawdown:         ${drawdown:>10.2f}")

print("\n" + "="*70)

if return_pct > 0:
    print("[OK] PROFITABLE!")
    print(f"   Strong momentum runs work!")
    print(f"   Made ${total_return:,.2f} ({return_pct:.2f}%)")
elif return_pct > -15:
    print("[~] CLOSE TO BREAKEVEN")
    print(f"   Lost ${abs(total_return):,.2f} ({abs(return_pct):.2f}%)")
    print(f"   Strong momentum helps but not enough")
else:
    print("[X] STILL LOSING")
    print(f"   Lost ${abs(total_return):,.2f} ({abs(return_pct):.2f}%)")

print("="*70 + "\n")

if len(trades_df) > 0:
    trades_df.to_csv('backtest_strong_momentum_results.csv', index=False)
    print("[OK] Results saved to backtest_strong_momentum_results.csv\n")

