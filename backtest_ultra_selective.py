"""
ULTRA-SELECTIVE BACKTEST
Only trade when MULTIPLE factors align for highest probability
Path to Profitability: Quality over Quantity
"""

import pandas as pd
import numpy as np
import joblib
import math
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("\n" + "="*70)
print("IGNITION AI - ULTRA-SELECTIVE STRATEGY")
print("Path to Profitability: Quality Over Quantity")
print("="*70)

# Load 2023-24 data
print("\nLoading 2023-24 season data...")
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')
features_df = features_df.sort_values('game_id')
unique_games = features_df['game_id'].unique()
n_games = len(unique_games)

# Split: First 33% for training, rest for testing
train_cutoff = int(n_games * 0.33)
train_games = set(unique_games[:train_cutoff])
test_games = set(unique_games[train_cutoff:])

train_df = features_df[features_df['game_id'].isin(train_games)].copy()
test_df = features_df[features_df['game_id'].isin(test_games)].copy()

print(f"\n  TRAIN: First {len(train_games)} games")
print(f"  TEST:  Next {len(test_games)} games")

# Train model
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

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

# ULTRA-SELECTIVE CRITERIA
print("\n" + "="*70)
print("ULTRA-SELECTIVE CRITERIA")
print("="*70)

def is_elite_opportunity(row, prediction):
    """
    Only trade when MULTIPLE factors align
    This is the path to profitability
    """
    
    # Factor 1: PURE RUN with significant score (7-0, 8-0, 9-0)
    run_score = row['run_score']
    opp_score = row['opp_score']
    
    if opp_score != 0:
        return False, "Not a pure run"
    
    if run_score < 7:
        return False, f"Run too small ({run_score}-0)"
    
    # Factor 2: HIGH MODEL CONFIDENCE (65%+)
    if prediction < 0.65:
        return False, f"Confidence too low ({prediction:.1%})"
    
    # Factor 3: STRONG DEFENSIVE PRESSURE (3+ combined)
    defensive_pressure = row.get('defensive_pressure', 0)
    if defensive_pressure < 3:
        return False, f"Low defense pressure ({defensive_pressure})"
    
    # Factor 4: EARLY GAME (Q1-Q2, fresh teams)
    period = row.get('period', 1)
    if period > 2:
        return False, f"Too late in game (Q{period})"
    
    # Factor 5: OFFENSIVE EFFICIENCY (3-pointers)
    offensive_eff = row.get('offensive_efficiency', 0)
    if offensive_eff < 3:
        return False, f"Low offensive efficiency ({offensive_eff})"
    
    # ALL FACTORS ALIGNED!
    return True, "ELITE"

print("\n  Elite Opportunity Criteria (ALL must be true):")
print("    1. Pure run: 7-0, 8-0, or 9-0 (NO opponent points)")
print("    2. Model confidence: 65%+ (high conviction)")
print("    3. Defensive pressure: 3+ steals/blocks/turnovers")
print("    4. Early game: Q1 or Q2 only (fresh teams)")
print("    5. Offensive efficiency: 3+ from three-pointers")
print("\n  Philosophy: Quality over Quantity")

# Filter test data
print("\n" + "="*70)
print("FINDING ELITE OPPORTUNITIES")
print("="*70)

micro_runs = test_df[test_df['is_micro_run'] == 1].copy()

# Generate predictions for all
X_test = micro_runs[feature_cols].values
X_test_scaled = scaler.transform(X_test)
predictions = model.predict_proba(X_test_scaled)[:, 1]
micro_runs['prediction'] = predictions

# Apply elite criteria
elite_opportunities = []
rejection_reasons = {}

for idx, row in micro_runs.iterrows():
    pred = row['prediction']
    is_elite, reason = is_elite_opportunity(row, pred)
    
    if is_elite:
        elite_opportunities.append(row)
    else:
        if reason not in rejection_reasons:
            rejection_reasons[reason] = 0
        rejection_reasons[reason] += 1

print(f"\n  Total micro-runs evaluated: {len(micro_runs):,}")
print(f"  Elite opportunities found: {len(elite_opportunities)}")
print(f"\n  Rejection breakdown:")
for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {reason}: {count:,}")

if len(elite_opportunities) == 0:
    print("\n  [!] NO ELITE OPPORTUNITIES FOUND")
    print("  Criteria are too strict. Try relaxing one constraint.")
    print("\n  Suggestions:")
    print("    - Lower pure run requirement to 6-0")
    print("    - Lower confidence to 60%")
    print("    - Lower defensive pressure to 2+")
    print("    - Include Q3")
    exit()

elite_df = pd.DataFrame(elite_opportunities)

# Select best per game
elite_df = elite_df.sort_values(['game_id', 'prediction'], ascending=[True, False])
trades_to_make = []
seen_games = set()

for idx, row in elite_df.iterrows():
    if row['game_id'] not in seen_games:
        trades_to_make.append(row)
        seen_games.add(row['game_id'])

print(f"\n  Elite trades (1 per game): {len(trades_to_make)}")

if len(trades_to_make) == 0:
    print("\n  [!] NO TRADES TO MAKE")
    exit()

# Analyze
actual_win_rate = pd.DataFrame(trades_to_make)['run_extends'].mean()
avg_confidence = pd.DataFrame(trades_to_make)['prediction'].mean()

print(f"\n  ELITE TRADE ANALYSIS:")
print(f"    Model prediction: {avg_confidence*100:.1f}%")
print(f"    Actual win rate:  {actual_win_rate*100:.1f}%")
print(f"    Calibration:      {(avg_confidence - actual_win_rate)*100:+.1f}% error")

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

# Simulate trades
print("\n" + "="*70)
print("SIMULATING ELITE TRADES")
print("="*70)

INITIAL_BANKROLL = 1000.0
POSITION_SIZE_PCT = 0.05  # 5% per trade (higher since fewer trades)

bankroll = INITIAL_BANKROLL
trades = []

print(f"\n  Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Position Size: {POSITION_SIZE_PCT*100:.0f}% (aggressive, few trades)")
print(f"  Simulating {len(trades_to_make)} elite trades...\n")

for i, row in enumerate(trades_to_make):
    position_size = bankroll * POSITION_SIZE_PCT
    predicted_prob = row['prediction']
    entry_fee = calculate_fee(position_size, predicted_prob)
    
    if position_size + entry_fee > bankroll:
        continue
    
    actual_outcome = row['run_extends']
    run_pattern = f"{int(row['run_score'])}-{int(row['opp_score'])}"
    def_pressure = int(row.get('defensive_pressure', 0))
    period = int(row.get('period', 1))
    
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
        'period': f"Q{period}",
        'def_pressure': def_pressure,
        'confidence': predicted_prob,
        'actual_outcome': actual_outcome,
        'exit_reason': exit_reason,
        'profit': profit,
        'profit_pct': (profit / position_size) * 100,
        'bankroll': bankroll
    }
    trades.append(trade)
    
    outcome = "WIN " if actual_outcome == 1 else "LOSS"
    print(f"  #{i+1:<3} | {outcome} | {run_pattern:<5} | {trade['period']} | D:{def_pressure} | "
          f"{predicted_prob:.1%} | {exit_reason:<10} | "
          f"P/L: ${profit:>6.2f} | Bank: ${bankroll:>8.2f}")

# Results
trades_df = pd.DataFrame(trades)

print("\n" + "="*70)
print("ULTRA-SELECTIVE RESULTS")
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
print(f"    Total Trades:         {len(trades_df):>10,} (elite only)")
print(f"    Winning Trades:       {len(wins):>10,} ({len(wins)/len(trades_df)*100:.1f}%)")
print(f"    Losing Trades:        {len(losses):>10,} ({len(losses)/len(trades_df)*100:.1f}%)")

print(f"\n  PROFIT/LOSS:")
avg_win = wins['profit'].mean() if len(wins) > 0 else 0
avg_loss = losses['profit'].mean() if len(losses) > 0 else 0
print(f"    Average Win:          ${avg_win:>10.2f}")
print(f"    Average Loss:         ${avg_loss:>10.2f}")
print(f"    Win/Loss Ratio:       {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.2f}:1")

# Risk metrics
returns = trades_df['profit'] / (trades_df['bankroll'].shift(1).fillna(INITIAL_BANKROLL) * POSITION_SIZE_PCT)
sharpe = (returns.mean() / returns.std()) * np.sqrt(len(trades_df)) if returns.std() > 0 and len(trades_df) > 1 else 0

print(f"\n  RISK METRICS:")
print(f"    Sharpe Ratio:         {sharpe:>10.2f}")
total_fees = trades_df['profit'].sum() - (bankroll - INITIAL_BANKROLL)
print(f"    Total Fees:           ${abs(total_fees):>10.2f}")

print(f"\n  EFFICIENCY:")
print(f"    Trades per game:      {len(trades_df)/len(test_games):.3f}")
print(f"    Total fees / capital: {abs(total_fees)/INITIAL_BANKROLL*100:.1f}%")

print("\n" + "="*70)

if return_pct > 0:
    print("[OK] PROFITABLE!")
    print(f"   Ultra-selective approach works!")
    print(f"   Made ${total_return:,.2f} ({return_pct:.2f}%) with only {len(trades_df)} trades")
elif return_pct > -5:
    print("[~] NEAR BREAKEVEN")
    print(f"   Return: {return_pct:+.2f}%")
    print(f"   Close to profitability with {len(trades_df)} trades")
    print(f"   Consider slightly relaxing criteria")
else:
    print("[X] UNPROFITABLE")
    print(f"   Lost ${abs(total_return):,.2f} ({abs(return_pct):.2f}%)")
    print(f"   Even ultra-selective doesn't work")
    print(f"   Win rate: {actual_win_rate*100:.1f}% (need 50%+)")

print("="*70 + "\n")

trades_df.to_csv('backtest_ultra_selective_results.csv', index=False)
print("[OK] Results saved to backtest_ultra_selective_results.csv\n")

