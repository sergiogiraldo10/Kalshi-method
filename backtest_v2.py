"""
Simple Backtest V2 - Test momentum trading strategy on 2023-24 season
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

print("\n" + "="*70)
print("IGNITION AI - BACKTEST V2 (2023-24 Season)")
print("="*70)

# Load models
print("\nLoading models...")
momentum_model = joblib.load('models/momentum_model_v2.pkl')
win_prob_model = joblib.load('models/win_probability_enhanced.pkl')
print("  [OK] Models loaded")

# Load 2023-24 features
print("\nLoading 2023-24 season features...")
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')
print(f"  [OK] Loaded {len(features_df):,} game moments from 2023-24")

# Trading parameters
INITIAL_BANKROLL = 1000.0
POSITION_SIZE_PCT = 0.02  # 2% of bankroll per trade
ENTRY_FEE = 0.015  # 1.5%
EXIT_FEE = 0.015   # 1.5%
TAKE_PROFIT_PCT = 0.10  # 10% profit target
STOP_LOSS_PCT = -0.05   # -5% stop loss
MIN_CONFIDENCE = 0.55   # Only trade if model predicts >55% chance of extension

print("\n" + "="*70)
print("TRADING PARAMETERS")
print("="*70)
print(f"  Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Position Size: {POSITION_SIZE_PCT*100:.1f}% of bankroll")
print(f"  Entry Fee: {ENTRY_FEE*100:.1f}%")
print(f"  Exit Fee: {EXIT_FEE*100:.1f}%")
print(f"  Take Profit: {TAKE_PROFIT_PCT*100:.1f}%")
print(f"  Stop Loss: {STOP_LOSS_PCT*100:.1f}%")
print(f"  Min Confidence: {MIN_CONFIDENCE*100:.0f}%")
print("="*70)

# Filter for micro-runs only (potential trading opportunities)
micro_runs = features_df[features_df['is_micro_run'] == 1].copy()
print(f"\n[INFO] Found {len(micro_runs):,} micro-run opportunities")

# Predict using momentum model
print("\n[INFO] Generating predictions...")
feature_cols = momentum_model['feature_cols']
X = micro_runs[feature_cols].values
X_scaled = momentum_model['scaler'].transform(X)
predictions = momentum_model['model'].predict_proba(X_scaled)[:, 1]
micro_runs['prediction'] = predictions

# Filter for high confidence predictions
high_confidence = micro_runs[micro_runs['prediction'] >= MIN_CONFIDENCE].copy()
print(f"[INFO] {len(high_confidence):,} opportunities meet confidence threshold")

# Simulate trading
bankroll = INITIAL_BANKROLL
trades = []
current_position = None

print("\n" + "="*70)
print("SIMULATING TRADES")
print("="*70 + "\n")

for idx, row in high_confidence.iterrows():
    # Skip if already in a position
    if current_position is not None:
        continue
    
    # Enter trade
    position_size = bankroll * POSITION_SIZE_PCT
    entry_fee = position_size * ENTRY_FEE
    total_cost = position_size + entry_fee
    
    if total_cost > bankroll:
        continue  # Not enough bankroll
    
    # Determine outcome (did run extend?)
    actual_outcome = row['run_extends']
    predicted_prob = row['prediction']
    
    # Simple exit simulation: if run extended, we win; if not, we lose
    if actual_outcome == 1:
        # Win - run extended as predicted
        profit_pct = np.random.uniform(0.05, TAKE_PROFIT_PCT)  # Random profit in range
        payout = position_size * (1 + profit_pct)
        exit_fee = payout * EXIT_FEE
        profit = payout - total_cost - exit_fee
    else:
        # Loss - run did not extend
        loss_pct = np.random.uniform(STOP_LOSS_PCT, -0.02)  # Random loss in range
        payout = position_size * (1 + loss_pct)
        exit_fee = payout * EXIT_FEE if payout > 0 else 0
        profit = payout - total_cost - exit_fee
    
    # Update bankroll
    bankroll += profit
    
    # Record trade
    trade = {
        'game_id': row['game_id'],
        'entry_prob': predicted_prob,
        'position_size': position_size,
        'entry_fee': entry_fee,
        'exit_fee': exit_fee,
        'outcome': 'WIN' if actual_outcome == 1 else 'LOSS',
        'profit': profit,
        'bankroll': bankroll,
        'run_score': row['run_score'],
        'opp_score': row['opp_score']
    }
    trades.append(trade)
    
    # Print first 10 trades and every 10th trade
    if len(trades) <= 10 or len(trades) % 10 == 0:
        print(f"  Trade #{len(trades):<3} | {trade['outcome']:<4} | "
              f"Run: {trade['run_score']:.0f}-{trade['opp_score']:.0f} | "
              f"Profit: ${trade['profit']:>7.2f} | "
              f"Bankroll: ${trade['bankroll']:>8.2f}")

# Convert to DataFrame
trades_df = pd.DataFrame(trades)

# Calculate metrics
print("\n" + "="*70)
print("BACKTEST RESULTS")
print("="*70)

final_bankroll = bankroll
total_return = final_bankroll - INITIAL_BANKROLL
return_pct = (total_return / INITIAL_BANKROLL) * 100

print(f"\n$$ FINANCIAL RESULTS:")
print(f"  Starting Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Final Bankroll:    ${final_bankroll:,.2f}")
print(f"  Total Return:      ${total_return:+,.2f}")
print(f"  Return %:          {return_pct:+.2f}%")

print(f"\n## TRADING STATISTICS:")
print(f"  Total Trades:      {len(trades_df)}")
print(f"  Winning Trades:    {(trades_df['outcome'] == 'WIN').sum()} ({(trades_df['outcome'] == 'WIN').mean()*100:.1f}%)")
print(f"  Losing Trades:     {(trades_df['outcome'] == 'LOSS').sum()} ({(trades_df['outcome'] == 'LOSS').mean()*100:.1f}%)")
print(f"  Win Rate:          {(trades_df['outcome'] == 'WIN').mean()*100:.1f}%")

print(f"\n$$ PROFIT/LOSS:")
print(f"  Average Win:       ${trades_df[trades_df['outcome'] == 'WIN']['profit'].mean():.2f}")
print(f"  Average Loss:      ${trades_df[trades_df['outcome'] == 'LOSS']['profit'].mean():.2f}")
print(f"  Largest Win:       ${trades_df['profit'].max():.2f}")
print(f"  Largest Loss:      ${trades_df['profit'].min():.2f}")
print(f"  Avg Profit/Trade:  ${trades_df['profit'].mean():.2f}")

# Risk metrics
returns = trades_df['profit'] / trades_df['position_size']
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(trades_df)) if returns.std() > 0 else 0

print(f"\n## RISK METRICS:")
print(f"  Sharpe Ratio:      {sharpe_ratio:.2f}")
print(f"  Max Drawdown:      ${(trades_df['bankroll'].cummax() - trades_df['bankroll']).max():.2f}")
print(f"  Total Fees Paid:   ${(trades_df['entry_fee'] + trades_df['exit_fee']).sum():.2f}")

print("\n" + "="*70)

if total_return > 0:
    print("[OK] PROFITABLE STRATEGY!")
    print(f"   You would have made ${total_return:,.2f} ({return_pct:.1f}%) trading 2023-24")
else:
    print("[X] LOSING STRATEGY")
    print(f"   You would have lost ${abs(total_return):,.2f} ({abs(return_pct):.1f}%) trading 2023-24")

print("="*70)

# Save results
trades_df.to_csv('backtest_results_v2.csv', index=False)
print("\n[OK] Results saved to backtest_results_v2.csv")

