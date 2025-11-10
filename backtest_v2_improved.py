"""
Backtest V2 Improved - More selective trading with realistic fee structure
"""
import pandas as pd
import numpy as np
import joblib
import math

print("\n" + "="*70)
print("IGNITION AI - BACKTEST V2 IMPROVED (2023-24 Season)")
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
TAKE_PROFIT_PCT = 0.15  # 15% profit target
STOP_LOSS_PCT = -0.05   # -5% stop loss
MIN_CONFIDENCE = 0.75   # INCREASED: Only trade if model predicts >75% chance

def calculate_fee(position_size, probability):
    """
    Calculate fee using: round_up(0.07 × C × P × (1-P))
    where C = position size, P = probability
    """
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100  # Round up to nearest cent

print("\n" + "="*70)
print("TRADING PARAMETERS")
print("="*70)
print(f"  Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Position Size: {POSITION_SIZE_PCT*100:.1f}% of bankroll")
print(f"  Fee Structure: round_up(0.07 × C × P × (1-P))")
print(f"  Take Profit: {TAKE_PROFIT_PCT*100:.1f}%")
print(f"  Stop Loss: {STOP_LOSS_PCT*100:.1f}%")
print(f"  Min Confidence: {MIN_CONFIDENCE*100:.0f}% (MORE SELECTIVE)")
print("="*70)

# Filter for micro-runs only
micro_runs = features_df[features_df['is_micro_run'] == 1].copy()
print(f"\n[INFO] Found {len(micro_runs):,} micro-run opportunities")

# Predict using momentum model
print("\n[INFO] Generating predictions...")
feature_cols = momentum_model['feature_cols']
X = micro_runs[feature_cols].values
X_scaled = momentum_model['scaler'].transform(X)
predictions = momentum_model['model'].predict_proba(X_scaled)[:, 1]
micro_runs['prediction'] = predictions

# Filter for HIGH confidence predictions (75%+)
high_confidence = micro_runs[micro_runs['prediction'] >= MIN_CONFIDENCE].copy()
print(f"[INFO] {len(high_confidence):,} opportunities meet HIGH confidence threshold (was 435 at 55%)")

# Simulate trading
bankroll = INITIAL_BANKROLL
trades = []
total_fees_paid = 0

print("\n" + "="*70)
print("SIMULATING TRADES (More Selective)")
print("="*70 + "\n")

for idx, row in high_confidence.iterrows():
    # Calculate position size
    position_size = bankroll * POSITION_SIZE_PCT
    predicted_prob = row['prediction']
    
    # Calculate entry fee using new structure
    entry_fee = calculate_fee(position_size, predicted_prob)
    total_cost = position_size + entry_fee
    
    if total_cost > bankroll:
        continue  # Not enough bankroll
    
    # Determine outcome
    actual_outcome = row['run_extends']
    
    # Simulate exit
    if actual_outcome == 1:
        # Win - run extended as predicted
        profit_pct = np.random.uniform(0.08, TAKE_PROFIT_PCT)
        payout = position_size * (1 + profit_pct)
        exit_fee = calculate_fee(payout, predicted_prob)
        profit = payout - total_cost - exit_fee
    else:
        # Loss - run did not extend
        loss_pct = np.random.uniform(STOP_LOSS_PCT, -0.02)
        payout = position_size * (1 + loss_pct)
        exit_fee = calculate_fee(payout, predicted_prob) if payout > 0 else 0
        profit = payout - total_cost - exit_fee
    
    # Update bankroll
    bankroll += profit
    total_fees_paid += entry_fee + exit_fee
    
    # Record trade
    trade = {
        'game_id': row['game_id'],
        'entry_prob': predicted_prob,
        'position_size': position_size,
        'entry_fee': entry_fee,
        'exit_fee': exit_fee,
        'total_fees': entry_fee + exit_fee,
        'outcome': 'WIN' if actual_outcome == 1 else 'LOSS',
        'profit': profit,
        'bankroll': bankroll,
        'run_score': row['run_score'],
        'opp_score': row['opp_score']
    }
    trades.append(trade)
    
    # Print all trades (since there should be fewer)
    if len(trades) <= 50 or len(trades) % 5 == 0:
        print(f"  Trade #{len(trades):<3} | {trade['outcome']:<4} | "
              f"Conf: {trade['entry_prob']:.1%} | "
              f"Run: {trade['run_score']:.0f}-{trade['opp_score']:.0f} | "
              f"Fees: ${trade['total_fees']:>5.2f} | "
              f"Profit: ${trade['profit']:>7.2f} | "
              f"Bankroll: ${trade['bankroll']:>8.2f}")

# Convert to DataFrame
trades_df = pd.DataFrame(trades)

# Calculate metrics
print("\n" + "="*70)
print("BACKTEST RESULTS - IMPROVED STRATEGY")
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
print(f"  Total Trades:      {len(trades_df)} (was 435)")
print(f"  Winning Trades:    {(trades_df['outcome'] == 'WIN').sum()} ({(trades_df['outcome'] == 'WIN').mean()*100:.1f}%)")
print(f"  Losing Trades:     {(trades_df['outcome'] == 'LOSS').sum()} ({(trades_df['outcome'] == 'LOSS').mean()*100:.1f}%)")
print(f"  Win Rate:          {(trades_df['outcome'] == 'WIN').mean()*100:.1f}%")

if len(trades_df) > 0:
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
    print(f"  Total Fees Paid:   ${trades_df['total_fees'].sum():.2f} ({trades_df['total_fees'].sum()/INITIAL_BANKROLL*100:.1f}% of capital)")
    print(f"  Avg Fee per Trade: ${trades_df['total_fees'].mean():.2f}")
    
    # Confidence analysis
    print(f"\n## CONFIDENCE ANALYSIS:")
    print(f"  Min Confidence:    {trades_df['entry_prob'].min():.1%}")
    print(f"  Max Confidence:    {trades_df['entry_prob'].max():.1%}")
    print(f"  Avg Confidence:    {trades_df['entry_prob'].mean():.1%}")

print("\n" + "="*70)

if len(trades_df) == 0:
    print("[!] NO TRADES TAKEN")
    print("   Model was too selective - no opportunities met 75%+ confidence threshold")
elif total_return > 0:
    print("[OK] PROFITABLE STRATEGY!")
    print(f"   You would have made ${total_return:,.2f} ({return_pct:.1f}%) trading 2023-24")
    print(f"   With only {len(trades_df)} trades (vs 435 before)")
else:
    print("[X] STILL LOSING STRATEGY")
    print(f"   You would have lost ${abs(total_return):,.2f} ({abs(return_pct):.1f}%) trading 2023-24")
    print(f"   But only {len(trades_df)} trades (vs 435 before)")

print("="*70)

# Save results
if len(trades_df) > 0:
    trades_df.to_csv('backtest_results_v2_improved.csv', index=False)
    print("\n[OK] Results saved to backtest_results_v2_improved.csv")

# Compare to previous version
print("\n" + "="*70)
print("COMPARISON TO PREVIOUS VERSION")
print("="*70)
print(f"  Previous (55% conf):  435 trades, -$149.36 (-14.9%), 42% win rate")
if len(trades_df) > 0:
    print(f"  Improved (75% conf):  {len(trades_df)} trades, ${total_return:+.2f} ({return_pct:+.1f}%), {(trades_df['outcome'] == 'WIN').mean()*100:.1f}% win rate")
else:
    print(f"  Improved (75% conf):  0 trades (too selective)")
print("="*70)

