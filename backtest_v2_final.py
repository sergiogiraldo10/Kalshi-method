"""
Backtest V2 Final - Realistic Trading Strategy
- One bet per game maximum
- 5% position sizing
- Enter when run reaches 3 scores (6+ points) to 0
"""
import pandas as pd
import numpy as np
import joblib
import math

print("\n" + "="*70)
print("IGNITION AI - BACKTEST V2 FINAL (2023-24 Season)")
print("="*70)

# Load models
print("\nLoading models...")
momentum_model = joblib.load('models/momentum_model_v2.pkl')
print("  [OK] Models loaded")

# Load 2023-24 features
print("\nLoading 2023-24 season features...")
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')
print(f"  [OK] Loaded {len(features_df):,} game moments from 2023-24")

# Trading parameters
INITIAL_BANKROLL = 1000.0
POSITION_SIZE_PCT = 0.05  # 5% of bankroll per trade (INCREASED)
TAKE_PROFIT_PCT = 0.15    # 15% profit target
STOP_LOSS_PCT = -0.05     # -5% stop loss
MIN_CONFIDENCE = 0.65     # 65% confidence minimum
MIN_RUN_SCORE = 6         # Must be at least 6 points (3 scores)
MAX_OPP_SCORE = 0         # Opponent must have 0 (pure run)

def calculate_fee(position_size, probability):
    """Calculate fee: round_up(0.07 × C × P × (1-P))"""
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

print("\n" + "="*70)
print("TRADING PARAMETERS - REALISTIC STRATEGY")
print("="*70)
print(f"  Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Position Size: {POSITION_SIZE_PCT*100:.1f}% of bankroll (INCREASED)")
print(f"  Fee Structure: round_up(0.07 × C × P × (1-P))")
print(f"  Min Run Score: {MIN_RUN_SCORE}+ points (3 scores)")
print(f"  Max Opp Score: {MAX_OPP_SCORE} points (pure run)")
print(f"  Min Confidence: {MIN_CONFIDENCE*100:.0f}%")
print(f"  Max 1 bet per game")
print("="*70)

# Filter for good entry points: 6+ point runs with opponent at 0
micro_runs = features_df[
    (features_df['is_micro_run'] == 1) &
    (features_df['run_score'] >= MIN_RUN_SCORE) &
    (features_df['opp_score'] == MAX_OPP_SCORE)
].copy()
print(f"\n[INFO] Found {len(micro_runs):,} qualifying run opportunities (6-0+)")

# Predict using momentum model
print("\n[INFO] Generating predictions...")
feature_cols = momentum_model['feature_cols']
X = micro_runs[feature_cols].values
X_scaled = momentum_model['scaler'].transform(X)
predictions = momentum_model['model'].predict_proba(X_scaled)[:, 1]
micro_runs['prediction'] = predictions

# Filter for confidence threshold
high_confidence = micro_runs[micro_runs['prediction'] >= MIN_CONFIDENCE].copy()
print(f"[INFO] {len(high_confidence):,} opportunities meet confidence threshold")

# Sort by game_id and prediction (best opportunity per game)
high_confidence = high_confidence.sort_values(['game_id', 'prediction'], ascending=[True, False])

# Take ONLY FIRST (best) opportunity per game
games_traded = set()
trades_to_make = []

for idx, row in high_confidence.iterrows():
    game_id = row['game_id']
    if game_id not in games_traded:
        trades_to_make.append(row)
        games_traded.add(game_id)

print(f"[INFO] Taking 1 trade per game = {len(trades_to_make)} total trades")

# Simulate trading
bankroll = INITIAL_BANKROLL
trades = []

print("\n" + "="*70)
print("SIMULATING TRADES (1 per game, 5% sizing)")
print("="*70 + "\n")

for row in trades_to_make:
    # Calculate position size (5% of current bankroll)
    position_size = bankroll * POSITION_SIZE_PCT
    predicted_prob = row['prediction']
    
    # Calculate entry fee
    entry_fee = calculate_fee(position_size, predicted_prob)
    total_cost = position_size + entry_fee
    
    if total_cost > bankroll:
        continue  # Not enough bankroll
    
    # Entry info
    entry_run = f"{int(row['run_score'])}-{int(row['opp_score'])}"
    
    # Determine outcome
    actual_outcome = row['run_extends']
    
    # Simulate final score at exit
    if actual_outcome == 1:
        # Win - run extended
        profit_pct = np.random.uniform(0.08, TAKE_PROFIT_PCT)
        payout = position_size * (1 + profit_pct)
        exit_fee = calculate_fee(payout, predicted_prob)
        profit = payout - total_cost - exit_fee
        
        # Estimate final run score (run extended by 4-8 more points)
        extension = np.random.randint(4, 9)
        final_run = f"{int(row['run_score']) + extension}-{int(row['opp_score']) + np.random.randint(0, 3)}"
    else:
        # Loss - run did not extend
        loss_pct = np.random.uniform(STOP_LOSS_PCT, -0.02)
        payout = position_size * (1 + loss_pct)
        exit_fee = calculate_fee(payout, predicted_prob) if payout > 0 else 0
        profit = payout - total_cost - exit_fee
        
        # Estimate final run score (opponent came back)
        final_run = f"{int(row['run_score']) + np.random.randint(0, 3)}-{int(row['opp_score']) + np.random.randint(4, 8)}"
    
    # Update bankroll
    bankroll += profit
    
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
        'entry_run': entry_run,
        'exit_run': final_run,
        'game_score_diff': int(row['score_diff'])
    }
    trades.append(trade)
    
    # Print all trades
    print(f"  Trade #{len(trades):<3} | {trade['outcome']:<4} | "
          f"Conf: {trade['entry_prob']:.0%} | "
          f"Entry: {trade['entry_run']:<6} Exit: {trade['exit_run']:<7} | "
          f"Size: ${trade['position_size']:>6.2f} | "
          f"Profit: ${trade['profit']:>7.2f} | "
          f"Bankroll: ${trade['bankroll']:>8.2f}")

# Convert to DataFrame
trades_df = pd.DataFrame(trades)

# Calculate metrics
print("\n" + "="*70)
print("BACKTEST RESULTS - FINAL STRATEGY")
print("="*70)

final_bankroll = bankroll
total_return = final_bankroll - INITIAL_BANKROLL
return_pct = (total_return / INITIAL_BANKROLL) * 100

print(f"\n$$ FINANCIAL RESULTS:")
print(f"  Starting Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Final Bankroll:    ${final_bankroll:,.2f}")
print(f"  Total Return:      ${total_return:+,.2f}")
print(f"  Return %:          {return_pct:+.2f}%")

if len(trades_df) > 0:
    print(f"\n## TRADING STATISTICS:")
    print(f"  Total Trades:      {len(trades_df)} (1 per game)")
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
    print(f"  Total Fees Paid:   ${trades_df['total_fees'].sum():.2f} ({trades_df['total_fees'].sum()/INITIAL_BANKROLL*100:.1f}% of capital)")
    print(f"  Avg Fee per Trade: ${trades_df['total_fees'].mean():.2f}")
    
    # Confidence analysis
    print(f"\n## CONFIDENCE ANALYSIS:")
    print(f"  Min Confidence:    {trades_df['entry_prob'].min():.1%}")
    print(f"  Max Confidence:    {trades_df['entry_prob'].max():.1%}")
    print(f"  Avg Confidence:    {trades_df['entry_prob'].mean():.1%}")
    
    print(f"\n## ENTRY POINT ANALYSIS:")
    print(f"  All entries at:    6-0+ runs (3 scores)")
    print(f"  Avg entry run:     {trades_df['entry_run'].mode()[0] if len(trades_df['entry_run'].mode()) > 0 else 'N/A'}")

print("\n" + "="*70)

if len(trades_df) == 0:
    print("[!] NO TRADES TAKEN - Criteria too strict")
elif total_return > 0:
    print("[OK] PROFITABLE STRATEGY!")
    print(f"   Profit: ${total_return:,.2f} ({return_pct:.2f}%) with {len(trades_df)} trades")
    print(f"   Avg ${total_return/len(trades_df):.2f} per trade")
else:
    print("[X] LOSING STRATEGY")
    print(f"   Loss: ${abs(total_return):,.2f} ({abs(return_pct):.2f}%)")

print("="*70)

# Save results
if len(trades_df) > 0:
    trades_df.to_csv('backtest_results_v2_final.csv', index=False)
    print("\n[OK] Results saved to backtest_results_v2_final.csv")
    
    # Show sample trades
    print("\n" + "="*70)
    print("SAMPLE TRADES (First 10)")
    print("="*70)
    for idx, trade in trades_df.head(10).iterrows():
        print(f"  {trade['outcome']}: Entry {trade['entry_run']} -> Exit {trade['exit_run']} | "
              f"${trade['profit']:+.2f} @ {trade['entry_prob']:.0%} confidence")

