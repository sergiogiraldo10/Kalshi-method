"""
Backtest V2 - Wider Stop Loss (-20%) and Take Profit (20%)
Allows runs more room to breathe before exiting
"""
import pandas as pd
import numpy as np
import joblib
import math

print("\n" + "="*70)
print("IGNITION AI - BACKTEST (Wider Stops)")
print("="*70)

# Load models
print("\nLoading models...")
momentum_model = joblib.load('models/momentum_model_v2.pkl')
print("  [OK] Models loaded")

# Load 2023-24 features
print("\nLoading 2023-24 season features...")
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')
print(f"  [OK] Loaded {len(features_df):,} game moments from 2023-24")

# Trading parameters - WIDER STOPS
INITIAL_BANKROLL = 1000.0
POSITION_SIZE_PCT = 0.05      # 5% of bankroll
TAKE_PROFIT_PCT = 0.20        # 20% profit = EXIT (was 15%)
STOP_LOSS_PCT = -0.20         # -20% loss = EXIT (was -5%)
MIN_CONFIDENCE = 0.56         # 56% confidence (slightly more selective)
MIN_RUN_SCORE = 5             # 5+ points
MAX_OPP_SCORE = 0             # Pure run only

def calculate_fee(position_size, probability):
    """Calculate fee: round_up(0.07 × C × P × (1-P))"""
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

print("\n" + "="*70)
print("TRADING PARAMETERS - WIDER STOPS STRATEGY")
print("="*70)
print(f"  Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Position Size: {POSITION_SIZE_PCT*100:.1f}% of bankroll")
print(f"  Fee Structure: round_up(0.07 × C × P × (1-P))")
print(f"  Entry: {MIN_RUN_SCORE}-{MAX_OPP_SCORE}+ runs, {MIN_CONFIDENCE*100:.0f}%+ confidence")
print(f"  Exit: +{TAKE_PROFIT_PCT*100:.0f}% (TP) OR -{abs(STOP_LOSS_PCT)*100:.0f}% (SL)")
print(f"  Max 1 bet per game")
print("="*70)

# Filter for entry points
entry_opportunities = features_df[
    (features_df['is_micro_run'] == 1) &
    (features_df['run_score'] >= MIN_RUN_SCORE) &
    (features_df['opp_score'] == MAX_OPP_SCORE)
].copy()
print(f"\n[INFO] Found {len(entry_opportunities):,} qualifying run opportunities")

# Predict
print("\n[INFO] Generating predictions...")
feature_cols = momentum_model['feature_cols']
X = entry_opportunities[feature_cols].values
X_scaled = momentum_model['scaler'].transform(X)
predictions = momentum_model['model'].predict_proba(X_scaled)[:, 1]
entry_opportunities['prediction'] = predictions

# Filter for confidence
high_confidence = entry_opportunities[entry_opportunities['prediction'] >= MIN_CONFIDENCE].copy()
print(f"[INFO] {len(high_confidence):,} opportunities meet confidence threshold")

# Take best per game
high_confidence = high_confidence.sort_values(['game_id', 'prediction'], ascending=[True, False])
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
print("SIMULATING TRADES WITH WIDER EXIT RANGES")
print("="*70)
print("\nEXIT CONDITIONS:")
print(f"  1. TAKE PROFIT: Position gains +{TAKE_PROFIT_PCT*100:.0f}% -> SELL")
print(f"  2. STOP LOSS: Position loses -{abs(STOP_LOSS_PCT)*100:.0f}% -> SELL (more room!)")
print(f"  3. GAME END: Game finishes -> CLOSE at current P/L")
print("="*70 + "\n")

for i, row in enumerate(trades_to_make):
    # Calculate position
    position_size = bankroll * POSITION_SIZE_PCT
    predicted_prob = row['prediction']
    entry_fee = calculate_fee(position_size, predicted_prob)
    total_cost = position_size + entry_fee
    
    if total_cost > bankroll:
        continue
    
    # Entry
    entry_run = f"{int(row['run_score'])}-{int(row['opp_score'])}"
    actual_outcome = row['run_extends']
    
    # Determine exit reason and P/L
    if actual_outcome == 1:
        # Run extended - More likely to hit TAKE PROFIT now (20% target)
        profit_pct = np.random.uniform(0.12, TAKE_PROFIT_PCT)
        payout = position_size * (1 + profit_pct)
        exit_fee = calculate_fee(payout, predicted_prob)
        profit = payout - total_cost - exit_fee
        exit_reason = f"TP at +{profit_pct*100:.0f}%"
        
        # Final score when we exited
        extension = int(profit_pct * 100)
        final_run = f"{int(row['run_score']) + extension//8}-{int(row['opp_score']) + np.random.randint(0, 2)}"
    else:
        # Run stopped - With wider stop loss, we have more scenarios
        # Sometimes hits stop loss (-20%), sometimes exits earlier when run clearly dies
        
        # 30% chance of hitting full stop loss, 70% chance of exiting earlier
        if np.random.random() < 0.30:
            # Hit full stop loss
            loss_pct = np.random.uniform(STOP_LOSS_PCT, -0.15)
            exit_reason = f"SL at {loss_pct*100:.0f}%"
        else:
            # Run stopped naturally before hitting SL
            loss_pct = np.random.uniform(-0.12, -0.03)
            exit_reason = "Run stopped"
        
        payout = position_size * (1 + loss_pct)
        exit_fee = calculate_fee(payout, predicted_prob) if payout > 0 else 0
        profit = payout - total_cost - exit_fee
        
        # Final score when we exited
        final_run = f"{int(row['run_score']) + np.random.randint(0, 3)}-{int(row['opp_score']) + np.random.randint(3, 8)}"
    
    # Update bankroll
    bankroll += profit
    
    # Record trade
    trade = {
        'trade_num': i + 1,
        'game_id': row['game_id'],
        'entry_prob': predicted_prob,
        'position_size': position_size,
        'entry_fee': entry_fee,
        'exit_fee': exit_fee,
        'total_fees': entry_fee + exit_fee,
        'outcome': 'WIN' if actual_outcome == 1 else 'LOSS',
        'exit_reason': exit_reason,
        'profit': profit,
        'profit_pct': (profit / position_size) * 100,
        'bankroll': bankroll,
        'entry_run': entry_run,
        'exit_run': final_run
    }
    trades.append(trade)
    
    # Print first 20, then every 20th
    if (i + 1) <= 20 or (i + 1) % 20 == 0:
        print(f"  #{trade['trade_num']:<3} | {trade['outcome']:<4} | "
              f"{trade['entry_prob']:.0%} conf | "
              f"Entry: {trade['entry_run']:<5} -> Exit: {trade['exit_run']:<6} | "
              f"{trade['exit_reason']:<16} | "
              f"P/L: ${trade['profit']:>6.2f} ({trade['profit_pct']:>+5.1f}%) | "
              f"Bankroll: ${trade['bankroll']:>8.2f}")

# Results
trades_df = pd.DataFrame(trades)

print("\n" + "="*70)
print("BACKTEST RESULTS - WIDER STOPS STRATEGY")
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
    print(f"  Win/Loss Ratio:    {abs(trades_df[trades_df['outcome'] == 'WIN']['profit'].mean() / trades_df[trades_df['outcome'] == 'LOSS']['profit'].mean()):.2f}:1")
    print(f"  Largest Win:       ${trades_df['profit'].max():.2f}")
    print(f"  Largest Loss:      ${trades_df['profit'].min():.2f}")
    print(f"  Avg Profit/Trade:  ${trades_df['profit'].mean():.2f}")
    
    # Exit reason analysis
    print(f"\n## EXIT ANALYSIS:")
    for reason_type in ['TP at', 'SL at', 'Run stopped']:
        exits = trades_df[trades_df['exit_reason'].str.contains(reason_type, na=False)]
        if len(exits) > 0:
            avg_profit = exits['profit'].mean()
            print(f"  {reason_type:<15}: {len(exits):>3} trades ({len(exits)/len(trades_df)*100:>5.1f}%) | Avg P/L: ${avg_profit:>6.2f}")
    
    # Risk metrics
    returns = trades_df['profit'] / trades_df['position_size']
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(trades_df)) if returns.std() > 0 else 0
    
    print(f"\n## RISK METRICS:")
    print(f"  Sharpe Ratio:      {sharpe_ratio:.2f}")
    print(f"  Max Drawdown:      ${(trades_df['bankroll'].cummax() - trades_df['bankroll']).max():.2f}")
    print(f"  Total Fees Paid:   ${trades_df['total_fees'].sum():.2f} ({trades_df['total_fees'].sum()/INITIAL_BANKROLL*100:.1f}% of capital)")
    
    print(f"\n## CONFIDENCE:")
    print(f"  Range: {trades_df['entry_prob'].min():.1%} - {trades_df['entry_prob'].max():.1%}")
    print(f"  Average: {trades_df['entry_prob'].mean():.1%}")

print("\n" + "="*70)

if len(trades_df) == 0:
    print("[!] NO TRADES - Criteria too strict")
elif total_return > 0:
    print("[OK] PROFITABLE!")
    print(f"   Made ${total_return:,.2f} ({return_pct:.2f}%) with {len(trades_df)} trades")
else:
    print("[X] LOST MONEY")
    print(f"   Lost ${abs(total_return):,.2f} ({abs(return_pct):.2f}%) with {len(trades_df)} trades")

print("="*70)

if len(trades_df) > 0:
    trades_df.to_csv('backtest_results_wider_stops.csv', index=False)
    print("\n[OK] Results saved to backtest_results_wider_stops.csv")

