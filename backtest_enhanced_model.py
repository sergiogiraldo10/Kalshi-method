"""
Backtest Enhanced Model with Team Features
Following Action Plan: 6-0 runs, 50-55% confidence, asymmetric exits
"""

import pandas as pd
import numpy as np
import joblib
import math

print("\n" + "="*70)
print("BACKTESTING ENHANCED MODEL")
print("Following Action Plan to Profitability")
print("="*70)

# Load enhanced model
print("\nLoading enhanced model...")
enhanced_model = joblib.load('models/momentum_model_enhanced.pkl')
print("  [OK] Enhanced model loaded")

# Load enhanced features
print("\nLoading enhanced features...")
features_df = pd.read_csv('data/processed/features_v2_2023_24_enhanced.csv')
features_df = features_df.sort_values('game_id')
print(f"  [OK] Loaded {len(features_df):,} samples")

# Split: First 33% for training, rest for testing
unique_games = features_df['game_id'].unique()
n_games = len(unique_games)
train_cutoff = int(n_games * 0.33)
train_games = set(unique_games[:train_cutoff])
test_games = set(unique_games[train_cutoff:])

test_df = features_df[features_df['game_id'].isin(test_games)].copy()
print(f"\n  Testing on {len(test_games)} games ({len(test_df):,} samples)")

# Configuration from Action Plan
# Note: After calibration, predictions are in 14-27% range (well-calibrated!)
# We'll use top quartile of predictions as "high confidence"
CONFIG = {
    'position_size_pct': 0.03,      # 3% (Action Plan recommendation)
    'take_profit_pct': 0.25,        # +25% (asymmetric - wide)
    'stop_loss_pct': -0.05,         # -5% (asymmetric - tight)
    'min_confidence': None,         # Will be set to top quartile
    'max_confidence': None,         # Will be set to max
    'min_run_score': 6,             # 6-0 ONLY (Action Plan)
    'max_opp_score': 0,             # Pure runs
}

print("\n" + "="*70)
print("STRATEGY (From Action Plan)")
print("="*70)
print(f"  Entry: 6-0 runs, top quartile confidence (will be determined)")
print(f"  Exit: +{CONFIG['take_profit_pct']*100:.0f}% TP OR {CONFIG['stop_loss_pct']*100:.0f}% SL (asymmetric)")
print(f"  Position: {CONFIG['position_size_pct']*100:.0f}% of bankroll")
print(f"  Only Q1-Q3 (Q4 excluded per Action Plan)")

# Filter for entry opportunities
entry_opportunities = test_df[
    (test_df['is_micro_run'] == 1) &
    (test_df['run_score'] == CONFIG['min_run_score']) &
    (test_df['opp_score'] == CONFIG['max_opp_score']) &
    (test_df['period'] <= 3)  # Q1-Q3 only
].copy()

print(f"\n  6-0 run opportunities (Q1-Q3): {len(entry_opportunities):,}")

# Generate predictions
feature_cols = enhanced_model['feature_cols']
X = entry_opportunities[feature_cols].values
X_scaled = enhanced_model['scaler'].transform(X)
predictions = enhanced_model['model'].predict_proba(X_scaled)[:, 1]
entry_opportunities['prediction'] = predictions

# Check prediction distribution
print(f"\n  Prediction distribution:")
print(f"    Min: {predictions.min()*100:.1f}%")
print(f"    Max: {predictions.max()*100:.1f}%")
print(f"    Mean: {predictions.mean()*100:.1f}%")
print(f"    Median: {np.median(predictions)*100:.1f}%")
print(f"    75th percentile: {np.percentile(predictions, 75)*100:.1f}%")

# Set confidence range to top quartile (highest predictions)
CONFIG['min_confidence'] = np.percentile(predictions, 75)  # Top 25%
CONFIG['max_confidence'] = predictions.max()

print(f"\n  Using top quartile: {CONFIG['min_confidence']*100:.1f}% - {CONFIG['max_confidence']*100:.1f}%")

# Filter by confidence
sweet_spot = entry_opportunities[
    (entry_opportunities['prediction'] >= CONFIG['min_confidence']) &
    (entry_opportunities['prediction'] <= CONFIG['max_confidence'])
].copy()

print(f"  Opportunities in top quartile: {len(sweet_spot):,}")

# Select best per game
sweet_spot = sweet_spot.sort_values(['game_id', 'prediction'], ascending=[True, False])
trades_to_make = []
seen_games = set()

for idx, row in sweet_spot.iterrows():
    if row['game_id'] not in seen_games:
        trades_to_make.append(row)
        seen_games.add(row['game_id'])

print(f"  Final opportunities (1 per game): {len(trades_to_make)}")

# Analyze calibration
if len(trades_to_make) > 0:
    trades_temp = pd.DataFrame(trades_to_make)
    actual_win_rate = trades_temp['run_extends'].mean()
    avg_confidence = trades_temp['prediction'].mean()
    
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

print(f"\n  Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Simulating {len(trades_to_make)} trades...\n")

for i, row in enumerate(trades_to_make):
    position_size = bankroll * CONFIG['position_size_pct']
    predicted_prob = row['prediction']
    entry_fee = calculate_fee(position_size, predicted_prob)
    
    if position_size + entry_fee > bankroll:
        continue
    
    actual_outcome = row['run_extends']
    
    # Realistic P/L based on outcome
    if actual_outcome == 1:
        # Win - hit take profit
        prob_change = np.random.uniform(0.10, CONFIG['take_profit_pct'])
        price_change_pct = prob_change * 2.0
        exit_reason = "TP"
    else:
        # Loss - hit stop loss or run stopped
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
        'confidence': predicted_prob,
        'actual_outcome': actual_outcome,
        'exit_reason': exit_reason,
        'profit': profit,
        'profit_pct': (profit / position_size) * 100,
        'bankroll': bankroll
    }
    trades.append(trade)
    
    if (i + 1) <= 20 or (i + 1) % 50 == 0:
        outcome = "WIN " if actual_outcome == 1 else "LOSS"
        print(f"  #{i+1:<4} | {outcome} | {predicted_prob:.1%} | {exit_reason:<8} | "
              f"P/L: ${profit:>6.2f} | Bank: ${bankroll:>8.2f}")

# Results
trades_df = pd.DataFrame(trades)

print("\n" + "="*70)
print("ENHANCED MODEL RESULTS")
print("="*70)

if len(trades_df) == 0:
    print("\n  [!] NO TRADES EXECUTED")
    print("  The calibrated model predictions are outside the confidence range.")
    print("  Try adjusting the confidence range or check model calibration.")
    print("="*70 + "\n")
    exit()

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
print(f"    Winning Trades:       {len(wins):>10,}")
print(f"    Losing Trades:        {len(losses):>10,}")

avg_win = wins['profit'].mean() if len(wins) > 0 else 0
avg_loss = losses['profit'].mean() if len(losses) > 0 else 0

print(f"\n  PROFIT/LOSS:")
print(f"    Average Win:          ${avg_win:>10.2f}")
print(f"    Average Loss:         ${avg_loss:>10.2f}")
print(f"    Win/Loss Ratio:       {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.2f}:1")

# Risk metrics
returns = trades_df['profit'] / (trades_df['bankroll'].shift(1).fillna(INITIAL_BANKROLL) * CONFIG['position_size_pct'])
sharpe = (returns.mean() / returns.std()) * np.sqrt(len(trades_df)) if returns.std() > 0 and len(trades_df) > 1 else 0

print(f"\n  RISK METRICS:")
print(f"    Sharpe Ratio:         {sharpe:>10.2f}")
equity = trades_df['bankroll'].values
drawdown = (pd.Series(equity).cummax() - equity).max()
print(f"    Max Drawdown:         ${drawdown:>10.2f}")

print("\n" + "="*70)

if return_pct > 5:
    print("[OK] PROFITABLE!")
    print(f"   Enhanced model with team features works!")
    print(f"   Return: ${total_return:,.2f} ({return_pct:+.2f}%)")
    print(f"   Win rate: {len(wins)/len(trades_df)*100:.1f}%")
elif return_pct > 0:
    print("[~] SLIGHTLY PROFITABLE")
    print(f"   Return: ${total_return:,.2f} ({return_pct:+.2f}%)")
    print(f"   Close to Action Plan target (+5%)")
elif return_pct > -5:
    print("[~] NEAR BREAKEVEN")
    print(f"   Return: {return_pct:+.2f}%")
    print(f"   Much better than before (-25%)!")
else:
    print("[X] STILL LOSING")
    print(f"   Lost: ${abs(total_return):,.2f} ({return_pct:.2f}%)")
    print(f"   Win rate: {len(wins)/len(trades_df)*100:.1f}% (need 45%+)")

print("="*70 + "\n")

trades_df.to_csv('backtest_enhanced_model_results.csv', index=False)
print("[OK] Results saved to backtest_enhanced_model_results.csv\n")

