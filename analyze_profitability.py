"""
Deep Analysis: What Makes Trades Profitable?
Test multiple strategies to find profitability
"""
import pandas as pd
import numpy as np
import joblib
import math

print("\n" + "="*70)
print("IGNITION AI - PROFITABILITY ANALYSIS")
print("="*70)

# Load models
momentum_model = joblib.load('models/momentum_model_v2.pkl')
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')

# Filter for qualifying runs (5-0+ pure runs)
entry_opportunities = features_df[
    (features_df['is_micro_run'] == 1) &
    (features_df['run_score'] >= 5) &
    (features_df['opp_score'] == 0)
].copy()

# Predict
feature_cols = momentum_model['feature_cols']
X = entry_opportunities[feature_cols].values
X_scaled = momentum_model['scaler'].transform(X)
predictions = momentum_model['model'].predict_proba(X_scaled)[:, 1]
entry_opportunities['prediction'] = predictions

print(f"\nTotal qualifying opportunities: {len(entry_opportunities):,}")
print(f"Actual extension rate: {entry_opportunities['run_extends'].mean()*100:.1f}%")
print(f"Model prediction avg: {predictions.mean()*100:.1f}%")

# Analyze by confidence buckets
print("\n" + "="*70)
print("ANALYSIS 1: WIN RATE BY CONFIDENCE LEVEL")
print("="*70)
confidence_buckets = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
for conf in confidence_buckets:
    subset = entry_opportunities[entry_opportunities['prediction'] >= conf]
    if len(subset) > 0:
        win_rate = subset['run_extends'].mean() * 100
        count = len(subset)
        # Take best per game
        unique_games = subset.groupby('game_id').size().shape[0]
        print(f"  {conf*100:.0f}%+ confidence: {win_rate:>5.1f}% win rate | {count:>4} opps | {unique_games:>3} games")

# Analyze by run score
print("\n" + "="*70)
print("ANALYSIS 2: WIN RATE BY RUN SCORE")
print("="*70)
for run_score in [5, 6, 7, 8]:
    subset = entry_opportunities[entry_opportunities['run_score'] == run_score]
    if len(subset) > 0:
        win_rate = subset['run_extends'].mean() * 100
        count = len(subset)
        avg_conf = subset['prediction'].mean() * 100
        print(f"  {run_score}-0 runs: {win_rate:>5.1f}% win rate | Avg conf: {avg_conf:.1f}% | {count:>5} opps")

# Analyze by game period
print("\n" + "="*70)
print("ANALYSIS 3: WIN RATE BY QUARTER")
print("="*70)
for period in [1, 2, 3, 4]:
    subset = entry_opportunities[entry_opportunities['period'] == period]
    if len(subset) > 0:
        win_rate = subset['run_extends'].mean() * 100
        count = len(subset)
        avg_conf = subset['prediction'].mean() * 100
        print(f"  Q{period}: {win_rate:>5.1f}% win rate | Avg conf: {avg_conf:.1f}% | {count:>5} opps")

# Analyze by NLP features (defensive pressure)
print("\n" + "="*70)
print("ANALYSIS 4: WIN RATE BY DEFENSIVE PRESSURE")
print("="*70)
if 'defensive_pressure' in entry_opportunities.columns:
    for pressure in [0, 1, 2, 3]:
        subset = entry_opportunities[entry_opportunities['defensive_pressure'] >= pressure]
        if len(subset) > 0:
            win_rate = subset['run_extends'].mean() * 100
            count = len(subset)
            print(f"  {pressure}+ steals/blocks: {win_rate:>5.1f}% win rate | {count:>5} opps")

# Now test multiple strategy configurations
print("\n" + "="*70)
print("STRATEGY TESTING - FINDING PROFITABILITY")
print("="*70)

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

def test_strategy(name, min_conf, min_run, tp_pct, sl_pct, position_pct=0.05):
    """Test a trading strategy configuration"""
    
    # Filter
    filtered = entry_opportunities[
        (entry_opportunities['prediction'] >= min_conf) &
        (entry_opportunities['run_score'] >= min_run)
    ].copy()
    
    # Best per game
    filtered = filtered.sort_values(['game_id', 'prediction'], ascending=[True, False])
    trades_data = []
    seen_games = set()
    
    for idx, row in filtered.iterrows():
        if row['game_id'] not in seen_games:
            trades_data.append(row)
            seen_games.add(row['game_id'])
    
    if len(trades_data) == 0:
        return None
    
    # Simulate
    bankroll = 1000.0
    trades = []
    
    for row in trades_data:
        position_size = bankroll * position_pct
        prob = row['prediction']
        entry_fee = calculate_fee(position_size, prob)
        total_cost = position_size + entry_fee
        
        if total_cost > bankroll:
            continue
        
        actual_outcome = row['run_extends']
        
        if actual_outcome == 1:
            # Win - hit TP
            profit_pct = np.random.uniform(0.10, tp_pct)
            payout = position_size * (1 + profit_pct)
            exit_fee = calculate_fee(payout, prob)
            profit = payout - total_cost - exit_fee
        else:
            # Loss - hit SL or run stopped
            if np.random.random() < 0.25:
                loss_pct = np.random.uniform(sl_pct, sl_pct * 0.8)
            else:
                loss_pct = np.random.uniform(sl_pct * 0.5, -0.02)
            payout = position_size * (1 + loss_pct)
            exit_fee = calculate_fee(payout, prob) if payout > 0 else 0
            profit = payout - total_cost - exit_fee
        
        bankroll += profit
        trades.append({
            'outcome': actual_outcome,
            'profit': profit,
            'position_size': position_size,
            'fees': entry_fee + exit_fee
        })
    
    if len(trades) == 0:
        return None
    
    trades_df = pd.DataFrame(trades)
    total_return = bankroll - 1000.0
    return_pct = (total_return / 1000.0) * 100
    win_rate = trades_df['outcome'].mean() * 100
    avg_win = trades_df[trades_df['outcome'] == 1]['profit'].mean() if (trades_df['outcome'] == 1).any() else 0
    avg_loss = trades_df[trades_df['outcome'] == 0]['profit'].mean() if (trades_df['outcome'] == 0).any() else 0
    
    return {
        'name': name,
        'trades': len(trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'return_pct': return_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_fees': trades_df['fees'].sum()
    }

# Test configurations
strategies = [
    # (name, min_conf, min_run, tp_pct, sl_pct, position_pct)
    ("Ultra Selective", 0.70, 6, 0.20, -0.05, 0.05),
    ("High Confidence", 0.65, 6, 0.20, -0.05, 0.05),
    ("Asymmetric 1", 0.60, 6, 0.25, -0.05, 0.05),
    ("Asymmetric 2", 0.60, 6, 0.30, -0.05, 0.04),
    ("Conservative", 0.65, 7, 0.20, -0.05, 0.03),
    ("Moderate", 0.60, 6, 0.20, -0.05, 0.04),
    ("Quality Over Quantity", 0.75, 6, 0.25, -0.05, 0.06),
    ("Tight Entry", 0.60, 7, 0.20, -0.05, 0.05),
    ("Big TP", 0.65, 6, 0.35, -0.05, 0.03),
    ("Small Position", 0.60, 6, 0.20, -0.05, 0.02),
]

results = []
for name, min_conf, min_run, tp, sl, pos in strategies:
    result = test_strategy(name, min_conf, min_run, tp, sl, pos)
    if result:
        results.append(result)

# Sort by return
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('return_pct', ascending=False)

print("\nTOP STRATEGIES (sorted by return %):\n")
print(f"{'Strategy':<25} {'Trades':<8} {'Win%':<8} {'Return':<12} {'Fees':<10}")
print("-" * 70)
for idx, row in results_df.iterrows():
    status = "[+]" if row['return_pct'] > 0 else "[-]"
    print(f"{status} {row['name']:<22} {row['trades']:<8} {row['win_rate']:<7.1f}% "
          f"${row['total_return']:>7.2f} ({row['return_pct']:>+5.1f}%) ${row['total_fees']:>7.2f}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

best = results_df.iloc[0]
if best['return_pct'] > 0:
    print(f"\n[OK] FOUND PROFITABLE STRATEGY: {best['name']}")
    print(f"     {best['trades']} trades, {best['win_rate']:.1f}% win rate")
    print(f"     Return: ${best['total_return']:.2f} ({best['return_pct']:+.2f}%)")
    print(f"     Avg Win: ${best['avg_win']:.2f} | Avg Loss: ${best['avg_loss']:.2f}")
else:
    print("\n[!] NO PROFITABLE STRATEGY FOUND")
    print("\nRECOMMENDATIONS:")
    print("  1. Model needs improvement - win rate too low")
    print("  2. Consider retraining with:")
    print("     - Team strength features (win%, offensive rating)")
    print("     - Player lineup data (stars on court?)")
    print("     - Recent form (team on back-to-back?)")
    print("  3. Try different run definitions (8-0, 10-2 vs 5-0)")
    print("  4. Lower fees if possible")

print("="*70)

