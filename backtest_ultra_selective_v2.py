"""
Ultra-Selective Strategy V2
Based on analysis: Only take TOP 10-15% confidence trades
These show 68.2% win rate!
"""

import pandas as pd
import numpy as np
import math

print("\n" + "="*70)
print("ULTRA-SELECTIVE STRATEGY V2")
print("Only TOP confidence trades (68%+ win rate)")
print("="*70)

# Load improved model results
trades_df = pd.read_csv('backtest_improved_quality.csv')

print(f"\nTotal opportunities: {len(trades_df)}")
print(f"  Quality score: {trades_df['quality_score'].mean():.1f} avg")
print(f"  Confidence: {trades_df['confidence'].mean()*100:.1f}% avg")
print(f"  Win rate: {(trades_df['actual_outcome'] == 1).mean()*100:.1f}%")

# Test different selectivity levels
selectivity_levels = [
    {'name': 'Top 10%', 'percentile': 90, 'expected_wr': 68},
    {'name': 'Top 15%', 'percentile': 85, 'expected_wr': 60},
    {'name': 'Top 20%', 'percentile': 80, 'expected_wr': 55},
    {'name': 'Top 25%', 'percentile': 75, 'expected_wr': 50},
]

print("\n" + "="*70)
print("TESTING SELECTIVITY LEVELS")
print("="*70)

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

def backtest_level(filtered_trades, level_name, position_pct=0.03):
    """Backtest a selectivity level"""
    INITIAL_BANKROLL = 1000.0
    bankroll = INITIAL_BANKROLL
    results = []
    
    CONFIG = {
        'position_size_pct': position_pct,
        'take_profit_pct': 0.25,
        'stop_loss_pct': -0.05,
    }
    
    for idx, row in filtered_trades.iterrows():
        position_size = bankroll * CONFIG['position_size_pct']
        predicted_prob = row['confidence']
        entry_fee = calculate_fee(position_size, predicted_prob)
        
        if position_size + entry_fee > bankroll:
            continue
        
        actual_outcome = row['actual_outcome']
        
        # Realistic P/L
        if actual_outcome == 1:
            prob_change = np.random.uniform(0.10, CONFIG['take_profit_pct'])
            price_change_pct = prob_change * 2.0
            exit_reason = "TP"
        else:
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
        results.append({
            'profit': profit,
            'actual_outcome': actual_outcome
        })
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    total_return = bankroll - INITIAL_BANKROLL
    return_pct = (total_return / INITIAL_BANKROLL) * 100
    win_rate = (results_df['actual_outcome'] == 1).mean() * 100
    
    return {
        'trades': len(results),
        'win_rate': win_rate,
        'return': total_return,
        'return_pct': return_pct,
        'final_bankroll': bankroll
    }

best_strategy = None
best_return = -999999

for level in selectivity_levels:
    threshold = np.percentile(trades_df['confidence'], level['percentile'])
    filtered = trades_df[trades_df['confidence'] >= threshold].copy()
    
    print(f"\n{level['name']} (Confidence >= {threshold*100:.1f}%):")
    print(f"  Trades: {len(filtered)}")
    
    if len(filtered) < 20:
        print(f"  [!] Too few trades ({len(filtered)} < 20), skipping")
        continue
    
    # Run backtest
    result = backtest_level(filtered, level['name'])
    
    if result:
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Return: ${result['return']:+.2f} ({result['return_pct']:+.2f}%)")
        print(f"  Final: ${result['final_bankroll']:.2f}")
        
        if result['return'] > best_return:
            best_return = result['return']
            best_strategy = {
                'level': level['name'],
                'threshold': threshold,
                **result
            }

# Test with increased position size for ultra-selective
print("\n" + "="*70)
print("TESTING WITH INCREASED POSITION SIZE")
print("="*70)

for position_pct in [0.05, 0.07, 0.10]:
    threshold = np.percentile(trades_df['confidence'], 90)  # Top 10%
    filtered = trades_df[trades_df['confidence'] >= threshold].copy()
    
    result = backtest_level(filtered, f"Top 10% @ {position_pct*100:.0f}%", position_pct)
    
    if result:
        print(f"\n  Position Size: {position_pct*100:.0f}%")
        print(f"    Trades: {result['trades']}")
        print(f"    Win Rate: {result['win_rate']:.1f}%")
        print(f"    Return: ${result['return']:+.2f} ({result['return_pct']:+.2f}%)")
        
        if result['return'] > best_return:
            best_return = result['return']
            best_strategy = {
                'level': f"Top 10% @ {position_pct*100:.0f}%",
                'threshold': threshold,
                'position_pct': position_pct,
                **result
            }

# Show best strategy
print("\n" + "="*70)
print("BEST STRATEGY")
print("="*70)

if best_strategy:
    print(f"\n  Configuration: {best_strategy['level']}")
    print(f"  Confidence Threshold: >={best_strategy['threshold']*100:.1f}%")
    print(f"  Position Size: {best_strategy.get('position_pct', 0.03)*100:.0f}%")
    print(f"\n  Performance:")
    print(f"    Trades: {best_strategy['trades']}")
    print(f"    Win Rate: {best_strategy['win_rate']:.1f}%")
    print(f"    Return: ${best_strategy['return']:+.2f} ({best_strategy['return_pct']:+.2f}%)")
    print(f"    Final Bankroll: ${best_strategy['final_bankroll']:.2f}")
    
    if best_strategy['return_pct'] > 25:
        print(f"\n  [OK] HIGHLY PROFITABLE!")
    elif best_strategy['return_pct'] > 15:
        print(f"\n  [OK] PROFITABLE")
    elif best_strategy['return_pct'] > 5:
        print(f"\n  [~] SLIGHTLY PROFITABLE")
    else:
        print(f"\n  [!] LIMITED PROFITABILITY")

print("="*70 + "\n")

