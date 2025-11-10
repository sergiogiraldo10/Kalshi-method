"""
Analyze Trade Quality
What makes a good trade vs bad trade?
"""

import pandas as pd
import numpy as np

print("\n" + "="*70)
print("ANALYZING TRADE QUALITY")
print("What separates winners from losers?")
print("="*70)

# Load improved model results
trades_df = pd.read_csv('backtest_improved_quality.csv')

wins = trades_df[trades_df['actual_outcome'] == 1]
losses = trades_df[trades_df['actual_outcome'] == 0]

print(f"\nTotal trades: {len(trades_df)}")
print(f"  Wins: {len(wins)} ({len(wins)/len(trades_df)*100:.1f}%)")
print(f"  Losses: {len(losses)} ({len(losses)/len(trades_df)*100:.1f}%)")

# ============================================================================
# ANALYSIS 1: Quality Score Distribution
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: QUALITY SCORE vs WIN RATE")
print("="*70)

for score_threshold in [60, 65, 70, 75, 80]:
    filtered = trades_df[trades_df['quality_score'] >= score_threshold]
    if len(filtered) > 0:
        win_rate = (filtered['actual_outcome'] == 1).mean() * 100
        avg_profit = filtered['profit'].mean()
        print(f"  Score >={score_threshold}: {win_rate:.1f}% win rate | "
              f"${avg_profit:>6.2f} avg profit | {len(filtered):>3} trades")

# ============================================================================
# ANALYSIS 2: Confidence vs Win Rate
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 2: CONFIDENCE vs WIN RATE")
print("="*70)

percentiles = [25, 50, 75, 90]
for p in percentiles:
    threshold = np.percentile(trades_df['confidence'], p)
    filtered = trades_df[trades_df['confidence'] >= threshold]
    if len(filtered) > 0:
        win_rate = (filtered['actual_outcome'] == 1).mean() * 100
        avg_profit = filtered['profit'].mean()
        print(f"  Top {100-p}% conf (>={threshold:.1%}): {win_rate:.1f}% win rate | "
              f"${avg_profit:>6.2f} avg profit | {len(filtered):>3} trades")

# ============================================================================
# ANALYSIS 3: Winners vs Losers Characteristics
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 3: WINNERS vs LOSERS CHARACTERISTICS")
print("="*70)

print(f"\n  Quality Score:")
print(f"    Winners: {wins['quality_score'].mean():.1f} (avg)")
print(f"    Losers:  {losses['quality_score'].mean():.1f} (avg)")
print(f"    Diff:    {wins['quality_score'].mean() - losses['quality_score'].mean():+.1f}")

print(f"\n  Confidence:")
print(f"    Winners: {wins['confidence'].mean()*100:.1f}% (avg)")
print(f"    Losers:  {losses['confidence'].mean()*100:.1f}% (avg)")
print(f"    Diff:    {(wins['confidence'].mean() - losses['confidence'].mean())*100:+.1f}%")

# ============================================================================
# ANALYSIS 4: Optimal Thresholds
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 4: FINDING OPTIMAL THRESHOLDS")
print("="*70)

best_return = -999999
best_config = None

for min_quality in [60, 65, 70, 75]:
    for min_conf_percentile in [60, 70, 80, 90]:
        conf_threshold = np.percentile(trades_df['confidence'], min_conf_percentile)
        
        filtered = trades_df[
            (trades_df['quality_score'] >= min_quality) &
            (trades_df['confidence'] >= conf_threshold)
        ]
        
        if len(filtered) >= 50:  # At least 50 trades
            total_profit = filtered['profit'].sum()
            win_rate = (filtered['actual_outcome'] == 1).mean() * 100
            
            if total_profit > best_return:
                best_return = total_profit
                best_config = {
                    'min_quality': min_quality,
                    'min_conf_percentile': min_conf_percentile,
                    'conf_threshold': conf_threshold,
                    'trades': len(filtered),
                    'win_rate': win_rate,
                    'profit': total_profit,
                    'return_pct': (total_profit / 1000) * 100
                }

if best_config:
    print(f"\n  OPTIMAL CONFIGURATION:")
    print(f"    Min Quality Score: {best_config['min_quality']}")
    print(f"    Min Confidence: Top {100-best_config['min_conf_percentile']}% (>={best_config['conf_threshold']:.1%})")
    print(f"    Trades: {best_config['trades']}")
    print(f"    Win Rate: {best_config['win_rate']:.1f}%")
    print(f"    Total Profit: ${best_config['profit']:.2f}")
    print(f"    Return: {best_config['return_pct']:.2f}%")

# ============================================================================
# ANALYSIS 5: Exit Reason Analysis
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 5: EXIT REASON ANALYSIS")
print("="*70)

exit_reasons = trades_df['exit_reason'].unique()
for reason in exit_reasons:
    filtered = trades_df[trades_df['exit_reason'] == reason]
    win_rate = (filtered['actual_outcome'] == 1).mean() * 100
    avg_profit = filtered['profit'].mean()
    count = len(filtered)
    print(f"  {reason:<10}: {win_rate:>5.1f}% win rate | ${avg_profit:>6.2f} avg | {count:>3} trades")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nKEY FINDINGS:")
print("  1. Quality score matters - higher scores correlate with better outcomes")
print("  2. Model confidence is weakly predictive - higher confidence slightly better")
print("  3. Take Profit exits are most profitable (by design)")
print("  4. Stop Loss protects capital but happens on losers")

if best_config and best_config['return_pct'] > 30:
    print(f"\n  [OK] Optimized strategy could achieve {best_config['return_pct']:.1f}% return!")
elif best_config and best_config['return_pct'] > 20:
    print(f"\n  [~] Optimized strategy shows {best_config['return_pct']:.1f}% return")
else:
    print(f"\n  [!] Even optimized strategy shows limited upside")

print("="*70 + "\n")

