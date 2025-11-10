"""
Analyze Cross-Season Validation Results
Investigate why 2023-24 shows much higher win rate
"""

import pandas as pd
import numpy as np

print("\n" + "="*70)
print("ANALYZING CROSS-SEASON RESULTS")
print("="*70)

# Load results
results_df = pd.read_csv('cross_season_validation_results.csv')

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# Group by selectivity
for selectivity in results_df['selectivity'].unique():
    subset = results_df[results_df['selectivity'] == selectivity]
    
    print(f"\n{selectivity}:")
    print("-" * 70)
    
    for _, row in subset.iterrows():
        print(f"  {row['season'].replace('_', '-')}: {row['win_rate']:.1f}% win rate ({row['trades']} trades)")
    
    # Calculate stats
    print(f"\n  Average: {subset['win_rate'].mean():.1f}%")
    print(f"  Std Dev: {subset['win_rate'].std():.1f}%")
    print(f"  Min: {subset['win_rate'].min():.1f}%")
    print(f"  Max: {subset['win_rate'].max():.1f}%")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# Focus on ultra-selective
ultra = results_df[results_df['selectivity'] == 'Ultra-Selective (Top 10%)']

print("\n2023-24 shows 84.6% win rate vs 38-40% for other seasons.")
print("This is suspicious and likely indicates:")
print("\n1. MODEL TRAINED ON 2023-24 DATA")
print("   - The model was trained on 60% of 2023-24 games")
print("   - Even though we used a train/test split, the model learned")
print("   - patterns specific to that season")
print("\n2. REALISTIC WIN RATES: 2021-22 and 2022-23")
print("   - 2021-22: 38.5% win rate")
print("   - 2022-23: 39.6% win rate")
print("   - These are seasons the model NEVER saw during training")
print("   - **These are the REAL expected win rates**")

print("\n" + "="*70)
print("REALISTIC PERFORMANCE EXPECTATIONS")
print("="*70)

# Calculate realistic stats (excluding 2023-24)
realistic_ultra = ultra[ultra['season'] != '2023_24']

print(f"\nBased on UNSEEN seasons (2021-22, 2022-23):")
print(f"\n  Ultra-Selective (Top 10%):")
print(f"    Average Win Rate: {realistic_ultra['win_rate'].mean():.1f}%")
print(f"    Average Trades: {realistic_ultra['trades'].mean():.0f}")
print(f"    Average Return: +{realistic_ultra['return_pct'].mean():.1f}%")

# Do same for moderate
moderate = results_df[results_df['selectivity'] == 'Moderate (Top 25%)']
realistic_moderate = moderate[moderate['season'] != '2023_24']

print(f"\n  Moderate (Top 25%):")
print(f"    Average Win Rate: {realistic_moderate['win_rate'].mean():.1f}%")
print(f"    Average Trades: {realistic_moderate['trades'].mean():.0f}")
print(f"    Average Return: +{realistic_moderate['return_pct'].mean():.1f}%")

# Balanced
balanced = results_df[results_df['selectivity'] == 'Balanced (Top 30%)']
realistic_balanced = balanced[balanced['season'] != '2023_24']

print(f"\n  Balanced (Top 30%):")
print(f"    Average Win Rate: {realistic_balanced['win_rate'].mean():.1f}%")
print(f"    Average Trades: {realistic_balanced['trades'].mean():.0f}")
print(f"    Average Return: +{realistic_balanced['return_pct'].mean():.1f}%")

print("\n" + "="*70)
print("VERDICT FOR 2025-26 SEASON")
print("="*70)

avg_wr = realistic_ultra['win_rate'].mean()
avg_return = realistic_ultra['return_pct'].mean()

print(f"\nExpected performance for 2025-26:")
print(f"\n  Strategy: Ultra-Selective (Top 10% confidence)")
print(f"  Expected Win Rate: ~{avg_wr:.0f}%")
print(f"  Expected Trades: ~185-195 per season")
print(f"  Expected Return: +{avg_return:.0f}% to +{avg_return*1.2:.0f}%")
print(f"  (with position size optimization)")

if avg_wr >= 38:
    print(f"\n  [OK] STRATEGY IS VIABLE!")
    print(f"  38-40% win rate is above breakeven with proper position sizing")
    print(f"  and asymmetric exits (+25% TP, -5% SL)")
    print(f"\n  Risk-adjusted, this can be profitable if:")
    print(f"  - Position size: 3-5% of bankroll")
    print(f"  - Win/loss ratio maintained at 2.5:1+")
    print(f"  - Fees are kept low")
else:
    print(f"\n  [!] STRATEGY NEEDS IMPROVEMENT")
    print(f"  Win rate below 40% makes it difficult to be consistently profitable")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("""
1. RETRAIN ON ALL HISTORICAL DATA (2015-2023)
   - Train on 2015-2022 (8 seasons)
   - Test on 2022-23 (never seen)
   - Validate on 2023-24
   - This will improve generalization

2. USE REALISTIC WIN RATE EXPECTATIONS
   - Don't expect 68% win rate
   - Expect 38-42% win rate based on out-of-sample testing
   - This is still profitable with proper position sizing

3. FOR 2025-26 SEASON
   - Start with small position sizes (3%)
   - Monitor actual win rate over first 20-30 trades
   - Adjust confidence thresholds based on real results
   - Paper trade first before real money

4. IMPROVE MODEL
   - Add more training data (earlier seasons)
   - Add player-level features
   - Add lineup/rotation data
   - Consider ensemble models
""")

print("="*70 + "\n")

# Save realistic expectations
realistic_summary = {
    'strategy': ['Ultra-Selective', 'Moderate', 'Balanced'],
    'win_rate': [
        realistic_ultra['win_rate'].mean(),
        realistic_moderate['win_rate'].mean(),
        realistic_balanced['win_rate'].mean()
    ],
    'avg_trades': [
        realistic_ultra['trades'].mean(),
        realistic_moderate['trades'].mean(),
        realistic_balanced['trades'].mean()
    ],
    'avg_return_pct': [
        realistic_ultra['return_pct'].mean(),
        realistic_moderate['return_pct'].mean(),
        realistic_balanced['return_pct'].mean()
    ]
}

realistic_df = pd.DataFrame(realistic_summary)
realistic_df.to_csv('realistic_expectations_2025_26.csv', index=False)
print("[OK] Realistic expectations saved to: realistic_expectations_2025_26.csv\n")

