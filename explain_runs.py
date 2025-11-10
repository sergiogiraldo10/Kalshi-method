"""
Analyze what "runs" we're actually detecting in real games
"""
import pandas as pd
import numpy as np

# Load some test season features
df = pd.read_csv('data/processed/features_2020_21.csv')

print("="*60)
print("UNDERSTANDING MOMENTUM RUNS")
print("="*60)

# Look at micro-runs (4+ points)
micro_runs = df[df['is_micro_run'] == 1].copy()
print(f"\n1. MICRO-RUNS (4+ points in 2 minutes)")
print(f"   Total samples: {len(micro_runs):,}")
print(f"   Of all game moments: {len(micro_runs)/len(df)*100:.1f}%")

print(f"\n   Distribution of run sizes:")
print(micro_runs['max_current_run'].describe())

print(f"\n   Top run sizes:")
print(micro_runs['max_current_run'].value_counts().head(20).sort_index(ascending=False))

# Look at runs that extended
extended_runs = micro_runs[micro_runs['run_extends'] == 1]
print(f"\n2. RUNS THAT EXTENDED")
print(f"   Total: {len(extended_runs):,} ({len(extended_runs)/len(micro_runs)*100:.1f}% of micro-runs)")

print(f"\n   Initial run size distribution (before extension):")
print(extended_runs['max_current_run'].value_counts().head(15).sort_index(ascending=False))

print(f"\n3. WHAT MAKES A RUN \"EXTEND\"?")
print(f"   Current criteria:")
print(f"   - Momentum team scores 4+ more points in next 20 plays")
print(f"   - Total run becomes 8+ points")
print(f"   - Momentum team outscores opponent 1.5:1 OR opponent scores <=4")
print(f"")
print(f"   Examples of runs that extend: 4-0 -> 8-2, 6-0 -> 10-4, 8-2 -> 12-4")
print(f"   Examples of runs that stop: 4-0 -> 6-6, 6-0 -> 8-8, 4-2 -> 6-8")

# Look at game context
print(f"\n4. WHEN DO RUNS HAPPEN?")
print(f"\n   Close games (score diff <= 5):")
print(f"   {(extended_runs['is_close_game'] == 1).sum():,} / {len(extended_runs):,} extended runs")

print(f"\n   Clutch time (last 5 min, close game):")
print(f"   {(extended_runs['is_clutch_time'] == 1).sum():,} / {len(extended_runs):,} extended runs")

print(f"\n   By period:")
for period in sorted(extended_runs['period'].unique()):
    count = (extended_runs['period'] == period).sum()
    print(f"   Period {int(period)}: {count:,} extended runs")

print("\n" + "="*60)

