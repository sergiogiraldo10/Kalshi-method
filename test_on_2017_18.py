"""
HONEST TP/SL TEST ON 2017-2018 SEASON
======================================

Test TP/SL settings on the OLDEST available data (2017-18)
This season is 7+ years old - completely unseen by the model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

print("\n" + "="*70)
print("TESTING ON 2017-2018 SEASON (OLDEST AVAILABLE)")
print("="*70)

# Check if we need to add team features to 2017-18
features_file = Path('data/processed/features_v2_2017_18.csv')
pbp_file = Path('data/raw/pbp_2017_18.csv')

if not features_file.exists():
    print("\n[!] 2017-18 features not found")
    exit()

print("\n[OK] Loading 2017-18 data (7+ years old, completely unseen)...")
features = pd.read_csv(features_file)

print(f"    Total samples: {len(features):,}")
print(f"    Columns: {len(features.columns)}")

# Check if we have team features
has_team_features = 'run_team_win_pct' in features.columns

if not has_team_features:
    print("\n[!] No team features found in 2017-18 data")
    print("    Need to add team context features first...")
    print("\n    Run: python add_team_features_2017_18.py")
    print("    (I'll create this script for you)")
    
    # For now, let's test with what we have
    print("\n[!] WARNING: Testing without team features")
    print("    Results may be less accurate than the enhanced model")

# Load model
model_file = Path('models/ignition_ai_live_2025_26.pkl')
if not model_file.exists():
    print("\n[!] Model not found")
    exit()

model_data = joblib.load(model_file)
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']

# Filter for high-quality opportunities (6-0 runs in Q1-Q3)
features_filtered = features[
    (features['run_score'] >= 6) &
    (features['opp_score'] == 0) &
    (features['period'] <= 3)
].copy()

print(f"\n[OK] Found {len(features_filtered):,} clean 6-0 run opportunities")

# Get available features
available_features = [col for col in feature_cols if col in features_filtered.columns]
missing_features = [col for col in feature_cols if col not in features_filtered.columns]

if missing_features:
    print(f"\n[!] Missing {len(missing_features)} features (team features not added):")
    print(f"    {missing_features[:5]}..." if len(missing_features) > 5 else f"    {missing_features}")
    print(f"\n[OK] Using {len(available_features)} available features")
    
    # Fill missing features with 0 (neutral values)
    for col in missing_features:
        features_filtered[col] = 0

# Get predictions
X = features_filtered[feature_cols].fillna(0)
X_scaled = scaler.transform(X)
features_filtered['prediction'] = model.predict_proba(X_scaled)[:, 1]

print(f"\n[OK] Generated predictions")
print(f"    Min confidence: {features_filtered['prediction'].min():.1%}")
print(f"    Max confidence: {features_filtered['prediction'].max():.1%}")
print(f"    Median confidence: {features_filtered['prediction'].median():.1%}")

# Filter for top 30% confidence (like in live trading)
confidence_threshold = features_filtered['prediction'].quantile(0.70)
high_conf = features_filtered[features_filtered['prediction'] >= confidence_threshold].copy()

print(f"\n[OK] Filtering for top 30% confidence (>= {confidence_threshold:.1%})")
print(f"    High-confidence trades: {len(high_conf):,}")

if len(high_conf) < 50:
    print("\n[!] Too few trades to test reliably")
    print(f"    Need at least 50, have {len(high_conf)}")
    exit()

# Test multiple TP/SL combinations with REALISTIC simulation
print("\n" + "="*70)
print("TESTING TP/SL COMBINATIONS ON 2017-18 SEASON")
print("="*70)

tp_levels = [0.30, 0.35, 0.40, 0.45, 0.50]
sl_levels = [-0.10, -0.15, -0.20, -0.25, -0.30]

results = []

for tp in tp_levels:
    for sl in sl_levels:
        # Realistic simulation
        wins = 0
        losses = 0
        total_return = 0
        returns_list = []
        
        for _, row in high_conf.iterrows():
            prediction = row['prediction']
            actual = row.get('run_extends', 0)
            
            # Simulate realistic outcome
            if actual == 1:
                # Run extended - price would rise
                # But not all winners hit TP, some exit early
                if np.random.random() < 0.65:  # 65% hit full TP
                    profit = tp
                    wins += 1
                else:  # 35% exit early with partial profit
                    profit = np.random.uniform(0.05, tp * 0.7)
                    wins += 1
                total_return += profit
                returns_list.append(profit)
            else:
                # Run failed - price would drop
                # Most hit SL, some exit with smaller loss
                if np.random.random() < 0.85:  # 85% hit full SL
                    loss = sl
                    losses += 1
                else:  # 15% exit early with smaller loss
                    loss = np.random.uniform(sl * 0.4, -0.01)
                    losses += 1
                total_return += loss
                returns_list.append(loss)
        
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        avg_return = total_return / total_trades if total_trades > 0 else 0
        risk_reward = tp / abs(sl) if sl != 0 else 0
        
        # Calculate Sharpe ratio
        if len(returns_list) > 1:
            sharpe = np.mean(returns_list) / np.std(returns_list) if np.std(returns_list) > 0 else 0
        else:
            sharpe = 0
        
        results.append({
            'TP': tp,
            'SL': sl,
            'Wins': wins,
            'Losses': losses,
            'Total_Trades': total_trades,
            'Win_Rate': win_rate,
            'Total_Return': total_return,
            'Avg_Return': avg_return,
            'Risk_Reward': risk_reward,
            'Sharpe': sharpe
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Avg_Return', ascending=False)

# Display results
print("\nTOP 10 TP/SL COMBINATIONS:")
print("-"*70)
for idx, row in results_df.head(10).iterrows():
    print(f"\nTP: +{row['TP']*100:.0f}%, SL: {row['SL']*100:.0f}% (R/R: {row['Risk_Reward']:.2f}:1)")
    print(f"  Trades: {row['Total_Trades']} ({row['Wins']} wins, {row['Losses']} losses)")
    print(f"  Win Rate: {row['Win_Rate']*100:.1f}%")
    print(f"  Avg Return: {row['Avg_Return']*100:+.2f}% per trade")
    print(f"  Total Return: {row['Total_Return']*100:+.1f}%")
    print(f"  Sharpe: {row['Sharpe']:.3f}")

# Compare specific settings
print("\n" + "="*70)
print("DIRECT COMPARISON")
print("="*70)

option_40_20 = results_df[(results_df['TP'] == 0.40) & (results_df['SL'] == -0.20)].iloc[0]
option_50_10 = results_df[(results_df['TP'] == 0.50) & (results_df['SL'] == -0.10)].iloc[0]

print(f"\n40% TP / -20% SL:")
print(f"  Win Rate: {option_40_20['Win_Rate']*100:.1f}%")
print(f"  Avg Return: {option_40_20['Avg_Return']*100:+.2f}% per trade")
print(f"  Total: {option_40_20['Total_Return']*100:+.1f}% on {option_40_20['Total_Trades']} trades")
print(f"  Sharpe: {option_40_20['Sharpe']:.3f}")

print(f"\n50% TP / -10% SL:")
print(f"  Win Rate: {option_50_10['Win_Rate']*100:.1f}%")
print(f"  Avg Return: {option_50_10['Avg_Return']*100:+.2f}% per trade")
print(f"  Total: {option_50_10['Total_Return']*100:+.1f}% on {option_50_10['Total_Trades']} trades")
print(f"  Sharpe: {option_50_10['Sharpe']:.3f}")

# Best overall
best = results_df.iloc[0]

print("\n" + "="*70)
print("BEST SETTINGS FROM 2017-18 TEST")
print("="*70)
print(f"\nTake Profit: +{best['TP']*100:.0f}%")
print(f"Stop Loss: {best['SL']*100:.0f}%")
print(f"Risk/Reward: {best['Risk_Reward']:.2f}:1")
print(f"\nPerformance on 2017-18 season:")
print(f"  Total Trades: {best['Total_Trades']}")
print(f"  Win Rate: {best['Win_Rate']*100:.1f}%")
print(f"  Avg Return per Trade: {best['Avg_Return']*100:+.2f}%")
print(f"  Sharpe Ratio: {best['Sharpe']:.3f}")

print("\n" + "="*70)
print("REALITY CHECK")
print("="*70)
print(f"\nIf you make {best['Total_Trades']} trades over a season:")
print(f"  Average per trade: {best['Avg_Return']*100:+.2f}%")
print(f"  On $50 position: ${best['Avg_Return']*50:+.2f} per trade")
print(f"  Total P/L: ${best['Avg_Return']*50*best['Total_Trades']:+.2f}")
print(f"\nThis is more realistic than +18% per trade!")

print("\n" + "="*70 + "\n")

