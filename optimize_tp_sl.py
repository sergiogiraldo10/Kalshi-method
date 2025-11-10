"""
OPTIMIZE TAKE PROFIT AND STOP LOSS
===================================

Simulate different TP/SL combinations on historical data
to find the most profitable settings for Kalshi trading
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

print("\n" + "="*70)
print("OPTIMIZING TAKE PROFIT & STOP LOSS FOR KALSHI")
print("="*70)

# Load enhanced features and model
features_file = Path('data/processed/features_v2_2023_24_enhanced.csv')
model_file = Path('models/ignition_ai_live_2025_26.pkl')

if not features_file.exists():
    print("\n[!] Features file not found")
    exit()

if not model_file.exists():
    print("\n[!] Model file not found")
    exit()

print("\n[OK] Loading data...")
features = pd.read_csv(features_file)
model_data = joblib.load(model_file)
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']

# Filter for high-quality opportunities
features = features[
    (features['run_score'] >= 6) &
    (features['opp_score'] == 0) &
    (features['period'] <= 3)
].copy()

print(f"[OK] Found {len(features):,} opportunities to test")

# Get predictions
X = features[feature_cols].fillna(0)
X_scaled = scaler.transform(X)
features['prediction'] = model.predict_proba(X_scaled)[:, 1]

# Filter for top confidence (like in live trading)
confidence_threshold = np.percentile(features['prediction'], 70)
features = features[features['prediction'] >= confidence_threshold].copy()

print(f"[OK] Testing on {len(features):,} high-confidence trades")

# Test different TP/SL combinations
tp_levels = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  # +25% to +50%
sl_levels = [-0.10, -0.15, -0.20, -0.25, -0.30]  # -10% to -30%

print("\n" + "="*70)
print("TESTING TP/SL COMBINATIONS")
print("="*70)

results = []

for tp in tp_levels:
    for sl in sl_levels:
        # Simulate trades
        total_return = 0
        wins = 0
        losses = 0
        
        for _, row in features.iterrows():
            actual = row['run_extends']
            
            # Simulate outcome
            if actual == 1:
                # Win - take profit
                total_return += tp
                wins += 1
            else:
                # Loss - stop loss
                total_return += sl
                losses += 1
        
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        avg_return = total_return / total_trades if total_trades > 0 else 0
        risk_reward = tp / abs(sl) if sl != 0 else 0
        
        results.append({
            'TP': tp,
            'SL': sl,
            'Wins': wins,
            'Losses': losses,
            'Win_Rate': win_rate,
            'Total_Return': total_return,
            'Avg_Return': avg_return,
            'Risk_Reward': risk_reward,
            'Score': avg_return * win_rate  # Combined metric
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Score', ascending=False)

# Show top 10 combinations
print("\nTOP 10 TP/SL COMBINATIONS:")
print("-"*70)
for idx, row in results_df.head(10).iterrows():
    print(f"\nTP: +{row['TP']*100:.0f}%, SL: {row['SL']*100:.0f}% (R/R: {row['Risk_Reward']:.2f}:1)")
    print(f"  Win Rate: {row['Win_Rate']*100:.1f}% ({row['Wins']}/{row['Wins']+row['Losses']} trades)")
    print(f"  Avg Return: {row['Avg_Return']*100:+.2f}% per trade")
    print(f"  Total Return: {row['Total_Return']*100:+.1f}%")
    print(f"  Score: {row['Score']:.4f}")

# Best overall
best = results_df.iloc[0]

print("\n" + "="*70)
print("RECOMMENDED SETTINGS FOR KALSHI")
print("="*70)
print(f"\nTake Profit: +{best['TP']*100:.0f}%")
print(f"Stop Loss: {best['SL']*100:.0f}%")
print(f"Risk/Reward: {best['Risk_Reward']:.2f}:1")
print(f"\nExpected Performance:")
print(f"  Win Rate: {best['Win_Rate']*100:.1f}%")
print(f"  Avg Return per Trade: {best['Avg_Return']*100:+.2f}%")
print(f"  Total Return (100 trades): {best['Avg_Return']*100*100:+.1f}%")

print("\n" + "="*70)
print("ENTRY PRICE EXAMPLES")
print("="*70)

entry_prices = [35, 40, 45, 50]
for entry in entry_prices:
    tp_price = int(entry * (1 + best['TP']))
    sl_price = int(entry * (1 + best['SL']))
    print(f"\nEntry @ {entry}¢:")
    print(f"  Take Profit: {tp_price}¢ (sell at this price)")
    print(f"  Stop Loss: {sl_price}¢ (sell at this price)")

print("\n" + "="*70)

# Save optimal settings
optimal_settings = {
    'take_profit_pct': best['TP'],
    'stop_loss_pct': best['SL'],
    'risk_reward': best['Risk_Reward'],
    'expected_win_rate': best['Win_Rate'],
    'expected_return_per_trade': best['Avg_Return']
}

joblib.dump(optimal_settings, 'optimal_kalshi_settings.pkl')
print("\n[OK] Optimal settings saved to: optimal_kalshi_settings.pkl")
print("="*70 + "\n")

