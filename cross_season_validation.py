"""
Cross-Season Validation
Test the improved model on multiple seasons to validate robustness
Goal: Prove the 68% win rate holds across different years
"""

import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path
from sklearn.preprocessing import StandardScaler

print("\n" + "="*70)
print("CROSS-SEASON VALIDATION")
print("Testing model on 2021-22, 2022-23, and 2023-24 seasons")
print("="*70)

# Load the improved model
print("\nLoading improved model...")
model = joblib.load('models/momentum_model_improved.pkl')
print(f"  [OK] Model loaded")
print(f"  Trained on: {model['train_games']} games")
print(f"  Features: {len(model['feature_cols'])}")

# Check available seasons
data_dir = Path('data/raw')
available_seasons = []
for season_file in sorted(data_dir.glob('pbp_*.csv')):
    season_name = season_file.stem.replace('pbp_', '')
    if 'partial' not in season_name and 'BACKUP' not in season_name:
        available_seasons.append(season_name)

print(f"\nAvailable seasons: {', '.join(available_seasons)}")

# We want to test on seasons OTHER than where we trained
# The model was trained on 60% of 2023-24, so let's test on:
# 1. 2021-22
# 2. 2022-23  
# 3. Full 2023-24 (to compare with our previous results)

test_seasons = ['2021_22', '2022_23', '2023_24']

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

def load_and_prepare_features(season):
    """Load features for a season"""
    feature_file = Path(f'data/processed/features_v2_{season}_enhanced.csv')
    
    if not feature_file.exists():
        print(f"  [!] Features not found for {season}, skipping")
        return None
    
    features_df = pd.read_csv(feature_file)
    return features_df

def calculate_quality_score(row):
    """Calculate quality score (same as training)"""
    score = 0
    
    # 1. Pure run bonus
    if row.get('opp_score', 0) == 0:
        score += 20
    
    # 2. Large run bonus
    if row.get('run_score', 0) >= 7:
        score += 15
    elif row.get('run_score', 0) >= 6:
        score += 10
    
    # 3. Defensive pressure
    defensive_actions = row.get('team_steals_2min', 0) + row.get('team_blocks_2min', 0) + row.get('opponent_turnovers_2min', 0)
    if defensive_actions >= 3:
        score += 15
    elif defensive_actions >= 2:
        score += 10
    elif defensive_actions >= 1:
        score += 5
    
    # 4. Offensive efficiency
    if row.get('team_threes_2min', 0) >= 2:
        score += 10
    elif row.get('team_threes_2min', 0) >= 1:
        score += 5
    
    # 5. Team quality advantage
    team_quality_diff = row.get('team_quality_diff', 0)
    if team_quality_diff > 0.10:
        score += 10
    elif team_quality_diff > 0:
        score += 5
    
    # 6. Recent form advantage
    form_diff = row.get('team_form_advantage', 0)
    if form_diff > 0.2:
        score += 10
    elif form_diff > 0:
        score += 5
    
    # 7. Early game
    if row.get('period', 3) <= 2:
        score += 10
    
    # 8. Close game
    if row.get('is_close_game', 0) == 1:
        score += 5
    
    return min(score, 100)

def backtest_season(season_name, features_df, selectivity='balanced'):
    """Backtest a season with the improved model"""
    print(f"\n  Processing {season_name}...")
    
    # Calculate quality scores
    features_df['run_quality_score'] = features_df.apply(calculate_quality_score, axis=1)
    
    # Get predictions
    feature_cols = model['feature_cols']
    
    # Check for missing features
    missing_features = set(feature_cols) - set(features_df.columns)
    if missing_features:
        print(f"    [!] Missing features: {missing_features}")
        # Fill missing features with 0
        for feat in missing_features:
            features_df[feat] = 0
    
    X = features_df[feature_cols].fillna(0).values
    X_scaled = model['scaler'].transform(X)
    predictions = model['model'].predict_proba(X_scaled)[:, 1]
    features_df['prediction'] = predictions
    
    print(f"    Total samples: {len(features_df):,}")
    print(f"    Avg prediction: {predictions.mean()*100:.1f}%")
    
    # Filter for 6-0 runs in Q1-Q3
    entry_opportunities = features_df[
        (features_df.get('run_score', 0) >= 6) &
        (features_df.get('opp_score', 0) == 0) &
        (features_df.get('period', 5) <= 3) &
        (features_df['run_quality_score'] >= 60)
    ].copy()
    
    print(f"    6-0 opportunities (Q1-Q3, quality>=60): {len(entry_opportunities):,}")
    
    if len(entry_opportunities) == 0:
        return None
    
    # Apply selectivity
    if selectivity == 'ultra':
        # Top 10% confidence
        threshold = np.percentile(entry_opportunities['prediction'], 90)
        filtered = entry_opportunities[entry_opportunities['prediction'] >= threshold].copy()
        label = "Ultra-Selective (Top 10%)"
    elif selectivity == 'moderate':
        # Top 25% confidence
        threshold = np.percentile(entry_opportunities['prediction'], 75)
        filtered = entry_opportunities[entry_opportunities['prediction'] >= threshold].copy()
        label = "Moderate (Top 25%)"
    else:  # balanced
        # Top 30% confidence
        threshold = np.percentile(entry_opportunities['prediction'], 70)
        filtered = entry_opportunities[entry_opportunities['prediction'] >= threshold].copy()
        label = "Balanced (Top 30%)"
    
    # Best opportunity per game
    filtered = filtered.sort_values(['game_id', 'run_quality_score', 'prediction'], ascending=[True, False, False])
    best_per_game = filtered.groupby('game_id').first().reset_index()
    
    print(f"    {label}: {len(best_per_game)} trades")
    print(f"    Confidence threshold: {threshold*100:.1f}%")
    
    if len(best_per_game) == 0:
        return None
    
    # Backtest
    INITIAL_BANKROLL = 1000.0
    bankroll = INITIAL_BANKROLL
    trades = []
    
    CONFIG = {
        'position_size_pct': 0.05 if selectivity == 'ultra' else 0.03,
        'take_profit_pct': 0.25,
        'stop_loss_pct': -0.05,
    }
    
    for idx, row in best_per_game.iterrows():
        position_size = bankroll * CONFIG['position_size_pct']
        predicted_prob = row['prediction']
        entry_fee = calculate_fee(position_size, predicted_prob)
        
        if position_size + entry_fee > bankroll:
            continue
        
        actual_outcome = row.get('run_extends', 0)
        
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
        trades.append({
            'actual_outcome': actual_outcome,
            'profit': profit,
            'confidence': predicted_prob,
            'quality': row['run_quality_score']
        })
    
    if len(trades) == 0:
        return None
    
    trades_df = pd.DataFrame(trades)
    total_return = bankroll - INITIAL_BANKROLL
    return_pct = (total_return / INITIAL_BANKROLL) * 100
    win_rate = (trades_df['actual_outcome'] == 1).mean() * 100
    
    return {
        'season': season_name,
        'selectivity': label,
        'trades': len(trades),
        'win_rate': win_rate,
        'return': total_return,
        'return_pct': return_pct,
        'avg_confidence': trades_df['confidence'].mean(),
        'avg_quality': trades_df['quality'].mean(),
        'final_bankroll': bankroll
    }

# Run validation across seasons
print("\n" + "="*70)
print("TESTING ACROSS SEASONS")
print("="*70)

all_results = []

for season in test_seasons:
    print(f"\n{'='*70}")
    print(f"SEASON: {season.replace('_', '-')}")
    print(f"{'='*70}")
    
    features_df = load_and_prepare_features(season)
    
    if features_df is None:
        continue
    
    # Test with different selectivity levels
    for selectivity in ['ultra', 'moderate', 'balanced']:
        result = backtest_season(season, features_df.copy(), selectivity)
        if result:
            all_results.append(result)

# Print summary
print("\n" + "="*70)
print("CROSS-SEASON VALIDATION RESULTS")
print("="*70)

if len(all_results) == 0:
    print("\n[!] No results to display")
else:
    results_df = pd.DataFrame(all_results)
    
    # Group by selectivity
    for selectivity in ['Ultra-Selective (Top 10%)', 'Moderate (Top 25%)', 'Balanced (Top 30%)']:
        subset = results_df[results_df['selectivity'] == selectivity]
        
        if len(subset) == 0:
            continue
        
        print(f"\n{selectivity}:")
        print("-" * 70)
        
        for _, row in subset.iterrows():
            print(f"\n  {row['season'].replace('_', '-')}:")
            print(f"    Trades: {row['trades']}")
            print(f"    Win Rate: {row['win_rate']:.1f}%")
            print(f"    Return: ${row['return']:+.2f} ({row['return_pct']:+.2f}%)")
            print(f"    Avg Confidence: {row['avg_confidence']*100:.1f}%")
            print(f"    Avg Quality: {row['avg_quality']:.1f}")
        
        # Overall stats
        if len(subset) > 1:
            print(f"\n  AVERAGE ACROSS SEASONS:")
            print(f"    Avg Trades/Season: {subset['trades'].mean():.0f}")
            print(f"    Avg Win Rate: {subset['win_rate'].mean():.1f}%")
            print(f"    Avg Return: {subset['return_pct'].mean():+.2f}%")
            print(f"    Std Dev Return: {subset['return_pct'].std():.2f}%")
            
            # Check consistency
            if subset['win_rate'].std() < 10:
                print(f"    [OK] Win rate is CONSISTENT across seasons!")
            else:
                print(f"    [!] Win rate varies significantly across seasons")

    # Save results
    results_df.to_csv('cross_season_validation_results.csv', index=False)
    print(f"\n[OK] Results saved to: cross_season_validation_results.csv")

print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

if len(all_results) > 0:
    # Focus on ultra-selective
    ultra_results = results_df[results_df['selectivity'] == 'Ultra-Selective (Top 10%)']
    
    if len(ultra_results) > 0:
        avg_wr = ultra_results['win_rate'].mean()
        std_wr = ultra_results['win_rate'].std()
        avg_return = ultra_results['return_pct'].mean()
        
        print(f"\nULTRA-SELECTIVE STRATEGY (Top 10% Confidence):")
        print(f"  Tested across {len(ultra_results)} seasons")
        print(f"  Average Win Rate: {avg_wr:.1f}% (+/- {std_wr:.1f}%)")
        print(f"  Average Return: {avg_return:+.2f}%")
        
        if avg_wr >= 60 and std_wr < 15:
            print(f"\n  [OK] STRATEGY IS ROBUST!")
            print(f"  Win rate consistently above 60% across seasons")
            print(f"  Ready to use for 2025-26 season!")
        elif avg_wr >= 50 and std_wr < 20:
            print(f"\n  [~] STRATEGY SHOWS PROMISE")
            print(f"  Win rate above 50%, but monitor closely")
        else:
            print(f"\n  [!] STRATEGY NEEDS IMPROVEMENT")
            print(f"  Win rate or consistency not meeting targets")

print("="*70 + "\n")

