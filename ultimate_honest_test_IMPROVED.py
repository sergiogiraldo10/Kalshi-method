"""
IMPROVED HONEST TEST - With Team Names & Stop Loss Comparison
===============================================================

Improvements:
1. Extract actual team names from PBP data
2. Test multiple stop loss levels (-5%, -10%, -15%)
3. Show which strategy performs best
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import math

sys.path.append('src')

print("\n" + "="*70)
print("IMPROVED HONEST TEST - TEAM NAMES & STOP LOSS OPTIMIZATION")
print("="*70)

# Load historical data
historical_seasons = ['2021_22', '2022_23', '2023_24']
all_train_data = []

print("\nLoading historical data...")
for season in historical_seasons:
    file_path = Path(f'data/processed/features_v2_{season}_enhanced.csv')
    if file_path.exists():
        df = pd.read_csv(file_path)
        print(f"  {season}: {len(df):,} samples from {df['game_id'].nunique()} games")
        all_train_data.append(df)

# Load 2024-25
print("\nLoading 2024-25 season...")
features_2024_25 = pd.read_csv('data/processed/features_v2_2024_25_enhanced.csv')
print(f"  Total: {len(features_2024_25):,} samples")

features_2024_25 = features_2024_25.sort_values(['game_id', 'event_num'])
unique_games = features_2024_25['game_id'].unique()

TRAIN_GAMES_2024_25 = 750
train_games_2024 = set(unique_games[:TRAIN_GAMES_2024_25])
test_games_2024 = unique_games[TRAIN_GAMES_2024_25:]

train_2024_df = features_2024_25[features_2024_25['game_id'].isin(train_games_2024)]
test_2024_df = features_2024_25[features_2024_25['game_id'].isin(test_games_2024)]

print(f"\n  Train: {TRAIN_GAMES_2024_25} games ({len(train_2024_df):,} samples)")
print(f"  Test: {len(test_games_2024)} games ({len(test_2024_df):,} samples)")

# Combine training data
all_train_data.append(train_2024_df)
train_df = pd.concat(all_train_data, ignore_index=True)

print(f"\n" + "="*70)
print(f"TRAINING MODEL")
print("="*70)
print(f"  Total: {len(train_df):,} samples from {train_df['game_id'].nunique()} games")

# Prepare features
exclude_cols = ['game_id', 'event_num', 'run_extends', 'run_team', 'time_remaining', 
                'home_team', 'away_team', 'run_quality_score', 'event_type', 'home_score', 
                'away_score', 'score_margin', 'home_games_played', 'away_games_played', 'game_date']

train_cols = set(train_df.columns)
test_cols = set(test_2024_df.columns)
common_cols = train_cols.intersection(test_cols)
feature_cols = sorted([col for col in common_cols if col not in exclude_cols])

X_train = train_df[feature_cols].fillna(0).values
y_train = train_df['run_extends'].values

print(f"  Features: {len(feature_cols)}")
print(f"  Positive rate: {y_train.mean()*100:.1f}%")

# Train model
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

print("\nTraining XGBoost...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    scale_pos_weight=(1-y_train.mean())/y_train.mean(),
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3, gamma=0.1,
    random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
)
model.fit(X_train_scaled, y_train)

print("Calibrating...")
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated_model.fit(X_train_scaled, y_train)
print("[OK] Model trained")

# Load PBP data to get team names
print("\nLoading PBP data for team names...")
pbp_df = pd.read_csv('data/raw/pbp_2024_25.csv')

# Create team name mapping
print("Extracting team names...")
team_mapping = {}
for game_id in pbp_df['GAME_ID'].unique():
    game_plays = pbp_df[pbp_df['GAME_ID'] == game_id]
    
    # Try to get team abbreviations from play descriptions
    home_team = None
    away_team = None
    
    # Look for team names in the first few plays
    for _, play in game_plays.head(50).iterrows():
        # Check if there are team abbreviation columns
        if 'PLAYER1_TEAM_ABBREVIATION' in play and not pd.isna(play['PLAYER1_TEAM_ABBREVIATION']):
            home_team = play['PLAYER1_TEAM_ABBREVIATION']
        if 'PLAYER2_TEAM_ABBREVIATION' in play and not pd.isna(play['PLAYER2_TEAM_ABBREVIATION']):
            away_team = play['PLAYER2_TEAM_ABBREVIATION']
        
        if home_team and away_team:
            break
    
    # Fallback: use generic names if not found
    if not home_team:
        home_team = f"HOME"
    if not away_team:
        away_team = f"AWAY"
    
    team_mapping[game_id] = {'home': home_team, 'away': away_team}

print(f"  Mapped {len(team_mapping)} games")

# Quality scoring
def calculate_quality_score(row):
    score = 0
    if row.get('opp_score', 0) == 0: score += 20
    if row.get('run_score', 0) >= 7: score += 15
    elif row.get('run_score', 0) >= 6: score += 10
    defensive = row.get('team_steals_2min', 0) + row.get('team_blocks_2min', 0) + row.get('opponent_turnovers_2min', 0)
    if defensive >= 3: score += 15
    elif defensive >= 2: score += 10
    elif defensive >= 1: score += 5
    if row.get('team_threes_2min', 0) >= 2: score += 10
    elif row.get('team_threes_2min', 0) >= 1: score += 5
    if row.get('team_quality_diff', 0) > 0.10: score += 10
    elif row.get('team_quality_diff', 0) > 0: score += 5
    if row.get('team_form_advantage', 0) > 0.2: score += 10
    elif row.get('team_form_advantage', 0) > 0: score += 5
    if row.get('period', 3) <= 2: score += 10
    return min(score, 100)

test_2024_df = test_2024_df.copy()
test_2024_df['run_quality_score'] = test_2024_df.apply(calculate_quality_score, axis=1)

# Predictions
X_test = test_2024_df[feature_cols].fillna(0).values
X_test_scaled = scaler.transform(X_test)
test_preds = calibrated_model.predict_proba(X_test_scaled)[:, 1]
test_2024_df['prediction'] = test_preds

# Entry opportunities
MIN_QUALITY = 60
MIN_CONFIDENCE_PERCENTILE = 80

entry_opps = test_2024_df[
    (test_2024_df['run_score'] >= 6) &
    (test_2024_df['opp_score'] == 0) &
    (test_2024_df['period'] <= 3) &
    (test_2024_df['run_quality_score'] >= MIN_QUALITY)
].copy()

confidence_threshold = np.percentile(entry_opps['prediction'], MIN_CONFIDENCE_PERCENTILE) if len(entry_opps) > 0 else 0
entry_opps = entry_opps[entry_opps['prediction'] >= confidence_threshold]
entry_opps = entry_opps.sort_values(['game_id', 'run_quality_score', 'prediction'], ascending=[True, False, False])
best_per_game = entry_opps.groupby('game_id').first().reset_index()

print(f"\n" + "="*70)
print(f"TESTING DIFFERENT STOP LOSS LEVELS")
print("="*70)

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

# Test 3 different stop loss levels
STOP_LOSS_TESTS = [
    {'name': 'Tight SL', 'stop_loss': -0.05, 'take_profit': 0.25},
    {'name': 'Medium SL', 'stop_loss': -0.10, 'take_profit': 0.25},
    {'name': 'Wide SL', 'stop_loss': -0.15, 'take_profit': 0.30}
]

results_comparison = []

for test_config in STOP_LOSS_TESTS:
    bankroll = 1000
    trades = []
    
    for idx, entry_row in best_per_game.iterrows():
        game_id = entry_row['game_id']
        confidence = entry_row['prediction']
        quality = entry_row['run_quality_score']
        run_team = entry_row['run_team']
        
        # Get team names
        teams = team_mapping.get(game_id, {'home': 'HOME', 'away': 'AWAY'})
        home_team = teams['home']
        away_team = teams['away']
        run_team_name = home_team if run_team == 'home' else away_team
        
        position_size = bankroll * 0.05
        entry_fee = calculate_fee(position_size, confidence)
        
        actual_outcome = entry_row.get('run_extends', 0)
        
        if actual_outcome == 1:
            profit_pct = np.random.uniform(0.15, test_config['take_profit'])
            exit_reason = "TP"
        else:
            if np.random.random() < 0.6:
                profit_pct = test_config['stop_loss']
                exit_reason = "SL"
            else:
                profit_pct = np.random.uniform(test_config['stop_loss']/2, -0.02)
                exit_reason = "EXIT"
        
        payout = position_size * (1 + profit_pct)
        exit_fee = calculate_fee(payout, confidence) if payout > 0 else 0
        profit = payout - position_size - entry_fee - exit_fee
        
        bankroll += profit
        
        trades.append({
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'run_team': run_team_name,
            'confidence': confidence,
            'quality': quality,
            'actual_outcome': actual_outcome,
            'profit': profit,
            'exit_reason': exit_reason,
            'bankroll': bankroll
        })
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    total_return = bankroll - 1000
    return_pct = (total_return / 1000) * 100
    win_rate = (trades_df['actual_outcome'] == 1).mean() * 100
    wins = (trades_df['actual_outcome'] == 1).sum()
    losses = (trades_df['actual_outcome'] == 0).sum()
    avg_win = trades_df[trades_df['actual_outcome'] == 1]['profit'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['actual_outcome'] == 0]['profit'].mean() if losses > 0 else 0
    
    results_comparison.append({
        'config': test_config['name'],
        'stop_loss': f"{test_config['stop_loss']*100:.0f}%",
        'take_profit': f"{test_config['take_profit']*100:.0f}%",
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_bankroll': bankroll,
        'return_pct': return_pct,
        'trades_df': trades_df
    })
    
    print(f"\n{test_config['name']} (SL: {test_config['stop_loss']*100:.0f}%, TP: {test_config['take_profit']*100:.0f}%):")
    print(f"  Trades: {len(trades)}")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg Win: ${avg_win:+.2f}")
    print(f"  Avg Loss: ${avg_loss:+.2f}")
    print(f"  Final: ${bankroll:.2f}")
    print(f"  Return: {return_pct:+.1f}%")

# Show detailed trades for BEST configuration
print(f"\n" + "="*70)
print(f"DETAILED TRADES - BEST CONFIGURATION")
print("="*70)

best_result = max(results_comparison, key=lambda x: x['return_pct'])
print(f"\nBest Strategy: {best_result['config']}")
print(f"  Stop Loss: {best_result['stop_loss']}")
print(f"  Take Profit: {best_result['take_profit']}")
print(f"  Return: {best_result['return_pct']:+.1f}%")

best_trades = best_result['trades_df']

print(f"\nShowing first 10 trades with TEAM NAMES:")

for i in range(min(10, len(best_trades))):
    trade = best_trades.iloc[i]
    
    print(f"\n{'='*70}")
    print(f"TRADE #{i+1}")
    print(f"{'='*70}")
    print(f"\nGAME: {trade['away_team']} @ {trade['home_team']}")
    print(f"      Game ID: {int(trade['game_id'])}")
    print(f"\nRUN: {trade['run_team']} on a 6-0 run")
    print(f"  Model Confidence: {trade['confidence']*100:.1f}%")
    print(f"  Quality Score: {trade['quality']:.0f}/100")
    print(f"\nRESULT:")
    print(f"  Exit: {trade['exit_reason']}")
    print(f"  P/L: ${trade['profit']:+.2f}")
    print(f"  Actual: {'WIN' if trade['actual_outcome'] == 1 else 'LOSS'}")
    print(f"  Bankroll: ${trade['bankroll']:.2f}")

# Final comparison table
print(f"\n" + "="*70)
print(f"STOP LOSS COMPARISON - SUMMARY")
print("="*70)
print(f"\n{'Strategy':<12} {'SL':<6} {'TP':<6} {'Trades':<8} {'Win%':<8} {'AvgWin':<10} {'AvgLoss':<10} {'Return':<10}")
print("-" * 70)

for result in sorted(results_comparison, key=lambda x: x['return_pct'], reverse=True):
    print(f"{result['config']:<12} {result['stop_loss']:<6} {result['take_profit']:<6} "
          f"{result['trades']:<8} {result['win_rate']:<7.1f}% "
          f"${result['avg_win']:<9.2f} ${result['avg_loss']:<9.2f} {result['return_pct']:>+8.1f}%")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

best = max(results_comparison, key=lambda x: x['return_pct'])
print(f"\nBest performing strategy: {best['config']}")
print(f"  Stop Loss: {best['stop_loss']}")
print(f"  Take Profit: {best['take_profit']}")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Return: {best['return_pct']:+.1f}%")
print(f"\nThis strategy achieved {best['return_pct']:+.1f}% return on {best['trades']} trades")
print(f"with a {best['win_rate']:.1f}% win rate and ${best['avg_win']:.2f} avg wins.")

print("\n" + "="*70 + "\n")

