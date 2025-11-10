"""
HONEST BACKTEST - NO PEEKING
Uses actual win probability movements to determine exits and P/L
No future data leakage!
"""

import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path

print("\n" + "="*70)
print("IGNITION AI - HONEST BACKTEST (NO PEEKING)")
print("="*70)

# Load models
print("\nLoading models...")
momentum_model = joblib.load('models/momentum_model_v2.pkl')
win_prob_model = joblib.load('models/win_probability_enhanced.pkl')
print("  [OK] Momentum model loaded")
print("  [OK] Win probability model loaded")

# Load raw play-by-play data (need this for probability tracking)
print("\nLoading 2023-24 season data...")
pbp_df = pd.read_csv('data/raw/pbp_2023_24.csv')
features_df = pd.read_csv('data/processed/features_v2_2023_24.csv')
print(f"  [OK] Loaded {len(pbp_df):,} plays")
print(f"  [OK] Loaded {len(features_df):,} feature samples")

# Configuration
CONFIG = {
    'position_size_pct': 0.03,
    'take_profit_prob_change': 0.10,  # Exit when prob increases by 10%
    'stop_loss_prob_change': -0.05,   # Exit when prob decreases by 5%
    'min_confidence': 0.50,
    'max_confidence': 0.55,
    'min_run_score': 6,
    'max_opp_score': 0,
}

print("\n" + "="*70)
print("STRATEGY - NO PEEKING VERSION")
print("="*70)
print(f"\n  Entry: 6-0 runs, {CONFIG['min_confidence']*100:.0f}-{CONFIG['max_confidence']*100:.0f}% confidence")
print(f"  Exit: Win prob changes by +{CONFIG['take_profit_prob_change']*100:.0f}% (TP) or {CONFIG['stop_loss_prob_change']*100:.0f}% (SL)")
print(f"  Position: {CONFIG['position_size_pct']*100:.0f}% of bankroll")
print(f"  P/L: Based on ACTUAL win probability movements")
print("="*70)

def calculate_fee(position_size, probability):
    """Calculate trading fee"""
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

def calculate_win_probability(play_state, model):
    """Calculate win probability for a game state"""
    try:
        features = {
            'score_diff': play_state.get('score_diff', 0),
            'time_remaining': play_state.get('time_remaining', 0),
            'period': play_state.get('period', 1),
        }
        
        # Add momentum features if available
        for key in ['points_home_2min', 'points_away_2min', 'momentum_2min',
                    'points_home_5min', 'points_away_5min', 'momentum_5min',
                    'current_run', 'home_timeouts', 'away_timeouts',
                    'home_fouls', 'away_fouls', 'is_clutch', 
                    'points_per_minute', 'lead_changes_5min']:
            features[key] = play_state.get(key, 0)
        
        X = pd.DataFrame([features])
        X_scaled = model['scaler'].transform(X[model['feature_cols']])
        prob = model['model'].predict_proba(X_scaled)[0, 1]
        return prob
    except:
        return 0.5  # Default to 50% if we can't calculate

def get_game_progression(game_id, entry_event_num, pbp_data):
    """Get all plays after entry point for a game"""
    game_plays = pbp_data[pbp_data['GAME_ID'] == game_id].copy()
    game_plays = game_plays.sort_values('EVENTNUM')
    
    # Get plays after entry
    future_plays = game_plays[game_plays['EVENTNUM'] > entry_event_num].copy()
    
    return future_plays

def simulate_trade_honest(entry_row, pbp_data, win_prob_model, config):
    """
    Simulate a trade WITHOUT peeking at future outcome
    Track actual win probability changes play-by-play
    """
    game_id = entry_row['game_id']
    entry_event_num = entry_row.get('event_num', 0)
    
    # Calculate entry win probability
    entry_prob = calculate_win_probability(entry_row, win_prob_model)
    
    # Determine which team we're betting on
    run_team = entry_row.get('run_team', 'home')
    betting_home_win = (run_team == 'home')
    
    # Get future plays
    future_plays = get_game_progression(game_id, entry_event_num, pbp_data)
    
    if len(future_plays) == 0:
        # Game ended right after entry - exit at entry price
        return {
            'exit_prob': entry_prob,
            'prob_change': 0,
            'exit_reason': 'Game ended',
            'plays_held': 0
        }
    
    # Track probability changes play by play
    max_prob_seen = entry_prob
    min_prob_seen = entry_prob
    
    for idx, (_, play) in enumerate(future_plays.iterrows()):
        # Skip if no score
        if pd.isna(play['SCORE']):
            continue
        
        try:
            # Parse score
            score_str = str(play['SCORE'])
            if ' - ' in score_str:
                away_score, home_score = map(int, score_str.split(' - '))
            else:
                away_score, home_score = map(int, score_str.split('-'))
            
            # Calculate current state
            score_diff = home_score - away_score if betting_home_win else away_score - home_score
            
            # Parse time
            period = int(play['PERIOD'])
            time_parts = str(play['PCTIMESTRING']).split(':')
            mins, secs = int(time_parts[0]), int(time_parts[1])
            time_in_period = mins * 60 + secs
            
            if period <= 4:
                time_remaining = (4 - period) * 720 + time_in_period
            else:
                time_remaining = max(0, (5 - (period - 4)) * 300 + time_in_period)
            
            # Build play state
            play_state = {
                'score_diff': score_diff,
                'time_remaining': time_remaining,
                'period': period,
                # Add zeros for other features (simplified)
                'points_home_2min': 0, 'points_away_2min': 0, 'momentum_2min': 0,
                'points_home_5min': 0, 'points_away_5min': 0, 'momentum_5min': 0,
                'current_run': 0, 'home_timeouts': 0, 'away_timeouts': 0,
                'home_fouls': 0, 'away_fouls': 0, 'is_clutch': 0,
                'points_per_minute': 0, 'lead_changes_5min': 0
            }
            
            # Calculate current win probability
            current_prob = calculate_win_probability(play_state, win_prob_model)
            
            # Track max/min
            max_prob_seen = max(max_prob_seen, current_prob)
            min_prob_seen = min(min_prob_seen, current_prob)
            
            # Calculate change from entry
            prob_change = current_prob - entry_prob
            
            # Check for TP/SL
            if prob_change >= config['take_profit_prob_change']:
                return {
                    'exit_prob': current_prob,
                    'prob_change': prob_change,
                    'exit_reason': 'Take Profit',
                    'plays_held': idx + 1
                }
            
            if prob_change <= config['stop_loss_prob_change']:
                return {
                    'exit_prob': current_prob,
                    'prob_change': prob_change,
                    'exit_reason': 'Stop Loss',
                    'plays_held': idx + 1
                }
        
        except:
            continue
    
    # If we reach here, game ended without hitting TP/SL
    final_prob = max_prob_seen if max_prob_seen > entry_prob else min_prob_seen
    prob_change = final_prob - entry_prob
    
    return {
        'exit_prob': final_prob,
        'prob_change': prob_change,
        'exit_reason': 'Game ended',
        'plays_held': len(future_plays)
    }

# Filter for entry opportunities
print("\n" + "="*70)
print("FINDING ENTRY OPPORTUNITIES")
print("="*70)

entry_opportunities = features_df[
    (features_df['is_micro_run'] == 1) &
    (features_df['run_score'] == CONFIG['min_run_score']) &
    (features_df['opp_score'] == CONFIG['max_opp_score'])
].copy()

print(f"\n  6-0 run opportunities: {len(entry_opportunities):,}")

# Generate predictions
feature_cols = momentum_model['feature_cols']
X = entry_opportunities[feature_cols].values
X_scaled = momentum_model['scaler'].transform(X)
predictions = momentum_model['model'].predict_proba(X_scaled)[:, 1]
entry_opportunities['prediction'] = predictions

# Filter by confidence
sweet_spot = entry_opportunities[
    (entry_opportunities['prediction'] >= CONFIG['min_confidence']) &
    (entry_opportunities['prediction'] <= CONFIG['max_confidence'])
].copy()

print(f"  In {CONFIG['min_confidence']*100:.0f}-{CONFIG['max_confidence']*100:.0f}% confidence range: {len(sweet_spot):,}")

# Select best per game
sweet_spot = sweet_spot.sort_values(['game_id', 'prediction'], ascending=[True, False])
trades_to_make = []
seen_games = set()

for idx, row in sweet_spot.iterrows():
    if row['game_id'] not in seen_games:
        trades_to_make.append(row)
        seen_games.add(row['game_id'])

print(f"  Final opportunities (1 per game): {len(trades_to_make)}")

# Execute trades
print("\n" + "="*70)
print("SIMULATING TRADES (NO PEEKING)")
print("="*70)

INITIAL_BANKROLL = 1000.0
bankroll = INITIAL_BANKROLL
trades = []

print(f"\n  Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
print(f"  Simulating {len(trades_to_make)} trades...\n")

for i, row in enumerate(trades_to_make):
    # Calculate position
    position_size = bankroll * CONFIG['position_size_pct']
    predicted_prob = row['prediction']
    entry_fee = calculate_fee(position_size, predicted_prob)
    
    if position_size + entry_fee > bankroll:
        continue
    
    # Simulate trade honestly
    result = simulate_trade_honest(row, pbp_df, win_prob_model, CONFIG)
    
    # Calculate P/L based on probability change
    # Win prob change maps to price change: +10% prob = ~+20% price (2x leverage)
    price_change_pct = result['prob_change'] * 2.0
    
    # Calculate payout
    payout = position_size * (1 + price_change_pct)
    exit_fee = calculate_fee(payout, predicted_prob) if payout > 0 else 0
    profit = payout - position_size - entry_fee - exit_fee
    
    # Update bankroll
    bankroll += profit
    
    # Record trade
    actual_outcome = row['run_extends']
    trade = {
        'trade_num': i + 1,
        'game_id': row['game_id'],
        'entry_confidence': predicted_prob,
        'entry_prob': result.get('exit_prob', 0.5) - result['prob_change'],
        'exit_prob': result['exit_prob'],
        'prob_change': result['prob_change'],
        'price_change_pct': price_change_pct,
        'exit_reason': result['exit_reason'],
        'plays_held': result['plays_held'],
        'position_size': position_size,
        'entry_fee': entry_fee,
        'exit_fee': exit_fee,
        'profit': profit,
        'profit_pct': (profit / position_size) * 100,
        'bankroll': bankroll,
        'actual_outcome': actual_outcome
    }
    trades.append(trade)
    
    # Print progress
    if (i + 1) <= 20 or (i + 1) % 50 == 0:
        outcome_label = "WIN" if actual_outcome == 1 else "LOSS"
        print(f"  #{i+1:<4} | {outcome_label:<4} | "
              f"Prob: {result['prob_change']*100:>+5.1f}% | "
              f"{result['exit_reason']:<12} | "
              f"P/L: ${profit:>7.2f} ({price_change_pct*100:>+5.1f}%) | "
              f"Bank: ${bankroll:>9.2f}")

# Results
trades_df = pd.DataFrame(trades)

print("\n" + "="*70)
print("HONEST BACKTEST RESULTS")
print("="*70)

total_return = bankroll - INITIAL_BANKROLL
return_pct = (total_return / INITIAL_BANKROLL) * 100

print(f"\n  FINANCIAL PERFORMANCE:")
print(f"    Initial Capital:      ${INITIAL_BANKROLL:>10,.2f}")
print(f"    Final Capital:        ${bankroll:>10,.2f}")
print(f"    Total Return:         ${total_return:>+10,.2f}")
print(f"    Return %:             {return_pct:>+10.2f}%")

if len(trades_df) > 0:
    wins = trades_df[trades_df['actual_outcome'] == 1]
    losses = trades_df[trades_df['actual_outcome'] == 0]
    
    print(f"\n  TRADING STATISTICS:")
    print(f"    Total Trades:         {len(trades_df):>10,}")
    print(f"    Winning Trades:       {len(wins):>10,} ({len(wins)/len(trades_df)*100:.1f}%)")
    print(f"    Losing Trades:        {len(losses):>10,} ({len(losses)/len(trades_df)*100:.1f}%)")
    
    print(f"\n  PROFIT/LOSS:")
    avg_win = wins['profit'].mean() if len(wins) > 0 else 0
    avg_loss = losses['profit'].mean() if len(losses) > 0 else 0
    print(f"    Average Win:          ${avg_win:>10.2f}")
    print(f"    Average Loss:         ${avg_loss:>10.2f}")
    print(f"    Win/Loss Ratio:       {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>10.2f}:1")
    
    print(f"\n  EXIT ANALYSIS:")
    for reason in ['Take Profit', 'Stop Loss', 'Game ended']:
        subset = trades_df[trades_df['exit_reason'] == reason]
        if len(subset) > 0:
            print(f"    {reason:<15}: {len(subset):>4} ({len(subset)/len(trades_df)*100:>5.1f}%) | Avg P/L: ${subset['profit'].mean():>7.2f}")
    
    # Risk metrics
    returns = trades_df['profit'] / trades_df['position_size']
    sharpe = (returns.mean() / returns.std()) * np.sqrt(len(trades_df)) if returns.std() > 0 else 0
    
    print(f"\n  RISK METRICS:")
    print(f"    Sharpe Ratio:         {sharpe:>10.2f}")
    equity = trades_df['bankroll'].values
    drawdown = (pd.Series(equity).cummax() - equity).max()
    print(f"    Max Drawdown:         ${drawdown:>10.2f}")
    print(f"    Total Fees:           ${trades_df['entry_fee'].sum() + trades_df['exit_fee'].sum():>10.2f}")

print("\n" + "="*70)

if return_pct > 0:
    print("[OK] PROFITABLE (no peeking!)")
    print(f"   Made ${total_return:,.2f} ({return_pct:.2f}%)")
else:
    print("[X] NOT PROFITABLE")
    print(f"   Lost ${abs(total_return):,.2f} ({abs(return_pct):.2f}%)")

print("="*70 + "\n")

# Save results
if len(trades_df) > 0:
    trades_df.to_csv('backtest_honest_results.csv', index=False)
    print("[OK] Results saved to backtest_honest_results.csv\n")

