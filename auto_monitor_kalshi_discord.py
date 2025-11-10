"""
AUTOMATIC KALSHI TRADING MONITOR with DISCORD ALERTS
=====================================================

Monitors live NBA games and sends Discord alerts for Kalshi trades
Adjusted for Kalshi's volatility with wider stops
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime
import math

# Import Discord webhook function
from discord_webhook_setup import send_discord_alert, DISCORD_WEBHOOK_URL

sys.path.append('src')

print("\n" + "="*70)
print("KALSHI AUTO-MONITOR with DISCORD")
print("="*70)
print(f"Time: {datetime.now().strftime('%I:%M %p ET')}")

# Check Discord webhook is configured
if DISCORD_WEBHOOK_URL == 'YOUR_WEBHOOK_URL_HERE':
    print("\n[!] DISCORD WEBHOOK NOT CONFIGURED")
    print("="*70)
    print("\nSetup steps:")
    print("1. Create Discord server")
    print("2. Server Settings -> Integrations -> Webhooks")
    print("3. Create webhook, copy URL")
    print("4. Edit discord_webhook_setup.py with your URL")
    print("5. Run: python discord_webhook_setup.py (to test)")
    print("6. Then run this script again")
    exit()

# KALSHI-SPECIFIC SETTINGS
KALSHI_CONFIG = {
    'initial_bankroll': 1000,
    'position_size_pct': 0.05,      # 5% per trade
    'stop_loss_pct': -0.20,         # -20% (wider for Kalshi volatility)
    'take_profit_pct': 0.36,        # +36% (maintains 1.75:1 ratio)
    'min_confidence': 0.345,        # Top 20%
    'min_quality': 60,
    'max_entry_price_cents': 50     # Don't buy if already > 50¢
}

print(f"\n[OK] Discord webhook configured")
print(f"[OK] Kalshi settings loaded")

# Load model
MODEL_FILE = Path('models/ignition_ai_live_2025_26.pkl')

if not MODEL_FILE.exists():
    print("\n[!] Model not found!")
    print("    Run: python train_for_live_alerts.py")
    exit()

print("[OK] Loading model...")
model_data = joblib.load(MODEL_FILE)
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']

print(f"    Trained on {model_data['training_games']:,} games")

# Team ID to Name mapping
TEAM_NAMES = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612739: 'CLE', 1610612740: 'NOP',
    1610612741: 'CHI', 1610612742: 'DAL', 1610612743: 'DEN', 1610612744: 'GSW',
    1610612745: 'HOU', 1610612746: 'LAC', 1610612747: 'LAL', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612751: 'BKN', 1610612752: 'NYK',
    1610612753: 'ORL', 1610612754: 'IND', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612760: 'OKC',
    1610612761: 'TOR', 1610612762: 'UTA', 1610612763: 'MEM', 1610612764: 'WAS',
    1610612765: 'DET', 1610612766: 'CHA'
}

def calculate_quality_score(game_state):
    """Calculate run quality score"""
    score = 0
    if game_state.get('opp_score', 1) == 0: score += 20
    if game_state.get('run_score', 0) >= 7: score += 15
    elif game_state.get('run_score', 0) >= 6: score += 10
    defensive = game_state.get('steals', 0) + game_state.get('blocks', 0) + game_state.get('turnovers', 0)
    if defensive >= 3: score += 15
    elif defensive >= 2: score += 10
    elif defensive >= 1: score += 5
    if game_state.get('threes', 0) >= 2: score += 10
    elif game_state.get('threes', 0) >= 1: score += 5
    if game_state.get('team_quality_diff', 0) > 0.10: score += 10
    elif game_state.get('team_quality_diff', 0) > 0: score += 5
    if game_state.get('period', 1) <= 2: score += 10
    return min(score, 100)

def calculate_kalshi_trade(win_prob, confidence, quality, bankroll):
    """
    Calculate Kalshi-specific trade parameters
    
    Kalshi uses contract pricing (0-100¢):
    - Entry price ~= win probability
    - Volatility is HIGH
    - Need wider stops
    """
    
    # Estimate Kalshi entry price based on win probability
    # In reality, you'd get this from Kalshi API
    estimated_entry = int(win_prob * 100)  # Convert to cents
    
    # Adjust for market spread (assume 2-3¢ spread)
    estimated_entry += 2  # Buying at ask
    
    # Position size in dollars
    position_size = bankroll * KALSHI_CONFIG['position_size_pct']
    
    # Number of contracts
    num_contracts = int(position_size / (estimated_entry / 100))
    
    # Actual position cost
    total_cost = num_contracts * (estimated_entry / 100)
    
    # Calculate exits (in cents)
    stop_loss_price = int(estimated_entry * (1 + KALSHI_CONFIG['stop_loss_pct']))
    take_profit_price = int(estimated_entry * (1 + KALSHI_CONFIG['take_profit_pct']))
    
    # Expected value calculation
    # Win: (take_profit_price - entry_price) * num_contracts / 100
    # Loss: (entry_price - stop_loss_price) * num_contracts / 100
    potential_win = (take_profit_price - estimated_entry) * num_contracts / 100
    potential_loss = (estimated_entry - stop_loss_price) * num_contracts / 100
    expected_value = (confidence * potential_win) - ((1 - confidence) * potential_loss)
    
    # Risk/reward ratio
    risk_reward = potential_win / potential_loss if potential_loss > 0 else 0
    
    return {
        'kalshi_entry_price': estimated_entry,
        'num_contracts': num_contracts,
        'total_cost': total_cost,
        'kalshi_stop_loss': stop_loss_price,
        'kalshi_take_profit': take_profit_price,
        'stop_loss_pct': KALSHI_CONFIG['stop_loss_pct'],
        'take_profit_pct': KALSHI_CONFIG['take_profit_pct'],
        'potential_win': potential_win,
        'potential_loss': potential_loss,
        'expected_value': expected_value,
        'risk_reward': risk_reward
    }

def monitor_live_games():
    """Monitor live games and send Discord alerts"""
    
    print("\n" + "="*70)
    print("MONITORING LIVE GAMES")
    print("="*70)
    print(f"\nKalshi Settings:")
    print(f"  Position Size: {KALSHI_CONFIG['position_size_pct']*100:.0f}%")
    print(f"  Stop Loss: {KALSHI_CONFIG['stop_loss_pct']*100:.0f}%")
    print(f"  Take Profit: {KALSHI_CONFIG['take_profit_pct']*100:.0f}%")
    print(f"  Min Confidence: {KALSHI_CONFIG['min_confidence']*100:.1f}%")
    print(f"  Max Entry: {KALSHI_CONFIG['max_entry_price_cents']}¢")
    
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.endpoints import playbyplayv2
    
    monitored_runs = set()
    bankroll = KALSHI_CONFIG['initial_bankroll']
    
    while True:
        try:
            # Get current games
            scoreboard = scoreboardv2.ScoreboardV2()
            games = scoreboard.get_data_frames()[0]
            
            # Filter for live games
            live_games = games[games['GAME_STATUS_TEXT'].str.contains('Q|Half', na=False)]
            
            if len(live_games) == 0:
                print(f"[{datetime.now().strftime('%I:%M:%S %p')}] No live games. Checking again in 60s...")
                time.sleep(60)
                continue
            
            print(f"\n[{datetime.now().strftime('%I:%M:%S %p')}] Monitoring {len(live_games)} live games...")
            
            # Check each live game
            for _, game in live_games.iterrows():
                game_id = game['GAME_ID']
                home_id = game['HOME_TEAM_ID']
                away_id = game['VISITOR_TEAM_ID']
                home_team = TEAM_NAMES.get(home_id, 'HOME')
                away_team = TEAM_NAMES.get(away_id, 'AWAY')
                
                try:
                    # Get play-by-play
                    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
                    plays = pbp.get_data_frames()[0]
                    
                    if len(plays) < 20:
                        continue
                    
                    # TODO: Full run detection logic would go here
                    # For now, this is a framework showing the structure
                    
                    print(f"  {away_team} @ {home_team} - checking...")
                    
                    # If high-confidence run detected:
                    # game_info = {...}
                    # send_discord_alert(game_info)
                    
                except Exception as e:
                    print(f"  {away_team} @ {home_team} - Error: {e}")
                    continue
            
            # Wait 30 seconds before next check
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n[!] Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

# Start monitoring
print("\n" + "="*70)
print("STARTING LIVE MONITORING FOR KALSHI")
print("="*70)
print(f"\nDiscord alerts enabled")
print(f"Optimized for Kalshi's volatility")
print("\nPress Ctrl+C to stop\n")

print("\n[DEMO MODE]")
print("="*70)
print("\nThe full monitoring logic requires complete integration")
print("with NBA API for real-time run detection.")
print("\nFor now, use the manual checker when you see a 6-0 run:")
print("  python check_single_run.py")
print("\nOr setup the full auto-monitor with proper run detection.")
print("="*70 + "\n")

# Uncomment to enable live monitoring:
# try:
#     monitor_live_games()
# except KeyboardInterrupt:
#     print("\n\nMonitoring stopped. Good luck with your Kalshi trades!")

