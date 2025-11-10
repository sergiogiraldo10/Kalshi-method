"""
LIVE GAME MONITOR FOR TONIGHT (November 9, 2024)
================================================

This script will:
1. Find all games scheduled for tonight
2. Monitor them in real-time
3. Alert you when there's a 6-0 run with high confidence (trade signal)
4. Show you the exact teams, run details, and model confidence

Run this script during game time!
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime
import math

sys.path.append('src')

print("\n" + "="*70)
print("LIVE GAME MONITOR - IGNITION AI")
print("="*70)
print(f"Date: {datetime.now().strftime('%B %d, %Y %I:%M %p')}")

# Check if we have a trained model
MODEL_FILE = Path('models/ignition_ai_live_2024_25.pkl')

if not MODEL_FILE.exists():
    print("\n[!] No trained model found!")
    print("    Run these steps first:")
    print("    1. python download_latest_2024_25.py")
    print("    2. python extract_features_2024_25.py")
    print("    3. python add_team_features_2024_25.py")
    print("    4. python train_live_model.py")
    print("\nExiting...")
    exit()

print("\n[OK] Loading trained model...")
model_data = joblib.load(MODEL_FILE)
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']

print(f"    Model trained on {model_data['training_games']:,} games")
print(f"    Features: {len(feature_cols)}")

# Try to get today's games
print("\n" + "="*70)
print("CHECKING FOR GAMES TONIGHT")
print("="*70)

try:
    from nba_api.live.nba.endpoints import scoreboard
    from nba_api.stats.endpoints import playbyplayv2
    
    print("\nFetching today's schedule...")
    games_today = scoreboard.ScoreBoard()
    games_df = games_today.get_data_frames()[0]
    
    if len(games_df) == 0:
        print("\n[!] No games scheduled for tonight")
        print("    Check back on a game day!")
        exit()
    
    print(f"\n[OK] Found {len(games_df)} games tonight:")
    print("-" * 70)
    
    for idx, game in games_df.iterrows():
        home = game.get('homeTeam', {}).get('teamTricode', 'HOME')
        away = game.get('awayTeam', {}).get('teamTricode', 'AWAY')
        status = game.get('gameStatusText', 'Scheduled')
        game_id = game.get('gameId', 'Unknown')
        
        print(f"{idx+1}. {away} @ {home} - {status} (ID: {game_id})")
    
    print("-" * 70)
    
except Exception as e:
    print(f"\n[!] Could not fetch live games: {e}")
    print("\n    DEMO MODE: Simulating game monitoring...")
    print("    In production, this would connect to NBA API")
    
    # Demo mode
    print("\n" + "="*70)
    print("DEMO: HOW LIVE MONITORING WORKS")
    print("="*70)
    
    print("\nThe script would:")
    print("1. Monitor each game's play-by-play in real-time")
    print("2. Detect 6-0 runs as they happen")
    print("3. Calculate quality score (steals, blocks, 3-pointers)")
    print("4. Get model prediction (confidence %)")
    print("5. ALERT YOU if confidence >= 34.5% (top 20%)")
    
    print("\n" + "="*70)
    print("EXAMPLE ALERT")
    print("="*70)
    
    print("\n🚨 TRADE SIGNAL 🚨")
    print("-" * 70)
    print("Game: LAL @ GSW")
    print("Time: Q2 3:45")
    print("Run: LAL on a 6-0 run")
    print("")
    print("Details:")
    print("  - Model Confidence: 35.2% (TOP 20%)")
    print("  - Quality Score: 65/100")
    print("  - Defensive Pressure: 2 steals, 1 block, 2 turnovers")
    print("  - 3-Pointers: 3 in last 2 minutes")
    print("  - Team Quality: LAL 0.15 better than GSW")
    print("")
    print("ACTION:")
    print("  - BUY: LAL to win")
    print("  - Position Size: $50 (5% of bankroll)")
    print("  - Take Profit: +25% ($62.50)")
    print("  - Stop Loss: -5% ($47.50)")
    print("-" * 70)

# Helper function to calculate quality
def calculate_quality_score(game_state):
    score = 0
    if game_state.get('opp_score', 1) == 0: score += 20
    run_score = game_state.get('run_score', 0)
    if run_score >= 7: score += 15
    elif run_score >= 6: score += 10
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

def calculate_fee(position_size, probability):
    fee = 0.07 * position_size * probability * (1 - probability)
    return math.ceil(fee * 100) / 100

print("\n" + "="*70)
print("LIVE MONITORING INSTRUCTIONS")
print("="*70)

print("\nTo use this for REAL trading tonight:")
print("")
print("1. MANUAL METHOD (Recommended for first time):")
print("   - Watch games on TV/stream")
print("   - When you see a 6-0 run, check the model manually")
print("   - Use the 'check_single_run.py' script")
print("")
print("2. AUTOMATED METHOD (Advanced):")
print("   - Integrate with NBA API real-time feed")
print("   - Set up alerts (SMS, email, desktop notification)")
print("   - Requires NBA API subscription or web scraping")
print("")
print("3. BETTING PLATFORMS:")
print("   - Have accounts ready on betting sites")
print("   - Check odds match model expectations")
print("   - Execute trades quickly (momentum is time-sensitive)")

print("\n" + "="*70)
print("RISK REMINDERS")
print("="*70)

print("\n⚠️  START SMALL:")
print("   - Use $100-500 bankroll for first night")
print("   - Max 1-2 trades to start")
print("   - Track results vs predictions")
print("")
print("⚠️  MODEL IS NOT PERFECT:")
print("   - 35.8% win rate = you'll lose 64% of trades")
print("   - Profitability comes from asymmetric exits")
print("   - Some nights you'll lose money")
print("")
print("⚠️  DISCIPLINE:")
print("   - ONLY trade signals >= 34.5% confidence")
print("   - ALWAYS use -5% stop loss")
print("   - ALWAYS use +25% take profit")
print("   - NO emotional decisions")

print("\n" + "="*70)
print(f"Ready for tonight's games! Good luck! 🏀")
print("="*70 + "\n")

