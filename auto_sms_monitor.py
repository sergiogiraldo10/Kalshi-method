"""
AUTOMATIC SMS ALERT SYSTEM
===========================

Monitors live NBA games and sends SMS alerts when high-confidence trade signals appear.

Phone: 973-294-8219
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime
import math
from twilio.rest import Client

sys.path.append('src')

print("\n" + "="*70)
print("AUTOMATIC SMS ALERT SYSTEM - IGNITION AI")
print("="*70)
print(f"Time: {datetime.now().strftime('%I:%M %p ET')}")

# Twilio Configuration (you'll need to set these up)
TWILIO_ACCOUNT_SID = 'YOUR_TWILIO_ACCOUNT_SID'  # Get from twilio.com
TWILIO_AUTH_TOKEN = 'YOUR_TWILIO_AUTH_TOKEN'
TWILIO_PHONE_NUMBER = '+1234567890'  # Your Twilio number
YOUR_PHONE_NUMBER = '+19732948219'

# Trading config
MIN_CONFIDENCE = 0.345  # Top 20%
MIN_QUALITY = 60
POSITION_SIZE_PCT = 0.05
TAKE_PROFIT_PCT = 0.25
STOP_LOSS_PCT = -0.05

# Check if Twilio is configured
if TWILIO_ACCOUNT_SID == 'YOUR_TWILIO_ACCOUNT_SID':
    print("\n[!] TWILIO NOT CONFIGURED YET")
    print("="*70)
    print("\nTo enable SMS alerts:")
    print("1. Sign up at twilio.com (free trial gives you $15 credit)")
    print("2. Get a phone number")
    print("3. Copy your Account SID and Auth Token")
    print("4. Edit this file and replace:")
    print("   - TWILIO_ACCOUNT_SID")
    print("   - TWILIO_AUTH_TOKEN")
    print("   - TWILIO_PHONE_NUMBER")
    print("\n   Then run this script again")
    print("\n" + "="*70)
    
    # Demo mode
    print("\nDEMO MODE: Showing what SMS alerts would look like...")
    print("="*70)
    
    example_alert = """
🏀 TRADE SIGNAL! 🏀

Game: LAL @ GSW
Score: 45-38 LAL leading
Quarter: 2 (5:30 left)

Run: LAL 6-0 run
Win Probability: 62.5%
Model Confidence: 35.2% (HIGH)
Quality: 65/100

TRADE:
- BUY: Lakers to win
- Size: $50 (5%)
- Take Profit: +25% ($62.50)
- Stop Loss: -5% ($47.50)

Expected Value: +$2.18

Act fast - momentum is time-sensitive!
"""
    
    print(example_alert)
    print("="*70)
    print("\nThis alert would be sent to: 973-294-8219")
    print("\nWithout Twilio setup, you can still use the manual checker:")
    print("  python check_single_run.py")
    exit()

# Initialize Twilio
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("[OK] Twilio connected")
except Exception as e:
    print(f"[!] Twilio error: {e}")
    exit()

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

def send_sms_alert(game_info):
    """Send SMS alert for trade signal"""
    
    message = f"""
🏀 IGNITION AI ALERT 🏀

{game_info['away_team']} @ {game_info['home_team']}
Score: {game_info['away_score']}-{game_info['home_score']} 
Q{game_info['quarter']} {game_info['time_left']}

RUN: {game_info['run_team']} on {game_info['run_score']}-{game_info['opp_score']} run
Win Prob: {game_info['win_probability']*100:.1f}%
Confidence: {game_info['confidence']*100:.1f}% (TOP 20%)
Quality: {game_info['quality']}/100

TRADE:
BUY: {game_info['run_team']} to win
Size: ${game_info['position_size']:.2f}
TP: +25% (${game_info['take_profit']:.2f})
SL: -5% (${game_info['stop_loss']:.2f})

Expected: +${game_info['expected_value']:.2f}

ACT FAST!
"""
    
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=YOUR_PHONE_NUMBER
        )
        print(f"[OK] SMS sent to {YOUR_PHONE_NUMBER}")
        return True
    except Exception as e:
        print(f"[!] SMS failed: {e}")
        return False

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

def monitor_live_games():
    """Monitor live games and send alerts"""
    
    print("\n" + "="*70)
    print("MONITORING LIVE GAMES")
    print("="*70)
    
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.endpoints import playbyplayv2
    
    monitored_runs = set()  # Track runs we've already alerted on
    
    while True:
        try:
            # Get current games
            scoreboard = scoreboardv2.ScoreboardV2()
            games = scoreboard.get_data_frames()[0]
            
            # Filter for live games only
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
                    
                    # Get last 20 plays to detect runs
                    recent_plays = plays.tail(20)
                    
                    # Simple run detection (you'd expand this with your full feature engineering)
                    # For demo: just check if there's been a scoring run
                    
                    # ... (You'd add your full run detection logic here)
                    # For now, this is a framework
                    
                    print(f"  {away_team} @ {home_team} - checking...")
                    
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
print("STARTING LIVE MONITORING")
print("="*70)
print(f"\nPhone: {YOUR_PHONE_NUMBER}")
print(f"Min Confidence: {MIN_CONFIDENCE*100:.1f}%")
print(f"Min Quality: {MIN_QUALITY}")
print("\nPress Ctrl+C to stop\n")

try:
    monitor_live_games()
except KeyboardInterrupt:
    print("\n\nMonitoring stopped. Good luck with your trades!")

