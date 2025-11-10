"""
DISCORD WEBHOOK SETUP
=====================

Free, instant alerts to your Discord server!
"""

import requests
from datetime import datetime

# ============================================================================
# SETUP: Replace this with your Discord webhook URL
# ============================================================================
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1437218996230422540/G7tLYbCACSl-L07QrqC6pFuOpubXERWS3Zp00aZ-aa4TwH7kLsGadTGwCdW5hvE7jDqe'

# Example: 'https://discord.com/api/webhooks/123456789/abcdefg...'

# To get your webhook URL:
# 1. Open Discord, create a server (or use existing)
# 2. Go to Server Settings -> Integrations -> Webhooks
# 3. Click "New Webhook"
# 4. Name it "Ignition AI"
# 5. Copy the Webhook URL
# 6. Paste it above

# ============================================================================
# Phone number removed per user request
YOUR_PHONE = None

def send_discord_alert(game_info):
    """
    Send trade alert to Discord
    """
    
    # Format the message with rich embed
    embed = {
        "title": "🏀 IGNITION AI - TRADE SIGNAL 🏀",
        "description": f"High-confidence momentum trade detected!",
        "color": 15258703,  # Orange color
        "fields": [
            {
                "name": "📍 Game",
                "value": f"**{game_info['away_team']} @ {game_info['home_team']}**\nScore: {game_info['away_score']}-{game_info['home_score']}",
                "inline": False
            },
            {
                "name": "🏃 Momentum Run",
                "value": f"**{game_info['run_team']}** on a **{game_info['run_score']}-{game_info['opp_score']} run**\nQ{game_info['quarter']} - {game_info['time_left']} remaining",
                "inline": False
            },
            {
                "name": "📊 Model Analysis",
                "value": f"Win Probability: **{game_info['win_probability']*100:.1f}%**\nConfidence: **{game_info['confidence']*100:.1f}%** (TOP 20%)\nQuality: **{game_info['quality']}/100**",
                "inline": False
            },
            {
                "name": "💰 KALSHI TRADE",
                "value": f"**BUY:** {game_info['run_team']} wins\n**Entry:** {game_info['kalshi_entry_price']}¢ per contract\n**Contracts:** {game_info['num_contracts']}\n**Total Cost:** ${game_info['total_cost']:.2f}",
                "inline": True
            },
            {
                "name": "🎯 Exit Strategy",
                "value": f"**Take Profit:** {game_info['kalshi_take_profit']}¢ (+{game_info['take_profit_pct']*100:.0f}%)\n**Stop Loss:** {game_info['kalshi_stop_loss']}¢ ({game_info['stop_loss_pct']*100:.0f}%)",
                "inline": True
            },
            {
                "name": "💡 Expected Value",
                "value": f"**${game_info['expected_value']:+.2f}** per trade\n({game_info['risk_reward']:.1f}:1 risk/reward)",
                "inline": False
            },
            {
                "name": "⚡ ACTION REQUIRED",
                "value": f"Go to Kalshi NOW and buy contracts!\nTime-sensitive - momentum runs don't last!",
                "inline": False
            }
        ],
        "footer": {
            "text": f"Ignition AI | {datetime.now().strftime('%I:%M:%S %p EST')}"
        }
    }
    
    data = {
        "username": "Ignition AI",
        "embeds": [embed]
    }
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        if response.status_code == 204:
            print(f"[OK] Discord alert sent!")
            return True
        else:
            print(f"[!] Discord error: {response.status_code}")
            return False
    except Exception as e:
        print(f"[!] Failed to send Discord alert: {e}")
        return False

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_discord_webhook():
    """Test your Discord webhook with a demo alert"""
    
    if DISCORD_WEBHOOK_URL == 'YOUR_WEBHOOK_URL_HERE':
        print("\n" + "="*70)
        print("DISCORD WEBHOOK NOT CONFIGURED")
        print("="*70)
        print("\nTo setup:")
        print("1. Open Discord, create a server")
        print("2. Server Settings -> Integrations -> Webhooks")
        print("3. Create new webhook, copy URL")
        print("4. Edit this file and paste URL at line 15")
        print("\nThen run: python discord_webhook_setup.py")
        return
    
    print("\n" + "="*70)
    print("TESTING DISCORD WEBHOOK")
    print("="*70)
    print(f"\nSending test alert to Discord...")
    
    # Demo trade signal
    demo_game = {
        'away_team': 'LAL Lakers',
        'home_team': 'GSW Warriors',
        'away_score': 45,
        'home_score': 38,
        'quarter': 2,
        'time_left': '5:30',
        'run_team': 'LAL',
        'run_score': 6,
        'opp_score': 0,
        'win_probability': 0.625,
        'confidence': 0.352,
        'quality': 65,
        'kalshi_entry_price': 40,
        'num_contracts': 100,
        'total_cost': 40.00,
        'kalshi_take_profit': 55,
        'kalshi_stop_loss': 30,
        'take_profit_pct': 0.375,
        'stop_loss_pct': -0.25,
        'expected_value': 2.18,
        'risk_reward': 1.5
    }
    
    success = send_discord_alert(demo_game)
    
    if success:
        print("\n[OK] Test alert sent!")
        print("Check your Discord server for the message")
    else:
        print("\n[!] Test failed. Check your webhook URL")
    
    print("="*70 + "\n")

# Run test if executed directly
if __name__ == '__main__':
    test_discord_webhook()

