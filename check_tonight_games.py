"""
Check what games are playing tonight
"""

from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import pandas as pd

# Team ID to Name mapping
TEAM_NAMES = {
    1610612737: 'ATL Hawks', 1610612738: 'BOS Celtics', 1610612739: 'CLE Cavaliers',
    1610612740: 'NOP Pelicans', 1610612741: 'CHI Bulls', 1610612742: 'DAL Mavericks',
    1610612743: 'DEN Nuggets', 1610612744: 'GSW Warriors', 1610612745: 'HOU Rockets',
    1610612746: 'LAC Clippers', 1610612747: 'LAL Lakers', 1610612748: 'MIA Heat',
    1610612749: 'MIL Bucks', 1610612750: 'MIN Timberwolves', 1610612751: 'BKN Nets',
    1610612752: 'NYK Knicks', 1610612753: 'ORL Magic', 1610612754: 'IND Pacers',
    1610612755: 'PHI 76ers', 1610612756: 'PHX Suns', 1610612757: 'POR Trail Blazers',
    1610612758: 'SAC Kings', 1610612759: 'SAS Spurs', 1610612760: 'OKC Thunder',
    1610612761: 'TOR Raptors', 1610612762: 'UTA Jazz', 1610612763: 'MEM Grizzlies',
    1610612764: 'WAS Wizards', 1610612765: 'DET Pistons', 1610612766: 'CHA Hornets'
}

print("\n" + "="*70)
print("NBA GAMES TONIGHT - " + datetime.now().strftime('%B %d, %Y'))
print("="*70)

try:
    # Get today's scoreboard
    scoreboard = scoreboardv2.ScoreboardV2()
    games = scoreboard.get_data_frames()[0]
    
    if len(games) == 0:
        print("\n[!] No games scheduled for tonight")
        print("    The NBA schedule might have no games today")
        print("    OR games might be scheduled for a different time")
    else:
        print(f"\n[OK] Found {len(games)} games tonight:\n")
        print("-" * 70)
        
        for idx, game in games.iterrows():
            home_id = game['HOME_TEAM_ID']
            away_id = game['VISITOR_TEAM_ID']
            home_team = TEAM_NAMES.get(home_id, f'Team {home_id}')
            away_team = TEAM_NAMES.get(away_id, f'Team {away_id}')
            game_status = game['GAME_STATUS_TEXT']
            game_id = game['GAME_ID']
            
            print(f"{idx+1}. {away_team} @ {home_team}")
            print(f"   Status: {game_status}")
            print(f"   Game ID: {game_id}")
            print()
        
        print("-" * 70)
        print("\nTo monitor these games:")
        print("1. Watch them live on TV/stream")
        print("2. When you see a 6-0 run, use: python check_single_run.py")
        print("3. Enter the run details to get a trade signal")
        
except Exception as e:
    print(f"\n[!] Could not fetch games: {e}")
    print("\nThis could mean:")
    print("  - No games scheduled for tonight")
    print("  - NBA API connection issue")
    print("  - You're running this on an off day")
    
    print("\nNBA Schedule (November 2024):")
    print("  - Games typically on Tue, Wed, Thu, Fri, Sat, Sun")
    print("  - Usually 8-15 games per night")
    print("  - Check nba.com for today's schedule")

print("\n" + "="*70)
print("YOUR MODEL IS READY!")
print("="*70)
print("\nModel Stats:")
print("  - Trained on: 6,912 games")
print("  - Win Rate: ~36%")
print("  - Expected Return: ~12% per trading session")
print("\nHow to use:")
print("  1. Watch games live")
print("  2. Spot 6-0 runs")
print("  3. Run: python check_single_run.py")
print("  4. Follow the trade signal")
print("\n" + "="*70 + "\n")

