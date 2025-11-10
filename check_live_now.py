"""
CHECK LIVE GAMES RIGHT NOW
==========================
Shows current scores and quarters for all live games
"""

from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timezone, timedelta
import time

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
print("CHECKING LIVE GAMES NOW")
print("="*70)

est = timezone(timedelta(hours=-5))
current_time = datetime.now(est)
print(f"Current Time: {current_time.strftime('%I:%M %p EST on %B %d, %Y')}")

try:
    print("\nFetching live game data...")
    scoreboard = scoreboardv2.ScoreboardV2()
    games = scoreboard.get_data_frames()[0]
    
    # All games today
    print(f"\nTotal games today: {len(games)}")
    
    # Live games
    live_games = games[games['GAME_STATUS_TEXT'].str.contains('Q|Half', na=False)]
    
    if len(live_games) == 0:
        print("\n[!] No games currently live (in Q1-Q4 or Halftime)")
        print("\nAll games today:")
        for _, game in games.iterrows():
            home = TEAM_NAMES.get(game['HOME_TEAM_ID'], 'HOME')
            away = TEAM_NAMES.get(game['VISITOR_TEAM_ID'], 'AWAY')
            status = game['GAME_STATUS_TEXT']
            print(f"  {away} @ {home} - {status}")
    else:
        print(f"\n{'='*70}")
        print(f"LIVE GAMES NOW ({len(live_games)} games)")
        print("="*70)
        
        for _, game in live_games.iterrows():
            home_id = game['HOME_TEAM_ID']
            away_id = game['VISITOR_TEAM_ID']
            home = TEAM_NAMES.get(home_id, 'HOME')
            away = TEAM_NAMES.get(away_id, 'AWAY')
            away_score = int(game.get('VISITOR_TEAM_SCORE', 0))
            home_score = int(game.get('HOME_TEAM_SCORE', 0))
            status = game['GAME_STATUS_TEXT']
            
            print(f"\n{away} {away_score} @ {home} {home_score}")
            print(f"  Status: {status}")
            print(f"  Game ID: {game['GAME_ID']}")
        
        print("\n" + "="*70)
        print("TO TRADE THESE GAMES:")
        print("="*70)
        print("\n1. Watch the game on TV/stream")
        print("2. When you see a 6-0 run:")
        print("   python check_single_run.py")
        print("3. Enter the details")
        print("4. Get Discord alert")
        print("5. Trade on Kalshi!")
    
    # Upcoming games
    upcoming = games[games['GAME_STATUS_TEXT'].str.contains('pm ET|am ET', na=False)]
    if len(upcoming) > 0:
        print("\n" + "="*70)
        print("UPCOMING GAMES")
        print("="*70)
        for _, game in upcoming.iterrows():
            home = TEAM_NAMES.get(game['HOME_TEAM_ID'], 'HOME')
            away = TEAM_NAMES.get(game['VISITOR_TEAM_ID'], 'AWAY')
            status = game['GAME_STATUS_TEXT']
            print(f"  {away} @ {home} - {status}")
    
    print("\n" + "="*70 + "\n")
    
except Exception as e:
    print(f"\n[!] Error: {e}")
    print("\nThis might mean:")
    print("- NBA API is slow")
    print("- Games haven't started yet")
    print("- Network issues")
    print("\nTry again in a minute")

