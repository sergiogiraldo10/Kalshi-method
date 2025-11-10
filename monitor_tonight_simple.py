"""
SIMPLE GAME MONITOR FOR TONIGHT
================================

Checks live games every 30 seconds and prints basic stats
You still need to spot the 6-0 runs manually and use check_single_run.py
"""

from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import time

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

print("\n" + "="*70)
print("TONIGHT'S GAME MONITOR")
print("="*70)
print(f"Started: {datetime.now().strftime('%I:%M %p ET')}")
print("\nThis will show you live scores every 30 seconds")
print("YOU still need to spot 6-0 runs and use check_single_run.py")
print("\nPress Ctrl+C to stop\n")

def get_live_games():
    """Get current live games"""
    try:
        scoreboard = scoreboardv2.ScoreboardV2()
        games_data = scoreboard.get_data_frames()[0]
        
        # Get line score for detailed quarter info
        line_scores = scoreboard.get_data_frames()[1]
        
        return games_data, line_scores
    except Exception as e:
        print(f"[!] Error fetching games: {e}")
        return None, None

def monitor_games():
    """Monitor live games"""
    
    last_scores = {}
    
    while True:
        try:
            games_data, line_scores = get_live_games()
            
            if games_data is None:
                print(f"[{datetime.now().strftime('%I:%M:%S %p')}] Error fetching data, retrying in 30s...")
                time.sleep(30)
                continue
            
            # Filter for live games
            live_games = games_data[games_data['GAME_STATUS_TEXT'].str.contains('Q|Half|Final', na=False)]
            
            if len(live_games) == 0:
                print(f"[{datetime.now().strftime('%I:%M:%S %p')}] No live games yet. Waiting...")
                time.sleep(30)
                continue
            
            print("\n" + "="*70)
            print(f"LIVE GAMES - {datetime.now().strftime('%I:%M:%S %p ET')}")
            print("="*70)
            
            for _, game in live_games.iterrows():
                game_id = game['GAME_ID']
                home_id = game['HOME_TEAM_ID']
                away_id = game['VISITOR_TEAM_ID']
                home_team = TEAM_NAMES.get(home_id, 'HOME')
                away_team = TEAM_NAMES.get(away_id, 'AWAY')
                
                away_score = game.get('VISITOR_TEAM_SCORE', 0)
                home_score = game.get('HOME_TEAM_SCORE', 0)
                status = game['GAME_STATUS_TEXT']
                
                # Check for scoring bursts
                last_key = f"{game_id}"
                if last_key in last_scores:
                    last_away, last_home = last_scores[last_key]
                    away_diff = away_score - last_away
                    home_diff = home_score - last_home
                    
                    momentum = ""
                    if away_diff >= 6 and home_diff == 0:
                        momentum = f" [!!!] {away_team} ON A RUN! Use check_single_run.py NOW!"
                    elif home_diff >= 6 and away_diff == 0:
                        momentum = f" [!!!] {home_team} ON A RUN! Use check_single_run.py NOW!"
                    elif away_diff >= 4 or home_diff >= 4:
                        momentum = f" [!] Scoring burst detected"
                    
                    print(f"\n{away_team} {away_score} @ {home_team} {home_score} - {status}{momentum}")
                else:
                    print(f"\n{away_team} {away_score} @ {home_team} {home_score} - {status}")
                
                # Update last scores
                last_scores[last_key] = (away_score, home_score)
            
            print("\n" + "="*70)
            print("Checking again in 30 seconds...")
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n[!] Monitor stopped")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")
            print("Retrying in 30 seconds...")
            time.sleep(30)

if __name__ == '__main__':
    try:
        monitor_games()
    except KeyboardInterrupt:
        print("\n\nMonitor stopped. Good luck tonight!")

