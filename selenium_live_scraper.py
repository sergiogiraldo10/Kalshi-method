"""
SELENIUM LIVE NBA SCRAPER
==========================

Educational tool for scraping live NBA scores using Selenium.
For personal use only. Use responsibly.

Advantages over NBA API:
- 5-10 second delay (vs 30 seconds)
- More reliable during games
- Can see exactly what fans see

Note: Respect the site's resources - use reasonable delays
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
from datetime import datetime, timezone, timedelta
import re

print("\n" + "="*70)
print("SELENIUM LIVE NBA SCRAPER")
print("="*70)
print("\nSetting up browser (auto-installing ChromeDriver if needed)...")

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in background
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--window-size=1920,1080')
# Look more human-like
chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# Team mappings
TEAM_ABBREV = {
    'ATL': 'ATL Hawks', 'BOS': 'BOS Celtics', 'BKN': 'BKN Nets', 'CHA': 'CHA Hornets',
    'CHI': 'CHI Bulls', 'CLE': 'CLE Cavaliers', 'DAL': 'DAL Mavericks', 'DEN': 'DEN Nuggets',
    'DET': 'DET Pistons', 'GSW': 'GSW Warriors', 'HOU': 'HOU Rockets', 'IND': 'IND Pacers',
    'LAC': 'LAC Clippers', 'LAL': 'LAL Lakers', 'MEM': 'MEM Grizzlies', 'MIA': 'MIA Heat',
    'MIL': 'MIL Bucks', 'MIN': 'MIN Timberwolves', 'NOP': 'NOP Pelicans', 'NYK': 'NYK Knicks',
    'OKC': 'OKC Thunder', 'ORL': 'ORL Magic', 'PHI': 'PHI 76ers', 'PHX': 'PHX Suns',
    'POR': 'POR Trail Blazers', 'SAC': 'SAC Kings', 'SAS': 'SAS Spurs', 'TOR': 'TOR Raptors',
    'UTA': 'UTA Jazz', 'WAS': 'WAS Wizards'
}

class NBALiveScraper:
    def __init__(self):
        """Initialize the scraper"""
        try:
            # Auto-install and use ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            print("[OK] Browser initialized (headless mode)")
        except Exception as e:
            print(f"\n[!] Error initializing browser: {e}")
            print("\nMake sure you have:")
            print("  1. Chrome installed on your computer")
            print("  2. Selenium installed: pip install selenium webdriver-manager")
            print("\nTroubleshooting:")
            print("  - Check Chrome version: chrome://version")
            print("  - Try running in non-headless mode (comment out --headless)")
            raise
    
    def get_live_scores_espn(self):
        """Scrape live scores from ESPN"""
        try:
            url = "https://www.espn.com/nba/scoreboard"
            print(f"\n[OK] Loading ESPN scoreboard...")
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(3)  # Respectful delay
            
            games = []
            
            # Find all game cards
            try:
                game_elements = self.driver.find_elements(By.CLASS_NAME, "Scoreboard")
                
                if not game_elements:
                    game_elements = self.driver.find_elements(By.CLASS_NAME, "ScoreboardScoreCell")
                
                print(f"[OK] Found {len(game_elements)} game elements")
                
                # Parse each game
                for game_elem in game_elements:
                    try:
                        # Get teams
                        teams = game_elem.find_elements(By.CLASS_NAME, "ScoreCell__TeamName")
                        if len(teams) < 2:
                            continue
                        
                        away_team = teams[0].text.strip()
                        home_team = teams[1].text.strip()
                        
                        # Get scores
                        scores = game_elem.find_elements(By.CLASS_NAME, "ScoreCell__Score")
                        if len(scores) < 2:
                            continue
                        
                        away_score = scores[0].text.strip()
                        home_score = scores[1].text.strip()
                        
                        # Get status/quarter
                        try:
                            status_elem = game_elem.find_element(By.CLASS_NAME, "ScoreCell__NetworkItem")
                            status = status_elem.text.strip()
                        except:
                            status = "Unknown"
                        
                        # Only include live games
                        if away_score and home_score and away_score.isdigit():
                            games.append({
                                'away_team': TEAM_ABBREV.get(away_team, away_team),
                                'home_team': TEAM_ABBREV.get(home_team, home_team),
                                'away_score': int(away_score),
                                'home_score': int(home_score),
                                'status': status,
                                'is_live': any(q in status for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Half', 'OT'])
                            })
                    except Exception as e:
                        continue
                
                return games
                
            except Exception as e:
                print(f"[!] Error parsing games: {e}")
                return []
        
        except Exception as e:
            print(f"[!] Error loading ESPN: {e}")
            return []
    
    def get_live_scores_nba(self):
        """Scrape live scores from NBA.com (backup method)"""
        try:
            url = "https://www.nba.com/games"
            print(f"\n[OK] Loading NBA.com scoreboard...")
            self.driver.get(url)
            
            # Wait for dynamic content
            time.sleep(5)  # NBA.com is heavily JS-based
            
            games = []
            
            # NBA.com structure (this may need updates if they change layout)
            try:
                # Look for game cards
                game_cards = self.driver.find_elements(By.CSS_SELECTOR, "[class*='GameCard']")
                
                print(f"[OK] Found {len(game_cards)} games on NBA.com")
                
                for card in game_cards:
                    try:
                        # Extract team names and scores
                        # Note: NBA.com structure changes frequently
                        text = card.text
                        # Basic parsing (would need to be more robust)
                        lines = text.split('\n')
                        
                        # This is a simplified parser
                        # Real implementation would need detailed CSS selectors
                        
                    except:
                        continue
                
                return games
            
            except Exception as e:
                print(f"[!] Error parsing NBA.com: {e}")
                return []
        
        except Exception as e:
            print(f"[!] Error loading NBA.com: {e}")
            return []
    
    def close(self):
        """Clean up"""
        try:
            self.driver.quit()
            print("\n[OK] Browser closed")
        except:
            pass

def test_scraper():
    """Test the scraper"""
    scraper = None
    try:
        scraper = NBALiveScraper()
        
        print("\n" + "="*70)
        print("TESTING LIVE SCRAPER")
        print("="*70)
        
        # Try ESPN first
        games = scraper.get_live_scores_espn()
        
        if games:
            print(f"\n[OK] Found {len(games)} games")
            print("\n" + "="*70)
            print("LIVE GAMES")
            print("="*70)
            
            live_games = [g for g in games if g['is_live']]
            
            if live_games:
                for game in live_games:
                    print(f"\n{game['away_team']} {game['away_score']} @ {game['home_team']} {game['home_score']}")
                    print(f"  Status: {game['status']}")
            else:
                print("\n[!] No live games right now")
                print("\nAll games today:")
                for game in games:
                    print(f"  {game['away_team']} @ {game['home_team']} - {game['status']}")
        else:
            print("\n[!] No games found")
            print("This could mean:")
            print("  - No games today")
            print("  - Site structure changed")
            print("  - Connection issues")
        
        print("\n" + "="*70)
    
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if scraper:
            scraper.close()

if __name__ == '__main__':
    print("\nNOTE: This is for educational/personal use only")
    print("Use responsibly with reasonable delays")
    print("="*70)
    
    test_scraper()

