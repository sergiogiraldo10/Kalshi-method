"""Debug the resume logic"""
import pandas as pd
import glob
from nba_api.stats.endpoints import leaguegamefinder

# Load the latest partial file
pattern = 'data/raw/pbp_2020_21_partial_*.csv'
partial_files = glob.glob(pattern)
latest_file = max(partial_files, key=lambda x: int(x.split('_partial_')[1].split('.csv')[0]))

print(f"Latest partial file: {latest_file}")

# Load already downloaded games
df = pd.read_csv(latest_file)
already_downloaded = df['GAME_ID'].unique().tolist()
print(f"\nGames already downloaded: {len(already_downloaded)}")

# Get all games for the season
print("\nFetching all games for 2020-21 season...")
gamefinder = leaguegamefinder.LeagueGameFinder(
    season_nullable='2020-21',
    season_type_nullable='Regular Season'
)
games = gamefinder.get_data_frames()[0]
all_game_ids = games['GAME_ID'].unique()

print(f"Total games in 2020-21 season: {len(all_game_ids)}")

# Calculate remaining
remaining = [gid for gid in all_game_ids if gid not in already_downloaded]
print(f"Remaining to download: {len(remaining)}")

print(f"\nResume will start from game #{len(already_downloaded) + 1} of {len(all_game_ids)}")
print(f"That's {len(remaining)} games left to download")

