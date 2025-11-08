"""
NBA Play-by-Play Data Acquisition
Downloads historical NBA play-by-play data using the nba_api library
"""

import os
import time
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder
from nba_api.stats.static import teams
import json

class NBADataAcquisition:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def get_season_games(self, season='2023-24'):
        """
        Get all game IDs for a specific season
        Season format: '2023-24' for 2023-2024 season
        """
        print(f"Fetching games for season {season}...")
        
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            games = gamefinder.get_data_frames()[0]
            
            # Get unique game IDs (each game appears twice, once for each team)
            game_ids = games['GAME_ID'].unique()
            
            print(f"Found {len(game_ids)} games for season {season}")
            return game_ids
            
        except Exception as e:
            print(f"Error fetching games for season {season}: {e}")
            return []
    
    def download_play_by_play(self, game_id, retry_count=3):
        """
        Download play-by-play data for a single game
        """
        for attempt in range(retry_count):
            try:
                pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
                df = pbp.get_data_frames()[0]
                time.sleep(0.6)  # Rate limiting - NBA API recommends 600ms between requests
                return df
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  Error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed after {retry_count} attempts: {e}")
                    return None
    
    def download_season(self, season='2023-24'):
        """
        Download all play-by-play data for a season
        """
        print(f"\n{'='*60}")
        print(f"Starting download for season {season}")
        print(f"{'='*60}\n")
        
        game_ids = self.get_season_games(season)
        
        if game_ids is None or len(game_ids) == 0:
            print(f"No games found for season {season}")
            return
        
        all_pbp_data = []
        successful_downloads = 0
        failed_downloads = 0
        
        for i, game_id in enumerate(game_ids):
            print(f"[{i+1}/{len(game_ids)}] Downloading game {game_id}...", end=' ')
            
            pbp_df = self.download_play_by_play(game_id)
            
            if pbp_df is not None and len(pbp_df) > 0:
                pbp_df['SEASON'] = season
                all_pbp_data.append(pbp_df)
                successful_downloads += 1
                print(f"[OK] {len(pbp_df)} plays")
            else:
                failed_downloads += 1
                print("[FAILED]")
            
            # Save progress every 50 games
            if (i + 1) % 50 == 0:
                self._save_intermediate(all_pbp_data, season, i + 1)
        
        # Final save
        if all_pbp_data:
            combined_df = pd.concat(all_pbp_data, ignore_index=True)
            output_file = os.path.join(self.output_dir, f'pbp_{season.replace("-", "_")}.csv')
            combined_df.to_csv(output_file, index=False)
            
            print(f"\n{'='*60}")
            print(f"Season {season} complete!")
            print(f"  Successful: {successful_downloads} games")
            print(f"  Failed: {failed_downloads} games")
            print(f"  Total plays: {len(combined_df):,}")
            print(f"  Saved to: {output_file}")
            print(f"{'='*60}\n")
        else:
            print(f"No data downloaded for season {season}")
    
    def _save_intermediate(self, data_list, season, count):
        """Save intermediate progress"""
        if data_list:
            combined_df = pd.concat(data_list, ignore_index=True)
            output_file = os.path.join(self.output_dir, f'pbp_{season.replace("-", "_")}_partial_{count}.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"\n  [Checkpoint] Saved {count} games to {output_file}")
    
    def download_multiple_seasons(self, seasons):
        """
        Download multiple seasons
        seasons: list of season strings, e.g., ['2017-18', '2018-19', ...]
        """
        print(f"\n{'#'*60}")
        print(f"# NBA Data Acquisition")
        print(f"# Downloading {len(seasons)} seasons")
        print(f"# Estimated time: {len(seasons) * 30} minutes")
        print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        for season in seasons:
            season_start = time.time()
            self.download_season(season)
            season_elapsed = time.time() - season_start
            print(f"Season {season} took {season_elapsed/60:.1f} minutes\n")
        
        total_elapsed = time.time() - start_time
        print(f"\n{'#'*60}")
        print(f"# All downloads complete!")
        print(f"# Total time: {total_elapsed/60:.1f} minutes")
        print(f"{'#'*60}\n")


def main():
    """
    Main function to download all required seasons
    """
    # Initialize downloader
    downloader = NBADataAcquisition(output_dir='data/raw')
    
    # Define seasons to download (2017-2024 + current season)
    # Note: NBA seasons are formatted as YYYY-YY (e.g., '2023-24' for 2023-2024 season)
    seasons = [
        '2017-18',  # Training data
        '2018-19',  # Training data
        '2019-20',  # Training data
        '2020-21',  # Training data
        '2021-22',  # Training data
        '2022-23',  # Validation data
        '2023-24',  # Test data
        '2024-25',  # Additional test data (current season)
    ]
    
    print("This will download approximately 8 seasons of NBA play-by-play data.")
    print("Estimated time: 3-4 hours")
    print("Estimated data size: ~500MB")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Download all seasons
    downloader.download_multiple_seasons(seasons)
    
    # Print summary
    print("\nData acquisition complete!")
    print("\nNext steps:")
    print("1. Run data validation (src/data_validation.py)")
    print("2. Process and engineer features (src/feature_engineering.py)")


if __name__ == '__main__':
    main()

