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
                time.sleep(0.6)  # Reduced rate limiting - faster download (was 0.6s)
                return df
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  Error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed after {retry_count} attempts: {e}")
                    return None
    
    def _load_existing_data(self, season):
        """Load existing partial files and return already downloaded game IDs"""
        import glob
        pattern = os.path.join(self.output_dir, f'pbp_{season.replace("-", "_")}_partial_*.csv')
        partial_files = glob.glob(pattern)
        
        # Filter out backup files
        partial_files = [f for f in partial_files if 'BACKUP' not in f]
        
        if not partial_files:
            return [], []
        
        # Find the latest partial file
        latest_file = max(partial_files, key=lambda x: int(x.split('_partial_')[1].split('.csv')[0]))
        
        print(f"  Found existing progress: {os.path.basename(latest_file)}")
        
        # Load the data
        df = pd.read_csv(latest_file)
        # Convert game IDs to strings to match API format (e.g., '0022001069')
        already_downloaded_ids = [str(int(gid)).zfill(10) if len(str(int(gid))) < 10 else str(int(gid)) for gid in df['GAME_ID'].unique()]
        
        print(f"  Already downloaded: {len(already_downloaded_ids)} games")
        print(f"  [NOTE] Will skip these games and only download new ones")
        
        # DON'T return the dataframe - we only need the game IDs for skipping
        # Returning the df causes duplicates when resuming
        return already_downloaded_ids, []
    
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
        
        # Load existing progress
        already_downloaded, all_pbp_data = self._load_existing_data(season)
        
        # Skip already downloaded games
        remaining_game_ids = [gid for gid in game_ids if gid not in already_downloaded]
        
        if not remaining_game_ids:
            print(f"All games already downloaded for season {season}!")
            return
        
        print(f"  Remaining to download: {len(remaining_game_ids)} games\n")
        
        already_count = len(already_downloaded)
        successful_downloads = already_count
        failed_downloads = 0
        
        for i, game_id in enumerate(remaining_game_ids):
            current_num = already_count + i + 1
            total_games = len(game_ids)
            print(f"[{current_num}/{total_games}] Downloading game {game_id}...", end=' ')
            
            pbp_df = self.download_play_by_play(game_id)
            
            if pbp_df is not None and len(pbp_df) > 0:
                pbp_df['SEASON'] = season
                all_pbp_data.append(pbp_df)
                successful_downloads += 1
                print(f"[OK] {len(pbp_df)} plays")
            else:
                failed_downloads += 1
                print("[FAILED]")
            
            # Save progress every 50 games (attempts, not successes)
            if (i + 1) % 50 == 0:
                self._save_intermediate(all_pbp_data, season, successful_downloads + failed_downloads, already_count)
        
        # Final save
        if all_pbp_data or already_count > 0:
            # If resuming, merge old data with new data
            if already_count > 0:
                import glob
                pattern = os.path.join(self.output_dir, f'pbp_{season.replace("-", "_")}_partial_*.csv')
                partial_files = glob.glob(pattern)
                # Filter out backup files
                partial_files = [f for f in partial_files if 'BACKUP' not in f]
                if partial_files:
                    latest_file = max(partial_files, key=lambda x: int(x.split('_partial_')[1].split('.csv')[0]))
                    print(f"\n  Loading existing data from {os.path.basename(latest_file)}...")
                    old_df = pd.read_csv(latest_file)
                    all_pbp_data.insert(0, old_df)
            
            if all_pbp_data:
                combined_df = pd.concat(all_pbp_data, ignore_index=True)
                # Remove any duplicates (just in case)
                combined_df = combined_df.drop_duplicates(subset=['GAME_ID', 'EVENTNUM'], keep='first')
                
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
                print(f"No new data downloaded for season {season}")
        else:
            print(f"No data downloaded for season {season}")
    
    def _save_intermediate(self, new_data_list, season, total_attempts, already_count=0):
        """Save intermediate progress - ONLY saves cumulative data including old progress"""
        if new_data_list or already_count > 0:
            all_parts = []
            
            # Load existing data if any
            if already_count > 0:
                import glob
                pattern = os.path.join(self.output_dir, f'pbp_{season.replace("-", "_")}_partial_*.csv')
                partial_files = glob.glob(pattern)
                # Filter out backup files
                partial_files = [f for f in partial_files if 'BACKUP' not in f]
                if partial_files:
                    latest_file = max(partial_files, key=lambda x: int(x.split('_partial_')[1].split('.csv')[0]))
                    old_df = pd.read_csv(latest_file)
                    all_parts.append(old_df)
            
            # Add new data
            if new_data_list:
                all_parts.extend(new_data_list)
            
            if all_parts:
                combined_df = pd.concat(all_parts, ignore_index=True)
                # Remove duplicates
                combined_df = combined_df.drop_duplicates(subset=['GAME_ID', 'EVENTNUM'], keep='first')
                
                output_file = os.path.join(self.output_dir, f'pbp_{season.replace("-", "_")}_partial_{total_attempts}.csv')
                combined_df.to_csv(output_file, index=False)
                unique_games = combined_df['GAME_ID'].nunique()
                print(f"\n  [Checkpoint] Saved {unique_games} games ({len(combined_df):,} plays) to {os.path.basename(output_file)}")
    
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
    
    # Define seasons to download (2020-2025)
    # Note: NBA seasons are formatted as YYYY-YY (e.g., '2023-24' for 2023-2024 season)
    # 2017-2020 already downloaded from Kaggle
    seasons = [
        '2020-21',  # Training data
        '2021-22',  # Training data
        '2022-23',  # Validation data
        '2023-24',  # Test data
        '2024-25',  # Additional test data (current season)
    ]
    
    print("This will download 5 seasons of NBA play-by-play data (2020-2025).")
    print("Note: 2017-2020 already available from Kaggle data")
    print("Estimated time: 2-3 hours")
    print("Estimated data size: ~300MB")
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

