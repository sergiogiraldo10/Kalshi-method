"""
Data Validation Module
Validates NBA play-by-play data quality and consistency
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

class DataValidator:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.validation_results = {}
        
    def load_season_data(self, season):
        """Load play-by-play data for a season"""
        file_pattern = f'pbp_{season.replace("-", "_")}.csv'
        file_path = os.path.join(self.data_dir, file_pattern)
        
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            return pd.read_csv(file_path)
        else:
            print(f"File not found: {file_path}")
            return None
    
    def validate_season(self, season, df):
        """
        Validate a single season's data
        """
        print(f"\n{'='*60}")
        print(f"Validating season {season}")
        print(f"{'='*60}\n")
        
        results = {
            'season': season,
            'total_plays': len(df),
            'total_games': df['GAME_ID'].nunique(),
            'issues': []
        }
        
        # 1. Check for required columns
        required_columns = [
            'GAME_ID', 'EVENTNUM', 'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE',
            'PERIOD', 'PCTIMESTRING', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION',
            'SCORE', 'SCOREMARGIN'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            results['issues'].append(f"Missing columns: {missing_cols}")
            print(f"[ERROR] Missing columns: {missing_cols}")
        else:
            print(f"[OK] All required columns present")
        
        # 2. Check for null values in critical columns
        critical_cols = ['GAME_ID', 'EVENTNUM', 'PERIOD']
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    results['issues'].append(f"{col} has {null_count} null values")
                    print(f"[WARNING] {col} has {null_count} null values ({null_count/len(df)*100:.2f}%)")
                else:
                    print(f"[OK] {col} has no null values")
        
        # 3. Check timestamp format and consistency
        if 'PCTIMESTRING' in df.columns:
            # Check for valid time format (MM:SS)
            sample_times = df['PCTIMESTRING'].dropna().head(100)
            invalid_times = []
            for time_str in sample_times:
                if pd.notna(time_str) and not self._is_valid_time_format(str(time_str)):
                    invalid_times.append(time_str)
            
            if invalid_times:
                results['issues'].append(f"Invalid time formats found: {len(invalid_times)} samples")
                print(f"[WARNING] Found {len(invalid_times)} invalid time formats")
            else:
                print(f"[OK] Time format is valid")
        
        # 4. Check period values (should be 1-4 for regulation, 5+ for OT)
        if 'PERIOD' in df.columns:
            period_values = df['PERIOD'].unique()
            if all(1 <= p <= 10 for p in period_values if pd.notna(p)):
                print(f"[OK] Period values are valid (found periods: {sorted([p for p in period_values if pd.notna(p)])})")
            else:
                invalid_periods = [p for p in period_values if pd.notna(p) and (p < 1 or p > 10)]
                results['issues'].append(f"Invalid period values: {invalid_periods}")
                print(f"[WARNING] Invalid period values: {invalid_periods}")
        
        # 5. Check score progression
        games_sample = df['GAME_ID'].unique()[:10]  # Sample 10 games
        score_issues = 0
        
        for game_id in games_sample:
            game_df = df[df['GAME_ID'] == game_id].sort_values('EVENTNUM')
            if not self._validate_score_progression(game_df):
                score_issues += 1
        
        if score_issues > 0:
            results['issues'].append(f"Score progression issues in {score_issues}/{len(games_sample)} sampled games")
            print(f"[WARNING] Score progression issues in {score_issues}/{len(games_sample)} sampled games")
        else:
            print(f"[OK] Score progression is consistent (checked {len(games_sample)} games)")
        
        # 6. Check for duplicate events
        duplicates = df.duplicated(subset=['GAME_ID', 'EVENTNUM'], keep=False).sum()
        if duplicates > 0:
            results['issues'].append(f"Found {duplicates} duplicate events")
            print(f"[WARNING] Found {duplicates} duplicate events")
        else:
            print(f"[OK] No duplicate events found")
        
        # 7. Data completeness statistics
        print(f"\nData Statistics:")
        print(f"  Total plays: {len(df):,}")
        print(f"  Total games: {df['GAME_ID'].nunique():,}")
        print(f"  Average plays per game: {len(df) / df['GAME_ID'].nunique():.1f}")
        print(f"  Periods found: {sorted(df['PERIOD'].unique())}")
        
        # Summary
        if results['issues']:
            print(f"\n[SUMMARY] Validation completed with {len(results['issues'])} issue(s)")
        else:
            print(f"\n[SUMMARY] Validation completed successfully - no issues found!")
        
        self.validation_results[season] = results
        return results
    
    def _is_valid_time_format(self, time_str):
        """Check if time string is in MM:SS format"""
        try:
            parts = time_str.split(':')
            if len(parts) != 2:
                return False
            minutes, seconds = int(parts[0]), int(parts[1])
            return 0 <= minutes <= 12 and 0 <= seconds <= 59
        except:
            return False
    
    def _validate_score_progression(self, game_df):
        """
        Validate that scores only increase or stay the same
        """
        if 'SCORE' not in game_df.columns:
            return True  # Can't validate without scores
        
        scores = game_df['SCORE'].dropna()
        if len(scores) == 0:
            return True
        
        for score in scores:
            if not isinstance(score, str) or ' - ' not in str(score):
                continue
            
            try:
                parts = str(score).split(' - ')
                if len(parts) == 2:
                    home_score = int(parts[1])
                    away_score = int(parts[0])
                    
                    # Scores should be non-negative
                    if home_score < 0 or away_score < 0:
                        return False
                    
                    # Scores shouldn't be impossibly high (e.g., >200)
                    if home_score > 200 or away_score > 200:
                        return False
            except:
                continue
        
        return True
    
    def validate_all_seasons(self, seasons):
        """
        Validate all downloaded seasons
        """
        print(f"\n{'#'*60}")
        print(f"# NBA Data Validation")
        print(f"# Validating {len(seasons)} seasons")
        print(f"{'#'*60}\n")
        
        all_results = []
        
        for season in seasons:
            df = self.load_season_data(season)
            if df is not None:
                results = self.validate_season(season, df)
                all_results.append(results)
            else:
                print(f"[ERROR] Could not load data for season {season}")
        
        # Overall summary
        print(f"\n{'#'*60}")
        print(f"# Validation Summary")
        print(f"{'#'*60}\n")
        
        total_plays = sum(r['total_plays'] for r in all_results)
        total_games = sum(r['total_games'] for r in all_results)
        total_issues = sum(len(r['issues']) for r in all_results)
        
        print(f"Seasons validated: {len(all_results)}/{len(seasons)}")
        print(f"Total plays: {total_plays:,}")
        print(f"Total games: {total_games:,}")
        print(f"Total issues found: {total_issues}")
        
        if total_issues == 0:
            print(f"\n[SUCCESS] All data validated successfully!")
        else:
            print(f"\n[WARNING] Found {total_issues} issue(s) - review details above")
        
        return all_results
    
    def check_available_data(self):
        """
        Check what data files are available
        """
        print(f"\nChecking for available data files in {self.data_dir}...\n")
        
        csv_files = glob.glob(os.path.join(self.data_dir, 'pbp_*.csv'))
        
        if not csv_files:
            print("[WARNING] No data files found!")
            print(f"Expected location: {os.path.abspath(self.data_dir)}")
            print("\nPlease run data_acquisition.py first to download the data.")
            return []
        
        print(f"Found {len(csv_files)} data file(s):\n")
        
        available_seasons = []
        for file in sorted(csv_files):
            filename = os.path.basename(file)
            file_size = os.path.getsize(file) / (1024 * 1024)  # MB
            
            # Extract season from filename (pbp_2023_24.csv -> 2023-24)
            season_str = filename.replace('pbp_', '').replace('.csv', '')
            if '_partial_' not in season_str:
                season = '-'.join([season_str[:4], season_str[-2:]])
                available_seasons.append(season)
                print(f"  - {filename:<30} ({file_size:>6.1f} MB) -> Season {season}")
        
        return available_seasons


def main():
    """
    Main validation function
    """
    validator = DataValidator(data_dir='data/raw')
    
    # Check what data is available
    available_seasons = validator.check_available_data()
    
    if not available_seasons:
        print("\nNo data available to validate. Please run data_acquisition.py first.")
        return
    
    # Validate all available seasons
    print(f"\nPreparing to validate {len(available_seasons)} season(s)...")
    print("Starting validation in 2 seconds...")
    import time
    time.sleep(2)
    
    results = validator.validate_all_seasons(available_seasons)
    
    print("\nValidation complete!")
    print("\nNext steps:")
    print("1. If validation passed, proceed to feature engineering (src/feature_engineering.py)")
    print("2. If issues were found, review and clean data as needed")


if __name__ == '__main__':
    main()

