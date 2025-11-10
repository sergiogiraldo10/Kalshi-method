"""
Download ONLY 2024-25 Season
Quick script to get current season data
"""

import os
import sys
sys.path.append('src')

from data_acquisition import NBADataAcquisition

def main():
    print("\n" + "="*70)
    print("DOWNLOADING 2024-25 SEASON ONLY")
    print("="*70)
    
    # Initialize downloader
    downloader = NBADataAcquisition(output_dir='data/raw')
    
    # Download only 2024-25
    season = '2024-25'
    
    print(f"\nDownloading {season} season...")
    print("This is the current season, so it will have partial data")
    print("(games played so far in 2024-25)\n")
    
    downloader.download_season(season)
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    
    # Check what we got
    output_file = f'data/raw/pbp_{season.replace("-", "_")}.csv'
    if os.path.exists(output_file):
        import pandas as pd
        df = pd.read_csv(output_file)
        print(f"\n  File: {output_file}")
        print(f"  Games: {df['GAME_ID'].nunique()}")
        print(f"  Total plays: {len(df):,}")
        print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    print("\n  [OK] Ready to test model on 2024-25 season!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()

