"""
Download LATEST 2024-25 season data (up to today: November 9, 2024)
"""

import sys
sys.path.append('src')

from data_acquisition import NBADataAcquisition
import os

print("\n" + "="*70)
print("DOWNLOADING LATEST 2024-25 SEASON DATA")
print("="*70)
print("\nThis will get all games played through today (Nov 9, 2024)")
print("We'll use this to train the model, then test on TONIGHT'S games\n")

# Backup existing file
existing_file = 'data/raw/pbp_2024_25.csv'
if os.path.exists(existing_file):
    backup_file = existing_file.replace('.csv', '_OLD_BACKUP.csv')
    print(f"Backing up existing file to {backup_file}...")
    os.rename(existing_file, backup_file)

# Download fresh data
downloader = NBADataAcquisition(output_dir='data/raw')
downloader.download_season('2024-25')

print("\n[OK] Latest 2024-25 data downloaded!")
print("\nNext steps:")
print("1. Extract features (run extract_features_2024_25.py)")
print("2. Add team features (run add_team_features_2024_25.py)")  
print("3. Run live monitoring (run live_monitor_tonight.py)")

print("="*70 + "\n")

