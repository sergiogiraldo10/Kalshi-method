"""
Download 2025-26 season data (current season)
"""

import sys
sys.path.append('src')

from data_acquisition import NBADataAcquisition
import os

print("\n" + "="*70)
print("DOWNLOADING 2025-26 SEASON DATA")
print("="*70)
print("\nGetting all games played so far this season...")

# Download 2025-26
downloader = NBADataAcquisition(output_dir='data/raw')
downloader.download_season('2025-26')

print("\n[OK] 2025-26 season data downloaded!")
print("\nNext steps:")
print("1. python extract_features_2025_26.py")
print("2. python add_team_features_2025_26.py")
print("3. python train_for_live_alerts.py")
print("4. python auto_sms_monitor.py (starts monitoring)")

print("="*70 + "\n")

