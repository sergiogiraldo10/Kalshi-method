"""Debug why checkpoint says 900 games"""
from src.data_acquisition import NBADataAcquisition

# Simulate what happens when starting 2021-22 season
downloader = NBADataAcquisition(output_dir='data/raw')

# Check what _load_existing_data returns for 2021-22
already_downloaded, _ = downloader._load_existing_data('2021-22')

print("="*60)
print("2021-22 SEASON - EXISTING DATA CHECK")
print("="*60)
print(f"\nGames marked as 'already downloaded': {len(already_downloaded)}")
print(f"Sample IDs: {already_downloaded[:5] if already_downloaded else 'None'}")

# This number becomes 'already_count' in download_season
print(f"\nThis becomes 'already_count' = {len(already_downloaded)}")
print("\nIf already_count > 0, checkpoints will load old partial files.")
print("If already_count == 0, checkpoints should only have NEW data.")
print("="*60)

