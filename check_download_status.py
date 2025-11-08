"""
Quick script to check download status
"""
import os
import glob
from datetime import datetime

data_dir = 'data/raw'

print("="*60)
print("Download Status Check")
print("="*60)
print(f"\nChecking directory: {os.path.abspath(data_dir)}\n")

# Check if directory exists
if not os.path.exists(data_dir):
    print(f"[X] Directory doesn't exist: {data_dir}")
    print("   Creating directory...")
    os.makedirs(data_dir, exist_ok=True)
    print("   [OK] Directory created\n")

# Check for any CSV files
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

if not csv_files:
    print("[X] No files found yet.")
    print("\nPossible reasons:")
    print("  1. Download hasn't started")
    print("  2. Download is in progress (saves every 50 games)")
    print("  3. Download failed or was stopped")
    print("\n[INFO] To start download, run: python src/data_acquisition.py")
else:
    print(f"[OK] Found {len(csv_files)} file(s):\n")
    
    for file in sorted(csv_files):
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
        time_ago = datetime.now() - mod_time
        
        status = "[COMPLETE]" if "_partial_" not in file else "[IN PROGRESS]"
        
        print(f"{status} {os.path.basename(file)}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_ago.seconds//60} minutes ago)")
        print()
    
    # Check for partial files (active download)
    partial_files = [f for f in csv_files if '_partial_' in f]
    if partial_files:
        print("[WARNING] Partial files found - download is in progress!")
        print("   Files are saved every 50 games.")
    else:
        print("[OK] All files appear complete (no partial files)")
    
    # Expected files
    expected_seasons = ['2017-18', '2018-19', '2019-20', '2020-21', 
                       '2021-22', '2022-23', '2023-24', '2024-25']
    expected_files = [f'pbp_{s.replace("-", "_")}.csv' for s in expected_seasons]
    
    missing = []
    for exp_file in expected_files:
        if not any(exp_file in f for f in csv_files):
            missing.append(exp_file)
    
    if missing:
        print(f"\n[INFO] Still missing {len(missing)} season(s):")
        for m in missing:
            print(f"   - {m}")

print("\n" + "="*60)

