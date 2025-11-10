"""
Check V2 feature extraction progress
"""
import os
import glob
from datetime import datetime

print("="*60)
print("V2 FEATURE EXTRACTION PROGRESS")
print("="*60)

seasons = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']

files = glob.glob('data/processed/features_v2_*.csv')
files.sort(key=lambda x: os.path.getmtime(x))

print(f"\nCompleted: {len(files)}/{len(seasons)} seasons\n")

for f in files:
    size_mb = os.path.getsize(f) / (1024*1024)
    mod_time = datetime.fromtimestamp(os.path.getmtime(f))
    season = os.path.basename(f).replace('features_v2_', '').replace('.csv', '').replace('_', '-')
    print(f"  [OK] {season:<10} {size_mb:>6.1f} MB  (completed {mod_time.strftime('%I:%M %p')})")

remaining = [s for s in seasons if not any(s.replace('-','_') in f for f in files)]
if remaining:
    print(f"\nStill processing:")
    for s in remaining:
        print(f"  [ ] {s}")

print("\n" + "="*60)
print(f"Progress: {len(files)/len(seasons)*100:.0f}% complete")
print("="*60)

