"""
Clean up all partial files now that we have complete season files
"""
import os
import glob

print("="*60)
print("CLEANING UP PARTIAL FILES")
print("="*60)

# Find all partial files
partial_files = glob.glob('data/raw/*_partial_*.csv')

print(f"\nFound {len(partial_files)} partial files")
print("\nDeleting partial files...")

deleted_count = 0
for file in partial_files:
    try:
        os.remove(file)
        deleted_count += 1
        if deleted_count % 10 == 0:
            print(f"  Deleted {deleted_count}/{len(partial_files)}...")
    except Exception as e:
        print(f"  Error deleting {file}: {e}")

print(f"\n[OK] Deleted {deleted_count} partial files")
print("\nFinal season files remaining:")

# List final files
final_files = glob.glob('data/raw/pbp_20*.csv')
final_files = [f for f in final_files if 'partial' not in f and 'BACKUP' not in f]

for f in sorted(final_files):
    size_mb = os.path.getsize(f) / (1024*1024)
    print(f"  {os.path.basename(f):<25} {size_mb:>8.2f} MB")

print("\n" + "="*60)
print("CLEANUP COMPLETE")
print("="*60)

