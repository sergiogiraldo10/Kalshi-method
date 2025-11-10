"""
Quick comparison between Kaggle and NBA API data
"""
import pandas as pd

print("="*60)
print("DATA SOURCE COMPARISON")
print("="*60)

# Load NBA API data (partial download)
print("\n1. NBA API DATA (from partial download):")
nba_api_df = pd.read_csv('data/raw/pbp_2017_18_partial_1200.csv')
print(f"   Total plays: {len(nba_api_df):,}")
print(f"   Unique games: {nba_api_df['GAME_ID'].nunique()}")
print(f"   Date range: {nba_api_df['GAME_ID'].min()} to {nba_api_df['GAME_ID'].max()}")
print(f"   Columns: {len(nba_api_df.columns)}")

# Load Kaggle data
print("\n2. KAGGLE DATA (converted):")
kaggle_df = pd.read_csv('data/raw/pbp_2017_18.csv')
print(f"   Total plays: {len(kaggle_df):,}")
print(f"   Unique games: {kaggle_df['GAME_ID'].nunique()}")
print(f"   Date range: {kaggle_df['GAME_ID'].min()} to {kaggle_df['GAME_ID'].max()}")
print(f"   Columns: {len(kaggle_df.columns)}")

# Sample comparison - first game from NBA API
print("\n3. SAMPLE GAME COMPARISON:")
sample_game_id = nba_api_df['GAME_ID'].iloc[100]
print(f"\n   NBA API Sample Game: {sample_game_id}")

nba_sample = nba_api_df[nba_api_df['GAME_ID'] == sample_game_id]
print(f"   Total plays: {len(nba_sample)}")
print(f"   Periods: {nba_sample['PERIOD'].max()}")
print(f"   Final score: {nba_sample['SCORE'].iloc[-1] if pd.notna(nba_sample['SCORE'].iloc[-1]) else 'N/A'}")

print("\n   Sample plays (NBA API):")
for idx, row in nba_sample.head(5).iterrows():
    desc = row['HOMEDESCRIPTION'] if pd.notna(row['HOMEDESCRIPTION']) else row['VISITORDESCRIPTION']
    print(f"   Q{row['PERIOD']} {row['PCTIMESTRING']}: {desc[:50] if pd.notna(desc) else 'N/A'}")

# Compare play counts by event type
print("\n4. EVENT TYPE DISTRIBUTION:")
print("\n   NBA API:")
event_counts_api = nba_api_df['EVENTMSGTYPE'].value_counts().head(10)
for event_type, count in event_counts_api.items():
    print(f"   Type {event_type}: {count:,} plays")

print("\n   Kaggle (converted):")
event_counts_kaggle = kaggle_df['EVENTMSGTYPE'].value_counts().head(10)
for event_type, count in event_counts_kaggle.items():
    print(f"   Type {event_type}: {count:,} plays")

# Check data quality
print("\n5. DATA QUALITY CHECK:")
print(f"\n   NBA API:")
print(f"   - Missing scores: {nba_api_df['SCORE'].isna().sum():,}")
print(f"   - Missing player names: {nba_api_df['PLAYER1_NAME'].isna().sum():,}")
print(f"   - Has player IDs: {(nba_api_df['PLAYER1_ID'] > 0).sum():,}")

print(f"\n   Kaggle:")
print(f"   - Missing scores: {kaggle_df['SCORE'].isna().sum():,}")
print(f"   - Missing player names: {kaggle_df['PLAYER1_NAME'].isna().sum():,}")
print(f"   - Has player IDs: {(kaggle_df['PLAYER1_ID'] > 0).sum():,}")

print("\n" + "="*60)
print("VERDICT:")
print("="*60)
print("\nKaggle Data Pros:")
print("  + Complete seasons (no missing games)")
print("  + Parsed event types already extracted")
print("  + Reliable historical data")
print("\nKaggle Data Cons:")
print("  - No player IDs (only names)")
print("  - Generated game IDs (not official NBA IDs)")
print("  - Some metadata missing")
print("\nNBA API Data Pros:")
print("  + Official NBA game IDs")
print("  + Official player IDs")
print("  + More metadata")
print("\nNBA API Data Cons:")
print("  - Slow download (timeout issues)")
print("  - Incomplete coverage due to errors")
print("\nRECOMMENDATION: Use Kaggle data for training!")
print("The Kaggle data has all the essential information needed")
print("for momentum detection: plays, scores, timing, events.")
print("="*60)

