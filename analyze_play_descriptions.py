"""
Analyze play-by-play descriptions to understand what causes runs
"""
import pandas as pd
import re

# Load one game
df = pd.read_csv('data/raw/pbp_2020_21.csv')
game_id = df['GAME_ID'].iloc[0]
game_df = df[df['GAME_ID'] == game_id].copy()
game_df = game_df.sort_values('EVENTNUM')

print("="*70)
print("ANALYZING PLAY DESCRIPTIONS - What Causes Runs?")
print("="*70)

# Define keywords to look for
keywords = {
    'MISS': 0,
    'BLOCK': 0,
    'STEAL': 0,
    'Turnover': 0,
    'REBOUND': 0,
    'Foul': 0,
    'Free Throw': 0,
    '3PT': 0,
    'Layup': 0,
    'Dunk': 0
}

# Count occurrences
for idx, row in game_df.iterrows():
    home_desc = str(row.get('HOMEDESCRIPTION', ''))
    away_desc = str(row.get('VISITORDESCRIPTION', ''))
    neutral_desc = str(row.get('NEUTRALDESCRIPTION', ''))
    
    full_desc = f"{home_desc} {away_desc} {neutral_desc}"
    
    for keyword in keywords:
        if keyword in full_desc:
            keywords[keyword] += 1

print("\n1. EVENT TYPE FREQUENCY IN THIS GAME:\n")
for keyword, count in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
    print(f"   {keyword:<15} {count:>3} occurrences")

# Sample plays
print("\n2. SAMPLE PLAYS (showing what descriptions look like):\n")
sample_count = 0
for idx, row in game_df.head(30).iterrows():
    home_desc = str(row.get('HOMEDESCRIPTION', ''))
    away_desc = str(row.get('VISITORDESCRIPTION', ''))
    score = str(row.get('SCORE', ''))
    
    if home_desc != 'nan' or away_desc != 'nan':
        if home_desc != 'nan':
            print(f"   {score:<10} HOME: {home_desc}")
        if away_desc != 'nan':
            print(f"   {score:<10} AWAY: {away_desc}")
        sample_count += 1
        if sample_count >= 15:
            break

print("\n3. FEATURES WE CAN EXTRACT FROM PLAY DESCRIPTIONS:\n")
features = [
    "- Missed shots in last 2 min (defensive stops)",
    "- Blocks in last 2 min (defensive momentum)",
    "- Steals in last 2 min (creating turnovers)",
    "- Turnovers by each team (ball security)",
    "- 3-pointers made (can spark runs quickly)",
    "- Fouls (disrupting opponent's flow)",
    "- Fast break points (transition scoring)",
    "- Second chance points (offensive rebounds)",
    "- Free throw efficiency (clutch execution)"
]

for feature in features:
    print(f"   {feature}")

print("\n4. PROPOSED: NLP-ENHANCED MOMENTUM FEATURES")
print("\n   Instead of just 'Team A scored 10, Team B scored 2',")
print("   we can add context like:")
print("   - 'Team A had 3 steals leading to fast breaks'")
print("   - 'Team B missed 5 straight shots'")
print("   - 'Team A made 2 blocks during this run'")
print("   - 'Team A hit 3 three-pointers'")

print("\n" + "="*70)

