"""
Kaggle to NBA API Format Converter
Converts Basketball-Reference scraped data (Kaggle) to NBA API format
"""

import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime

class KaggleToNBAConverter:
    def __init__(self):
        # Event type mapping
        self.event_type_map = {
            'shot': 1,      # Field goal made
            'miss': 2,      # Field goal missed
            'free_throw': 3,
            'rebound': 4,
            'turnover': 5,
            'foul': 6,
            'violation': 7,
            'substitution': 8,
            'timeout': 9,
            'jump_ball': 10,
            'period_start': 12,
            'period_end': 13
        }
    
    def generate_game_id(self, url, date):
        """Generate a consistent game ID from URL"""
        # Extract date and team from URL
        # Format: /boxscores/201710170CLE.html -> 0021700001 (example)
        if 'boxscores' in url:
            parts = url.split('/')[-1].replace('.html', '')
            # Use first 8 chars as date, last 3 as team identifier
            return f"002{parts[:6]}{hash(parts) % 1000:03d}"
        return f"00217{hash(url) % 100000:05d}"
    
    def parse_time_to_clock(self, sec_left):
        """Convert seconds left to MM:SS format"""
        if pd.isna(sec_left):
            return "0:00"
        minutes = int(sec_left) // 60
        seconds = int(sec_left) % 60
        return f"{minutes}:{seconds:02d}"
    
    def determine_event_type(self, row):
        """Determine NBA API event type from Kaggle row"""
        # Check for specific events in order of priority
        if not pd.isna(row['JumpballAwayPlayer']) or not pd.isna(row['JumpballHomePlayer']):
            return 10  # Jump ball
        if not pd.isna(row['Shooter']):
            return 1 if row['ShotOutcome'] == 'make' else 2
        if not pd.isna(row['FreeThrowShooter']):
            return 3
        if not pd.isna(row['Rebounder']):
            return 4
        if not pd.isna(row['TurnoverPlayer']):
            return 5
        if not pd.isna(row['Fouler']):
            return 6
        if not pd.isna(row['ViolationPlayer']):
            return 7
        if not pd.isna(row['EnterGame']) or not pd.isna(row['LeaveGame']):
            return 8
        if not pd.isna(row['TimeoutTeam']):
            return 9
        
        # Check text descriptions
        away_play = str(row['AwayPlay']).lower() if pd.notna(row['AwayPlay']) else ''
        home_play = str(row['HomePlay']).lower() if pd.notna(row['HomePlay']) else ''
        
        if 'start of' in away_play or 'start of' in home_play:
            return 12
        if 'end of' in away_play or 'end of' in home_play:
            return 13
        
        return 0  # Unknown
    
    def extract_player_name(self, player_str):
        """Extract clean player name from 'Name - suffix' format"""
        if pd.isna(player_str):
            return None
        if ' - ' in str(player_str):
            return str(player_str).split(' - ')[0]
        return str(player_str)
    
    def create_description(self, row):
        """Create NBA API style description from Kaggle parsed data"""
        # Use AwayPlay or HomePlay
        if pd.notna(row['AwayPlay']):
            return str(row['AwayPlay'])
        elif pd.notna(row['HomePlay']):
            return str(row['HomePlay'])
        return ""
    
    def convert_file(self, kaggle_file, output_file):
        """Convert a single Kaggle CSV to NBA API format"""
        print(f"\nConverting: {kaggle_file}")
        print(f"Output: {output_file}")
        
        # Read Kaggle data
        df = pd.read_csv(kaggle_file)
        print(f"  Loaded {len(df):,} plays")
        
        # Create new dataframe with NBA API structure
        nba_df = pd.DataFrame()
        
        # Generate game IDs
        nba_df['GAME_ID'] = df.apply(lambda row: self.generate_game_id(row['URL'], row['Date']), axis=1)
        
        # Event number (sequential per game)
        nba_df['EVENTNUM'] = df.groupby(df.apply(lambda row: self.generate_game_id(row['URL'], row['Date']), axis=1)).cumcount() + 1
        
        # Event type
        nba_df['EVENTMSGTYPE'] = df.apply(self.determine_event_type, axis=1)
        
        # Event action type (simplified - would need more detailed mapping)
        nba_df['EVENTMSGACTIONTYPE'] = 0
        
        # Period
        nba_df['PERIOD'] = df['Quarter']
        
        # Time
        nba_df['WCTIMESTRING'] = ''  # Wall clock time not available
        nba_df['PCTIMESTRING'] = df['SecLeft'].apply(self.parse_time_to_clock)
        
        # Descriptions
        nba_df['HOMEDESCRIPTION'] = df.apply(lambda row: str(row['HomePlay']) if pd.notna(row['HomePlay']) else '', axis=1)
        nba_df['NEUTRALDESCRIPTION'] = ''
        nba_df['VISITORDESCRIPTION'] = df.apply(lambda row: str(row['AwayPlay']) if pd.notna(row['AwayPlay']) else '', axis=1)
        
        # Score
        nba_df['SCORE'] = df.apply(lambda row: f"{row['AwayScore']}-{row['HomeScore']}" if pd.notna(row['AwayScore']) and pd.notna(row['HomeScore']) else '', axis=1)
        nba_df['SCOREMARGIN'] = df.apply(lambda row: int(row['HomeScore']) - int(row['AwayScore']) if pd.notna(row['HomeScore']) and pd.notna(row['AwayScore']) else '', axis=1)
        
        # Player 1 (primary player)
        nba_df['PERSON1TYPE'] = 0
        nba_df['PLAYER1_ID'] = 0  # Player IDs not available in Kaggle data
        nba_df['PLAYER1_NAME'] = df.apply(lambda row: self.extract_player_name(
            row['Shooter'] if pd.notna(row['Shooter']) else 
            row['FreeThrowShooter'] if pd.notna(row['FreeThrowShooter']) else
            row['Rebounder'] if pd.notna(row['Rebounder']) else
            row['Fouler'] if pd.notna(row['Fouler']) else
            row['TurnoverPlayer'] if pd.notna(row['TurnoverPlayer']) else None
        ), axis=1)
        nba_df['PLAYER1_TEAM_ID'] = 0
        nba_df['PLAYER1_TEAM_CITY'] = ''
        nba_df['PLAYER1_TEAM_NICKNAME'] = ''
        nba_df['PLAYER1_TEAM_ABBREVIATION'] = df.apply(lambda row: 
            row['AwayTeam'] if pd.notna(row['AwayPlay']) else 
            row['HomeTeam'] if pd.notna(row['HomePlay']) else '', axis=1)
        
        # Player 2 (secondary player - assist, block, etc.)
        nba_df['PERSON2TYPE'] = 0
        nba_df['PLAYER2_ID'] = 0
        nba_df['PLAYER2_NAME'] = df.apply(lambda row: self.extract_player_name(
            row['Assister'] if pd.notna(row['Assister']) else 
            row['Blocker'] if pd.notna(row['Blocker']) else
            row['Fouled'] if pd.notna(row['Fouled']) else
            row['TurnoverCauser'] if pd.notna(row['TurnoverCauser']) else None
        ), axis=1)
        nba_df['PLAYER2_TEAM_ID'] = 0
        nba_df['PLAYER2_TEAM_CITY'] = ''
        nba_df['PLAYER2_TEAM_NICKNAME'] = ''
        nba_df['PLAYER2_TEAM_ABBREVIATION'] = ''
        
        # Player 3 (tertiary player - jump ball possession, etc.)
        nba_df['PERSON3TYPE'] = 0
        nba_df['PLAYER3_ID'] = 0
        nba_df['PLAYER3_NAME'] = df.apply(lambda row: self.extract_player_name(
            row['JumpballPoss'] if pd.notna(row.get('JumpballPoss')) else None
        ), axis=1)
        nba_df['PLAYER3_TEAM_ID'] = 0
        nba_df['PLAYER3_TEAM_CITY'] = ''
        nba_df['PLAYER3_TEAM_NICKNAME'] = ''
        nba_df['PLAYER3_TEAM_ABBREVIATION'] = ''
        
        # Video flag
        nba_df['VIDEO_AVAILABLE_FLAG'] = 0
        
        # Season (extract from filename or date)
        season = self.extract_season(kaggle_file)
        nba_df['SEASON'] = season
        
        # Save
        nba_df.to_csv(output_file, index=False)
        print(f"  Converted {len(nba_df):,} plays")
        print(f"  Unique games: {nba_df['GAME_ID'].nunique()}")
        
        return nba_df
    
    def extract_season(self, filename):
        """Extract season from filename like 'NBA_PBP_2017-18.csv'"""
        import re
        match = re.search(r'(\d{4})-(\d{2})', filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return "Unknown"
    
    def convert_all_kaggle_files(self, kaggle_dir='data/Kaggle', output_dir='data/raw'):
        """Convert all Kaggle files in directory"""
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all Kaggle files
        kaggle_files = glob.glob(os.path.join(kaggle_dir, 'NBA_PBP_*.csv'))
        
        if not kaggle_files:
            print(f"No Kaggle files found in {kaggle_dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(kaggle_files)} Kaggle files to convert")
        print(f"{'='*60}")
        
        for kaggle_file in sorted(kaggle_files):
            # Extract season from filename
            season = self.extract_season(kaggle_file)
            
            # Create output filename
            output_file = os.path.join(output_dir, f'pbp_{season.replace("-", "_")}.csv')
            
            # Skip if already exists
            if os.path.exists(output_file):
                print(f"\n[SKIP] {output_file} already exists")
                continue
            
            # Convert
            try:
                self.convert_file(kaggle_file, output_file)
                print(f"  [OK] Saved to {output_file}")
            except Exception as e:
                print(f"  [ERROR] Failed to convert: {e}")
        
        print(f"\n{'='*60}")
        print("Conversion complete!")
        print(f"{'='*60}\n")


def main():
    """Main conversion function"""
    converter = KaggleToNBAConverter()
    
    print("\n" + "="*60)
    print("Kaggle to NBA API Format Converter")
    print("="*60)
    
    # Convert all files
    converter.convert_all_kaggle_files(
        kaggle_dir='data/Kaggle',
        output_dir='data/raw'
    )
    
    print("\nNext steps:")
    print("1. Download remaining seasons (2021-2025) using nba_api")
    print("2. Run data validation: python src/data_validation.py")


if __name__ == '__main__':
    main()

