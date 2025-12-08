# NBA Props Predictor - Data Collection (FIXED HOME/AWAY)
# Run this in Google Colab or Jupyter Notebook

# STEP 2: Imports
import pandas as pd
import numpy as np
from datetime import datetime
import time
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from nba_api.stats.static import players, teams
import warnings
warnings.filterwarnings('ignore')

print("✓ All imports successful!")
print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

# STEP 3: Configuration
# Using NBA API for historical data (reliable and complete)
# Will use ESPN scraping for current 2025-26 season (NBA API not updating)
SEASONS = ['2022-23', '2023-24', '2024-25']  # Historical seasons via NBA API
OUTPUT_FILE = 'nba_player_game_logs.csv'
TEAM_STATS_FILE = 'nba_team_stats.csv'
CURRENT_SEASON = '2025-26'  # Will collect separately via ESPN

TARGET_PLAYERS = [
    # Superstars - USE EXACT NAMES WITH UNICODE CHARACTERS
    'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
    'Luka Dončić', 'Nikola Jokić', 'Joel Embiid', 'Jayson Tatum',
    'Damian Lillard', 'Anthony Davis', 'Devin Booker', 'Donovan Mitchell',
    'Jaylen Brown', 'Trae Young', 'Anthony Edwards', 'Shai Gilgeous-Alexander',
    'Jimmy Butler III', 'Paul George', 'Tyrese Haliburton', 'De\'Aaron Fox',
    'Domantas Sabonis', 'Bam Adebayo', 'Julius Randle', 'DeMar DeRozan',
    'Pascal Siakam', 'LaMelo Ball', 'James Harden', 'Karl-Anthony Towns',
    'Nikola Vučević', 'Jalen Brunson', 'Fred VanVleet', 'Tyler Herro',
    'Victor Wembanyama', 'Paolo Banchero', 'Franz Wagner', 'Scottie Barnes',
    'Cade Cunningham', 'Alperen Sengun', 'Jaren Jackson Jr.', 'Mikal Bridges',
    'Darius Garland', 'Lauri Markkanen', 'Desmond Bane', 'Jalen Williams',
    'OG Anunoby', 'Jarrett Allen', 'Kristaps Porziņģis', 'CJ McCollum'
]

print(f"Configuration:")
print(f"  Seasons: {SEASONS}")
print(f"  Target players: {len(TARGET_PLAYERS)}")
print(f"  Output file: {OUTPUT_FILE}")

# STEP 4: Get active players
all_active = players.get_active_players()
active_df = pd.DataFrame(all_active)

target_df = active_df[active_df['full_name'].isin(TARGET_PLAYERS)].copy()

found_players = set(target_df['full_name'].tolist())
missing_players = set(TARGET_PLAYERS) - found_players

print(f"\n✓ Found {len(target_df)} out of {len(TARGET_PLAYERS)} target players")
if missing_players:
    print(f"\n⚠ Missing players:")
    for player in missing_players:
        print(f"  - {player}")

print(f"\nPlayers to collect:")
print(target_df[['full_name', 'id']])

# STEP 5: Collect player game logs
def get_player_season_logs(player_id, player_name, season):
    time.sleep(1.0)  # Increased from 0.6 to 1.0 second to avoid timeouts
    
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season',
            timeout=60  # Increased timeout from default 30 to 60 seconds
        )
        df = gamelog.get_data_frames()[0]
        
        if len(df) > 0:
            df['PLAYER_ID'] = player_id
            df['PLAYER_NAME'] = player_name
            df['SEASON'] = season
            return df
        return None
        
    except Exception as e:
        print(f"  ✗ Error for {player_name} ({season}): {e}")
        print(f"    Retrying in 3 seconds...")
        time.sleep(3)  # Wait 3 seconds before continuing
        
        # Retry once
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season',
                timeout=60
            )
            df = gamelog.get_data_frames()[0]
            
            if len(df) > 0:
                df['PLAYER_ID'] = player_id
                df['PLAYER_NAME'] = player_name
                df['SEASON'] = season
                print(f"    ✓ Retry successful")
                return df
        except Exception as e2:
            print(f"    ✗ Retry also failed: {e2}")
        
        return None

all_game_logs = []
total_requests = len(target_df) * len(SEASONS)
current_request = 0

print(f"\nStarting collection of {total_requests} requests...\n")

for idx, row in target_df.iterrows():
    player_id = row['id']
    player_name = row['full_name']
    
    for season in SEASONS:
        current_request += 1
        print(f"[{current_request}/{total_requests}] {player_name} - {season}", end='')
        
        df = get_player_season_logs(player_id, player_name, season)
        
        if df is not None:
            all_game_logs.append(df)
            print(f" ✓ ({len(df)} games)")
        else:
            print(" ✗ (no data)")

print(f"\n{'='*60}")
print(f"Collection complete! Total dataframes: {len(all_game_logs)}")

# STEP 6: Combine data
if all_game_logs:
    player_data = pd.concat(all_game_logs, ignore_index=True)
    
    print(f"\nCombined dataset shape: {player_data.shape}")
    print(f"Date range: {player_data['GAME_DATE'].min()} to {player_data['GAME_DATE'].max()}")
    print(f"Unique players: {player_data['PLAYER_NAME'].nunique()}")
    print(f"Total games: {len(player_data)}")
    
    print("\nColumn names:")
    print(player_data.columns.tolist())
    
    print("\nSample data:")
    print(player_data[['GAME_DATE', 'PLAYER_NAME', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST']].head())
else:
    raise Exception("ERROR: No data collected!")

# STEP 7: Parse opponent AND home/away status CORRECTLY
print("\n" + "="*60)
print("PARSING MATCHUP DATA (OPPONENT + HOME/AWAY)")
print("="*60)

def parse_matchup_info(matchup):
    """
    Extract opponent and home/away status from MATCHUP string
    NBA API Format: 'TEAM vs. OPP' (home) or 'TEAM @ OPP' (away)
    Returns: (opponent, is_home, standardized_matchup)
    """
    if pd.isna(matchup) or matchup == '':
        return None, None, None
    
    matchup = str(matchup).strip()
    
    # Home game (vs.)
    if 'vs.' in matchup:
        parts = matchup.split('vs.')
        team = parts[0].strip()
        opponent = parts[1].strip()
        is_home = 1
        standardized = f"{team} vs {opponent}"
    # Away game (@)
    elif '@' in matchup:
        parts = matchup.split('@')
        team = parts[0].strip()
        opponent = parts[1].strip()
        is_home = 0
        standardized = f"{team} @ {opponent}"
    else:
        # Fallback - couldn't parse
        return None, None, matchup
    
    return opponent, is_home, standardized

# Apply parsing
print("Extracting opponent and home/away status from MATCHUP column...")
matchup_data = player_data['MATCHUP'].apply(parse_matchup_info)
player_data['OPPONENT'] = matchup_data.apply(lambda x: x[0])
player_data['IS_HOME'] = matchup_data.apply(lambda x: x[1])
player_data['MATCHUP'] = matchup_data.apply(lambda x: x[2])

# Verify parsing
print("\n✓ Matchup parsing complete")
print(f"  Total games: {len(player_data)}")
print(f"  Successfully parsed: {player_data['OPPONENT'].notna().sum()}")
print(f"  Failed to parse: {player_data['OPPONENT'].isna().sum()}")

# Check home/away distribution
home_games = (player_data['IS_HOME'] == 1).sum()
away_games = (player_data['IS_HOME'] == 0).sum()
print(f"\n  Home games: {home_games} ({home_games/len(player_data)*100:.1f}%)")
print(f"  Away games: {away_games} ({away_games/len(player_data)*100:.1f}%)")

# Show samples
print("\nSample parsed matchups:")
sample_cols = ['PLAYER_NAME', 'GAME_DATE', 'MATCHUP', 'OPPONENT', 'IS_HOME', 'WL']
print(player_data[sample_cols].head(10))

print("\nHome game examples:")
print(player_data[player_data['IS_HOME'] == 1][sample_cols].head(5))

print("\nAway game examples:")
print(player_data[player_data['IS_HOME'] == 0][sample_cols].head(5))

# STEP 8: Get team defensive stats
print("\n\n" + "="*60)
print("COLLECTING TEAM STATISTICS")
print("="*60)

def get_team_defensive_stats(season):
    """
    Get team defensive stats including DEF_RATING and PACE
    Note: We need to try multiple measure types to get all stats
    """
    time.sleep(1.0)  # Increased from 0.6 to 1.0 second
    
    try:
        # First try: Get Base stats (has some defensive metrics)
        team_stats_base = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Base',
            timeout=60  # Added timeout parameter
        )
        df_base = team_stats_base.get_data_frames()[0]
        
        # Second try: Get Advanced stats (has DEF_RATING and PACE)
        time.sleep(1.0)  # Increased delay
        team_stats_adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Advanced',
            timeout=60  # Added timeout parameter
        )
        df_adv = team_stats_adv.get_data_frames()[0]
        
        # Merge the two dataframes to get all columns
        if 'TEAM_ID' in df_base.columns and 'TEAM_ID' in df_adv.columns:
            # Keep only what we need from advanced stats
            adv_cols = ['TEAM_ID', 'DEF_RATING', 'PACE']
            adv_cols = [col for col in adv_cols if col in df_adv.columns]
            
            df = df_base.merge(df_adv[adv_cols], on='TEAM_ID', how='left')
        else:
            # If merge fails, just use advanced
            df = df_adv
        
        # Verify we got all 30 teams
        if len(df) < 30:
            print(f"  ⚠ Warning: Only {len(df)} teams returned for {season} (expected 30)")
            missing_count = 30 - len(df)
            print(f"    {missing_count} team(s) missing - will be filled with league averages")
        else:
            print(f"  ✓ All 30 teams found for {season}")
        
        # Check if we have the critical columns
        if 'DEF_RATING' not in df.columns or 'PACE' not in df.columns:
            print(f"  ⚠ Warning: Missing DEF_RATING or PACE columns")
            print(f"    Available columns: {df.columns.tolist()}")
        
        df['SEASON'] = season
        return df
        
    except Exception as e:
        print(f"  ✗ Error fetching team stats for {season}: {e}")
        return None

all_team_stats = []
for season in SEASONS:
    print(f"Fetching team stats for {season}...")
    team_df = get_team_defensive_stats(season)
    if team_df is not None:
        all_team_stats.append(team_df)

if all_team_stats:
    team_stats = pd.concat(all_team_stats, ignore_index=True)
    print(f"\n✓ Team stats collected: {len(team_stats)} team-seasons")
    print("\nTeam stats columns available:")
    print(team_stats.columns.tolist())
    
    # Show sample with actual column names
    display_cols = []
    if 'TEAM_NAME' in team_stats.columns:
        display_cols.append('TEAM_NAME')
    if 'TEAM_ABBREVIATION' in team_stats.columns:
        display_cols.append('TEAM_ABBREVIATION')
    elif 'TEAM_ID' in team_stats.columns:
        display_cols.append('TEAM_ID')
    if 'DEF_RATING' in team_stats.columns:
        display_cols.append('DEF_RATING')
    if 'PACE' in team_stats.columns:
        display_cols.append('PACE')
    if 'SEASON' in team_stats.columns:
        display_cols.append('SEASON')
    
    if display_cols:
        print("\nSample team stats:")
        print(team_stats[display_cols].head())
else:
    print("⚠ Warning: No team stats collected")
    team_stats = None

# STEP 9: Merge opponent defensive ratings
print("\n" + "="*60)
print("MERGING OPPONENT DEFENSIVE STATS")
print("="*60)

if team_stats is not None:
    # Find the correct team abbreviation column
    team_abbrev_col = None
    if 'TEAM_ABBREVIATION' in team_stats.columns:
        team_abbrev_col = 'TEAM_ABBREVIATION'
    elif 'TEAM_ABBRV' in team_stats.columns:
        team_abbrev_col = 'TEAM_ABBRV'
    elif 'TEAM_ID' in team_stats.columns:
        # Create mapping from team names to abbreviations
        from nba_api.stats.static import teams
        team_list = teams.get_teams()
        team_mapping = {t['full_name']: t['abbreviation'] for t in team_list}
        team_stats['TEAM_ABBREVIATION'] = team_stats['TEAM_NAME'].map(team_mapping)
        team_abbrev_col = 'TEAM_ABBREVIATION'
    
    if team_abbrev_col:
        print(f"Using team column: {team_abbrev_col}")
        
        # Prepare opponent stats
        opp_def_stats = team_stats[[team_abbrev_col, 'SEASON', 'DEF_RATING', 'PACE']].copy()
        opp_def_stats.columns = ['OPPONENT', 'SEASON', 'OPP_DEF_RATING', 'OPP_PACE']
        
        # Show what's available
        print(f"\nTeam stats available for merge:")
        print(f"  Unique teams: {opp_def_stats['OPPONENT'].nunique()}")
        # Filter out NaN values before sorting
        valid_teams = [t for t in opp_def_stats['OPPONENT'].unique() if pd.notna(t)]
        print(f"  Valid teams: {len(valid_teams)}")
        print(f"  Teams: {sorted(valid_teams)}")
        
        # Check for missing teams
        if len(valid_teams) < 30:
            print(f"\n  ⚠ Warning: Only {len(valid_teams)} teams found (expected 30)")
            print(f"    NaN values in OPPONENT: {opp_def_stats['OPPONENT'].isna().sum()}")
            
            # Identify which teams are missing
            all_nba_teams = ['ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 
                           'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 
                           'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 
                           'TOR', 'UTA', 'WAS']
            missing_teams = set(all_nba_teams) - set(valid_teams)
            if missing_teams:
                print(f"    Missing teams: {sorted(missing_teams)}")
                print(f"    These teams will use league average defensive stats")
        
        # Merge
        before_merge = len(player_data)
        player_data = player_data.merge(
            opp_def_stats,
            on=['OPPONENT', 'SEASON'],
            how='left'
        )
        after_merge = len(player_data)
        
        if before_merge == after_merge:
            print(f"✓ Merge successful - no rows lost")
        else:
            print(f"⚠ Warning: Row count changed from {before_merge} to {after_merge}")
        
        missing_stats = player_data['OPP_DEF_RATING'].isna().sum() if 'OPP_DEF_RATING' in player_data.columns else len(player_data)
        print(f"\n✓ Merged opponent stats")
        print(f"  Successfully matched: {len(player_data) - missing_stats} games ({(len(player_data) - missing_stats)/len(player_data)*100:.1f}%)")
        print(f"  Missing opponent data: {missing_stats} games ({missing_stats/len(player_data)*100:.1f}%)")
        
        # Show which columns were successfully merged
        merged_cols = []
        if 'OPP_DEF_RATING' in player_data.columns:
            merged_cols.append('OPP_DEF_RATING')
        if 'OPP_PACE' in player_data.columns:
            merged_cols.append('OPP_PACE')
        if 'OPP_OFF_RATING' in player_data.columns:
            merged_cols.append('OPP_OFF_RATING')
        print(f"  Opponent columns available: {merged_cols}")
        
        if missing_stats > 0:
            print("\n  Games missing opponent stats (top 10):")
            missing_breakdown = player_data[player_data['OPP_DEF_RATING'].isna()]['OPPONENT'].value_counts()
            print(missing_breakdown.head(10))
            
            # Fill missing with season-specific league averages
            print("\n  Filling missing opponent stats with league averages by season...")
            for season in player_data['SEASON'].unique():
                season_mask = player_data['SEASON'] == season
                missing_mask = player_data['OPP_DEF_RATING'].isna() & season_mask
                
                if missing_mask.sum() > 0:
                    # Calculate league average for this season
                    season_avg_def = player_data.loc[season_mask & ~player_data['OPP_DEF_RATING'].isna(), 'OPP_DEF_RATING'].mean()
                    season_avg_pace = player_data.loc[season_mask & ~player_data['OPP_PACE'].isna(), 'OPP_PACE'].mean()
                    
                    # Fill missing values
                    player_data.loc[missing_mask, 'OPP_DEF_RATING'] = season_avg_def
                    player_data.loc[missing_mask, 'OPP_PACE'] = season_avg_pace
                    
                    missing_teams = player_data.loc[missing_mask, 'OPPONENT'].unique()
                    print(f"    {season}: Filled {missing_mask.sum()} games for {list(missing_teams)}")
                    print(f"      DEF_RATING={season_avg_def:.2f}, PACE={season_avg_pace:.2f}")
            
            print(f"\n  ✓ All missing opponent stats filled with league averages")
    else:
        print("\n⚠ Could not find team abbreviation column, skipping opponent merge")
else:
    print("\n⚠ No team stats available, skipping opponent merge")

# STEP 10: Convert data types
print("\n" + "="*60)
print("CONVERTING DATA TYPES")
print("="*60)

player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'])
player_data = player_data.sort_values(['PLAYER_NAME', 'GAME_DATE'])

numeric_cols = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', 'FGM', 
                'FG3A', 'FG3M', 'FTA', 'FTM', 'PLUS_MINUS']

for col in numeric_cols:
    if col in player_data.columns:
        player_data[col] = pd.to_numeric(player_data[col], errors='coerce')

# Ensure IS_HOME is numeric
if 'IS_HOME' in player_data.columns:
    player_data['IS_HOME'] = pd.to_numeric(player_data['IS_HOME'], errors='coerce')

print("✓ Data types converted to numeric")

# STEP 11: Save data
print("\n" + "="*60)
print("SAVING DATA")
print("="*60)

player_data.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Player game logs saved to: {OUTPUT_FILE}")

if team_stats is not None:
    team_stats.to_csv(TEAM_STATS_FILE, index=False)
    print(f"✓ Team stats saved to: {TEAM_STATS_FILE}")

# STEP 12: Summary statistics
print(f"\n{'='*60}")
print("DATA COLLECTION SUMMARY")
print(f"{'='*60}")
print(f"Total games collected: {len(player_data):,}")
print(f"Unique players: {player_data['PLAYER_NAME'].nunique()}")
print(f"Date range: {player_data['GAME_DATE'].min().date()} to {player_data['GAME_DATE'].max().date()}")

print(f"\nHome/Away Distribution:")
if 'IS_HOME' in player_data.columns:
    print(f"  Home games (IS_HOME=1): {(player_data['IS_HOME'] == 1).sum()}")
    print(f"  Away games (IS_HOME=0): {(player_data['IS_HOME'] == 0).sum()}")
    print(f"  Unknown: {player_data['IS_HOME'].isna().sum()}")

print(f"\nGames per player:")
print(player_data.groupby('PLAYER_NAME').size().describe())

print(f"\nAverage stats:")
print(player_data[['PTS', 'REB', 'AST', 'MIN']].describe())

print(f"\n✓ Data collection complete!")
print(f"\nNext step: Run ESPN scraper for current season (2025-26)")