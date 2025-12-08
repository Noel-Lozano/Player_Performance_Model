# NBA Props Predictor - Data Exploration & Cleaning
# Notebook 02: Load, Clean, and Understand Your Data

# STEP 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Imports successful")

# STEP 2: LOAD + CLEAN + MERGE DATA CORRECTLY
print("\nLoading datasets...")

# Load historical game logs (NBA API)
historical_df = pd.read_csv("nba_player_game_logs.csv")
print(f"✓ Historical data loaded: {len(historical_df)} games")
print(f"  Columns: {list(historical_df.columns)}")

# Load ESPN current-season data
try:
    current_df = pd.read_csv("nba_current_season_2025_26.csv")
    print(f"✓ Current season loaded: {len(current_df)} games")
    print(f"  Columns: {list(current_df.columns)}")
except FileNotFoundError:
    print("⚠ No ESPN current-season file found, continuing with historical only.")
    current_df = None

# Load team stats (historical)
try:
    historical_team_stats = pd.read_csv('nba_team_stats.csv')
    print(f"✓ Historical team stats loaded: {len(historical_team_stats)} team-seasons")
except FileNotFoundError:
    print("⚠ No historical team stats found")
    historical_team_stats = None

# Load current season team stats (Hollinger)
try:
    current_team_stats = pd.read_csv('nba_team_stats_current_season.csv')
    print(f"✓ Current season team stats loaded: {len(current_team_stats)} teams")
except FileNotFoundError:
    print("⚠ No current season team stats found")
    current_team_stats = None

# --------------------------------------------------------------
# MERGE TEAM STATS (Historical + Current Season)
# --------------------------------------------------------------
print("\n" + "="*60)
print("MERGING TEAM STATS")
print("="*60)

if historical_team_stats is not None and current_team_stats is not None:
    # Check if current season already exists in historical
    if 'SEASON' in historical_team_stats.columns:
        current_season_val = '2025-26'
        if current_season_val in historical_team_stats['SEASON'].values:
            print(f"⚠ {current_season_val} already in historical data - replacing...")
            historical_team_stats = historical_team_stats[historical_team_stats['SEASON'] != current_season_val]
    
    # Align columns
    all_team_cols = list(set(historical_team_stats.columns) | set(current_team_stats.columns))
    
    # Add missing columns to historical
    for col in current_team_stats.columns:
        if col not in historical_team_stats.columns:
            historical_team_stats[col] = None
    
    # Add missing columns to current
    for col in historical_team_stats.columns:
        if col not in current_team_stats.columns:
            current_team_stats[col] = None
    
    # Align column order
    historical_team_stats = historical_team_stats[sorted(historical_team_stats.columns)]
    current_team_stats = current_team_stats[sorted(current_team_stats.columns)]
    
    # Merge
    team_stats = pd.concat([historical_team_stats, current_team_stats], ignore_index=True)
    team_stats = team_stats.sort_values(['SEASON', 'TEAM_ABBREVIATION'], ignore_index=True)
    
    print(f"✓ Team stats merged:")
    print(f"  Historical: {len(historical_team_stats)} team-seasons")
    print(f"  Current: {len(current_team_stats)} teams")
    print(f"  Total: {len(team_stats)} team-seasons")
    print(f"  Seasons: {sorted(team_stats['SEASON'].unique())}")
    
elif historical_team_stats is not None:
    team_stats = historical_team_stats
    print("⚠ Using only historical team stats (no current season)")
elif current_team_stats is not None:
    team_stats = current_team_stats
    print("⚠ Using only current season team stats (no historical)")
else:
    team_stats = None
    print("✗ No team stats available!")

# --------------------------------------------------------------
# 1) STANDARDIZE NAMES ON BOTH DATASETS
# --------------------------------------------------------------
name_fix = {
    'Nikola Jokić': 'Nikola Jokic',
    'Nikola Vučević': 'Nikola Vucevic',
    'Luka Dončić': 'Luka Doncic',
    'Kristaps Porziņģis': 'Kristaps Porzingis',
    'Dāvis Bertāns': 'Davis Bertans',
    'Bogdan Bogdanović': 'Bogdan Bogdanovic',
    'Nikola Jović': 'Nikola Jovic',
}

historical_df['PLAYER_NAME'] = historical_df['PLAYER_NAME'].replace(name_fix)

if current_df is not None:
    current_df['PLAYER_NAME'] = current_df['PLAYER_NAME'].replace(name_fix)

# --------------------------------------------------------------
# 2) REMOVE PLAYER_ID COLUMNS BEFORE MERGE
# --------------------------------------------------------------
print("\n" + "="*60)
print("REMOVING ID COLUMNS FOR NAME-ONLY TRACKING")
print("="*60)

# Drop ID columns from historical data
id_cols_historical = [col for col in historical_df.columns if 'ID' in col.upper()]
if id_cols_historical:
    print(f"Removing from historical data: {id_cols_historical}")
    historical_df = historical_df.drop(columns=id_cols_historical, errors='ignore')

# Drop ID columns from current data
if current_df is not None:
    id_cols_current = [col for col in current_df.columns if 'ID' in col.upper()]
    if id_cols_current:
        print(f"Removing from current data: {id_cols_current}")
        current_df = current_df.drop(columns=id_cols_current, errors='ignore')

# --------------------------------------------------------------
# 3) STANDARDIZE MATCHUP FORMAT FOR BOTH DATASETS
# --------------------------------------------------------------
print("\n" + "="*60)
print("STANDARDIZING MATCHUP FORMAT")
print("="*60)

def standardize_matchup(row):
    """
    Standardize matchup to format: TEAM_A vs TEAM_B or TEAM_A @ TEAM_B
    Shows who played whom clearly
    """
    if pd.isna(row.get('MATCHUP', '')):
        return ''
    
    matchup = str(row['MATCHUP'])
    
    # If matchup already in good format, return it
    if ' vs ' in matchup or ' @ ' in matchup:
        return matchup
    
    # Otherwise try to construct from available data
    team = row.get('TEAM', '')
    opponent = row.get('OPPONENT', '')
    is_home = row.get('IS_HOME', 0)
    
    if team and opponent:
        if is_home:
            return f"{team} vs {opponent}"
        else:
            return f"{team} @ {opponent}"
    
    return matchup

# Apply to historical data
print("Standardizing historical matchups...")
historical_df['MATCHUP'] = historical_df.apply(standardize_matchup, axis=1)

# Apply to current data
if current_df is not None:
    print("Standardizing current season matchups...")
    current_df['MATCHUP'] = current_df.apply(standardize_matchup, axis=1)

# Show sample matchups
print("\nSample historical matchups:")
print(historical_df[['PLAYER_NAME', 'GAME_DATE', 'MATCHUP']].head(10))

if current_df is not None:
    print("\nSample current season matchups:")
    print(current_df[['PLAYER_NAME', 'GAME_DATE', 'MATCHUP']].head(10))

# --------------------------------------------------------------
# 4) ALIGN COLUMNS (ADD MISSING COLUMNS)
# --------------------------------------------------------------
print("\n" + "="*60)
print("ALIGNING COLUMNS BETWEEN DATASETS")
print("="*60)

if current_df is not None:
    # Get all unique columns
    all_cols = list(set(historical_df.columns) | set(current_df.columns))
    
    # Add missing columns to historical
    missing_in_historical = [col for col in current_df.columns if col not in historical_df.columns]
    if missing_in_historical:
        print(f"Adding to historical: {missing_in_historical}")
        for col in missing_in_historical:
            historical_df[col] = None
    
    # Add missing columns to current
    missing_in_current = [col for col in historical_df.columns if col not in current_df.columns]
    if missing_in_current:
        print(f"Adding to current: {missing_in_current}")
        for col in missing_in_current:
            current_df[col] = None
    
    # Reorder both to match
    historical_df = historical_df[sorted(historical_df.columns)]
    current_df = current_df[sorted(current_df.columns)]
    
    # Merge (stack)
    df = pd.concat([historical_df, current_df], ignore_index=True)
    
    print(f"\n✓ Datasets merged successfully")
    print(f"  Historical games: {len(historical_df):,}")
    print(f"  Current season games: {len(current_df):,}")
    print(f"  Total games: {len(df):,}")
else:
    df = historical_df.copy()

# --------------------------------------------------------------
# 5) FIX DATE FORMAT + SORT CHRONOLOGICALLY
# --------------------------------------------------------------
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(by=["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

print(f"\n✓ Final merged dataset: {len(df)} total games")

# --------------------------------------------------------------
# 6) SHOW SAMPLE DATA FOR VERIFICATION
# --------------------------------------------------------------
print("\n" + "="*60)
print("SAMPLE MERGED DATA (Name-Based Tracking)")
print("="*60)

sample_cols = ['PLAYER_NAME', 'GAME_DATE', 'MATCHUP', 'TEAM', 'OPPONENT', 
               'PTS', 'REB', 'AST', 'MIN', 'SEASON']
available_sample_cols = [col for col in sample_cols if col in df.columns]

print("\nFirst 10 games:")
print(df[available_sample_cols].head(10))

print("\nLast 10 games (most recent):")
print(df[available_sample_cols].tail(10))

# Show data coverage by player
print("\n" + "="*60)
print("PLAYER DATA COVERAGE")
print("="*60)

player_summary = df.groupby('PLAYER_NAME').agg({
    'GAME_DATE': ['min', 'max', 'count'],
    'PTS': 'mean'
}).round(2)
player_summary.columns = ['First_Game', 'Last_Game', 'Total_Games', 'Avg_PTS']
player_summary = player_summary.sort_values('Total_Games', ascending=False)

print("\nTop 15 players by game count:")
print(player_summary.head(15))

# Check for duplicate games
print("\n" + "="*60)
print("CHECKING FOR DUPLICATE GAMES")
print("="*60)

duplicates = df.groupby(['PLAYER_NAME', 'GAME_DATE', 'MATCHUP']).size()
duplicates = duplicates[duplicates > 1].sort_values(ascending=False)

if len(duplicates) > 0:
    print(f"⚠ Found {len(duplicates)} duplicate game entries:")
    print(duplicates.head(10))
    print("\nRemoving duplicates (keeping first occurrence)...")
    df = df.drop_duplicates(subset=['PLAYER_NAME', 'GAME_DATE', 'MATCHUP'], keep='first')
    print(f"✓ Removed duplicates. New total: {len(df):,} games")
else:
    print("✓ No duplicate games found")

# STEP 4: Fix Team Abbreviations and Match Opponent Stats
print("\n" + "="*60)
print("FIXING OPPONENT DEFENSIVE STATS")
print("="*60)

# Use merged team stats
try:
    if team_stats is not None:
        print(f"✓ Using merged team stats: {len(team_stats)} team-seasons")

        # Check missing opponent stats before fixing
        missing_before = df['OPP_DEF_RATING'].isna().sum() if 'OPP_DEF_RATING' in df.columns else len(df)
        print(f"\nOpponent stats missing before fix: {missing_before} games ({missing_before/len(df)*100:.1f}%)")

        # Find correct team abbreviation column in team_stats
        team_col = None
        if 'TEAM_ABBREVIATION' in team_stats.columns:
            team_col = 'TEAM_ABBREVIATION'
        elif 'TEAM_ABBRV' in team_stats.columns:
            team_col = 'TEAM_ABBRV'
        else:
            # Create from team names using NBA API
            print("  Creating team abbreviations from team names...")
            from nba_api.stats.static import teams
            team_list = teams.get_teams()
            team_mapping_dict = {t['full_name']: t['abbreviation'] for t in team_list}
            team_stats['TEAM_ABBREVIATION'] = team_stats['TEAM_NAME'].map(team_mapping_dict)
            team_col = 'TEAM_ABBREVIATION'

        print(f"  Using team column: {team_col}")
        
        # Show unique teams in team_stats
        valid_teams = [t for t in team_stats[team_col].unique() if pd.notna(t)]
        print(f"\n  Teams in team_stats: {len(valid_teams)} unique teams")
        print(f"  Teams by season:")
        for season in sorted(team_stats['SEASON'].unique()):
            season_teams = team_stats[team_stats['SEASON'] == season][team_col].nunique()
            print(f"    {season}: {season_teams} teams")

        # Bidirectional team abbreviation mapping
        # Maps BOTH ESPN <-> NBA API formats
        team_abbrev_mapping = {
            # ESPN -> NBA API format
            'BKN': 'BRK',   # Brooklyn Nets
            'CHA': 'CHO',   # Charlotte Hornets
            'PHX': 'PHO',   # Phoenix Suns
            'SA': 'SAS',    # San Antonio Spurs (SHORT)
            'GS': 'GSW',    # Golden State Warriors (SHORT)
            'NO': 'NOP',    # New Orleans Pelicans (SHORT)
            'NY': 'NYK',    # New York Knicks (SHORT)
            'UTAH': 'UTA',  # Utah Jazz (LONG)
            'WSH': 'WAS',   # Washington Wizards (DIFF)
            # NBA API -> ESPN format (reverse mapping)
            'BRK': 'BKN',
            'CHO': 'CHA',
            'PHO': 'PHX',
            'SAS': 'SA',
            'GSW': 'GS',
            'NOP': 'NO',
            'NYK': 'NY',
            'UTA': 'UTAH',
            'WAS': 'WSH',
        }
        
        # Apply mapping to BOTH datasets for consistency
        # This ensures both use the same abbreviation format
        print("\n  Standardizing team abbreviations...")
        
        # Standardize team_stats to NBA API format (the format with more teams)
        team_stats['TEAM_STD'] = team_stats[team_col].copy()
        for espn_abbr, nba_abbr in team_abbrev_mapping.items():
            team_stats.loc[team_stats[team_col] == espn_abbr, 'TEAM_STD'] = nba_abbr
        
        # Standardize opponents in game logs to NBA API format
        df['OPPONENT_STD'] = df['OPPONENT'].copy()
        for espn_abbr, nba_abbr in team_abbrev_mapping.items():
            df.loc[df['OPPONENT'] == espn_abbr, 'OPPONENT_STD'] = nba_abbr
        
        # Show what we have after standardization
        print(f"    Team stats unique teams after standardization: {team_stats['TEAM_STD'].nunique()}")
        print(f"    Game logs unique opponents after standardization: {df['OPPONENT_STD'].nunique()}")
        
        # Check which teams are in team_stats
        available_teams = set(team_stats['TEAM_STD'].unique())
        game_opponents = set(df['OPPONENT_STD'].dropna().unique())
        missing_from_stats = game_opponents - available_teams
        
        if missing_from_stats:
            print(f"\n  ⚠ Teams in game logs but NOT in team_stats:")
            for team in sorted(missing_from_stats):
                count = df[df['OPPONENT_STD'] == team].shape[0]
                seasons = df[df['OPPONENT_STD'] == team]['SEASON'].unique()
                print(f"    {team}: {count} games across seasons {list(seasons)}")

        # Extract opponent from MATCHUP if not already present
        if 'OPPONENT' not in df.columns or df['OPPONENT'].isna().sum() > 0:
            print("\n  Extracting opponents from MATCHUP column...")
            def extract_opponent(matchup):
                if pd.isna(matchup):
                    return None
                matchup = str(matchup)
                if ' vs ' in matchup:
                    return matchup.split(' vs ')[-1].strip()
                elif ' @ ' in matchup:
                    return matchup.split(' @ ')[-1].strip()
                elif 'vs.' in matchup:
                    return matchup.split('vs.')[-1].strip()
                elif '@' in matchup:
                    return matchup.split('@')[-1].strip()
                return None
            df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent)
        
        print(f"\n  Unique opponents in game logs: {df['OPPONENT'].nunique()}")

        # Standardize both team_stats and game logs to common format
        team_stats['TEAM_STD'] = team_stats[team_col].replace(team_abbrev_mapping)
        df['OPPONENT_STD'] = df['OPPONENT'].replace(team_abbrev_mapping)
        
        # Prepare opponent stats for merging
        # Only use DEF_RATING and PACE (OFF_RATING not needed for opponent stats)
        merge_cols = ['TEAM_STD', 'SEASON']
        if 'DEF_RATING' in team_stats.columns:
            merge_cols.append('DEF_RATING')
        if 'PACE' in team_stats.columns:
            merge_cols.append('PACE')
        
        opp_stats = team_stats[merge_cols].copy()
        
        # Rename columns for opponent merge
        rename_map = {'TEAM_STD': 'OPPONENT_STD', 'SEASON': 'SEASON'}
        if 'DEF_RATING' in opp_stats.columns:
            rename_map['DEF_RATING'] = 'OPP_DEF_RATING'
        if 'PACE' in opp_stats.columns:
            rename_map['PACE'] = 'OPP_PACE'
        
        opp_stats = opp_stats.rename(columns=rename_map)

        # Remove old opponent stats columns if they exist
        df = df.drop(columns=['OPP_DEF_RATING', 'OPP_PACE'], errors='ignore')

        # Merge with standardized team abbreviations
        df = df.merge(opp_stats, on=['OPPONENT_STD', 'SEASON'], how='left')

        # Check improvement
        still_missing = df['OPP_DEF_RATING'].isna().sum()
        fixed_count = missing_before - still_missing
        
        if fixed_count > 0:
            print(f"\n✓ Successfully matched {fixed_count} additional games!")
        elif fixed_count == 0:
            print(f"\n⚠ No improvement in matching (still {still_missing} missing)")
        else:
            print(f"\n⚠ Matching got worse! Lost {abs(fixed_count)} matches")
        
        print(f"  Before: {missing_before} missing ({missing_before/len(df)*100:.1f}%)")
        print(f"  After: {still_missing} missing ({still_missing/len(df)*100:.1f}%)")

        # Show which opponents are still missing
        if still_missing > 0:
            print("\n  Opponents still missing stats:")
            missing_df = df[df['OPP_DEF_RATING'].isna()][['OPPONENT', 'OPPONENT_STD', 'SEASON']].copy()
            missing_summary = missing_df.groupby(['OPPONENT', 'OPPONENT_STD', 'SEASON']).size().sort_values(ascending=False)
            print(missing_summary.head(10))

        # Fill remaining missing with season-specific league averages
        if still_missing > 0:
            print("\n  Filling missing values with season-specific league averages...")
            for season in df['SEASON'].unique():
                season_mask = df['SEASON'] == season
                season_avg_def = df.loc[season_mask, 'OPP_DEF_RATING'].mean()
                season_avg_pace = df.loc[season_mask, 'OPP_PACE'].mean()
                
                if pd.notna(season_avg_def):
                    missing_count = (season_mask & df['OPP_DEF_RATING'].isna()).sum()
                    if missing_count > 0:
                        df.loc[season_mask & df['OPP_DEF_RATING'].isna(), 'OPP_DEF_RATING'] = season_avg_def
                        df.loc[season_mask & df['OPP_PACE'].isna(), 'OPP_PACE'] = season_avg_pace
                        print(f"    {season}: Filled {missing_count} games (DEF={season_avg_def:.2f}, PACE={season_avg_pace:.2f})")

        print("\n✓ Opponent defensive ratings distribution:")
        print(df['OPP_DEF_RATING'].describe())

    else:
        print("⚠ No team stats available, skipping opponent stats merge")
        
except Exception as e:
    print(f"\n✗ Error during opponent stats merge: {e}")
    import traceback
    traceback.print_exc()

    # Check missing opponent stats before fixing
    missing_before = df['OPP_DEF_RATING'].isna().sum() if 'OPP_DEF_RATING' in df.columns else len(df)
    print(f"\nOpponent stats missing before fix: {missing_before} games ({missing_before/len(df)*100:.1f}%)")

    # Find correct team abbreviation column in team_stats
    team_col = None
    if 'TEAM_ABBREVIATION' in team_stats.columns:
        team_col = 'TEAM_ABBREVIATION'
    elif 'TEAM_ABBRV' in team_stats.columns:
        team_col = 'TEAM_ABBRV'
    else:
        # Create from team names using NBA API
        print("  Creating team abbreviations from team names...")
        from nba_api.stats.static import teams
        team_list = teams.get_teams()
        team_mapping_dict = {t['full_name']: t['abbreviation'] for t in team_list}
        team_stats['TEAM_ABBREVIATION'] = team_stats['TEAM_NAME'].map(team_mapping_dict)
        team_col = 'TEAM_ABBREVIATION'

    print(f"  Using team column: {team_col}")
    
    # Show unique teams in team_stats
    print(f"\n  Teams in team_stats file: {sorted(team_stats[team_col].unique())}")

    # ESPN to NBA API team abbreviation mapping (BIDIRECTIONAL)
    # This maps BOTH directions: ESPN -> NBA API and NBA API -> ESPN
    team_abbrev_mapping = {
        # ESPN -> NBA API format
        'BKN': 'BRK',   # Brooklyn Nets
        'CHA': 'CHO',   # Charlotte Hornets
        'PHX': 'PHO',   # Phoenix Suns
        'SA': 'SAS',    # San Antonio Spurs
        'GS': 'GSW',    # Golden State Warriors
        'NO': 'NOP',    # New Orleans Pelicans
        'NY': 'NYK',    # New York Knicks
        'UTAH': 'UTA',  # Utah Jazz
        'WSH': 'WAS',   # Washington Wizards
        # NBA API -> ESPN format (reverse mapping)
        'BRK': 'BKN',
        'CHO': 'CHA',
        'PHO': 'PHX',
        'SAS': 'SA',
        'GSW': 'GS',
        'NOP': 'NO',
        'NYK': 'NY',
        'UTA': 'UTAH',
        'WAS': 'WSH',
    }

    # Extract opponent from MATCHUP if not already present
    if 'OPPONENT' not in df.columns or df['OPPONENT'].isna().sum() > 0:
        print("  Extracting opponents from MATCHUP column...")
        def extract_opponent(matchup):
            if pd.isna(matchup):
                return None
            matchup = str(matchup)
            if ' vs ' in matchup:
                return matchup.split(' vs ')[-1].strip()
            elif ' @ ' in matchup:
                return matchup.split(' @ ')[-1].strip()
            elif 'vs.' in matchup:
                return matchup.split('vs.')[-1].strip()
            elif '@' in matchup:
                return matchup.split('@')[-1].strip()
            return None
        df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent)
    
    print(f"\n  Unique opponents in game logs: {sorted(df['OPPONENT'].dropna().unique())}")

    # Standardize team_stats abbreviations to match game logs
    # Try both directions of mapping on team_stats
    team_stats['TEAM_STD'] = team_stats[team_col].replace(team_abbrev_mapping)
    
    # Standardize game logs opponents to match team_stats
    df['OPPONENT_STD'] = df['OPPONENT'].replace(team_abbrev_mapping)
    
    # Try merge with standardized abbreviations
    opp_stats = team_stats[['TEAM_STD', 'SEASON', 'DEF_RATING', 'PACE']].copy()
    opp_stats.columns = ['OPPONENT_STD', 'SEASON', 'OPP_DEF_RATING_NEW', 'OPP_PACE_NEW']

    # Remove old opponent stats columns if they exist
    df = df.drop(columns=['OPP_DEF_RATING', 'OPP_PACE'], errors='ignore')

    # Merge with standardized team abbreviations
    df = df.merge(opp_stats, on=['OPPONENT_STD', 'SEASON'], how='left')
    df = df.rename(columns={
        'OPP_DEF_RATING_NEW': 'OPP_DEF_RATING',
        'OPP_PACE_NEW': 'OPP_PACE'
    })

    # Check improvement
    still_missing = df['OPP_DEF_RATING'].isna().sum()
    fixed_count = missing_before - still_missing
    
    if fixed_count > 0:
        print(f"\n✓ Successfully matched {fixed_count} additional games!")
    elif fixed_count == 0:
        print(f"\n⚠ No improvement in matching (still {still_missing} missing)")
    else:
        print(f"\n⚠ Matching got worse! Lost {abs(fixed_count)} matches")
    
    print(f"  Before: {missing_before} missing ({missing_before/len(df)*100:.1f}%)")
    print(f"  After: {still_missing} missing ({still_missing/len(df)*100:.1f}%)")

    # Show which opponents are still missing (with both original and standardized names)
    if still_missing > 0:
        print("\n  Opponents still missing stats:")
        missing_df = df[df['OPP_DEF_RATING'].isna()][['OPPONENT', 'OPPONENT_STD', 'SEASON']].copy()
        missing_summary = missing_df.groupby(['OPPONENT', 'OPPONENT_STD']).size().sort_values(ascending=False)
        print(missing_summary.head(10))
        
        # Check if these teams exist in team_stats
        missing_teams = missing_df['OPPONENT_STD'].unique()
        available_teams = set(team_stats['TEAM_STD'].unique())
        print("\n  Missing teams analysis:")
        for team in missing_teams[:5]:
            if pd.notna(team):
                in_stats = "YES" if team in available_teams else "NO"
                print(f"    {team}: in team_stats = {in_stats}")

    # Fill remaining missing with league averages BY SEASON
    if still_missing > 0:
        print("\n  Filling missing values with season-specific league averages...")
        for season in df['SEASON'].unique():
            season_mask = df['SEASON'] == season
            season_avg_def = df.loc[season_mask, 'OPP_DEF_RATING'].mean()
            season_avg_pace = df.loc[season_mask, 'OPP_PACE'].mean()
            
            if pd.notna(season_avg_def):
                df.loc[season_mask & df['OPP_DEF_RATING'].isna(), 'OPP_DEF_RATING'] = season_avg_def
                df.loc[season_mask & df['OPP_PACE'].isna(), 'OPP_PACE'] = season_avg_pace
                print(f"    {season}: DEF_RATING={season_avg_def:.2f}, PACE={season_avg_pace:.2f}")

    print("\n✓ Opponent defensive ratings distribution:")
    print(df['OPP_DEF_RATING'].describe())

except FileNotFoundError:
    print("⚠ Team stats file not found, skipping opponent stats merge")

# STEP 5: Data Cleaning
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

original_rows = len(df)

# Drop columns with >90% missing (not useful for modeling)
high_missing_threshold = 0.9
cols_to_check = df.columns
high_missing_cols = []

for col in cols_to_check:
    missing_pct = df[col].isna().sum() / len(df)
    if missing_pct > high_missing_threshold:
        high_missing_cols.append(col)

if high_missing_cols:
    df = df.drop(columns=high_missing_cols)
    print(f"✓ Dropped {len(high_missing_cols)} columns with >{high_missing_threshold*100}% missing:")
    for col in high_missing_cols:
        print(f"  - {col}")

# Fill shooting percentages with 0 (for games with 0 attempts)
shooting_cols = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
for col in shooting_cols:
    if col in df.columns:
        df[col].fillna(0, inplace=True)
print(f"\n✓ Filled shooting percentages (0 for no attempts)")

# Drop rows missing critical stats
critical_cols = ['PTS', 'REB', 'AST', 'MIN', 'PLAYER_NAME', 'GAME_DATE']
existing_critical = [col for col in critical_cols if col in df.columns]
before_drop = len(df)
df = df.dropna(subset=existing_critical)
dropped_rows = before_drop - len(df)
if dropped_rows > 0:
    print(f"✓ Dropped {dropped_rows} rows missing critical stats ({dropped_rows/original_rows*100:.1f}%)")

# Fill remaining common missing values
if 'IS_HOME' in df.columns:
    df['IS_HOME'].fillna(0, inplace=True)

if 'PLUS_MINUS' in df.columns:
    df['PLUS_MINUS'].fillna(0, inplace=True)

if 'WL' in df.columns:
    df['WL'].fillna('L', inplace=True)

print(f"\n{'='*60}")
print("CLEANED DATASET SUMMARY")
print(f"{'='*60}")
print(f"Original rows: {original_rows:,}")
print(f"Cleaned rows: {len(df):,}")
print(f"Rows removed: {original_rows - len(df):,} ({(original_rows - len(df))/original_rows*100:.1f}%)")
print(f"\nMissing values remaining: {df.isnull().sum().sum()}")

remaining_missing = df.isnull().sum()
if remaining_missing.sum() > 0:
    print("\nColumns still with missing values:")
    missing_df = remaining_missing[remaining_missing > 0].sort_values(ascending=False)
    for col, count in missing_df.items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
else:
    print("\n✓ No missing values in critical columns!")

# STEP 6: Basic Data Info
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")
print(f"Unique players: {df['PLAYER_NAME'].nunique()}")
if 'SEASON' in df.columns:
    print(f"Seasons: {sorted(df['SEASON'].unique())}")

print("\nKey columns available:")
key_cols = ['PLAYER_NAME', 'GAME_DATE', 'MATCHUP', 'TEAM', 'OPPONENT', 
            'PTS', 'REB', 'AST', 'MIN', 'OPP_DEF_RATING', 'OPP_PACE', 'IS_HOME', 'WL']
for col in key_cols:
    status = "✓" if col in df.columns else "✗"
    print(f"  {status} {col}")

# STEP 7: Games Per Player
print("\n" + "="*60)
print("GAMES PER PLAYER")
print("="*60)

games_per_player = df.groupby('PLAYER_NAME').size().sort_values(ascending=False)
print(f"\nTop 10 players by games:")
print(games_per_player.head(10))

print(f"\nBottom 10 players by games:")
print(games_per_player.tail(10))

print(f"\nGames per player statistics:")
print(games_per_player.describe())

# Visualize
plt.figure(figsize=(14, 6))
games_per_player.head(20).plot(kind='bar', color='steelblue')
plt.title('Top 20 Players by Number of Games', fontsize=14, fontweight='bold')
plt.xlabel('Player', fontsize=12)
plt.ylabel('Number of Games', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# STEP 8: Statistical Summary
print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)

key_stats = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
             'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OPP_DEF_RATING', 'OPP_PACE']
available_stats = [stat for stat in key_stats if stat in df.columns]

print("\nDescriptive statistics:")
print(df[available_stats].describe().round(2))

# STEP 9: Points Distribution
print("\n" + "="*60)
print("POINTS DISTRIBUTION")
print("="*60)

print(f"\nPoints statistics:")
print(df['PTS'].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['PTS'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_title('Distribution of Points Scored', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Points')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['PTS'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["PTS"].mean():.1f}')
axes[0].axvline(df['PTS'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["PTS"].median():.1f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(df['PTS'], vert=True)
axes[1].set_title('Points Box Plot', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Points')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# STEP 10: Top Performers
print("\n" + "="*60)
print("TOP PERFORMERS")
print("="*60)

# Calculate averages (min 20 games)
player_stats = df.groupby('PLAYER_NAME').agg({
    'PTS': ['mean', 'std', 'count'],
    'REB': 'mean',
    'AST': 'mean',
    'MIN': 'mean'
}).round(2)

player_stats.columns = ['PTS_AVG', 'PTS_STD', 'GAMES', 'REB_AVG', 'AST_AVG', 'MIN_AVG']
player_stats = player_stats[player_stats['GAMES'] >= 20].sort_values('PTS_AVG', ascending=False)

print("\nTop 15 scorers (min 20 games):")
print(player_stats.head(15))

# Visualize
plt.figure(figsize=(12, 6))
top_15 = player_stats.head(15)
plt.barh(range(len(top_15)), top_15['PTS_AVG'], xerr=top_15['PTS_STD'],
         alpha=0.7, capsize=5, color='coral')
plt.yticks(range(len(top_15)), top_15.index)
plt.xlabel('Average Points Per Game', fontsize=12)
plt.title('Top 15 Scorers (with Std Dev)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# STEP 11: Correlation Analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

corr_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN', 'FGA', 'FG_PCT',
              'FG3A', 'FG3_PCT', 'OPP_DEF_RATING', 'OPP_PACE']
corr_stats = [col for col in corr_stats if col in df.columns]

correlation_matrix = df[corr_stats].corr()

print("\nCorrelation with Points:")
print(correlation_matrix['PTS'].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# STEP 12: Save Cleaned Data
print("\n" + "="*60)
print("SAVING CLEANED DATA")
print("="*60)

# Save cleaned player game logs
player_output_file = 'nba_player_game_logs_cleaned.csv'
df.to_csv(player_output_file, index=False)
print(f"✓ Cleaned player game logs saved to: {player_output_file}")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")

# Save cleaned team stats (simplified format)
if team_stats is not None:
    team_output_file = 'nba_team_stats_cleaned.csv'
    
    # Select only essential columns for team stats
    essential_cols = ['TEAM_ABBREVIATION', 'SEASON', 'DEF_RATING', 'PACE']
    
    # Add TEAM name if available
    if 'TEAM' in team_stats.columns:
        essential_cols.insert(0, 'TEAM')
    elif 'TEAM_NAME' in team_stats.columns:
        team_stats['TEAM'] = team_stats['TEAM_NAME']
        essential_cols.insert(0, 'TEAM')
    
    # Filter to only columns that exist
    available_cols = [col for col in essential_cols if col in team_stats.columns]
    
    team_stats_clean = team_stats[available_cols].copy()
    
    # Remove any duplicates (same team, same season)
    team_stats_clean = team_stats_clean.drop_duplicates(subset=['TEAM_ABBREVIATION', 'SEASON'], keep='first')
    
    # Sort by season and team
    team_stats_clean = team_stats_clean.sort_values(['SEASON', 'TEAM_ABBREVIATION']).reset_index(drop=True)
    
    # Save
    team_stats_clean.to_csv(team_output_file, index=False)
    
    print(f"\n✓ Cleaned team stats saved to: {team_output_file}")
    print(f"  Rows: {len(team_stats_clean):,} (teams × seasons)")
    print(f"  Columns: {len(team_stats_clean.columns)}")
    print(f"  Format: {list(team_stats_clean.columns)}")
    
    # Show summary
    print(f"\n  Team stats summary:")
    print(f"    Unique teams: {team_stats_clean['TEAM_ABBREVIATION'].nunique()}")
    print(f"    Seasons: {sorted(team_stats_clean['SEASON'].unique())}")
    print(f"    Teams per season:")
    for season in sorted(team_stats_clean['SEASON'].unique()):
        count = len(team_stats_clean[team_stats_clean['SEASON'] == season])
        print(f"      {season}: {count} teams")
    
    # Show sample
    print(f"\n  Sample team stats:")
    print(team_stats_clean.head(10).to_string(index=False))
else:
    print("\n⚠ No team stats to save")

# STEP 13: Final Summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"""
Dataset Summary:
  • Total games: {len(df):,}
  • Unique players: {df['PLAYER_NAME'].nunique()}
  • Date range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}
  • Avg points: {df['PTS'].mean():.1f} ± {df['PTS'].std():.1f}
  • Avg rebounds: {df['REB'].mean():.1f} ± {df['REB'].std():.1f}
  • Avg assists: {df['AST'].mean():.1f} ± {df['AST'].std():.1f}

Data Quality:
  • Missing values: {df.isnull().sum().sum()}
  • Opponent stats coverage: {(1 - df['OPP_DEF_RATING'].isna().sum()/len(df))*100:.1f}%
  • Tracking method: Player name only (no IDs)
  • Matchup format: Standardized (TEAM vs/@ OPPONENT)

✓ Ready for feature engineering!
""")

print("\nNext step: Run 03_feature_engineering.ipynb")