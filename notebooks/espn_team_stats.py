import requests
import pandas as pd
from bs4 import BeautifulSoup

print("="*70)
print("ESPN HOLLINGER TEAM STATS SCRAPER - CURRENT SEASON")
print("="*70)

HOLLINGER_URL = "https://www.espn.com/nba/hollinger/teamstats"
OUTPUT_FILE = "nba_team_stats_current_season.csv"
CURRENT_SEASON = "2025-26"

# Team abbreviation mapping (ESPN names -> STANDARD NBA API abbreviations)
# Using NBA API format for consistency with historical data
TEAM_ABBR = {
    "Denver": "DEN",
    "New York": "NYK",      # Fixed: Use NYK not NY
    "Houston": "HOU",
    "Boston": "BOS",
    "Oklahoma City": "OKC",
    "LA Lakers": "LAL",
    "San Antonio": "SAS",   # Fixed: Use SAS not SA
    "Minnesota": "MIN",
    "Cleveland": "CLE",
    "Orlando": "ORL",
    "Miami": "MIA",
    "Detroit": "DET",
    "Milwaukee": "MIL",
    "Phoenix": "PHO",       # Fixed: Use PHO not PHX
    "Toronto": "TOR",
    "Atlanta": "ATL",
    "Philadelphia": "PHI",
    "Utah": "UTA",          # Fixed: Use UTA not UTAH
    "Chicago": "CHI",
    "Portland": "POR",
    "LA Clippers": "LAC",
    "Golden State": "GSW",  # Fixed: Use GSW not GS
    "Charlotte": "CHO",     # Fixed: Use CHO not CHA
    "New Orleans": "NOP",   # Fixed: Use NOP not NO
    "Memphis": "MEM",
    "Brooklyn": "BRK",      # Fixed: Use BRK not BKN
    "Washington": "WAS",    # Fixed: Use WAS not WSH
    "Sacramento": "SAC",
    "Indiana": "IND",
    "Dallas": "DAL"
}

def get_hollinger_stats():
    """
    Scrape ESPN Hollinger team stats for current season
    Returns: DataFrame with team defensive/offensive ratings and pace
    """
    print(f"\nFetching data from: {HOLLINGER_URL}")

    try:
        response = requests.get(
            HOLLINGER_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=30
        )
        response.raise_for_status()

        # Parse HTML with pandas read_html (handles tables automatically)
        tables = pd.read_html(response.text)

        print(f"✓ Found {len(tables)} table(s)")

        if not tables:
            raise Exception("No tables found on the page")

        # The Hollinger page has one main table with all the stats
        df = tables[0]

        print(f"  Table shape: {df.shape}")
        print(f"  Column types: {type(df.columns)}")
        print(f"  First few columns: {list(df.columns[:5])}")

        # Check if columns are numeric (0, 1, 2...) - means pandas didn't detect headers
        if isinstance(df.columns[0], int):
            print("  Columns are numeric - using first row as headers")
            # First row is the header
            df.columns = df.iloc[0]
            df = df[1:]  # Remove the header row from data
            df = df.reset_index(drop=True)
            print(f"  New columns after fixing: {list(df.columns)}")

        # Show sample data for debugging
        print(f"\n  Sample data (first 3 rows):")
        print(df.head(3))

        # Now find the columns we need
        team_col = None
        pace_col = None
        off_col = None
        def_col = None

        for col in df.columns:
            col_str = str(col).upper()
            if 'TEAM' in col_str and team_col is None:
                team_col = col
            elif 'PACE' in col_str and pace_col is None:
                pace_col = col
            elif 'OFF' in col_str and 'EFF' in col_str and off_col is None:
                off_col = col
            elif 'DEF' in col_str and 'EFF' in col_str and def_col is None:
                def_col = col

        print(f"\n  Mapped columns:")
        print(f"    TEAM: {team_col}")
        print(f"    PACE: {pace_col}")
        print(f"    OFF EFF: {off_col}")
        print(f"    DEF EFF: {def_col}")

        if not all([team_col, pace_col, off_col, def_col]):
            missing = []
            if not team_col: missing.append("TEAM")
            if not pace_col: missing.append("PACE")
            if not off_col: missing.append("OFF EFF")
            if not def_col: missing.append("DEF EFF")

            print(f"\n  ⚠ Could not auto-detect columns. Available columns:")
            for i, col in enumerate(df.columns):
                print(f"    [{i}] {col}")

            # Manual mapping based on typical Hollinger table structure
            # RK, TEAM, PACE, AST, TO, ORR, DRR, REBR, EFF FG%, TS%, OFF EFF, DEF EFF
            if len(df.columns) >= 12:
                print(f"\n  Attempting manual column mapping (standard Hollinger format)...")
                team_col = df.columns[1]   # Column 1 is TEAM
                pace_col = df.columns[2]   # Column 2 is PACE
                off_col = df.columns[10]   # Column 10 is OFF EFF
                def_col = df.columns[11]   # Column 11 is DEF EFF

                print(f"    Using columns: TEAM={team_col}, PACE={pace_col}, OFF={off_col}, DEF={def_col}")
            else:
                raise Exception(f"Could not find required columns: {missing}")

        # Select only the columns we need
        df_stats = df[[team_col, pace_col, off_col, def_col]].copy()
        df_stats.columns = ['TEAM', 'PACE', 'OFF_RATING', 'DEF_RATING']

        # Remove any additional header rows
        df_stats = df_stats[df_stats['TEAM'].astype(str).str.upper() != 'TEAM']
        df_stats = df_stats[df_stats['TEAM'].astype(str) != 'RK']

        # Reset index
        df_stats = df_stats.reset_index(drop=True)

        print(f"\n✓ Cleaned data: {len(df_stats)} teams")
        print(f"\n  Sample teams:")
        print(df_stats.head(5))

        # Add team abbreviations
        df_stats['TEAM_ABBREVIATION'] = df_stats['TEAM'].map(TEAM_ABBR)

        # Check for teams without abbreviations
        missing_abbr = df_stats[df_stats['TEAM_ABBREVIATION'].isna()]['TEAM'].tolist()
        if missing_abbr:
            print(f"\n⚠ Warning: Teams without abbreviations:")
            for team in missing_abbr:
                print(f"    - '{team}'")
            print(f"  These teams will be skipped")

        # Drop teams without abbreviations
        df_stats = df_stats.dropna(subset=['TEAM_ABBREVIATION'])

        # Add season
        df_stats['SEASON'] = CURRENT_SEASON

        # Convert to numeric
        numeric_cols = ['PACE', 'OFF_RATING', 'DEF_RATING']
        for col in numeric_cols:
            df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')

        # Reorder columns
        df_stats = df_stats[['TEAM_ABBREVIATION', 'SEASON', 'DEF_RATING', 'OFF_RATING', 'PACE', 'TEAM']]

        return df_stats

    except requests.RequestException as e:
        print(f"\n✗ Network error: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error parsing Hollinger stats: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    try:
        # Scrape current season stats
        df_current = get_hollinger_stats()

        print(f"\n{'='*70}")
        print("CURRENT SEASON STATS SUMMARY")
        print(f"{'='*70}")
        print(f"Teams collected: {len(df_current)}")
        print(f"Season: {CURRENT_SEASON}")

        # Check for all 30 teams
        if len(df_current) == 30:
            print(f"\n✓ All 30 NBA teams found!")
        else:
            print(f"\n⚠ Warning: Expected 30 teams, found {len(df_current)}")

        # Check data completeness
        missing_def = df_current['DEF_RATING'].isna().sum()
        missing_off = df_current['OFF_RATING'].isna().sum()
        missing_pace = df_current['PACE'].isna().sum()

        print(f"\nData completeness:")
        print(f"  DEF_RATING: {len(df_current) - missing_def}/{len(df_current)} teams")
        print(f"  OFF_RATING: {len(df_current) - missing_off}/{len(df_current)} teams")
        print(f"  PACE: {len(df_current) - missing_pace}/{len(df_current)} teams")

        if missing_def > 0 or missing_off > 0 or missing_pace > 0:
            print(f"\n⚠ Some stats are missing. Teams with missing data:")
            missing_teams = df_current[
                df_current['DEF_RATING'].isna() |
                df_current['OFF_RATING'].isna() |
                df_current['PACE'].isna()
            ]
            for _, row in missing_teams.iterrows():
                print(f"    - {row['TEAM_ABBREVIATION']}: ", end="")
                missing_fields = []
                if pd.isna(row['DEF_RATING']):
                    missing_fields.append('DEF_RATING')
                if pd.isna(row['OFF_RATING']):
                    missing_fields.append('OFF_RATING')
                if pd.isna(row['PACE']):
                    missing_fields.append('PACE')
                print(", ".join(missing_fields))

        print(f"\nTeams collected:")
        print(df_current[['TEAM_ABBREVIATION', 'TEAM']].to_string(index=False))

        print(f"\nStats summary:")
        print(df_current[['DEF_RATING', 'OFF_RATING', 'PACE']].describe().round(2))

        # Save current season data
        df_current.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Current season team stats saved to: {OUTPUT_FILE}")

        print(f"\n{'='*70}")
        print("HOLLINGER SCRAPER COMPLETE")
        print(f"{'='*70}")
        print(f"\n✓ Data ready for merging!")
        print(f"  Next: Run your data exploration script to merge all data sources")

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print(f"{'='*70}")
        print(f"✗ Failed to scrape Hollinger stats: {e}")
        print(f"\nPossible issues:")
        print(f"  - ESPN changed their HTML structure")
        print(f"  - Network connectivity problem")
        print(f"  - Page requires authentication")
        print(f"\nTry visiting {HOLLINGER_URL} in your browser to verify it's accessible.")