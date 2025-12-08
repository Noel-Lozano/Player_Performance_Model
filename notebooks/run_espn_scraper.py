# ESPN NBA Scraper - INCREMENTAL UPDATE VERSION

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta

print("✓ Imports successful")

CURRENT_SEASON = '2025-26'
OUTPUT_FILE = 'nba_current_season_2025_26.csv'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
SEASON_START = datetime(2025, 10, 21)  # Regular season start

ESPN_SCOREBOARD_API = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
ESPN_SUMMARY_API = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary'

# List of target players to track
TARGET_PLAYERS_LIST = [
    "LeBron James",
    "Stephen Curry",
    "Kevin Durant",
    "Giannis Antetokounmpo",
    "Luka Doncic",
    "Nikola Jokic",
    "Joel Embiid",
    "Jayson Tatum",
    "Damian Lillard",
    "Anthony Davis",
    "Devin Booker",
    "Donovan Mitchell",
    "Jaylen Brown",
    "Trae Young",
    "Anthony Edwards",
    "Shai Gilgeous-Alexander",
    "Jimmy Butler III",
    "Paul George",
    "Tyrese Haliburton",
    "De'Aaron Fox",
    "Domantas Sabonis",
    "Bam Adebayo",
    "Julius Randle",
    "DeMar DeRozan",
    "Pascal Siakam",
    "LaMelo Ball",
    "James Harden",
    "Karl-Anthony Towns",
    "Nikola Vucevic",
    "Jalen Brunson",
    "Fred VanVleet",
    "Tyler Herro",
    "Victor Wembanyama",
    "Paolo Banchero",
    "Franz Wagner",
    "Scottie Barnes",
    "Cade Cunningham",
    "Alperen Sengun",
    "Jaren Jackson Jr.",
    "Mikal Bridges",
    "Darius Garland",
    "Lauri Markkanen",
    "Desmond Bane",
    "Jalen Williams",
    "OG Anunoby",
    "Jarrett Allen",
    "Kristaps Porzingis",
    "CJ McCollum"
]

def get_games_for_date(date_str):
    url = f"{ESPN_SCOREBOARD_API}?dates={date_str}"
    headers = {'User-Agent': USER_AGENT}
    try:
        time.sleep(0.3)
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json().get('events', [])
    except Exception as e:
        print(f"Error fetching games for {date_str}: {e}")
    return []

def get_game_box_score(game_id):
    url = f"{ESPN_SUMMARY_API}?event={game_id}"
    headers = {'User-Agent': USER_AGENT}
    try:
        time.sleep(0.3)
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching game {game_id}: {e}")
    return None

def safe_float(s):
    """Convert string to float safely"""
    try:
        return float(str(s).strip())
    except:
        return 0.0

def parse_split(s):
    """Parse 'X-Y' format into two floats"""
    try:
        if "-" in str(s):
            a, b = str(s).split("-")
            return safe_float(a), safe_float(b)
        return safe_float(s), 0.0
    except:
        return 0.0, 0.0

def extract_player_stats_from_game(game_data, game_id):
    """
    Extract player stats with CORRECT ESPN column mapping:
    Index 0: MIN, 1: PTS, 2: FG, 3: 3PT, 4: FT, 5: REB, 6: AST, 7: TO,
    8: STL, 9: BLK, 10: OREB, 11: DREB, 12: PF, 13: +/-
    """
    player_stats = []
    if not game_data or 'boxscore' not in game_data:
        return player_stats

    game_date = game_data.get('header', {}).get('competitions', [{}])[0].get('date', '')
    boxscore = game_data['boxscore']

    # get team matchups for each game for feature engineering later
    matchups = game_data.get('header', {}).get('competitions', [{}])[0].get('competitors', [])

    home_team = ''
    away_team = ''

    for team in matchups:
        team_abbrev = team.get('team', {}).get('abbreviation', '')
        if team.get('homeAway') == 'home':
            home_team = team_abbrev
        else:
            away_team = team_abbrev

    if "players" not in boxscore:
        return player_stats

    # Create a set of target player names (case-insensitive)
    target_players_set = {name.lower() for name in TARGET_PLAYERS_LIST}

    # Track processed players to avoid duplicates
    processed_players = set()

    for team_entry in boxscore["players"]:
        team_abbrev = team_entry.get("team", {}).get("abbreviation", "")

        for stat_group in team_entry.get("statistics", []):
            athletes = stat_group.get("athletes", [])

            for athlete in athletes:
                athlete_name = athlete.get("athlete", {}).get("displayName", "")

                # Match by name only (case-insensitive)
                if athlete_name.lower() in target_players_set:
                    player_name = athlete_name
                else:
                    # Not a target player
                    continue

                # Create a unique identifier for this player in this game
                player_game_key = f"{player_name}_{game_id}"

                # Skip if we've already processed this player in this game
                if player_game_key in processed_players:
                    continue

                # Mark this player as processed for this game
                processed_players.add(player_game_key)

                stats = athlete.get("stats", [])

                # More lenient check for stats length
                if len(stats) < 10:  # Require at least basic stats
                    continue

                # Pad stats array if needed
                while len(stats) < 14:
                    stats.append('0')

                fgm, fga = parse_split(stats[2])
                fg3m, fg3a = parse_split(stats[3])
                ftm, fta = parse_split(stats[4])

                player_stats.append({
                    "PLAYER_NAME": player_name,
                    "GAME_ID": game_id,
                    "GAME_DATE": game_date[:10],
                    "TEAM": team_abbrev,
                    "SEASON": CURRENT_SEASON,
                    "IS_HOME": 1 if team_abbrev == home_team else 0,
                    "OPPONENT": away_team if team_abbrev == home_team else home_team,
                    "MATCHUP": f"{team_abbrev} vs {away_team if team_abbrev == home_team else home_team}",
                    "MIN": safe_float(stats[0]),
                    "PTS": safe_float(stats[1]),
                    "FGM": fgm,
                    "FGA": fga,
                    "FG3M": fg3m,
                    "FG3A": fg3a,
                    "FTM": ftm,
                    "FTA": fta,
                    "REB": safe_float(stats[5]),
                    "AST": safe_float(stats[6]),
                    "TOV": safe_float(stats[7]),
                    "STL": safe_float(stats[8]),
                    "BLK": safe_float(stats[9]),
                    "OREB": safe_float(stats[10]) if len(stats) > 10 else 0.0,
                    "DREB": safe_float(stats[11]) if len(stats) > 11 else 0.0,
                    "PF": safe_float(stats[12]) if len(stats) > 12 else 0.0,
                    "PLUS_MINUS": safe_float(stats[13]) if len(stats) > 13 else 0.0
                })

    return player_stats

# ============================
# SMART COLLECTION LOGIC
# ============================

print("=" * 70)
print("ESPN NBA SCRAPER - INCREMENTAL UPDATE")
print("=" * 70)

# Determine collection strategy
if os.path.exists(OUTPUT_FILE):
    # INCREMENTAL MODE - File exists
    existing_df = pd.read_csv(OUTPUT_FILE)
    existing_df['GAME_DATE'] = pd.to_datetime(existing_df['GAME_DATE'])

    last_game_date = existing_df['GAME_DATE'].max()
    days_since_update = (datetime.now() - last_game_date).days

    # Go back 60 days from today to ensure we have enough for rolling stats
    # But don't go before season start
    lookback_start = max(
        datetime.now() - timedelta(days=60),
        SEASON_START
    )

    # Start 3 days before last game to catch any late-updated scores
    start_date = max(
        last_game_date - timedelta(days=3),
        lookback_start
    )

    print(f"\n📊 INCREMENTAL UPDATE MODE")
    print(f"  Existing data: {len(existing_df):,} games")
    print(f"  Last game date: {last_game_date.date()}")
    print(f"  Days since last update: {days_since_update}")
    print(f"  Collection start: {start_date.date()} (includes 60-day lookback)")
    print(f"  Strategy: Update recent games + ensure 60-day window for rolling stats\n")

else:
    # FULL COLLECTION MODE - No existing file
    start_date = SEASON_START
    existing_df = None

    print(f"\n🆕 FULL SEASON COLLECTION MODE")
    print(f"  No existing data found")
    print(f"  Collection start: {start_date.date()} (season opener)")
    print(f"  Strategy: Collect all games from season start\n")

end_date = datetime.now()

# Main Collection Loop
print("Starting data collection...")
print(f"Date range: {start_date.date()} → {end_date.date()}")
print("-" * 70)

all_player_stats = []
dates_checked = 0
games_found = 0
current_date = start_date

while current_date <= end_date:
    date_str = current_date.strftime('%Y%m%d')
    dates_checked += 1

    if dates_checked % 10 == 0:
        print(f"  Checked {dates_checked} dates | Found {games_found} games | Collected {len(all_player_stats)} performances")

    games = get_games_for_date(date_str)
    if games:
        for game in games:
            game_id = game.get('id')
            if not game_id:
                continue
            games_found += 1

            game_data = get_game_box_score(game_id)
            if game_data:
                player_stats = extract_player_stats_from_game(game_data, game_id)
                all_player_stats.extend(player_stats)

    current_date += timedelta(days=1)

print(f"\n{'='*70}")
print("COLLECTION COMPLETE")
print(f"{'='*70}")
print(f"Dates checked: {dates_checked}")
print(f"Games found: {games_found}")
print(f"Player performances collected: {len(all_player_stats)}")

# Process and Save
if all_player_stats:
    new_df = pd.DataFrame(all_player_stats)
    new_df['GAME_DATE'] = pd.to_datetime(new_df['GAME_DATE'])

    # Calculate shooting percentages
    new_df['FG_PCT'] = new_df['FGM'] / new_df['FGA'].replace(0, np.nan)
    new_df['FG3_PCT'] = new_df['FG3M'] / new_df['FG3A'].replace(0, np.nan)
    new_df['FT_PCT'] = new_df['FTM'] / new_df['FTA'].replace(0, np.nan)

    if existing_df is not None:
        # MERGE MODE - Combine with existing data
        print(f"\n📥 Merging with existing data...")

        # Combine datasets
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Remove duplicates (same player, same game)
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=['PLAYER_NAME', 'GAME_DATE', 'GAME_ID'],
            keep='last'  # Keep most recent version
        )
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup

        # Sort by player and date
        combined_df = combined_df.sort_values(['PLAYER_NAME', 'GAME_DATE'])

        # Save merged data
        combined_df.to_csv(OUTPUT_FILE, index=False)

        new_games_added = len(combined_df) - len(existing_df)

        print(f"  Previous total: {len(existing_df):,} games")
        print(f"  New games collected: {len(new_df):,}")
        print(f"  Duplicates removed: {duplicates_removed:,}")
        print(f"  New unique games added: {new_games_added:,}")
        print(f"  Final total: {len(combined_df):,} games")

        final_df = combined_df

    else:
        # FIRST RUN MODE - Save all data
        new_df = new_df.sort_values(['PLAYER_NAME', 'GAME_DATE'])
        new_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n💾 Saved initial dataset: {len(new_df):,} games")
        final_df = new_df

    # Summary Statistics
    print(f"\n{'='*70}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print(f"  Total games: {len(final_df):,}")
    print(f"  Unique players: {final_df['PLAYER_NAME'].nunique()}")
    print(f"  Date range: {final_df['GAME_DATE'].min().date()} → {final_df['GAME_DATE'].max().date()}")
    print(f"  Games per player (avg): {len(final_df) / final_df['PLAYER_NAME'].nunique():.1f}")

    print(f"\n📊 Average Stats:")
    print(f"  Points: {final_df['PTS'].mean():.1f} ± {final_df['PTS'].std():.1f}")
    print(f"  Rebounds: {final_df['REB'].mean():.1f} ± {final_df['REB'].std():.1f}")
    print(f"  Assists: {final_df['AST'].mean():.1f} ± {final_df['AST'].std():.1f}")
    print(f"  FG%: {final_df['FG_PCT'].mean():.1%}")
    print(f"  3P%: {final_df['FG3_PCT'].mean():.1%}")

    print(f"\n✅ Ready for feature engineering with 60-day rolling stats!")

else:
    print("\n❌ ERROR: No data collected!")
    print("Possible issues:")
    print("  - No games in date range")
    print("  - API connectivity issues")
    print("  - Rate limiting")

print(f"\n{'='*70}")
print("ESPN SCRAPER COMPLETE")
print(f"{'='*70}\n")