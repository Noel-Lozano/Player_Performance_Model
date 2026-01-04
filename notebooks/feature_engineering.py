import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class NBAFeatureEngineer:
    """
    Leakage-safe, production-ready feature engineering for NBA prop modeling.
    """

    def __init__(self, player_logs_path: str, team_stats_path: str):
        print("Loading data files...")
        self.player_logs = pd.read_csv(player_logs_path)
        self.team_stats = pd.read_csv(team_stats_path)

        self.player_logs["GAME_DATE"] = pd.to_datetime(self.player_logs["GAME_DATE"])
        self.player_logs = self.player_logs.sort_values(
            ["PLAYER_NAME", "GAME_DATE"]
        ).reset_index(drop=True)

        print(f"Loaded {len(self.player_logs)} player game logs")
        print(f"Loaded {len(self.team_stats)} team stat records")

    # ------------------------------------------------------------------
    # Rolling stats + PRA
    # ------------------------------------------------------------------
    def create_rolling_features(self, df, windows=(5, 10, 15)):
        print("Creating rolling performance features...")

        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

        stats = [
            "PTS", "AST", "REB", "STL", "BLK",
            "FG3M", "MIN", "FGA", "FGM",
            "FTA", "FTM", "TOV", "PLUS_MINUS", "PRA"
        ]

        g = df.groupby("PLAYER_NAME")

        for w in windows:
            for s in stats:
                df[f"{s}_L{w}"] = g[s].shift(1).rolling(w, min_periods=1).mean()
                df[f"{s}_STD_L{w}"] = g[s].shift(1).rolling(w, min_periods=2).std()

        for s in ["PTS", "AST", "REB", "PRA"]:
            df[f"{s}_TREND"] = df[f"{s}_L5"] - df[f"{s}_L15"]

        return df

    # ------------------------------------------------------------------
    # Shooting efficiency (NO groupby.apply)
    # ------------------------------------------------------------------
    def create_shooting_efficiency_features(self, df, windows=(5, 10, 15)):
        print("Creating shooting efficiency features...")

        g = df.groupby("PLAYER_NAME")

        for w in windows:
            df[f"FG_PCT_L{w}"] = (
                g["FGM"].shift(1).rolling(w).sum() /
                g["FGA"].shift(1).rolling(w).sum().replace(0, np.nan)
            )

            df[f"FG3_PCT_L{w}"] = (
                g["FG3M"].shift(1).rolling(w).sum() /
                g["FG3A"].shift(1).rolling(w).sum().replace(0, np.nan)
            )

            df[f"FT_PCT_L{w}"] = (
                g["FTM"].shift(1).rolling(w).sum() /
                g["FTA"].shift(1).rolling(w).sum().replace(0, np.nan)
            )

        return df

    # ------------------------------------------------------------------
    # Usage & minutes
    # ------------------------------------------------------------------
    def create_usage_features(self, df):
        print("Creating usage & minutes features...")

        g = df.groupby("PLAYER_NAME")

        for w in (5, 10):
            df[f"MIN_L{w}"] = g["MIN"].shift(1).rolling(w).mean()

        df["FGA_PER_MIN"] = df["FGA"] / df["MIN"].replace(0, np.nan)
        df["FG3A_PER_MIN"] = df["FG3A"] / df["MIN"].replace(0, np.nan)

        for w in (5, 10):
            df[f"FGA_PER_MIN_L{w}"] = g["FGA_PER_MIN"].shift(1).rolling(w).mean()

        return df

    # ------------------------------------------------------------------
    # Rest / workload
    # ------------------------------------------------------------------
    def create_rest_features(self, df):
      print("Creating rest & workload features...")

      g = df.groupby("PLAYER_NAME")

      # Days since last game
      df["DAYS_REST"] = g["GAME_DATE"].diff().dt.days.fillna(7)

      # Schedule indicators
      df["IS_B2B"] = (df["DAYS_REST"] <= 1).astype(int)
      df["IS_RESTED"] = (df["DAYS_REST"] >= 3).astype(int)

      # Games played in last 7 days (manual, safe implementation)
      df["GAMES_L7"] = (
          g["GAME_DATE"]
          .apply(lambda x: x.shift(1).apply(
              lambda d: (x < d + pd.Timedelta(days=7)).sum() if pd.notna(d) else 0
          ))
          .reset_index(level=0, drop=True)
      )

      return df

    # ------------------------------------------------------------------
    # Home / away splits (NO backfill leakage)
    # ------------------------------------------------------------------
    def create_home_away_features(self, df):
      print("Creating home / away split features...")

      df = df.copy()
      stats = ["PTS", "AST", "REB", "PRA"]

      for s in stats:
          df[f"{s}_HOME_AVG"] = np.nan
          df[f"{s}_AWAY_AVG"] = np.nan

      for player, group in df.groupby("PLAYER_NAME"):
          idx = group.index

          for s in stats:
              # Home games
              home_mask = group["IS_HOME"] == 1
              home_vals = group.loc[home_mask, s].values
              home_idx = group.loc[home_mask].index

              home_means = [np.nan] + [
                  np.mean(home_vals[:i]) for i in range(1, len(home_vals))
              ]
              df.loc[home_idx, f"{s}_HOME_AVG"] = home_means

              # Away games
              away_mask = group["IS_HOME"] == 0
              away_vals = group.loc[away_mask, s].values
              away_idx = group.loc[away_mask].index

              away_means = [np.nan] + [
                  np.mean(away_vals[:i]) for i in range(1, len(away_vals))
              ]
              df.loc[away_idx, f"{s}_AWAY_AVG"] = away_means

      # Fill remaining NaNs with shifted season average
      for s in stats:
          season_avg = df.groupby("PLAYER_NAME")[s].shift(1).expanding().mean()
          df[f"{s}_HOME_AVG"].fillna(season_avg, inplace=True)
          df[f"{s}_AWAY_AVG"].fillna(season_avg, inplace=True)

      return df


    # ------------------------------------------------------------------
    # Opponent context
    # ------------------------------------------------------------------
    def create_opponent_features(self, df):
        print("Creating opponent features...")

        stats = ["PTS", "AST", "REB", "PRA"]

        for s in stats:
            df[f"{s}_VS_OPP"] = (
                df.groupby(["PLAYER_NAME", "OPPONENT"])[s]
                .shift(1)
                .expanding()
                .mean()
            )

        df["OPP_DEF_RATING_REL"] = (
            df["OPP_DEF_RATING"] -
            df.groupby("GAME_DATE")["OPP_DEF_RATING"].transform("mean")
        )

        df["OPP_PACE_REL"] = (
            df["OPP_PACE"] -
            df.groupby("GAME_DATE")["OPP_PACE"].transform("mean")
        )

        return df

    # ------------------------------------------------------------------
    # Season context
    # ------------------------------------------------------------------
    def create_season_features(self, df):
        print("Creating season context features...")

        stats = ["PTS", "AST", "REB", "STL", "BLK", "FG3M", "PRA"]

        for s in stats:
            df[f"{s}_SEASON_AVG"] = (
                df.groupby(["PLAYER_NAME", "SEASON"])[s]
                .shift(1)
                .expanding()
                .mean()
            )

        df["GAMES_PLAYED_SEASON"] = df.groupby(
            ["PLAYER_NAME", "SEASON"]
        ).cumcount()

        df["MONTH"] = df["GAME_DATE"].dt.month
        df["DAY_OF_WEEK"] = df["GAME_DATE"].dt.dayofweek

        return df

    # ------------------------------------------------------------------
    # Consistency & volatility
    # ------------------------------------------------------------------
    def create_consistency_features(self, df):
        print("Creating consistency features...")

        stats = ["PTS", "AST", "REB", "PRA"]

        for s in stats:
            for w in (10, 15):
                df[f"{s}_CV_L{w}"] = (
                    df[f"{s}_STD_L{w}"] /
                    df[f"{s}_L{w}"].replace(0, np.nan)
                )

                df[f"{s}_MIN_L{w}"] = (
                    df.groupby("PLAYER_NAME")[s]
                    .shift(1)
                    .rolling(w)
                    .min()
                )

                df[f"{s}_MAX_L{w}"] = (
                    df.groupby("PLAYER_NAME")[s]
                    .shift(1)
                    .rolling(w)
                    .max()
                )

        return df

    # ------------------------------------------------------------------
    def create_all_features(self):
        print("\nStarting feature engineering pipeline...\n")

        df = self.player_logs.copy()

        df = self.create_rolling_features(df)
        df = self.create_shooting_efficiency_features(df)
        df = self.create_usage_features(df)
        df = self.create_rest_features(df)
        df = self.create_home_away_features(df)
        df = self.create_opponent_features(df)
        df = self.create_season_features(df)
        df = self.create_consistency_features(df)

        print("\nFeature engineering complete.")
        print(f"Total columns: {len(df.columns)}")

        return df

# ==========================================================
# RUN FEATURE ENGINEERING
# ==========================================================
if __name__ == "__main__":

  print("\nInitializing Feature Engineer...\n")

  engineer = NBAFeatureEngineer(
      player_logs_path="nba_player_game_logs_cleaned.csv",
      team_stats_path="nba_team_stats_cleaned.csv"
  )

  print("\nRunning full feature engineering pipeline...\n")
  df_features = engineer.create_all_features()

  print("\nSaving engineered features to CSV...\n")
  df_features.to_csv("nba_features_engineered.csv", index=False)

  print("✓ Saved: nba_features_engineered.csv")
  print(f"✓ Shape: {df_features.shape}")

  print("\nSample columns:")
  for col in df_features.columns[:15]:
      print("  ", col)

  print("\nFeature engineering pipeline completed successfully.")
