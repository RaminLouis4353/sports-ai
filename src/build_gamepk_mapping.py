import os
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

GAME_FEATURES_FILE = "data/raw/mlb_multi_season_team_data.csv"
PITCHER_FILE = "data/raw/mlb_pitcher_game_logs.csv"
OUTPUT_FILE = "data/processed/mlb_gamepk_mapping.csv"


# ============================================================
# TEAM NAME -> MLB CODE
# ============================================================

TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


# ============================================================
# NORMALIZE TEAM
# ============================================================

def normalize_team(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in TEAM_MAP:
        return TEAM_MAP[value]

    if value in TEAM_MAP.values():
        return value

    return value


# ============================================================
# CREATE DATE KEY FROM PITCHER DATE
#
# Pitcher dates look like:
# 2022-04-07
# 2023-06-15
# etc.
#
# We intentionally DO NOT use pd.to_datetime() here because
# the pitcher CSV contains mixed date formatting that was
# causing the process to stop.
# ============================================================

def pitcher_date_key(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str[:10]
    )


# ============================================================
# CREATE DATE KEY FROM GAME FEATURES
#
# Game feature dates look like:
# Thursday, Apr 7
# Friday, Apr 8
#
# There is no year in the Date column, so we use Season.
# ============================================================

def game_date_key(date_series, season_series):
    """
    Convert game dates into YYYY-MM-DD.

    Handles normal dates:
        Thursday, Apr 7

    And doubleheader dates:
        Tuesday, Apr 19 (1)
        Tuesday, Apr 19 (2)
    """

    date_text = (
        date_series
        .astype(str)
        .str.strip()
    )

    # Remove doubleheader suffix.
    #
    # Tuesday, Apr 19 (1)
    # ->
    # Tuesday, Apr 19
    #
    # Tuesday, Apr 19 (2)
    # ->
    # Tuesday, Apr 19
    date_text = (
        date_text
        .str.replace(
            r"\s*\([12]\)\s*$",
            "",
            regex=True
        )
        .str.strip()
    )

    # Remove weekday.
    #
    # Tuesday, Apr 19
    # ->
    # Apr 19
    date_text = (
        date_text
        .str.replace(
            r"^[A-Za-z]+,\s*",
            "",
            regex=True
        )
        .str.strip()
    )

    # Extract month and day.
    extracted = date_text.str.extract(
        r"^([A-Za-z]{3})\s+(\d{1,2})$"
    )

    month = extracted[0]
    day = extracted[1]

    # Month name -> number.
    month_numbers = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }

    month_number = month.map(month_numbers)

    # Day -> two digits.
    day_number = (
        pd.to_numeric(
            day,
            errors="coerce"
        )
        .astype("Int64")
        .astype(str)
        .str.zfill(2)
    )

    # Season -> year.
    season_text = (
        season_series
        .astype(str)
        .str.strip()
    )

    # Build YYYY-MM-DD.
    result = (
        season_text
        + "-"
        + month_number.astype(str)
        + "-"
        + day_number
    )

    # Identify invalid rows.
    invalid = (
        month_number.isna()
        | day.isna()
        | season_series.isna()
    )

    result = result.astype(object)
    result.loc[invalid] = None

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("BUILDING GAMEPK MAPPING")
    print("=" * 60)
    print()

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(GAME_FEATURES_FILE):
        raise FileNotFoundError(
            f"Game features file not found: {GAME_FEATURES_FILE}"
        )

    if not os.path.exists(PITCHER_FILE):
        raise FileNotFoundError(
            f"Pitcher file not found: {PITCHER_FILE}"
        )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # ========================================================
    # LOAD GAME FEATURES
    # ========================================================

    print("Loading game features...")

    games = pd.read_csv(
        GAME_FEATURES_FILE
    )

    print(f"Game rows: {len(games)}")
    print()

    required_game_columns = [
        "Date",
        "Tm",
        "Home_Away",
        "Opp",
        "Season",
    ]

    missing_game_columns = [
        col
        for col in required_game_columns
        if col not in games.columns
    ]

    if missing_game_columns:
        raise ValueError(
            "Game features are missing required columns: "
            + ", ".join(missing_game_columns)
        )

    # ========================================================
    # LOAD PITCHER LOGS
    # ========================================================

    print("Loading pitcher logs...")

    pitchers = pd.read_csv(
        PITCHER_FILE
    )

    print(f"Pitcher rows: {len(pitchers)}")
    print()

    required_pitcher_columns = [
        "GamePk",
        "Date",
        "Team",
        "Opponent",
        "IsHome",
        "GamesStarted",
    ]

    missing_pitcher_columns = [
        col
        for col in required_pitcher_columns
        if col not in pitchers.columns
    ]

    if missing_pitcher_columns:
        raise ValueError(
            "Pitcher data is missing required columns: "
            + ", ".join(missing_pitcher_columns)
        )

    # ========================================================
    # SELECT STARTERS
    # ========================================================

    print("Selecting starting pitchers...")

    starters = pitchers.loc[
        pitchers["GamesStarted"] == 1
    ].copy()

    print(f"Starter rows: {len(starters)}")
    print(
        f"Starter GamePks: "
        f"{starters['GamePk'].nunique()}"
    )
    print()

    if len(starters) == 0:
        raise ValueError(
            "No starting pitchers found."
        )

    # ========================================================
    # NORMALIZE PITCHER TEAMS
    # ========================================================

    print("Normalizing pitcher teams...")

    starters.loc[:, "TeamCode"] = (
        starters["Team"].map(normalize_team)
    )

    starters.loc[:, "OpponentCode"] = (
        starters["Opponent"].map(normalize_team)
    )

    unmapped_team = starters["TeamCode"].isna().sum()
    unmapped_opponent = starters["OpponentCode"].isna().sum()

    print(
        f"Unmapped Team values: {unmapped_team}"
    )

    print(
        f"Unmapped Opponent values: {unmapped_opponent}"
    )

    if unmapped_team > 0:
        print()
        print("Unmapped pitcher Team values:")
        print(
            starters.loc[
                starters["TeamCode"].isna(),
                "Team"
            ]
            .value_counts()
            .to_string()
        )

    if unmapped_opponent > 0:
        print()
        print("Unmapped pitcher Opponent values:")
        print(
            starters.loc[
                starters["OpponentCode"].isna(),
                "Opponent"
            ]
            .value_counts()
            .to_string()
        )

    print()

    # ========================================================
    # CREATE HOME / AWAY CODES
    # ========================================================

    print("Creating home/away codes...")

    starters.loc[:, "HomeCode"] = starters[
        "TeamCode"
    ].where(
        starters["IsHome"],
        starters["OpponentCode"]
    )

    starters.loc[:, "AwayCode"] = starters[
        "OpponentCode"
    ].where(
        starters["IsHome"],
        starters["TeamCode"]
    )

    print("HomeCode created.")
    print("AwayCode created.")
    print()

    # ========================================================
    # CREATE PITCHER DATE KEY
    # ========================================================

    print("Creating DateKey...")

    starters.loc[:, "DateKey"] = pitcher_date_key(
        starters["Date"]
    )

    invalid_dates = (
        starters["DateKey"].isna()
        | starters["DateKey"].eq("")
        | starters["DateKey"].eq("nan")
    ).sum()

    print(f"Invalid dates: {invalid_dates}")
    print()

    if invalid_dates > 0:
        raise ValueError(
            "Pitcher data contains invalid dates."
        )

    # ========================================================
    # SAMPLE
    # ========================================================

    print("Sample normalized starters:")

    sample_columns = [
        "GamePk",
        "DateKey",
        "TeamCode",
        "OpponentCode",
        "HomeCode",
        "AwayCode",
        "IsHome",
    ]

    if "PitcherName" in starters.columns:
        sample_columns.append("PitcherName")

    print(
        starters[
            sample_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    print()

    # ========================================================
    # BUILD GAMEPK LOOKUP
    #
    # IMPORTANT:
    #
    # We build the lookup directly from GamePk.
    #
    # Each GamePk represents one actual MLB game.
    #
    # There can be multiple starter rows for the same GamePk,
    # so we collapse those rows into ONE row per GamePk.
    # ========================================================

    print("Building GamePk lookup...")

    pitcher_lookup = (
        starters[
            [
                "GamePk",
                "DateKey",
                "HomeCode",
                "AwayCode",
            ]
        ]
        .dropna(
            subset=[
                "GamePk",
                "DateKey",
                "HomeCode",
                "AwayCode",
            ]
        )
        .copy()
    )

    # Make GamePk numeric.
    pitcher_lookup.loc[:, "GamePk"] = pd.to_numeric(
        pitcher_lookup["GamePk"],
        errors="coerce"
    )

    pitcher_lookup = pitcher_lookup.dropna(
        subset=["GamePk"]
    )

    # Remove duplicate starter rows for same GamePk.
    pitcher_lookup = (
        pitcher_lookup
        .drop_duplicates(
            subset=["GamePk"]
        )
        .copy()
    )

    print(
        f"Unique GamePks: "
        f"{pitcher_lookup['GamePk'].nunique()}"
    )

    print(
        f"GamePk lookup rows: "
        f"{len(pitcher_lookup)}"
    )

    print()

    # ========================================================
    # CHECK DATE / TEAM DUPLICATES
    #
    # Doubleheaders will naturally have multiple GamePks
    # for the same date/home/away matchup.
    # ========================================================

    matchup_counts = (
        pitcher_lookup
        .groupby(
            [
                "DateKey",
                "HomeCode",
                "AwayCode",
            ]
        )["GamePk"]
        .nunique()
        .reset_index(
            name="GameCount"
        )
    )

    multiple_matchups = matchup_counts[
        matchup_counts["GameCount"] > 1
    ]

    print(
        "Date/team matchup keys with multiple GamePks: "
        f"{len(multiple_matchups)}"
    )

    print()

    # ========================================================
    # ASSIGN GAME NUMBER TO PITCHER GAMES
    #
    # For a normal game:
    #
    # Date + Home + Away = GameNumber 1
    #
    # For a doubleheader:
    #
    # Date + Home + Away = GameNumber 1
    # Date + Home + Away = GameNumber 2
    #
    # Sorting by GamePk provides a deterministic order.
    # ========================================================

    pitcher_lookup = pitcher_lookup.sort_values(
        [
            "DateKey",
            "HomeCode",
            "AwayCode",
            "GamePk",
        ]
    ).copy()

    pitcher_lookup.loc[:, "GameNumber"] = (
        pitcher_lookup
        .groupby(
            [
                "DateKey",
                "HomeCode",
                "AwayCode",
            ]
        )
        .cumcount()
        + 1
    )

    # ========================================================
    # CREATE PITCHER MATCH KEY
    # ========================================================

    pitcher_lookup.loc[:, "MatchKey"] = (
        pitcher_lookup["DateKey"].astype(str)
        + "|"
        + pitcher_lookup["HomeCode"].astype(str)
        + "|"
        + pitcher_lookup["AwayCode"].astype(str)
        + "|"
        + pitcher_lookup["GameNumber"].astype(str)
    )

    # ========================================================
    # NORMALIZE GAME FEATURES
    # ========================================================

    print("Normalizing game features...")

    games = games.copy()

    # --------------------------------------------------------
    # Team
    # --------------------------------------------------------

    games.loc[:, "TeamCode"] = (
        games["Tm"].map(normalize_team)
    )

    games.loc[:, "OpponentCode"] = (
        games["Opp"].map(normalize_team)
    )

    # --------------------------------------------------------
    # Home / Away
    #
    # The actual game-feature dataset has:
    #
    # Home_Away = Home
    # Home_Away = Away
    #
    # For a Home row:
    #
    # HomeCode = Team
    # AwayCode = Opp
    #
    # For an Away row:
    #
    # HomeCode = Opp
    # AwayCode = Team
    # --------------------------------------------------------

    home_away_text = (
        games["Home_Away"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    games.loc[:, "HomeCode"] = (
        games["TeamCode"].where(
            home_away_text.eq("home"),
            games["OpponentCode"]
        )
    )

    games.loc[:, "AwayCode"] = (
        games["OpponentCode"].where(
            home_away_text.eq("home"),
            games["TeamCode"]
        )
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    print("Creating game DateKey...")

    games.loc[:, "DateKey"] = game_date_key(
        games["Date"],
        games["Season"]
    )

    invalid_game_dates = (
        games["DateKey"].isna()
    ).sum()

    print(
        f"Invalid game dates: "
        f"{invalid_game_dates}"
    )

    if invalid_game_dates > 0:

        print()
        print("Examples of invalid game dates:")

        print(
            games.loc[
                games["DateKey"].isna(),
                [
                    "Date",
                    "Season",
                    "Tm",
                    "Home_Away",
                    "Opp",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Game features contain invalid dates."
        )

    print()

    # ========================================================
    # CREATE GAME NUMBER
    #
    # Because the source has one row per TEAM per game,
    # we first reduce it to one row per actual game.
    # ========================================================

    print("Creating GameNumber key...")

    # --------------------------------------------------------
    # Keep only rows where the team is listed as Home.
    #
    # This gives one row per game because every game has
    # exactly one home team row.
    # --------------------------------------------------------

    games_unique = games.loc[
        home_away_text.eq("home")
    ].copy()

    # If Home_Away has unexpected values, report them.
    unexpected_home_away = (
        ~home_away_text.isin(
            ["home", "away"]
        )
    ).sum()

    if unexpected_home_away > 0:

        print(
            "WARNING: unexpected Home_Away values: "
            f"{unexpected_home_away}"
        )

    # --------------------------------------------------------
    # Remove exact duplicate game rows.
    # --------------------------------------------------------

    games_unique = (
        games_unique
        .drop_duplicates(
            subset=[
                "Season",
                "DateKey",
                "HomeCode",
                "AwayCode",
            ]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Sort and assign GameNumber.
    # --------------------------------------------------------

    games_unique = games_unique.sort_values(
        [
            "Season",
            "DateKey",
            "HomeCode",
            "AwayCode",
        ]
    ).copy()

    games_unique.loc[:, "GameNumber"] = (
        games_unique
        .groupby(
            [
                "Season",
                "DateKey",
                "HomeCode",
                "AwayCode",
            ]
        )
        .cumcount()
        + 1
    )

    # --------------------------------------------------------
    # Match key.
    # --------------------------------------------------------

    games_unique.loc[:, "MatchKey"] = (
        games_unique["DateKey"].astype(str)
        + "|"
        + games_unique["HomeCode"].astype(str)
        + "|"
        + games_unique["AwayCode"].astype(str)
        + "|"
        + games_unique["GameNumber"].astype(str)
    )

    print(
        f"Unique game rows: {len(games_unique)}"
    )

    print()

    # ========================================================
    # MATCH GAMES TO GAMEPK
    # ========================================================

    print("Matching games to GamePk...")

    mapping_columns = [
        "MatchKey",
        "GamePk",
    ]

    mapping = pitcher_lookup[
        mapping_columns
    ].copy()

    # Safety check.
    duplicate_mapping_keys = (
        mapping["MatchKey"]
        .duplicated()
        .sum()
    )

    if duplicate_mapping_keys > 0:

        print(
            "WARNING: duplicate MatchKeys in "
            "pitcher lookup: "
            f"{duplicate_mapping_keys}"
        )

        mapping = (
            mapping
            .drop_duplicates(
                subset=["MatchKey"],
                keep="first"
            )
            .copy()
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    games_unique = games_unique.merge(
        mapping,
        on="MatchKey",
        how="left",
        validate="one_to_one"
    )

    # ========================================================
    # VALIDATE MAPPING
    # ========================================================

    print()
    print("Validating mapping...")

    matched = (
        games_unique["GamePk"]
        .notna()
        .sum()
    )

    unmatched = (
        games_unique["GamePk"]
        .isna()
        .sum()
    )

    total_unique_games = len(
        games_unique
    )

    print(
        f"Matched unique games: {matched}"
    )

    print(
        f"Unmatched unique games: {unmatched}"
    )

    print(
        f"Total unique games: {total_unique_games}"
    )

    # ========================================================
    # DUPLICATE GAMEPK CHECK
    # ========================================================

    duplicate_gamepks = (
        games_unique.loc[
            games_unique["GamePk"].notna(),
            "GamePk"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate GamePk assignments: "
        f"{duplicate_gamepks}"
    )

    # ========================================================
    # SHOW UNMATCHED GAMES
    # ========================================================

    if unmatched > 0:

        print()
        print("UNMATCHED GAMES:")

        unmatched_display = games_unique.loc[
            games_unique["GamePk"].isna(),
            [
                "Season",
                "DateKey",
                "HomeCode",
                "AwayCode",
                "GameNumber",
                "MatchKey",
            ]
        ].copy()

        unmatched_display = (
            unmatched_display
            .rename(
                columns={
                    "HomeCode": "Home",
                    "AwayCode": "Away",
                }
            )
        )

        print(
            unmatched_display
            .head(100)
            .to_string(index=False)
        )

        if len(unmatched_display) > 100:
            print()
            print(
                f"... {len(unmatched_display) - 100} "
                "additional unmatched games."
            )

    # ========================================================
    # DOUBLEHEADER CHECK
    # ========================================================

    print()
    print("Checking doubleheaders...")

    doubleheaders = (
        games_unique
        .groupby(
            [
                "DateKey",
                "HomeCode",
                "AwayCode",
            ]
        )["GameNumber"]
        .max()
        .reset_index(
            name="GameCount"
        )
    )

    doubleheaders = doubleheaders.loc[
        doubleheaders["GameCount"] > 1
    ]

    print(
        f"Doubleheader matchups: "
        f"{len(doubleheaders)}"
    )

    if len(doubleheaders) > 0:

        print()
        print("Sample doubleheaders:")

        doubleheader_keys = doubleheaders[
            [
                "DateKey",
                "HomeCode",
                "AwayCode",
            ]
        ].head(20)

        sample_doubleheaders = games_unique.merge(
            doubleheader_keys,
            on=[
                "DateKey",
                "HomeCode",
                "AwayCode",
            ],
            how="inner"
        )

        sample_doubleheaders = (
            sample_doubleheaders
            .sort_values(
                [
                    "DateKey",
                    "HomeCode",
                    "AwayCode",
                    "GameNumber",
                ]
            )
            [
                [
                    "DateKey",
                    "HomeCode",
                    "AwayCode",
                    "GameNumber",
                    "GamePk",
                ]
            ]
        )

        print(
            sample_doubleheaders
            .head(40)
            .to_string(index=False)
        )

    # ========================================================
    # CREATE FINAL OUTPUT
    #
    # We want one row per actual game, NOT one row per team.
    #
    # Therefore the final mapping is based on games_unique.
    # ========================================================

    final_mapping = games_unique[
        [
            "Season",
            "DateKey",
            "HomeCode",
            "AwayCode",
            "GameNumber",
            "GamePk",
            "MatchKey",
        ]
    ].copy()

    # Rename for clarity.
    final_mapping = final_mapping.rename(
        columns={
            "HomeCode": "Home",
            "AwayCode": "Away",
        }
    )

    # Sort.
    final_mapping = final_mapping.sort_values(
        [
            "Season",
            "DateKey",
            "Home",
            "Away",
            "GameNumber",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # FINAL SAFETY CHECKS
    # ========================================================

    print()
    print("Final safety checks...")

    # No duplicate MatchKeys.
    duplicate_matchkeys = (
        final_mapping["MatchKey"]
        .duplicated()
        .sum()
    )

    # No duplicate GamePk values among matched rows.
    matched_rows = final_mapping.loc[
        final_mapping["GamePk"].notna()
    ]

    duplicate_gamepk_final = (
        matched_rows["GamePk"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate MatchKeys: "
        f"{duplicate_matchkeys}"
    )

    print(
        f"Duplicate matched GamePks: "
        f"{duplicate_gamepk_final}"
    )

    if duplicate_matchkeys > 0:
        raise ValueError(
            "Duplicate MatchKeys found in final mapping."
        )

    if duplicate_gamepk_final > 0:
        raise ValueError(
            "Duplicate GamePk assignments found."
        )

    # ========================================================
    # SAVE
    # ========================================================

    print()
    print("Saving mapping...")

    final_mapping.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_matched = (
        final_mapping["GamePk"]
        .notna()
        .sum()
    )

    final_unmatched = (
        final_mapping["GamePk"]
        .isna()
        .sum()
    )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"Rows: {len(final_mapping)}"
    )

    print(
        f"Matched: {final_matched}"
    )

    print(
        f"Unmatched: {final_unmatched}"
    )

    print(
        f"Duplicate GamePk assignments: "
        f"{duplicate_gamepk_final}"
    )

    print(
        f"Doubleheader matchups: "
        f"{len(doubleheaders)}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()