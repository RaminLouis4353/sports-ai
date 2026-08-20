import pandas as pd

GAME_INPUT = "data/processed/mlb_multi_season_features.csv"
PITCHER_INPUT = "data/processed/mlb_pitcher_recent_features.csv"
OUTPUT = "data/processed/mlb_features_with_recent_pitchers.csv"

print("=" * 60)
print("MERGING RECENT STARTING PITCHER FEATURES")
print("=" * 60)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print()
print("Loading game features...")
games = pd.read_csv(GAME_INPUT)
print("Game rows:", len(games))

print()
print("Loading pitcher features...")
pitchers = pd.read_csv(PITCHER_INPUT)
print("Pitcher rows:", len(pitchers))

# ------------------------------------------------------------
# TEAM MAPPING
# ------------------------------------------------------------

team_map = {
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

print()
print("Mapping pitcher teams...")

pitchers = pitchers.copy()
pitchers["TeamCode"] = pitchers["Team"].map(team_map)

unmapped = pitchers["TeamCode"].isna().sum()

print("Unmapped teams:", unmapped)

if unmapped > 0:
    print(
        pitchers.loc[
            pitchers["TeamCode"].isna(),
            "Team"
        ].unique()
    )
    raise ValueError("Some pitcher teams could not be mapped.")

# ------------------------------------------------------------
# DATE
# ------------------------------------------------------------

print()
print("Converting dates...")

games = games.copy()
games["Date"] = pd.to_datetime(
    games["Date"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

pitchers["Date"] = pd.to_datetime(
    pitchers["Date"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

if games["Date"].isna().any():
    raise ValueError("Game data contains invalid dates.")

if pitchers["Date"].isna().any():
    raise ValueError("Pitcher data contains invalid dates.")

print("Game date example:", repr(games["Date"].iloc[0]))
print("Pitcher date example:", repr(pitchers["Date"].iloc[0]))

# ------------------------------------------------------------
# STARTING PITCHERS
# ------------------------------------------------------------

print()
print("Selecting starting pitchers...")

starters = pitchers[
    pitchers["GamesStarted"] == 1
].copy()

print("Starting pitcher rows:", len(starters))

# One starter per GamePk + TeamCode.
starters = starters.drop_duplicates(
    subset=["GamePk", "TeamCode"],
    keep="first"
).copy()

print("Unique game/team starter rows:", len(starters))

# ------------------------------------------------------------
# IMPORTANT:
# CREATE DATE + TEAM KEYS
#
# GamePk is NOT present in the game feature file.
# Therefore the merge must use Date + Team.
# ------------------------------------------------------------

starters["MergeDate"] = starters["Date"]
starters["MergeTeam"] = starters["TeamCode"]

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

home_feature_map = {
    "PitcherID": "Home_PitcherID",
    "PitcherName": "Home_PitcherName",
    "DaysRest": "Home_DaysRest",

    "Last3_ERA": "Home_Last3_ERA",
    "Last3_IP": "Home_Last3_IP",
    "Last3_K": "Home_Last3_K",
    "Last3_BB": "Home_Last3_BB",
    "Last3_H": "Home_Last3_H",
    "Last3_HR": "Home_Last3_HR",
    "Last3_WHIP": "Home_Last3_WHIP",
    "Last3_K9": "Home_Last3_K9",
    "Last3_BB9": "Home_Last3_BB9",
    "Last3_HR9": "Home_Last3_HR9",

    "Last5_ERA": "Home_Last5_ERA",
    "Last5_IP": "Home_Last5_IP",
    "Last5_K": "Home_Last5_K",
    "Last5_BB": "Home_Last5_BB",
    "Last5_H": "Home_Last5_H",
    "Last5_HR": "Home_Last5_HR",
    "Last5_WHIP": "Home_Last5_WHIP",
    "Last5_K9": "Home_Last5_K9",
    "Last5_BB9": "Home_Last5_BB9",
    "Last5_HR9": "Home_Last5_HR9",
}

away_feature_map = {
    "PitcherID": "Away_PitcherID",
    "PitcherName": "Away_PitcherName",
    "DaysRest": "Away_DaysRest",

    "Last3_ERA": "Away_Last3_ERA",
    "Last3_IP": "Away_Last3_IP",
    "Last3_K": "Away_Last3_K",
    "Last3_BB": "Away_Last3_BB",
    "Last3_H": "Away_Last3_H",
    "Last3_HR": "Away_Last3_HR",
    "Last3_WHIP": "Away_Last3_WHIP",
    "Last3_K9": "Away_Last3_K9",
    "Last3_BB9": "Away_Last3_BB9",
    "Last3_HR9": "Away_Last3_HR9",

    "Last5_ERA": "Away_Last5_ERA",
    "Last5_IP": "Away_Last5_IP",
    "Last5_K": "Away_Last5_K",
    "Last5_BB": "Away_Last5_BB",
    "Last5_H": "Away_Last5_H",
    "Last5_HR": "Away_Last5_HR",
    "Last5_WHIP": "Away_Last5_WHIP",
    "Last5_K9": "Away_Last5_K9",
    "Last5_BB9": "Away_Last5_BB9",
    "Last5_HR9": "Away_Last5_HR9",
}

# ------------------------------------------------------------
# HOME STARTERS
# ------------------------------------------------------------

print()
print("Preparing home starters...")

home = starters[
    starters["IsHome"] == True
].copy()

home = home.rename(
    columns=home_feature_map
)

home = home[
    [
        "MergeDate",
        "MergeTeam",

        "Home_PitcherID",
        "Home_PitcherName",
        "Home_DaysRest",

        "Home_Last3_ERA",
        "Home_Last3_IP",
        "Home_Last3_K",
        "Home_Last3_BB",
        "Home_Last3_H",
        "Home_Last3_HR",
        "Home_Last3_WHIP",
        "Home_Last3_K9",
        "Home_Last3_BB9",
        "Home_Last3_HR9",

        "Home_Last5_ERA",
        "Home_Last5_IP",
        "Home_Last5_K",
        "Home_Last5_BB",
        "Home_Last5_H",
        "Home_Last5_HR",
        "Home_Last5_WHIP",
        "Home_Last5_K9",
        "Home_Last5_BB9",
        "Home_Last5_HR9",
    ]
].copy()

home = home.drop_duplicates(
    subset=["MergeDate", "MergeTeam"],
    keep="first"
).copy()

print("Home starter rows:", len(home))

# ------------------------------------------------------------
# AWAY STARTERS
# ------------------------------------------------------------

print()
print("Preparing away starters...")

away = starters[
    starters["IsHome"] == False
].copy()

away = away.rename(
    columns=away_feature_map
)

away = away[
    [
        "MergeDate",
        "MergeTeam",

        "Away_PitcherID",
        "Away_PitcherName",
        "Away_DaysRest",

        "Away_Last3_ERA",
        "Away_Last3_IP",
        "Away_Last3_K",
        "Away_Last3_BB",
        "Away_Last3_H",
        "Away_Last3_HR",
        "Away_Last3_WHIP",
        "Away_Last3_K9",
        "Away_Last3_BB9",
        "Away_Last3_HR9",

        "Away_Last5_ERA",
        "Away_Last5_IP",
        "Away_Last5_K",
        "Away_Last5_BB",
        "Away_Last5_H",
        "Away_Last5_HR",
        "Away_Last5_WHIP",
        "Away_Last5_K9",
        "Away_Last5_BB9",
        "Away_Last5_HR9",
    ]
].copy()

away = away.drop_duplicates(
    subset=["MergeDate", "MergeTeam"],
    keep="first"
).copy()

print("Away starter rows:", len(away))

# ------------------------------------------------------------
# MERGE HOME
# ------------------------------------------------------------

print()
print("Merging home starters...")

games["MergeDate"] = games["Date"]
games["MergeHome"] = games["Home"]

games = games.merge(
    home,
    left_on=["MergeDate", "MergeHome"],
    right_on=["MergeDate", "MergeTeam"],
    how="left",
    validate="many_to_one"
)

games = games.drop(
    columns=["MergeTeam"],
    errors="ignore"
)

print("Home merge complete.")

# ------------------------------------------------------------
# MERGE AWAY
# ------------------------------------------------------------

print()
print("Merging away starters...")

games["MergeAway"] = games["Away"]

games = games.merge(
    away,
    left_on=["MergeDate", "MergeAway"],
    right_on=["MergeDate", "MergeTeam"],
    how="left",
    validate="many_to_one"
)

games = games.drop(
    columns=["MergeTeam"],
    errors="ignore"
)

print("Away merge complete.")

# ------------------------------------------------------------
# REMOVE TEMP COLUMNS
# ------------------------------------------------------------

games = games.drop(
    columns=[
        "MergeDate",
        "MergeHome",
        "MergeAway",
    ],
    errors="ignore"
)

# ------------------------------------------------------------
# DIFFERENCE FEATURES
# ------------------------------------------------------------

print()
print("Calculating pitcher difference features...")

games["Pitcher_DaysRest_Diff"] = (
    games["Home_DaysRest"] -
    games["Away_DaysRest"]
)

games["Pitcher_Last3_ERA_Diff"] = (
    games["Home_Last3_ERA"] -
    games["Away_Last3_ERA"]
)

games["Pitcher_Last3_WHIP_Diff"] = (
    games["Home_Last3_WHIP"] -
    games["Away_Last3_WHIP"]
)

games["Pitcher_Last3_K9_Diff"] = (
    games["Home_Last3_K9"] -
    games["Away_Last3_K9"]
)

games["Pitcher_Last3_BB9_Diff"] = (
    games["Home_Last3_BB9"] -
    games["Away_Last3_BB9"]
)

games["Pitcher_Last3_HR9_Diff"] = (
    games["Home_Last3_HR9"] -
    games["Away_Last3_HR9"]
)

games["Pitcher_Last5_ERA_Diff"] = (
    games["Home_Last5_ERA"] -
    games["Away_Last5_ERA"]
)

games["Pitcher_Last5_WHIP_Diff"] = (
    games["Home_Last5_WHIP"] -
    games["Away_Last5_WHIP"]
)

games["Pitcher_Last5_K9_Diff"] = (
    games["Home_Last5_K9"] -
    games["Away_Last5_K9"]
)

games["Pitcher_Last5_BB9_Diff"] = (
    games["Home_Last5_BB9"] -
    games["Away_Last5_BB9"]
)

games["Pitcher_Last5_HR9_Diff"] = (
    games["Home_Last5_HR9"] -
    games["Away_Last5_HR9"]
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print()
print("Validating merge...")

home_found = games["Home_PitcherID"].notna().sum()
away_found = games["Away_PitcherID"].notna().sum()

both_found = (
    games["Home_PitcherID"].notna()
    &
    games["Away_PitcherID"].notna()
).sum()

print(
    "Home starters matched:",
    home_found,
    "/",
    len(games)
)

print(
    "Away starters matched:",
    away_found,
    "/",
    len(games)
)

print(
    "Games with both starters:",
    both_found,
    "/",
    len(games)
)

# ------------------------------------------------------------
# DUPLICATE VALIDATION
# ------------------------------------------------------------

print()
print("Checking MatchKey duplicates...")

if "MatchKey" in games.columns:
    duplicate_matchkeys = games["MatchKey"].duplicated().sum()
    unique_matchkeys = games["MatchKey"].nunique()

    print("Unique MatchKeys:", unique_matchkeys)
    print("Duplicate MatchKeys:", duplicate_matchkeys)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print()
print("Saving...")

games.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 60)
print("RESULT")
print("=" * 60)
print("Rows:", len(games))
print("Columns:", len(games.columns))
print("Saved:", OUTPUT)
print("=" * 60)
print("COMPLETE")
print("=" * 60)