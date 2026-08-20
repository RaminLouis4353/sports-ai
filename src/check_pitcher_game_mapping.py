import pandas as pd

INPUT = "data/raw/mlb_pitcher_game_logs.csv"

print("=" * 60)
print("CHECKING PITCHER GAME MAPPING")
print("=" * 60)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print()
print("Loading pitcher data...")

p = pd.read_csv(INPUT)

print("Total rows:", len(p))
print("Unique GamePks:", p["GamePk"].nunique())

# ------------------------------------------------------------
# STARTERS
# ------------------------------------------------------------

print()
print("Selecting starters...")

s = p.loc[p["GamesStarted"] == 1].copy()

print("Starter rows:", len(s))
print("Starter GamePks:", s["GamePk"].nunique())

# ------------------------------------------------------------
# TEAM MAP
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
print("Mapping teams...")

s.loc[:, "TeamCode"] = s["Team"].map(team_map)

print(
    "Unmapped teams:",
    s["TeamCode"].isna().sum()
)

# ------------------------------------------------------------
# HOME / AWAY
# ------------------------------------------------------------

print()
print("Creating home/away codes...")

s.loc[:, "HomeCode"] = s["TeamCode"].where(
    s["IsHome"],
    s["Opponent"]
)

s.loc[:, "AwayCode"] = s["Opponent"].where(
    s["IsHome"],
    s["TeamCode"]
)

print("HomeCode created.")
print("AwayCode created.")

# ------------------------------------------------------------
# DATE
# ------------------------------------------------------------

print()
print("Checking dates...")

# Dates are already YYYY-MM-DD.
# Do NOT use pd.to_datetime() here.

s.loc[:, "DateKey"] = s["Date"].astype(str).str.strip()

print("DateKey created.")

print(
    "Invalid/empty dates:",
    (
        s["DateKey"].isna()
        | (s["DateKey"] == "")
    ).sum()
)

# ------------------------------------------------------------
# SAMPLE
# ------------------------------------------------------------

print()
print("Sample mapped starters:")

print(
    s[
        [
            "GamePk",
            "DateKey",
            "TeamCode",
            "Opponent",
            "HomeCode",
            "AwayCode",
            "IsHome",
            "PitcherName",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# GROUP
# ------------------------------------------------------------

print()
print("Grouping games...")

x = (
    s.groupby(
        [
            "DateKey",
            "HomeCode",
            "AwayCode",
        ]
    )
    .agg(
        GameCount=("GamePk", "nunique"),
        StarterRows=("GamePk", "size"),
    )
    .reset_index()
)

print("Groups:", len(x))

multiple = x.loc[x["GameCount"] > 1].copy()

print(
    "Date/team matchups with multiple GamePks:",
    len(multiple)
)

print()
print("First multiple-game matchups:")

if len(multiple) > 0:
    print(
        multiple
        .head(20)
        .to_string(index=False)
    )
else:
    print("NONE")

print()
print("=" * 60)
print("CHECK COMPLETE")
print("=" * 60)