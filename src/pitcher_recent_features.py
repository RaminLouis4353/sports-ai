import pandas as pd
import numpy as np
from datetime import datetime

INPUT = "data/raw/mlb_pitcher_game_logs.csv"
OUTPUT = "data/processed/mlb_pitcher_recent_features.csv"

print("=" * 50)
print("RECENT PITCHER FEATURE ENGINEERING")
print("=" * 50)

print("Loading pitcher logs...")

p = pd.read_csv(INPUT)

print("Rows:", len(p))

# --------------------------------------------------
# Sort using the original date strings.
# YYYY-MM-DD sorts correctly chronologically.
# --------------------------------------------------

print("Sorting data...")

p = p.sort_values(
    ["PitcherID", "Date"],
    kind="mergesort"
).reset_index(drop=True)

print("Data sorted.")

# --------------------------------------------------
# Convert baseball innings notation
# --------------------------------------------------

def convert_ip(x):

    if pd.isna(x):
        return 0.0

    x = float(x)

    whole = int(x)
    remainder = int(round((x - whole) * 10))

    if remainder == 1:
        return whole + (1 / 3)

    elif remainder == 2:
        return whole + (2 / 3)

    return float(whole)


print("Converting innings...")

p.loc[:, "IP"] = p["InningsPitched"].map(convert_ip)

print("Innings converted.")

# --------------------------------------------------
# Calculate days rest using Python only
# No pandas datetime conversion.
# --------------------------------------------------

print("Calculating days rest...")

days_rest = [0] * len(p)

last_date = {}

for i in range(len(p)):

    pitcher = p.iloc[i]["PitcherID"]
    date_string = str(p.iloc[i]["Date"])

    current_date = datetime.strptime(
        date_string,
        "%Y-%m-%d"
    )

    if pitcher in last_date:

        days_rest[i] = (
            current_date - last_date[pitcher]
        ).days

    else:

        days_rest[i] = 0

    last_date[pitcher] = current_date

p.loc[:, "DaysRest"] = days_rest

print("Days rest calculated.")

# --------------------------------------------------
# Recent pitcher features
# --------------------------------------------------

print("Calculating recent pitcher features...")

pitcher_groups = p.groupby(
    "PitcherID",
    sort=False
)

for n in [3, 5]:

    print("  Last", n, "starts...")

    for col, short_name in [
        ("EarnedRuns", "ER"),
        ("IP", "IP"),
        ("Strikeouts", "K"),
        ("Walks", "BB"),
        ("Hits", "H"),
        ("HomeRuns", "HR")
    ]:

        previous = pitcher_groups[col].shift(1)

        rolling = (
            previous
            .groupby(p["PitcherID"], sort=False)
            .rolling(
                n,
                min_periods=1
            )
            .sum()
            .reset_index(
                level=0,
                drop=True
            )
        )

        p.loc[:, f"Last{n}_{short_name}"] = (
            rolling.to_numpy()
        )

    ip = p[f"Last{n}_IP"].replace(
        0,
        np.nan
    )

    p.loc[:, f"Last{n}_ERA"] = (
        p[f"Last{n}_ER"] * 9 / ip
    ).fillna(0)

    p.loc[:, f"Last{n}_WHIP"] = (
        (
            p[f"Last{n}_H"] +
            p[f"Last{n}_BB"]
        ) / ip
    ).fillna(0)

    p.loc[:, f"Last{n}_K9"] = (
        p[f"Last{n}_K"] * 9 / ip
    ).fillna(0)

    p.loc[:, f"Last{n}_BB9"] = (
        p[f"Last{n}_BB"] * 9 / ip
    ).fillna(0)

    p.loc[:, f"Last{n}_HR9"] = (
        p[f"Last{n}_HR"] * 9 / ip
    ).fillna(0)

# --------------------------------------------------
# Output
# --------------------------------------------------

print("Building output...")

output_cols = [
    "PitcherID",
    "PitcherName",
    "Season",
    "Date",
    "GamePk",
    "Team",
    "Opponent",
    "IsHome",
    "GamesStarted",
    "DaysRest"
]

for n in [3, 5]:

    output_cols += [
        f"Last{n}_ERA",
        f"Last{n}_IP",
        f"Last{n}_K",
        f"Last{n}_BB",
        f"Last{n}_H",
        f"Last{n}_HR",
        f"Last{n}_WHIP",
        f"Last{n}_K9",
        f"Last{n}_BB9",
        f"Last{n}_HR9"
    ]

result = p.loc[:, output_cols].copy()

result = result.replace(
    [np.inf, -np.inf],
    np.nan
)

result = result.fillna(0)

# --------------------------------------------------
# Save
# --------------------------------------------------

print("Saving...")

result.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 50)
print("RESULT")
print("=" * 50)

print("Rows:", len(result))
print(
    "Pitchers:",
    result["PitcherID"].nunique()
)

print(
    "Date range:",
    result["Date"].min(),
    "to",
    result["Date"].max()
)

print(
    "Columns:",
    len(result.columns)
)

print()
print("Saved:", OUTPUT)

print()
print("=" * 50)
print("COMPLETE")
print("=" * 50)