import pandas as pd
from datetime import datetime

print("1 - Loading")

games = pd.read_csv(
    "data/processed/mlb_2024_games.csv"
)

print("2 - Loaded")

dates = []

for _, row in games.iterrows():

    text = str(row["Date"])
    text = text.replace("(1)", "")
    text = text.replace("(2)", "")
    text = text.strip()
    text = text.split(",", 1)[1].strip()

    dates.append(
        datetime.strptime(
            text + " " + str(int(row["Season"])),
            "%b %d %Y"
        )
    )

print("3 - Dates converted")

game_dates = pd.Series(
    dates,
    index=games.index,
    name="GameDate"
)

print("4 - Series created")

games = games.copy()

print("5 - Copy created")

games = pd.concat(
    [games, game_dates],
    axis=1
)

print("6 - Concat worked")

print(
    games[["Date", "GameDate"]].head()
)

print("7 - TEST COMPLETE")