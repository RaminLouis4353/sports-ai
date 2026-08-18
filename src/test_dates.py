import pandas as pd
from datetime import datetime

print("Loading...")

games = pd.read_csv(
    "data/processed/mlb_2024_games.csv"
)

print("Rows:", len(games))

dates = []

for index, row in games.iterrows():

    text = str(row["Date"])

    text = text.replace("(1)", "")
    text = text.replace("(2)", "")
    text = text.strip()

    text = text.split(",", 1)[1].strip()

    date = datetime.strptime(
        text + " " + str(int(row["Season"])),
        "%b %d %Y"
    )

    dates.append(date)

    if (index + 1) % 500 == 0:
        print(
            "Dates built:",
            len(dates)
        )

print("ALL DATES BUILT")

print("Assigning dates...")

games["Date"] = dates

print("ASSIGNMENT WORKS")

print(
    games["Date"].head()
)

print(
    games["Date"].tail()
)

print("TEST COMPLETE")