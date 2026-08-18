import pandas as pd
from datetime import datetime

games = pd.read_csv(
    "data/processed/mlb_2024_games.csv"
)

for index, row in games.iterrows():

    text = str(row["Date"])

    text = text.replace("(1)", "")
    text = text.replace("(2)", "")
    text = text.strip()

    if "," in text:
        text = text.split(",", 1)[1].strip()

    try:
        date = datetime.strptime(
            text + " " + str(int(row["Season"])),
            "%b %d %Y"
        )

    except Exception as e:
        print("")
        print("================================")
        print("BAD DATE FOUND")
        print("================================")
        print("Row:", index)
        print("Original:", repr(row["Date"]))
        print("Cleaned:", repr(text))
        print("Season:", row["Season"])
        print("Error:", e)
        break

else:
    print("ALL DATES ARE VALID")