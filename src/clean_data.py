"""
MLB Sports Analytics Project
Convert team-level records into one record per game.
"""

from pathlib import Path
import pandas as pd


def main():
    input_file = Path("data/raw/mlb_2024_team_data.csv")
    output_file = Path("data/processed/mlb_2024_games.csv")

    print("Loading raw MLB data...")

    df = pd.read_csv(input_file)

    print(f"Raw rows: {len(df)}")

    # Keep completed games only.
    df = df[df["W/L"].notna()].copy()

    # Create a consistent game ID.
    # Sorting the two team names means:
    # NYY vs HOU and HOU vs NYY get the same ID.
    df["team_pair"] = df.apply(
        lambda row: "_".join(sorted([row["Tm"], row["Opp"]])),
        axis=1
    )

    df["game_key"] = (
        df["Season"].astype(str)
        + "_"
        + df["Date"].astype(str)
        + "_"
        + df["team_pair"]
    )

    # Count how many records belong to each game.
    game_counts = df["game_key"].value_counts()

    print("\nGame record counts:")
    print(game_counts.value_counts().sort_index())

    # Keep only games where we have both teams.
    valid_games = game_counts[game_counts == 2].index

    df = df[df["game_key"].isin(valid_games)].copy()

    print(f"\nRows after keeping complete games: {len(df)}")

    # Build one row per game.
    games = []

    for game_key, group in df.groupby("game_key"):

        if len(group) != 2:
            continue

        home_rows = group[group["Home_Away"] == "Home"]

        if len(home_rows) != 1:
            continue

        home = home_rows.iloc[0]
        away = group[group["Home_Away"] == "@"].iloc[0]

        # Determine the winner.
        if home["W/L"].startswith("W"):
            winner = home["Tm"]
        else:
            winner = away["Tm"]

        games.append({
            "Season": home["Season"],
            "Date": home["Date"],
            "Home": home["Tm"],
            "Away": away["Tm"],
            "HomeScore": home["R"],
            "AwayScore": away["R"],
            "Winner": winner
        })

    games_df = pd.DataFrame(games)

    # Save the clean game-level dataset.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    games_df.to_csv(output_file, index=False)

    print(f"\nUnique games: {len(games_df)}")
    print(f"Saved to: {output_file}")

    print("\nFirst 10 games:")
    print(games_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()