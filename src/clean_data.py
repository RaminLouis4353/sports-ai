"""
MLB Sports Analytics Project
Convert team-level records into one record per game.
Handles doubleheaders correctly.
"""

from pathlib import Path
import pandas as pd


def clean_date_for_key(value):
    """
    Clean the date while preserving doubleheader identifiers.

    Examples:
        Friday, Apr 12
        Sunday, Sep 22 (1)
        Sunday, Sep 22 (2)
    """

    return str(value).strip()


def main():

    input_file = Path("data/raw/mlb_multi_season_team_data.csv")
    output_file = Path("data/processed/mlb_multi_season_games.csv")

    print("========================================")
    print("MLB GAME DATA CLEANING")
    print("========================================")

    print("\nLoading raw MLB data...")

    df = pd.read_csv(input_file)

    print(f"Raw rows: {len(df)}")

    # ---------------------------------------------------------
    # Clean basic fields
    # ---------------------------------------------------------

    df = df.copy()

    df.loc[:, "Tm"] = df["Tm"].astype(str).str.strip()
    df.loc[:, "Opp"] = df["Opp"].astype(str).str.strip()
    df.loc[:, "Home_Away"] = df["Home_Away"].astype(str).str.strip()
    df.loc[:, "W/L"] = df["W/L"].astype(str).str.strip()

    # Keep completed games only.
    df = df[df["W/L"].notna()].copy()

    print(f"Rows after removing incomplete records: {len(df)}")

    # ---------------------------------------------------------
    # Clean dates
    # ---------------------------------------------------------

    df.loc[:, "Date"] = df["Date"].apply(clean_date_for_key)

    # ---------------------------------------------------------
    # Create consistent team pair
    # ---------------------------------------------------------

    df.loc[:, "team_pair"] = df.apply(
        lambda row: "_".join(
            sorted([row["Tm"], row["Opp"]])
        ),
        axis=1
    )

    # ---------------------------------------------------------
    # Create game key
    #
    # IMPORTANT:
    # The complete Date string is preserved.
    #
    # This means:
    #
    # Sep 22 (1)
    # Sep 22 (2)
    #
    # remain separate games.
    # ---------------------------------------------------------

    df.loc[:, "game_key"] = (
        df["Season"].astype(str)
        + "_"
        + df["Date"].astype(str)
        + "_"
        + df["team_pair"].astype(str)
    )

    # ---------------------------------------------------------
    # Check game record counts
    # ---------------------------------------------------------

    game_counts = df["game_key"].value_counts()

    print("\nGame record counts:")
    print(game_counts.value_counts().sort_index())

    # ---------------------------------------------------------
    # Keep games where exactly two team records exist
    # ---------------------------------------------------------

    valid_games = game_counts[
        game_counts == 2
    ].index

    df = df[
        df["game_key"].isin(valid_games)
    ].copy()

    print(
        f"\nRows after keeping complete games: {len(df)}"
    )

    # ---------------------------------------------------------
    # Build one row per game
    # ---------------------------------------------------------

    games = []

    for game_key, group in df.groupby("game_key"):

        if len(group) != 2:
            continue

        # Find home team
        home_rows = group[
            group["Home_Away"] == "Home"
        ]

        # Find away team
        away_rows = group[
            group["Home_Away"] == "@"
        ]

        if len(home_rows) != 1:
            continue

        if len(away_rows) != 1:
            continue

        home = home_rows.iloc[0]
        away = away_rows.iloc[0]

        # -----------------------------------------------------
        # Determine winner
        # -----------------------------------------------------

        home_result = str(home["W/L"]).strip()

        if home_result.startswith("W"):
            winner = home["Tm"]
        else:
            winner = away["Tm"]

        # -----------------------------------------------------
        # Create game record
        # -----------------------------------------------------

        games.append({
            "Season": home["Season"],
            "Date": home["Date"],
            "Home": home["Tm"],
            "Away": away["Tm"],
            "HomeScore": home["R"],
            "AwayScore": away["R"],
            "Winner": winner
        })

    # ---------------------------------------------------------
    # Create final dataframe
    # ---------------------------------------------------------

    games_df = pd.DataFrame(games)

    # Sort chronologically by season/date later during
    # feature engineering.
    games_df = games_df.reset_index(drop=True)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    games_df.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n========================================")
    print("DATA CLEANING COMPLETE")
    print("========================================")

    print(
        f"Unique games: {len(games_df)}"
    )

    print("\nGames by season:")

    print(
        games_df["Season"].value_counts().sort_index()
    )

    print(
        f"\nSaved to: {output_file}"
    )

    print("\nColumns:")

    for column in games_df.columns:
        print(f"- {column}")

    print("\nFirst 10 games:")

    print(
        games_df.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()