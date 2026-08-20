"""
Collect MLB game-level starting pitcher information.

Uses MLB Stats API.

Creates:
    data/raw/mlb_2022_game_pitchers.csv
    data/raw/mlb_2023_game_pitchers.csv
    data/raw/mlb_2024_game_pitchers.csv
    data/raw/mlb_game_pitchers.csv
"""

from pathlib import Path
import time
import requests
import pandas as pd


OUTPUT_DIR = Path("data/raw")

SEASONS = [2022, 2023, 2024]

API_URL = "https://statsapi.mlb.com/api/v1/schedule"


def get_team_abbreviation(team):
    """Return the MLB team abbreviation."""
    if not team:
        return None

    return team.get("abbreviation")


def get_pitcher(team_data):
    """Return probable/starting pitcher information."""
    pitcher = team_data.get("probablePitcher")

    if not pitcher:
        return None, None

    return (
        pitcher.get("id"),
        pitcher.get("fullName")
    )


def collect_season(season):

    print("\n========================================")
    print(f"COLLECTING {season}")
    print("========================================")

    start_date = f"{season}-03-01"
    end_date = f"{season}-11-30"

    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "probablePitcher,team"
    }

    print("Requesting MLB data...")

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    print("HTTP status:", response.status_code)

    response.raise_for_status()

    data = response.json()

    rows = []

    for date_entry in data.get("dates", []):

        games = date_entry.get("games", [])

        for game in games:

            # Only keep games that actually have a completed result.
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue

            teams = game.get("teams", {})

            away_data = teams.get("away", {})
            home_data = teams.get("home", {})

            away_team = get_team_abbreviation(
                away_data.get("team")
            )

            home_team = get_team_abbreviation(
                home_data.get("team")
            )

            # Ignore games where team information is unavailable.
            if not home_team or not away_team:
                continue

            home_pitcher_id, home_pitcher_name = get_pitcher(
                home_data
            )

            away_pitcher_id, away_pitcher_name = get_pitcher(
                away_data
            )

            rows.append({
                "Season": season,
                "Date": date_entry.get("date"),
                "GamePk": game.get("gamePk"),

                "Home": home_team,
                "Away": away_team,

                "HomeStartingPitcherID": home_pitcher_id,
                "HomeStartingPitcher": home_pitcher_name,

                "AwayStartingPitcherID": away_pitcher_id,
                "AwayStartingPitcher": away_pitcher_name
            })

    df = pd.DataFrame(rows)

    output_file = OUTPUT_DIR / f"mlb_{season}_game_pitchers.csv"

    df.to_csv(
        output_file,
        index=False
    )

    both_pitchers = (
        df["HomeStartingPitcher"].notna()
        & df["AwayStartingPitcher"].notna()
    ).sum()

    print()
    print("Completed games found:", len(df))
    print(
        "Games with both pitchers:",
        both_pitchers,
        "/",
        len(df)
    )

    print("Saved:", output_file)

    return df


def main():

    print("========================================")
    print("MLB GAME-LEVEL STARTING PITCHERS")
    print("========================================")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_data = []

    for season in SEASONS:

        try:
            df = collect_season(season)

            all_data.append(df)

        except Exception as error:

            print(
                f"ERROR collecting {season}:",
                error
            )

        time.sleep(1)

    if not all_data:
        print("\nERROR: No pitcher data collected.")
        return

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["Season", "GamePk"]
    )

    combined = combined.sort_values(
        ["Season", "Date", "GamePk"]
    ).reset_index(drop=True)

    combined_file = OUTPUT_DIR / "mlb_game_pitchers.csv"

    combined.to_csv(
        combined_file,
        index=False
    )

    both_pitchers = (
        combined["HomeStartingPitcher"].notna()
        & combined["AwayStartingPitcher"].notna()
    ).sum()

    print()
    print("========================================")
    print("PITCHER COLLECTION COMPLETE")
    print("========================================")

    print("Total games:", len(combined))

    print(
        "Games with both starting pitchers:",
        both_pitchers,
        "/",
        len(combined)
    )

    print("Saved to:", combined_file)

    print()
    print("Games by season:")

    print(
        combined.groupby("Season").size()
    )

    print()
    print("First 10 rows:")

    print(
        combined[
            [
                "Season",
                "Date",
                "GamePk",
                "Home",
                "Away",
                "HomeStartingPitcher",
                "AwayStartingPitcher"
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()