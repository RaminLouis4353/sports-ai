"""
MLB Sports Analytics Project
Download historical game data for all MLB teams.
"""

from pathlib import Path
import pandas as pd

from pybaseball import schedule_and_record

from teams import MLB_TEAMS


def download_team_data(season, team):
    """Download one team's schedule for a season."""
    print(f"Downloading {team} {season}...")

    try:
        games = schedule_and_record(season, team)

        if games is None or games.empty:
            print(f"  No data returned for {team}")
            return None

        games = games.copy()
        games["Season"] = season
        games["Team"] = team

        return games

    except Exception as error:
        print(f"  ERROR for {team}: {error}")
        return None


def main():
    season = 2024

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_games = []

    for team in MLB_TEAMS:
        games = download_team_data(season, team)

        if games is not None:
            all_games.append(games)

    if not all_games:
        print("No data was downloaded.")
        return

    combined = pd.concat(all_games, ignore_index=True)

    output_file = output_dir / f"mlb_{season}_team_data.csv"
    combined.to_csv(output_file, index=False)

    print("\n===== DOWNLOAD COMPLETE =====")
    print(f"Teams downloaded: {len(all_games)}")
    print(f"Rows collected: {len(combined)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()