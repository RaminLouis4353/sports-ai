"""
MLB Sports Analytics Project
Download historical game data for all MLB teams and multiple seasons.
"""

from pathlib import Path
import pandas as pd

from pybaseball import schedule_and_record

from teams import MLB_TEAMS


# Seasons we want to collect
SEASONS = [2022, 2023, 2024]


def download_team_data(season, team):
    """Download one team's schedule for a season."""

    print(f"Downloading {team} {season}...")

    try:
        games = schedule_and_record(season, team)

        if games is None or games.empty:
            print(f"  ERROR: No data returned for {team} {season}")
            return None

        games = games.copy()

        # Add identifiers
        games.loc[:, "Season"] = season
        games.loc[:, "Team"] = team

        # Keep completed games only.
        games = games[games["W/L"].notna()].copy()

        # Count the team's completed games.
        game_count = len(games)

        if game_count < 150:
            print(
                f"  WARNING: {team} {season} only has "
                f"{game_count} completed games"
            )
        else:
            print(
                f"  OK: {team} {season} -> "
                f"{game_count} completed games"
            )

        return games

    except Exception as error:
        print(f"  ERROR for {team} {season}: {error}")
        return None


def main():

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_games = []

    print("========================================")
    print("MLB HISTORICAL DATA COLLECTION")
    print("========================================")
    print(f"Seasons: {SEASONS}")
    print(f"Teams: {len(MLB_TEAMS)}")
    print()

    for season in SEASONS:

        print(f"\n========== SEASON {season} ==========")

        season_games = []

        for team in MLB_TEAMS:

            games = download_team_data(season, team)

            if games is not None:
                season_games.append(games)
                all_games.append(games)

        if not season_games:
            print(f"\nERROR: No data collected for {season}")
            continue

        # Combine this season.
        season_combined = pd.concat(
            season_games,
            ignore_index=True
        )

        # Save individual season file.
        season_file = (
            output_dir / f"mlb_{season}_team_data.csv"
        )

        season_combined.to_csv(
            season_file,
            index=False
        )

        print(f"\nSeason {season} complete:")
        print(
            f"  Teams downloaded: "
            f"{len(season_games)} / {len(MLB_TEAMS)}"
        )
        print(
            f"  Rows collected: "
            f"{len(season_combined)}"
        )
        print(f"  Saved to: {season_file}")

        # Show team counts for verification.
        print("\n  Team game counts:")

        team_counts = (
            season_combined["Team"]
            .value_counts()
            .sort_index()
        )

        print(team_counts.to_string())

    if not all_games:

        print("\nERROR: No data was downloaded.")
        return

    # Combine all seasons.
    combined = pd.concat(
        all_games,
        ignore_index=True
    )

    combined_file = (
        output_dir / "mlb_multi_season_team_data.csv"
    )

    combined.to_csv(
        combined_file,
        index=False
    )

    print("\n========================================")
    print("ALL SEASONS DOWNLOAD COMPLETE")
    print("========================================")

    print(f"Seasons: {SEASONS}")
    print(f"Total rows: {len(combined)}")
    print(f"Saved to: {combined_file}")

    # Final verification.
    print("\n===== FINAL SEASON ROW COUNTS =====")

    season_counts = (
        combined.groupby("Season")
        .size()
    )

    print(season_counts.to_string())

    print("\n===== FINAL TEAM COUNTS =====")

    team_counts = (
        combined.groupby(["Season", "Team"])
        .size()
    )

    print(team_counts.to_string())

    print("\n========================================")
    print("COLLECTION FINISHED")
    print("========================================")


if __name__ == "__main__":
    main()