from pathlib import Path
from datetime import datetime

import csv


INPUT_FILE = Path(
    "data/processed/mlb_2024_games.csv"
)

OUTPUT_FILE = Path(
    "data/processed/mlb_2024_features.csv"
)


def convert_date(date_text, season):

    text = str(date_text)

    text = text.replace("(1)", "")
    text = text.replace("(2)", "")
    text = text.strip()

    if "," in text:
        text = text.split(",", 1)[1].strip()

    return datetime.strptime(
        text + " " + str(int(float(season))),
        "%b %d %Y"
    )


def update_team_stats(
    stats,
    team,
    runs_scored,
    runs_allowed,
    won
):

    if team not in stats:

        stats[team] = {
            "games": 0,
            "wins": 0,
            "runs_scored": 0.0,
            "runs_allowed": 0.0,
            "recent_results": []
        }

    stats[team]["games"] += 1
    stats[team]["wins"] += int(won)

    stats[team]["runs_scored"] += float(
        runs_scored
    )

    stats[team]["runs_allowed"] += float(
        runs_allowed
    )

    stats[team]["recent_results"].append(
        int(won)
    )

    stats[team]["recent_results"] = (
        stats[team]["recent_results"][-5:]
    )


def get_team_features(stats, team):

    if team not in stats:

        return (
            0.5,
            0.0,
            0.0,
            0.5
        )

    s = stats[team]

    if s["games"] == 0:

        return (
            0.5,
            0.0,
            0.0,
            0.5
        )

    win_pct = (
        s["wins"] / s["games"]
    )

    avg_scored = (
        s["runs_scored"] / s["games"]
    )

    avg_allowed = (
        s["runs_allowed"] / s["games"]
    )

    recent_pct = (
        sum(s["recent_results"])
        / len(s["recent_results"])
    )

    return (
        win_pct,
        avg_scored,
        avg_allowed,
        recent_pct
    )


def main():

    print(
        "Loading game data...",
        flush=True
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        games = list(reader)

    print(
        f"Games loaded: {len(games)}",
        flush=True
    )

    # -----------------------------------------
    # CONVERT DATES
    # -----------------------------------------

    print(
        "Converting dates...",
        flush=True
    )

    for index, game in enumerate(games):

        game["_GameDate"] = convert_date(
            game["Date"],
            game["Season"]
        )

        if (index + 1) % 500 == 0:

            print(
                f"Converted {index + 1} / "
                f"{len(games)} dates",
                flush=True
            )

    print(
        "All dates converted.",
        flush=True
    )

    # -----------------------------------------
    # SORT GAMES
    # -----------------------------------------

    print(
        "Sorting games...",
        flush=True
    )

    games.sort(
        key=lambda x: x["_GameDate"]
    )

    print(
        "Games sorted.",
        flush=True
    )

    print(
        "Date range:",
        games[0]["_GameDate"],
        "to",
        games[-1]["_GameDate"],
        flush=True
    )

    # -----------------------------------------
    # BUILD FEATURES
    # -----------------------------------------

    print(
        "Building pre-game features...",
        flush=True
    )

    team_stats = {}

    feature_rows = []

    total = len(games)

    for index, game in enumerate(games):

        home = game["Home"]
        away = game["Away"]

        (
            home_win_pct,
            home_avg_scored,
            home_avg_allowed,
            home_recent_pct
        ) = get_team_features(
            team_stats,
            home
        )

        (
            away_win_pct,
            away_avg_scored,
            away_avg_allowed,
            away_recent_pct
        ) = get_team_features(
            team_stats,
            away
        )

        home_won = (
            home == game["Winner"]
        )

        feature_rows.append({

            "Date": game["_GameDate"].strftime(
                "%Y-%m-%d"
            ),

            "Home": home,

            "Away": away,

            "Home_WinPct": home_win_pct,

            "Home_AvgRunsScored":
                home_avg_scored,

            "Home_AvgRunsAllowed":
                home_avg_allowed,

            "Home_RecentWinPct":
                home_recent_pct,

            "Away_WinPct":
                away_win_pct,

            "Away_AvgRunsScored":
                away_avg_scored,

            "Away_AvgRunsAllowed":
                away_avg_allowed,

            "Away_RecentWinPct":
                away_recent_pct,

            "HomeWon":
                int(home_won)
        })

        # Update statistics AFTER the game
        # has been recorded.

        update_team_stats(
            team_stats,
            home,
            game["HomeScore"],
            game["AwayScore"],
            home_won
        )

        update_team_stats(
            team_stats,
            away,
            game["AwayScore"],
            game["HomeScore"],
            not home_won
        )

        if (index + 1) % 500 == 0:

            print(
                f"Processed {index + 1} / "
                f"{total} games",
                flush=True
            )

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    print(
        "Saving feature dataset...",
        flush=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "Date",
        "Home",
        "Away",
        "Home_WinPct",
        "Home_AvgRunsScored",
        "Home_AvgRunsAllowed",
        "Home_RecentWinPct",
        "Away_WinPct",
        "Away_AvgRunsScored",
        "Away_AvgRunsAllowed",
        "Away_RecentWinPct",
        "HomeWon"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            feature_rows
        )

    print("")
    print(
        "================================"
    )
    print(
        "FEATURE ENGINEERING COMPLETE"
    )
    print(
        "================================"
    )

    print(
        f"Games: {len(feature_rows)}"
    )

    print(
        f"Features: {len(fieldnames)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("")
    print(
        "First 5 games:"
    )

    for row in feature_rows[:5]:

        print(row)


if __name__ == "__main__":

    main()