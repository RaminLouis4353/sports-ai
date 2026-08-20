"""
Collect MLB starting-pitcher game logs.

Uses MLB Stats API.

The resulting data contains game-by-game pitching statistics
for pitchers appearing in our historical starting-pitcher dataset.
"""

from pathlib import Path
import time
import requests
import pandas as pd


INPUT_FILE = Path(
    "data/processed/mlb_games_with_pitchers.csv"
)

OUTPUT_FILE = Path(
    "data/raw/mlb_pitcher_game_logs.csv"
)


API_URL = (
    "https://statsapi.mlb.com/api/v1/people/"
    "{pitcher_id}/stats"
)


def get_pitcher_game_logs(pitcher_id, season):

    url = API_URL.format(
        pitcher_id=pitcher_id
    )

    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": season
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            print(
                f"    HTTP {response.status_code}"
            )
            return []

        data = response.json()

        stats = data.get(
            "stats",
            []
        )

        if not stats:
            return []

        return stats[0].get(
            "splits",
            []
        )

    except Exception as error:

        print(
            f"    ERROR: {error}"
        )

        return []


def main():

    print(
        "========================================"
    )
    print(
        "MLB PITCHER GAME LOG COLLECTION"
    )
    print(
        "========================================"
    )

    print(
        "\nLoading pitcher data..."
    )

    pitchers = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Games loaded: {len(pitchers)}"
    )

    # --------------------------------------------------
    # GET UNIQUE PITCHERS
    # --------------------------------------------------

    ids = pd.concat(
        [
            pitchers[
                "HomeStartingPitcherID"
            ],
            pitchers[
                "AwayStartingPitcherID"
            ]
        ]
    )

    ids = pd.to_numeric(
        ids,
        errors="coerce"
    )

    ids = sorted(
        ids.dropna().unique()
    )

    print(
        f"Unique pitchers: {len(ids)}"
    )

    # --------------------------------------------------
    # COLLECT GAME LOGS
    # --------------------------------------------------

    all_rows = []

    total = len(ids)

    for index, pitcher_id in enumerate(ids, start=1):

        print(
            f"\n[{index}/{total}] "
            f"Pitcher ID: {int(pitcher_id)}"
        )

        for season in [2022, 2023, 2024]:

            print(
                f"  Collecting {season}..."
            )

            splits = get_pitcher_game_logs(
                int(pitcher_id),
                season
            )

            print(
                f"    Games: {len(splits)}"
            )

            for split in splits:

                stat = split.get(
                    "stat",
                    {}
                )

                all_rows.append({

                    "PitcherID": int(
                        pitcher_id
                    ),

                    "PitcherName": (
                        split
                        .get("player", {})
                        .get("fullName")
                    ),

                    "Season": season,

                    "Date": split.get(
                        "date"
                    ),

                    "GamePk": (
                        split
                        .get("game", {})
                        .get("gamePk")
                    ),

                    "Team": (
                        split
                        .get("team", {})
                        .get("name")
                    ),

                    "Opponent": (
                        split
                        .get("opponent", {})
                        .get("name")
                    ),

                    "IsHome": split.get(
                        "isHome"
                    ),

                    "IsWin": split.get(
                        "isWin"
                    ),

                    "GamesStarted": stat.get(
                        "gamesStarted",
                        0
                    ),

                    "InningsPitched": stat.get(
                        "inningsPitched",
                        "0"
                    ),

                    "EarnedRuns": stat.get(
                        "earnedRuns",
                        0
                    ),

                    "Hits": stat.get(
                        "hits",
                        0
                    ),

                    "Walks": stat.get(
                        "baseOnBalls",
                        0
                    ),

                    "Strikeouts": stat.get(
                        "strikeOuts",
                        0
                    ),

                    "HomeRuns": stat.get(
                        "homeRuns",
                        0
                    ),

                    "Runs": stat.get(
                        "runs",
                        0
                    ),

                    "BattersFaced": stat.get(
                        "battersFaced",
                        0
                    ),

                    "Pitches": stat.get(
                        "numberOfPitches",
                        0
                    )
                })

            # Small delay so we don't hammer the API.
            time.sleep(0.15)

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    output = pd.DataFrame(
        all_rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n========================================"
    )
    print(
        "COLLECTION COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        f"Rows collected: {len(output)}"
    )

    print(
        f"Pitchers: {output['PitcherID'].nunique()}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    if len(output) > 0:

        print(
            "\nFirst 10 rows:"
        )

        print(
            output.head(10).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()