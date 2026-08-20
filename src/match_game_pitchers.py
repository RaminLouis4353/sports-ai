from pathlib import Path
import csv
import re


GAMES_FILE = Path(
    "data/processed/mlb_multi_season_games.csv"
)

PITCHERS_FILE = Path(
    "data/raw/mlb_game_pitchers.csv"
)

OUTPUT_FILE = Path(
    "data/processed/mlb_games_with_pitchers.csv"
)


# ============================================================
# TEAM ABBREVIATION NORMALIZATION
# ============================================================

TEAM_MAP = {
    "ARI": "ARI",
    "AZ": "ARI",

    "CHW": "CHW",
    "CWS": "CHW",

    "KCR": "KCR",
    "KC": "KCR",

    "SDP": "SDP",
    "SD": "SDP",

    "SFG": "SFG",
    "SF": "SFG",

    "TBR": "TBR",
    "TB": "TBR",

    "WSN": "WSN",
    "WSH": "WSN",
}


def clean_team(value):
    """
    Clean and normalize MLB team abbreviations.
    """

    if value is None:
        return ""

    value = str(value).strip().upper()

    if value in (
        "",
        "NONE",
        "NAN",
        "NULL"
    ):
        return ""

    return TEAM_MAP.get(
        value,
        value
    )


# ============================================================
# GAME NUMBER
# ============================================================

def get_game_number(date_text):
    """
    Extract doubleheader game number.

    Examples:

        Monday, May 30 (1) -> 1
        Monday, May 30 (2) -> 2
        Tuesday, May 3    -> 1
    """

    if date_text is None:
        return 1

    text = str(date_text).strip()

    match = re.search(
        r"\((1|2)\)",
        text
    )

    if match:
        return int(match.group(1))

    return 1


# ============================================================
# GAME DATE
# ============================================================

def game_date(date_text):
    """
    Convert:

        Friday, Apr 15

    or:

        Monday, May 30 (1)

    into:

        Apr 15
        May 30
    """

    if date_text is None:
        return ""

    date_text = str(
        date_text
    ).strip()

    # Remove weekday.
    if "," in date_text:
        date_text = date_text.split(
            ",",
            1
        )[1].strip()

    # Remove doubleheader marker.
    date_text = re.sub(
        r"\s*\([12]\)\s*$",
        "",
        date_text
    )

    return date_text.strip()


# ============================================================
# PITCHER DATE
# ============================================================

def pitcher_date(date_text):
    """
    Convert:

        2024-04-15

    into:

        Apr 15
    """

    if date_text is None:
        return ""

    date_text = str(
        date_text
    ).strip()

    parts = date_text.split("-")

    if len(parts) != 3:
        return ""

    month = parts[1]
    day = parts[2]

    months = {
        "01": "Jan",
        "02": "Feb",
        "03": "Mar",
        "04": "Apr",
        "05": "May",
        "06": "Jun",
        "07": "Jul",
        "08": "Aug",
        "09": "Sep",
        "10": "Oct",
        "11": "Nov",
        "12": "Dec",
    }

    month_name = months.get(
        month
    )

    if not month_name:
        return ""

    return (
        f"{month_name} "
        f"{int(day)}"
    )


# ============================================================
# GAME MATCHING KEY
# ============================================================

def make_game_key(row):
    season = str(
        row.get("Season", "")
    ).strip()

    date = game_date(
        row.get("Date", "")
    )

    home = clean_team(
        row.get("Home", "")
    )

    away = clean_team(
        row.get("Away", "")
    )

    if (
        not season
        or not date
        or not home
        or not away
    ):
        return ""

    return (
        f"{season}|"
        f"{date}|"
        f"{home}|"
        f"{away}"
    )


# ============================================================
# PITCHER MATCHING KEY
# ============================================================

def make_pitcher_key(row):
    season = str(
        row.get("Season", "")
    ).strip()

    date = pitcher_date(
        row.get("Date", "")
    )

    home = clean_team(
        row.get("Home", "")
    )

    away = clean_team(
        row.get("Away", "")
    )

    if (
        not season
        or not date
        or not home
        or not away
    ):
        return ""

    return (
        f"{season}|"
        f"{date}|"
        f"{home}|"
        f"{away}"
    )


# ============================================================
# NORMALIZE PITCHER ID
# ============================================================

def clean_id(value):
    if value is None:
        return ""

    value = str(
        value
    ).strip()

    if value in (
        "",
        "None",
        "none",
        "nan",
        "NaN",
        "NULL"
    ):
        return ""

    # Pandas-style floating ID:
    #
    # 425794.0 -> 425794
    #
    if value.endswith(".0"):
        value = value[:-2]

    return value


# ============================================================
# NORMALIZE PITCHER NAME
# ============================================================

def clean_name(value):
    if value is None:
        return ""

    value = str(
        value
    ).strip()

    if value.lower() in (
        "",
        "none",
        "nan",
        "null"
    ):
        return ""

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "MLB PITCHER DATA MATCHING"
    )

    print(
        "TEAM NORMALIZATION ENABLED"
    )

    print(
        "DOUBLEHEADER SUPPORT ENABLED"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # LOAD GAME DATA
    # --------------------------------------------------------

    print(
        "\nLoading game data..."
    )

    with open(
        GAMES_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        games = list(
            csv.DictReader(f)
        )

    print(
        f"Games loaded: {len(games)}"
    )

    # --------------------------------------------------------
    # LOAD PITCHER DATA
    # --------------------------------------------------------

    print(
        "\nLoading pitcher data..."
    )

    with open(
        PITCHERS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        pitchers = list(
            csv.DictReader(f)
        )

    print(
        f"Pitcher records loaded: "
        f"{len(pitchers)}"
    )

    # --------------------------------------------------------
    # CREATE PITCHER LOOKUP
    #
    # IMPORTANT:
    #
    # A key can occur more than once because of
    # doubleheaders.
    #
    # Therefore:
    #
    # key -> [pitcher1, pitcher2]
    #
    # NOT:
    #
    # key -> pitcher
    # --------------------------------------------------------

    print(
        "\nCreating pitcher lookup..."
    )

    pitcher_lookup = {}

    valid_pitcher_records = 0

    for row in pitchers:

        key = make_pitcher_key(
            row
        )

        if not key:
            continue

        valid_pitcher_records += 1

        if key not in pitcher_lookup:

            pitcher_lookup[key] = []

        pitcher_lookup[key].append(
            row
        )

    print(
        f"Valid pitcher records: "
        f"{valid_pitcher_records}"
    )

    print(
        f"Unique matchup/date keys: "
        f"{len(pitcher_lookup)}"
    )

    duplicate_keys = sum(
        1
        for records in pitcher_lookup.values()
        if len(records) > 1
    )

    duplicate_records = sum(
        len(records) - 1
        for records in pitcher_lookup.values()
        if len(records) > 1
    )

    print(
        f"Doubleheader keys: "
        f"{duplicate_keys}"
    )

    print(
        f"Additional duplicate records: "
        f"{duplicate_records}"
    )

    # --------------------------------------------------------
    # MATCH GAMES
    # --------------------------------------------------------

    print(
        "\nMatching pitchers to games..."
    )

    # Track which occurrence of a key we are currently using.
    #
    # Example:
    #
    # 2022|May 3|NYM|ATL
    #
    # Game 1 -> pitcher record 1
    # Game 2 -> pitcher record 2
    #
    pitcher_usage = {}

    home_matches = 0
    away_matches = 0
    both_matches = 0
    no_matches = 0

    doubleheader_games = 0
    doubleheader_matches = 0

    output_rows = []

    for row in games:

        key = make_game_key(
            row
        )

        game_number = get_game_number(
            row.get("Date", "")
        )

        row["MatchKey"] = key

        row["GameNumber"] = game_number

        # ----------------------------------------------------
        # FIND PITCHER RECORDS
        # ----------------------------------------------------

        records = pitcher_lookup.get(
            key,
            []
        )

        # ----------------------------------------------------
        # DETERMINE WHICH RECORD TO USE
        # ----------------------------------------------------

        pitcher = None

        if records:

            # Normal game:
            #
            # one pitcher record
            #
            # Doubleheader:
            #
            # two pitcher records
            #
            if len(records) == 1:

                pitcher = records[0]

            else:

                doubleheader_games += 1

                # Game 1 -> index 0
                # Game 2 -> index 1

                index = game_number - 1

                if (
                    index >= 0
                    and index < len(records)
                ):

                    pitcher = records[index]

                    doubleheader_matches += 1

                else:

                    # Fallback if unexpected
                    # game number occurs.

                    usage = pitcher_usage.get(
                        key,
                        0
                    )

                    if usage < len(records):

                        pitcher = records[usage]

                        pitcher_usage[key] = (
                            usage + 1
                        )

        # ----------------------------------------------------
        # NO PITCHER DATA
        # ----------------------------------------------------

        if pitcher is None:

            row[
                "HomeStartingPitcherID"
            ] = ""

            row[
                "HomeStartingPitcher"
            ] = ""

            row[
                "AwayStartingPitcherID"
            ] = ""

            row[
                "AwayStartingPitcher"
            ] = ""

            no_matches += 1

        # ----------------------------------------------------
        # PITCHER DATA FOUND
        # ----------------------------------------------------

        else:

            home_id = clean_id(
                pitcher.get(
                    "HomeStartingPitcherID",
                    ""
                )
            )

            away_id = clean_id(
                pitcher.get(
                    "AwayStartingPitcherID",
                    ""
                )
            )

            home_name = clean_name(
                pitcher.get(
                    "HomeStartingPitcher",
                    ""
                )
            )

            away_name = clean_name(
                pitcher.get(
                    "AwayStartingPitcher",
                    ""
                )
            )

            row[
                "HomeStartingPitcherID"
            ] = home_id

            row[
                "HomeStartingPitcher"
            ] = home_name

            row[
                "AwayStartingPitcherID"
            ] = away_id

            row[
                "AwayStartingPitcher"
            ] = away_name

            has_home = bool(
                home_id
            )

            has_away = bool(
                away_id
            )

            if has_home:

                home_matches += 1

            if has_away:

                away_matches += 1

            if (
                has_home
                and has_away
            ):

                both_matches += 1

            if (
                not has_home
                and not has_away
            ):

                no_matches += 1

        output_rows.append(
            row
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "MATCHING RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Total games: "
        f"{len(output_rows)}"
    )

    print(
        f"Home pitcher matched: "
        f"{home_matches}"
    )

    print(
        f"Away pitcher matched: "
        f"{away_matches}"
    )

    print(
        f"Both pitchers matched: "
        f"{both_matches}"
    )

    print(
        f"No pitchers matched: "
        f"{no_matches}"
    )

    print(
        f"Doubleheader games processed: "
        f"{doubleheader_games}"
    )

    print(
        f"Doubleheader pitcher matches: "
        f"{doubleheader_matches}"
    )

    if output_rows:

        rate = (
            both_matches
            / len(output_rows)
            * 100
        )

    else:

        rate = 0

    print(
        f"Both-pitcher match rate: "
        f"{rate:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print(
        "\nSaving output..."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        output_rows[0].keys()
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # SAMPLE MATCHED GAMES
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "SAMPLE MATCHED GAMES"
    )

    print(
        "========================================"
    )

    count = 0

    for row in output_rows:

        if (
            row[
                "HomeStartingPitcherID"
            ]
            and
            row[
                "AwayStartingPitcherID"
            ]
        ):

            print(
                f"{row['Season']} | "
                f"{row['Date']} | "
                f"Game {row['GameNumber']} | "
                f"{row['Away']} @ "
                f"{row['Home']} | "
                f"{row['AwayStartingPitcher']} "
                f"vs "
                f"{row['HomeStartingPitcher']}"
            )

            count += 1

            if count >= 10:
                break

    if count == 0:

        print(
            "No games matched."
        )

    # --------------------------------------------------------
    # DOUBLEHEADER SAMPLE
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "DOUBLEHEADER SAMPLE"
    )

    print(
        "========================================"
    )

    shown = 0

    for row in output_rows:

        if (
            row["GameNumber"] in (1, 2)
            and "(" in row["Date"]
            and row[
                "HomeStartingPitcherID"
            ]
            and row[
                "AwayStartingPitcherID"
            ]
        ):

            print(
                f"{row['Season']} | "
                f"{row['Date']} | "
                f"Game {row['GameNumber']} | "
                f"{row['Away']} @ "
                f"{row['Home']} | "
                f"{row['AwayStartingPitcher']} "
                f"vs "
                f"{row['HomeStartingPitcher']}"
            )

            shown += 1

            if shown >= 10:
                break

    if shown == 0:

        print(
            "No doubleheader samples found."
        )

    print(
        "\n========================================"
    )

    print(
        "PITCHER MATCHING COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()