from pathlib import Path
import csv


GAMES_FILE = Path("data/processed/mlb_multi_season_games.csv")
PITCHERS_FILE = Path("data/raw/mlb_game_pitchers.csv")
OUTPUT_FILE = Path("data/processed/mlb_games_with_pitchers.csv")


def clean_team(value):
    value = str(value).strip().upper()

    if value in ("", "NONE", "NAN", "NULL"):
        return ""

    return value


def game_date(date_text):
    """
    Convert:
        Friday, Apr 15

    Into:
        Apr 15
    """

    date_text = str(date_text).strip()

    if "," in date_text:
        date_text = date_text.split(",", 1)[1].strip()

    date_text = date_text.replace("(1)", "")
    date_text = date_text.replace("(2)", "")
    date_text = date_text.strip()

    return date_text


def pitcher_date(date_text):
    """
    Convert:
        2024-04-15

    Into:
        Apr 15
    """

    date_text = str(date_text).strip()

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

    month_name = months.get(month)

    if not month_name:
        return ""

    return f"{month_name} {int(day)}"


def make_game_key(row):
    season = str(row.get("Season", "")).strip()
    date = game_date(row.get("Date", ""))
    home = clean_team(row.get("Home", ""))
    away = clean_team(row.get("Away", ""))

    if not season or not date or not home or not away:
        return ""

    return f"{season}|{date}|{home}|{away}"


def make_pitcher_key(row):
    season = str(row.get("Season", "")).strip()
    date = pitcher_date(row.get("Date", ""))
    home = clean_team(row.get("Home", ""))
    away = clean_team(row.get("Away", ""))

    if not season or not date or not home or not away:
        return ""

    return f"{season}|{date}|{home}|{away}"


def main():

    print("========================================")
    print("MLB PITCHER DATA MATCHING")
    print("========================================")

    # --------------------------------------------------
    # LOAD GAME DATA
    # --------------------------------------------------

    print("\nLoading game data...")

    with open(
        GAMES_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        games = list(csv.DictReader(f))

    print(f"Games loaded: {len(games)}")

    # --------------------------------------------------
    # LOAD PITCHER DATA
    # --------------------------------------------------

    print("\nLoading pitcher data...")

    with open(
        PITCHERS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        pitchers = list(csv.DictReader(f))

    print(f"Pitcher records loaded: {len(pitchers)}")

    # --------------------------------------------------
    # GAME KEYS
    # --------------------------------------------------

    print("\nCreating game matching keys...")

    valid_games = 0

    for row in games:

        key = make_game_key(row)

        row["MatchKey"] = key

        if key:
            valid_games += 1

    print(f"Valid game keys: {valid_games}")

    # Show examples so we know the format
    print("\nExample game keys:")

    shown = 0

    for row in games:

        if row["MatchKey"]:

            print(
                f"  {row['MatchKey']}"
            )

            shown += 1

            if shown >= 5:
                break

    # --------------------------------------------------
    # PITCHER KEYS
    # --------------------------------------------------

    print("\nCreating pitcher matching keys...")

    pitcher_lookup = {}

    valid_pitchers = 0

    duplicates = 0

    for i, row in enumerate(pitchers):

        key = make_pitcher_key(row)

        if not key:
            continue

        valid_pitchers += 1

        if key in pitcher_lookup:
            duplicates += 1
            continue

        pitcher_lookup[key] = row

        if (i + 1) % 1000 == 0:
            print(
                f"  Processed {i + 1}/{len(pitchers)}"
            )

    print(
        f"Valid pitcher keys: {valid_pitchers}"
    )

    print(
        f"Unique pitcher game keys: "
        f"{len(pitcher_lookup)}"
    )

    print(
        f"Duplicate pitcher keys ignored: "
        f"{duplicates}"
    )

    # Show examples
    print("\nExample pitcher keys:")

    shown = 0

    for key in pitcher_lookup:

        print(f"  {key}")

        shown += 1

        if shown >= 5:
            break

    # --------------------------------------------------
    # MATCH
    # --------------------------------------------------

    print("\nMatching pitchers to games...")

    home_matches = 0
    away_matches = 0
    both_matches = 0
    no_matches = 0

    output_rows = []

    for row in games:

        key = row["MatchKey"]

        pitcher = pitcher_lookup.get(key)

        if pitcher is None:

            row["HomeStartingPitcherID"] = ""
            row["HomeStartingPitcher"] = ""
            row["AwayStartingPitcherID"] = ""
            row["AwayStartingPitcher"] = ""

            no_matches += 1

        else:

            home_pitcher = str(
                pitcher.get(
                    "HomeStartingPitcher",
                    ""
                )
            )

            away_pitcher = str(
                pitcher.get(
                    "AwayStartingPitcher",
                    ""
                )
            )

            home_id = str(
                pitcher.get(
                    "HomeStartingPitcherID",
                    ""
                )
            )

            away_id = str(
                pitcher.get(
                    "AwayStartingPitcherID",
                    ""
                )
            )

            if home_pitcher in ("None", "nan"):
                home_pitcher = ""

            if away_pitcher in ("None", "nan"):
                away_pitcher = ""

            if home_id in ("None", "nan"):
                home_id = ""

            if away_id in ("None", "nan"):
                away_id = ""

            row["HomeStartingPitcherID"] = home_id
            row["HomeStartingPitcher"] = home_pitcher
            row["AwayStartingPitcherID"] = away_id
            row["AwayStartingPitcher"] = away_pitcher

            has_home = bool(home_pitcher)
            has_away = bool(away_pitcher)

            if has_home:
                home_matches += 1

            if has_away:
                away_matches += 1

            if has_home and has_away:
                both_matches += 1

            if not has_home and not has_away:
                no_matches += 1

        output_rows.append(row)

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n========================================")
    print("MATCHING RESULTS")
    print("========================================")

    print(
        f"Total games: {len(output_rows)}"
    )

    print(
        f"Home pitcher matched: {home_matches}"
    )

    print(
        f"Away pitcher matched: {away_matches}"
    )

    print(
        f"Both pitchers matched: {both_matches}"
    )

    print(
        f"No pitchers matched: {no_matches}"
    )

    if len(output_rows) > 0:
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

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    print("\nSaving output...")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(output_rows[0].keys())

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
        writer.writerows(output_rows)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    # --------------------------------------------------
    # SAMPLE MATCHES
    # --------------------------------------------------

    print("\n========================================")
    print("SAMPLE MATCHED GAMES")
    print("========================================")

    count = 0

    for row in output_rows:

        if (
            row["HomeStartingPitcher"]
            and
            row["AwayStartingPitcher"]
        ):

            print(
                f"{row['Season']} | "
                f"{row['Date']} | "
                f"{row['Away']} @ {row['Home']} | "
                f"{row['AwayStartingPitcher']} "
                f"vs "
                f"{row['HomeStartingPitcher']}"
            )

            count += 1

            if count >= 10:
                break

    if count == 0:
        print("No games matched.")

    print("\n========================================")
    print("PITCHER MATCHING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()