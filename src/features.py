"""
MLB Sports Analytics Project

Build pre-game features from multiple MLB seasons.

Features include:
- Overall team performance
- Recent performance
- Run differential
- Home/road performance splits
- Matchup difference features
- Starting pitcher historical performance

IMPORTANT:
All features are calculated using information that was available
BEFORE the current game.

Doubleheaders:
- Multiple games between the same teams on the same date are preserved.
- GameNumber distinguishes Game 1, Game 2, etc.
- Team statistics are updated after each individual game.

Pitcher statistics:
- Only previous games for that pitcher are used.
- The current game's pitching performance is NEVER included.

Input:
    data/processed/mlb_games_with_pitchers.csv
    data/raw/mlb_pitcher_game_logs.csv

Output:
    data/processed/mlb_multi_season_features.csv
"""

from pathlib import Path
from datetime import datetime
import csv


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/mlb_games_with_pitchers.csv"
)

PITCHER_LOG_FILE = Path(
    "data/raw/mlb_pitcher_game_logs.csv"
)

OUTPUT_FILE = Path(
    "data/processed/mlb_multi_season_features.csv"
)


# ---------------------------------------------------------
# DATE CONVERSION
# ---------------------------------------------------------

def convert_date(date_text, season):

    text = str(date_text)

    # Remove doubleheader indicator.
    # Example:
    # Saturday, Apr 23 (1)
    # Saturday, Apr 23 (2)

    text = text.replace("(1)", "")
    text = text.replace("(2)", "")
    text = text.strip()

    if "," in text:
        text = text.split(",", 1)[1].strip()

    return datetime.strptime(
        text + " " + str(int(float(season))),
        "%b %d %Y"
    )


# ---------------------------------------------------------
# GAME NUMBER
# ---------------------------------------------------------

def get_game_number(date_text):

    """
    Extract the doubleheader game number.

    Examples:

        Saturday, Apr 23 (1) -> 1
        Saturday, Apr 23 (2) -> 2
        Sunday, Apr 24       -> 1

    Normal single games are treated as Game 1.
    """

    text = str(date_text).strip()

    if "(2)" in text:
        return 2

    if "(1)" in text:
        return 1

    return 1


# ---------------------------------------------------------
# PITCHER INNINGS CONVERSION
# ---------------------------------------------------------

def innings_to_float(value):

    """
    MLB stores innings such as:

        5.0 = 5 innings
        5.1 = 5 innings + 1 out
        5.2 = 5 innings + 2 outs

    This converts them into real decimal innings:

        5.0 -> 5.000
        5.1 -> 5.333
        5.2 -> 5.667
    """

    try:

        value = float(value)

        whole = int(value)

        remainder = round(
            (value - whole) * 10
        )

        if remainder == 1:

            return whole + (1.0 / 3.0)

        if remainder == 2:

            return whole + (2.0 / 3.0)

        return float(whole)

    except Exception:

        return 0.0


# ---------------------------------------------------------
# SAFE FLOAT
# ---------------------------------------------------------

def safe_float(value):

    try:

        return float(value)

    except Exception:

        return 0.0


# ---------------------------------------------------------
# TEAM STATISTICS
# ---------------------------------------------------------

def create_team_stats():

    return {

        "games": 0,
        "wins": 0,

        "runs_scored": 0.0,
        "runs_allowed": 0.0,

        "recent_results": [],

        # Home-only statistics

        "home_games": 0,
        "home_wins": 0,

        "home_runs_scored": 0.0,
        "home_runs_allowed": 0.0,

        "home_recent_results": [],

        # Road-only statistics

        "road_games": 0,
        "road_wins": 0,

        "road_runs_scored": 0.0,
        "road_runs_allowed": 0.0,

        "road_recent_results": []
    }


# ---------------------------------------------------------
# UPDATE TEAM STATISTICS
# ---------------------------------------------------------

def update_team_stats(
    stats,
    team,
    runs_scored,
    runs_allowed,
    won,
    location
):

    if team not in stats:

        stats[team] = create_team_stats()

    s = stats[team]

    runs_scored = float(runs_scored)
    runs_allowed = float(runs_allowed)
    won = int(won)

    # -----------------------------------------------------
    # OVERALL
    # -----------------------------------------------------

    s["games"] += 1

    s["wins"] += won

    s["runs_scored"] += runs_scored

    s["runs_allowed"] += runs_allowed

    s["recent_results"].append(won)

    s["recent_results"] = (
        s["recent_results"][-5:]
    )

    # -----------------------------------------------------
    # HOME
    # -----------------------------------------------------

    if location == "home":

        s["home_games"] += 1

        s["home_wins"] += won

        s["home_runs_scored"] += runs_scored

        s["home_runs_allowed"] += runs_allowed

        s["home_recent_results"].append(won)

        s["home_recent_results"] = (
            s["home_recent_results"][-5:]
        )

    # -----------------------------------------------------
    # ROAD
    # -----------------------------------------------------

    elif location == "road":

        s["road_games"] += 1

        s["road_wins"] += won

        s["road_runs_scored"] += runs_scored

        s["road_runs_allowed"] += runs_allowed

        s["road_recent_results"].append(won)

        s["road_recent_results"] = (
            s["road_recent_results"][-5:]
        )


# ---------------------------------------------------------
# GET OVERALL TEAM FEATURES
# ---------------------------------------------------------

def get_team_features(stats, team):

    if team not in stats:

        return (
            0.5,
            0.0,
            0.0,
            0.5,
            0.0,
            0.0
        )

    s = stats[team]

    if s["games"] == 0:

        return (
            0.5,
            0.0,
            0.0,
            0.5,
            0.0,
            0.0
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

    if s["recent_results"]:

        recent_pct = (
            sum(s["recent_results"])
            / len(s["recent_results"])
        )

    else:

        recent_pct = 0.5

    run_differential = (
        avg_scored - avg_allowed
    )

    recent_games = len(
        s["recent_results"]
    )

    if recent_games > 0:

        recent_run_differential = (
            sum(
                [
                    1 if result == 1 else -1
                    for result in s["recent_results"]
                ]
            )
            / recent_games
        )

    else:

        recent_run_differential = 0.0

    return (
        win_pct,
        avg_scored,
        avg_allowed,
        recent_pct,
        run_differential,
        recent_run_differential
    )


# ---------------------------------------------------------
# GET HOME FEATURES
# ---------------------------------------------------------

def get_home_features(stats, team):

    if team not in stats:

        return (
            0.5,
            0.0,
            0.0,
            0.5
        )

    s = stats[team]

    if s["home_games"] == 0:

        return (
            0.5,
            0.0,
            0.0,
            0.5
        )

    win_pct = (
        s["home_wins"]
        / s["home_games"]
    )

    avg_scored = (
        s["home_runs_scored"]
        / s["home_games"]
    )

    avg_allowed = (
        s["home_runs_allowed"]
        / s["home_games"]
    )

    if s["home_recent_results"]:

        recent_pct = (
            sum(s["home_recent_results"])
            / len(s["home_recent_results"])
        )

    else:

        recent_pct = 0.5

    return (
        win_pct,
        avg_scored,
        avg_allowed,
        recent_pct
    )


# ---------------------------------------------------------
# GET ROAD FEATURES
# ---------------------------------------------------------

def get_road_features(stats, team):

    if team not in stats:

        return (
            0.5,
            0.0,
            0.0,
            0.5
        )

    s = stats[team]

    if s["road_games"] == 0:

        return (
            0.5,
            0.0,
            0.0,
            0.5
        )

    win_pct = (
        s["road_wins"]
        / s["road_games"]
    )

    avg_scored = (
        s["road_runs_scored"]
        / s["road_games"]
    )

    avg_allowed = (
        s["road_runs_allowed"]
        / s["road_games"]
    )

    if s["road_recent_results"]:

        recent_pct = (
            sum(s["road_recent_results"])
            / len(s["road_recent_results"])
        )

    else:

        recent_pct = 0.5

    return (
        win_pct,
        avg_scored,
        avg_allowed,
        recent_pct
    )


# ---------------------------------------------------------
# PITCHER STATISTICS
# ---------------------------------------------------------

def create_pitcher_stats():

    return {

        "starts": 0,

        "innings": 0.0,

        "earned_runs": 0.0,

        "hits": 0.0,

        "walks": 0.0,

        "strikeouts": 0.0,

        "home_runs": 0.0,

        "runs": 0.0,

        "batters_faced": 0.0,

        "pitches": 0.0,

        "recent": []
    }


# ---------------------------------------------------------
# UPDATE PITCHER STATISTICS
# ---------------------------------------------------------

def update_pitcher_stats(
    stats,
    row
):

    pitcher_id = str(
        row["PitcherID"]
    ).strip()

    if not pitcher_id:
        return

    games_started = safe_float(
        row["GamesStarted"]
    )

    # Only use actual starting-pitcher appearances.

    if games_started <= 0:
        return

    if pitcher_id not in stats:

        stats[pitcher_id] = create_pitcher_stats()

    s = stats[pitcher_id]

    innings = innings_to_float(
        row["InningsPitched"]
    )

    earned_runs = safe_float(
        row["EarnedRuns"]
    )

    hits = safe_float(
        row["Hits"]
    )

    walks = safe_float(
        row["Walks"]
    )

    strikeouts = safe_float(
        row["Strikeouts"]
    )

    home_runs = safe_float(
        row["HomeRuns"]
    )

    runs = safe_float(
        row["Runs"]
    )

    batters_faced = safe_float(
        row["BattersFaced"]
    )

    pitches = safe_float(
        row["Pitches"]
    )

    s["starts"] += games_started

    s["innings"] += innings

    s["earned_runs"] += earned_runs

    s["hits"] += hits

    s["walks"] += walks

    s["strikeouts"] += strikeouts

    s["home_runs"] += home_runs

    s["runs"] += runs

    s["batters_faced"] += batters_faced

    s["pitches"] += pitches

    # Store last five starts.

    s["recent"].append({

        "innings": innings,
        "earned_runs": earned_runs,
        "hits": hits,
        "walks": walks,
        "strikeouts": strikeouts,
        "home_runs": home_runs,
        "runs": runs
    })

    s["recent"] = s["recent"][-5:]


# ---------------------------------------------------------
# GET PITCHER FEATURES
# ---------------------------------------------------------

def get_pitcher_features(
    stats,
    pitcher_id
):

    """
    Return historical pitcher statistics.

    Only previous starts are used.
    """

    default_values = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    )

    if pitcher_id is None:

        return default_values

    pitcher_id = str(
        pitcher_id
    ).strip()

    if not pitcher_id:

        return default_values

    if pitcher_id.endswith(".0"):

        pitcher_id = pitcher_id[:-2]

    if pitcher_id not in stats:

        return default_values

    s = stats[pitcher_id]

    starts = s["starts"]

    if starts == 0:

        return default_values

    innings = s["innings"]

    avg_innings = (
        innings / starts
    )

    avg_earned_runs = (
        s["earned_runs"] / starts
    )

    avg_hits = (
        s["hits"] / starts
    )

    avg_walks = (
        s["walks"] / starts
    )

    avg_strikeouts = (
        s["strikeouts"] / starts
    )

    avg_home_runs = (
        s["home_runs"] / starts
    )

    # -----------------------------------------------------
    # RATE STATISTICS
    # -----------------------------------------------------

    if innings > 0:

        whip = (
            s["walks"] + s["hits"]
        ) / innings

        k_per_9 = (
            s["strikeouts"] * 9
        ) / innings

        bb_per_9 = (
            s["walks"] * 9
        ) / innings

    else:

        whip = 0.0
        k_per_9 = 0.0
        bb_per_9 = 0.0

    # -----------------------------------------------------
    # RECENT ERA
    # -----------------------------------------------------

    recent = s["recent"]

    recent_innings = sum(
        item["innings"]
        for item in recent
    )

    recent_earned_runs = sum(
        item["earned_runs"]
        for item in recent
    )

    if recent_innings > 0:

        recent_era = (
            recent_earned_runs * 9
        ) / recent_innings

    else:

        recent_era = 0.0

    return (
        starts,
        avg_innings,
        avg_earned_runs,
        avg_hits,
        avg_walks,
        avg_strikeouts,
        avg_home_runs,
        whip,
        k_per_9,
        bb_per_9,
        recent_era
    )


# ---------------------------------------------------------
# LOAD PITCHER LOGS
# ---------------------------------------------------------

def load_pitcher_logs():

    print(
        "\nLoading pitcher game logs...",
        flush=True
    )

    with open(
        PITCHER_LOG_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    print(
        f"Pitcher log rows: {len(rows)}",
        flush=True
    )

    return rows


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "========================================"
    )

    print(
        "MLB MULTI-SEASON FEATURE ENGINEERING"
    )

    print(
        "WITH STARTING PITCHER FEATURES"
    )

    print(
        "DOUBLEHEADER SUPPORT ENABLED"
    )

    print(
        "========================================"
    )

    # -----------------------------------------------------
    # LOAD GAME DATA
    # -----------------------------------------------------

    print(
        "\nLoading game data...",
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

    if not games:

        print("No games found.")

        return

    # -----------------------------------------------------
    # LOAD PITCHER LOGS
    # -----------------------------------------------------

    pitcher_logs = load_pitcher_logs()

    # -----------------------------------------------------
    # CONVERT GAME DATES
    # -----------------------------------------------------

    print(
        "\nConverting game dates...",
        flush=True
    )

    for index, game in enumerate(games):

        game["_GameDate"] = convert_date(
            game["Date"],
            game["Season"]
        )

        game["_GameNumber"] = get_game_number(
            game["Date"]
        )

        if (index + 1) % 500 == 0:

            print(
                f"Converted {index + 1} / "
                f"{len(games)} dates",
                flush=True
            )

    print(
        "All game dates converted.",
        flush=True
    )

    # -----------------------------------------------------
    # CONVERT PITCHER DATES
    # -----------------------------------------------------

    print(
        "\nConverting pitcher dates...",
        flush=True
    )

    valid_pitcher_logs = []

    for row in pitcher_logs:

        try:

            row["_GameDate"] = datetime.strptime(
                row["Date"],
                "%Y-%m-%d"
            )

            valid_pitcher_logs.append(row)

        except Exception:

            continue

    pitcher_logs = valid_pitcher_logs

    print(
        f"Valid pitcher log rows: "
        f"{len(pitcher_logs)}",
        flush=True
    )

    # -----------------------------------------------------
    # SORT GAMES
    # -----------------------------------------------------

    print(
        "\nSorting games...",
        flush=True
    )

    games.sort(
        key=lambda x: (
            x["_GameDate"],
            int(float(x["Season"])),
            x["Home"],
            x["Away"],
            x["_GameNumber"]
        )
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

    # -----------------------------------------------------
    # SORT PITCHER LOGS
    # -----------------------------------------------------

    pitcher_logs.sort(
        key=lambda x: (
            x["_GameDate"],
            int(float(x["Season"]))
        )
    )

    # -----------------------------------------------------
    # BUILD FEATURES
    # -----------------------------------------------------

    print(
        "\nBuilding pre-game features...",
        flush=True
    )

    team_stats = {}

    pitcher_stats = {}

    feature_rows = []

    pitcher_log_index = 0

    total_pitcher_logs = len(
        pitcher_logs
    )

    total = len(games)

    current_season = None

    both_pitchers_count = 0

    # -----------------------------------------------------
    # PROCESS EACH GAME
    # -----------------------------------------------------

    for index, game in enumerate(games):

        game_date = game["_GameDate"]

        season = int(
            float(game["Season"])
        )

        home = game["Home"]

        away = game["Away"]

        game_number = game["_GameNumber"]

        # -------------------------------------------------
        # START NEW SEASON
        # -------------------------------------------------

        if season != current_season:

            current_season = season

            team_stats[season] = {}

            # Reset pitcher statistics at beginning
            # of each season.

            pitcher_stats = {}

            print(
                f"\nStarting season {season}...",
                flush=True
            )

        # -------------------------------------------------
        # ADD ONLY PREVIOUS PITCHER GAMES
        # -------------------------------------------------

        while (
            pitcher_log_index
            < total_pitcher_logs
            and pitcher_logs[
                pitcher_log_index
            ]["_GameDate"] < game_date
        ):

            log = pitcher_logs[
                pitcher_log_index
            ]

            log_season = int(
                float(log["Season"])
            )

            if log_season == season:

                update_pitcher_stats(
                    pitcher_stats,
                    log
                )

            pitcher_log_index += 1

        # -------------------------------------------------
        # GET PITCHER IDs
        # -------------------------------------------------

        home_pitcher_id = game.get(
            "HomeStartingPitcherID",
            ""
        )

        away_pitcher_id = game.get(
            "AwayStartingPitcherID",
            ""
        )

        # Normalize home pitcher ID.

        if home_pitcher_id is not None:

            home_pitcher_id = str(
                home_pitcher_id
            ).strip()

            if home_pitcher_id.endswith(".0"):

                home_pitcher_id = (
                    home_pitcher_id[:-2]
                )

        else:

            home_pitcher_id = ""

        # Normalize away pitcher ID.

        if away_pitcher_id is not None:

            away_pitcher_id = str(
                away_pitcher_id
            ).strip()

            if away_pitcher_id.endswith(".0"):

                away_pitcher_id = (
                    away_pitcher_id[:-2]
                )

        else:

            away_pitcher_id = ""

        # -------------------------------------------------
        # GET HOME PITCHER FEATURES
        # -------------------------------------------------

        (
            home_pitcher_starts,
            home_pitcher_avg_ip,
            home_pitcher_avg_er,
            home_pitcher_avg_hits,
            home_pitcher_avg_walks,
            home_pitcher_avg_k,
            home_pitcher_avg_hr,
            home_pitcher_whip,
            home_pitcher_k9,
            home_pitcher_bb9,
            home_pitcher_recent_era
        ) = get_pitcher_features(
            pitcher_stats,
            home_pitcher_id
        )

        # -------------------------------------------------
        # GET AWAY PITCHER FEATURES
        # -------------------------------------------------

        (
            away_pitcher_starts,
            away_pitcher_avg_ip,
            away_pitcher_avg_er,
            away_pitcher_avg_hits,
            away_pitcher_avg_walks,
            away_pitcher_avg_k,
            away_pitcher_avg_hr,
            away_pitcher_whip,
            away_pitcher_k9,
            away_pitcher_bb9,
            away_pitcher_recent_era
        ) = get_pitcher_features(
            pitcher_stats,
            away_pitcher_id
        )

        if (
            home_pitcher_starts > 0
            and away_pitcher_starts > 0
        ):

            both_pitchers_count += 1

        # -------------------------------------------------
        # OVERALL HOME FEATURES
        # -------------------------------------------------

        (
            home_win_pct,
            home_avg_scored,
            home_avg_allowed,
            home_recent_pct,
            home_run_differential,
            home_recent_run_differential
        ) = get_team_features(
            team_stats[season],
            home
        )

        # -------------------------------------------------
        # OVERALL AWAY FEATURES
        # -------------------------------------------------

        (
            away_win_pct,
            away_avg_scored,
            away_avg_allowed,
            away_recent_pct,
            away_run_differential,
            away_recent_run_differential
        ) = get_team_features(
            team_stats[season],
            away
        )

        # -------------------------------------------------
        # HOME TEAM HOME SPLIT
        # -------------------------------------------------

        (
            home_home_win_pct,
            home_home_avg_scored,
            home_home_avg_allowed,
            home_home_recent_pct
        ) = get_home_features(
            team_stats[season],
            home
        )

        # -------------------------------------------------
        # AWAY TEAM ROAD SPLIT
        # -------------------------------------------------

        (
            away_road_win_pct,
            away_road_avg_scored,
            away_road_avg_allowed,
            away_road_recent_pct
        ) = get_road_features(
            team_stats[season],
            away
        )

        # -------------------------------------------------
        # TEAM MATCHUP DIFFERENCES
        # -------------------------------------------------

        win_pct_diff = (
            home_win_pct
            - away_win_pct
        )

        avg_runs_scored_diff = (
            home_avg_scored
            - away_avg_scored
        )

        avg_runs_allowed_diff = (
            home_avg_allowed
            - away_avg_allowed
        )

        recent_win_pct_diff = (
            home_recent_pct
            - away_recent_pct
        )

        run_differential_diff = (
            home_run_differential
            - away_run_differential
        )

        recent_run_differential_diff = (
            home_recent_run_differential
            - away_recent_run_differential
        )

        # -------------------------------------------------
        # HOME / ROAD DIFFERENCES
        # -------------------------------------------------

        home_road_win_pct_diff = (
            home_home_win_pct
            - away_road_win_pct
        )

        home_road_runs_scored_diff = (
            home_home_avg_scored
            - away_road_avg_scored
        )

        home_road_runs_allowed_diff = (
            home_home_avg_allowed
            - away_road_avg_allowed
        )

        home_road_recent_win_pct_diff = (
            home_home_recent_pct
            - away_road_recent_pct
        )

        # -------------------------------------------------
        # PITCHER DIFFERENCES
        # -------------------------------------------------

        pitcher_starts_diff = (
            home_pitcher_starts
            - away_pitcher_starts
        )

        pitcher_avg_ip_diff = (
            home_pitcher_avg_ip
            - away_pitcher_avg_ip
        )

        pitcher_avg_er_diff = (
            home_pitcher_avg_er
            - away_pitcher_avg_er
        )

        pitcher_avg_hits_diff = (
            home_pitcher_avg_hits
            - away_pitcher_avg_hits
        )

        pitcher_avg_walks_diff = (
            home_pitcher_avg_walks
            - away_pitcher_avg_walks
        )

        pitcher_avg_k_diff = (
            home_pitcher_avg_k
            - away_pitcher_avg_k
        )

        pitcher_avg_hr_diff = (
            home_pitcher_avg_hr
            - away_pitcher_avg_hr
        )

        pitcher_whip_diff = (
            home_pitcher_whip
            - away_pitcher_whip
        )

        pitcher_k9_diff = (
            home_pitcher_k9
            - away_pitcher_k9
        )

        pitcher_bb9_diff = (
            home_pitcher_bb9
            - away_pitcher_bb9
        )

        pitcher_recent_era_diff = (
            home_pitcher_recent_era
            - away_pitcher_recent_era
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        home_won = (
            home == game["Winner"]
        )

        # -------------------------------------------------
        # STORE FEATURES
        # -------------------------------------------------

        feature_rows.append({

            "Season": season,

            "Date":
                game_date.strftime(
                    "%Y-%m-%d"
                ),

            "Home": home,

            "Away": away,

            "GameNumber": game_number,

            "MatchKey":
                game.get(
                    "MatchKey",
                    ""
                ),

            # ---------------------------------------------
            # HOME TEAM
            # ---------------------------------------------

            "Home_WinPct":
                home_win_pct,

            "Home_AvgRunsScored":
                home_avg_scored,

            "Home_AvgRunsAllowed":
                home_avg_allowed,

            "Home_RecentWinPct":
                home_recent_pct,

            "Home_RunDifferential":
                home_run_differential,

            "Home_RecentRunDifferential":
                home_recent_run_differential,

            # ---------------------------------------------
            # AWAY TEAM
            # ---------------------------------------------

            "Away_WinPct":
                away_win_pct,

            "Away_AvgRunsScored":
                away_avg_scored,

            "Away_AvgRunsAllowed":
                away_avg_allowed,

            "Away_RecentWinPct":
                away_recent_pct,

            "Away_RunDifferential":
                away_run_differential,

            "Away_RecentRunDifferential":
                away_recent_run_differential,

            # ---------------------------------------------
            # HOME SPLITS
            # ---------------------------------------------

            "Home_HomeWinPct":
                home_home_win_pct,

            "Home_HomeAvgRunsScored":
                home_home_avg_scored,

            "Home_HomeAvgRunsAllowed":
                home_home_avg_allowed,

            "Home_HomeRecentWinPct":
                home_home_recent_pct,

            # ---------------------------------------------
            # ROAD SPLITS
            # ---------------------------------------------

            "Away_RoadWinPct":
                away_road_win_pct,

            "Away_RoadAvgRunsScored":
                away_road_avg_scored,

            "Away_RoadAvgRunsAllowed":
                away_road_avg_allowed,

            "Away_RoadRecentWinPct":
                away_road_recent_pct,

            # ---------------------------------------------
            # TEAM DIFFERENCES
            # ---------------------------------------------

            "WinPct_Diff":
                win_pct_diff,

            "AvgRunsScored_Diff":
                avg_runs_scored_diff,

            "AvgRunsAllowed_Diff":
                avg_runs_allowed_diff,

            "RecentWinPct_Diff":
                recent_win_pct_diff,

            "RunDifferential_Diff":
                run_differential_diff,

            "RecentRunDifferential_Diff":
                recent_run_differential_diff,

            "HomeRoadWinPct_Diff":
                home_road_win_pct_diff,

            "HomeRoadRunsScored_Diff":
                home_road_runs_scored_diff,

            "HomeRoadRunsAllowed_Diff":
                home_road_runs_allowed_diff,

            "HomeRoadRecentWinPct_Diff":
                home_road_recent_win_pct_diff,

            # ---------------------------------------------
            # HOME STARTING PITCHER
            # ---------------------------------------------

            "Home_PitcherStarts":
                home_pitcher_starts,

            "Home_PitcherAvgIP":
                home_pitcher_avg_ip,

            "Home_PitcherAvgER":
                home_pitcher_avg_er,

            "Home_PitcherAvgHits":
                home_pitcher_avg_hits,

            "Home_PitcherAvgWalks":
                home_pitcher_avg_walks,

            "Home_PitcherAvgStrikeouts":
                home_pitcher_avg_k,

            "Home_PitcherAvgHomeRuns":
                home_pitcher_avg_hr,

            "Home_PitcherWHIP":
                home_pitcher_whip,

            "Home_PitcherK9":
                home_pitcher_k9,

            "Home_PitcherBB9":
                home_pitcher_bb9,

            "Home_PitcherRecentERA":
                home_pitcher_recent_era,

            # ---------------------------------------------
            # AWAY STARTING PITCHER
            # ---------------------------------------------

            "Away_PitcherStarts":
                away_pitcher_starts,

            "Away_PitcherAvgIP":
                away_pitcher_avg_ip,

            "Away_PitcherAvgER":
                away_pitcher_avg_er,

            "Away_PitcherAvgHits":
                away_pitcher_avg_hits,

            "Away_PitcherAvgWalks":
                away_pitcher_avg_walks,

            "Away_PitcherAvgStrikeouts":
                away_pitcher_avg_k,

            "Away_PitcherAvgHomeRuns":
                away_pitcher_avg_hr,

            "Away_PitcherWHIP":
                away_pitcher_whip,

            "Away_PitcherK9":
                away_pitcher_k9,

            "Away_PitcherBB9":
                away_pitcher_bb9,

            "Away_PitcherRecentERA":
                away_pitcher_recent_era,

            # ---------------------------------------------
            # PITCHER DIFFERENCES
            # ---------------------------------------------

            "PitcherStarts_Diff":
                pitcher_starts_diff,

            "PitcherAvgIP_Diff":
                pitcher_avg_ip_diff,

            "PitcherAvgER_Diff":
                pitcher_avg_er_diff,

            "PitcherAvgHits_Diff":
                pitcher_avg_hits_diff,

            "PitcherAvgWalks_Diff":
                pitcher_avg_walks_diff,

            "PitcherAvgStrikeouts_Diff":
                pitcher_avg_k_diff,

            "PitcherAvgHomeRuns_Diff":
                pitcher_avg_hr_diff,

            "PitcherWHIP_Diff":
                pitcher_whip_diff,

            "PitcherK9_Diff":
                pitcher_k9_diff,

            "PitcherBB9_Diff":
                pitcher_bb9_diff,

            "PitcherRecentERA_Diff":
                pitcher_recent_era_diff,

            # ---------------------------------------------
            # TARGET
            # ---------------------------------------------

            "HomeWon":
                int(home_won)
        })

        # -------------------------------------------------
        # UPDATE TEAM STATISTICS AFTER GAME
        # -------------------------------------------------

        update_team_stats(
            team_stats[season],
            home,
            game["HomeScore"],
            game["AwayScore"],
            home_won,
            "home"
        )

        update_team_stats(
            team_stats[season],
            away,
            game["AwayScore"],
            game["HomeScore"],
            not home_won,
            "road"
        )

        if (index + 1) % 500 == 0:

            print(
                f"Processed {index + 1} / "
                f"{total} games",
                flush=True
            )

    # -----------------------------------------------------
    # SAVE FEATURES
    # -----------------------------------------------------

    print(
        "\nSaving feature dataset...",
        flush=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [

        "Season",
        "Date",
        "Home",
        "Away",
        "GameNumber",
        "MatchKey",

        # Team features

        "Home_WinPct",
        "Home_AvgRunsScored",
        "Home_AvgRunsAllowed",
        "Home_RecentWinPct",
        "Home_RunDifferential",
        "Home_RecentRunDifferential",

        "Away_WinPct",
        "Away_AvgRunsScored",
        "Away_AvgRunsAllowed",
        "Away_RecentWinPct",
        "Away_RunDifferential",
        "Away_RecentRunDifferential",

        "Home_HomeWinPct",
        "Home_HomeAvgRunsScored",
        "Home_HomeAvgRunsAllowed",
        "Home_HomeRecentWinPct",

        "Away_RoadWinPct",
        "Away_RoadAvgRunsScored",
        "Away_RoadAvgRunsAllowed",
        "Away_RoadRecentWinPct",

        "WinPct_Diff",
        "AvgRunsScored_Diff",
        "AvgRunsAllowed_Diff",
        "RecentWinPct_Diff",
        "RunDifferential_Diff",
        "RecentRunDifferential_Diff",

        "HomeRoadWinPct_Diff",
        "HomeRoadRunsScored_Diff",
        "HomeRoadRunsAllowed_Diff",
        "HomeRoadRecentWinPct_Diff",

        # Home pitcher

        "Home_PitcherStarts",
        "Home_PitcherAvgIP",
        "Home_PitcherAvgER",
        "Home_PitcherAvgHits",
        "Home_PitcherAvgWalks",
        "Home_PitcherAvgStrikeouts",
        "Home_PitcherAvgHomeRuns",
        "Home_PitcherWHIP",
        "Home_PitcherK9",
        "Home_PitcherBB9",
        "Home_PitcherRecentERA",

        # Away pitcher

        "Away_PitcherStarts",
        "Away_PitcherAvgIP",
        "Away_PitcherAvgER",
        "Away_PitcherAvgHits",
        "Away_PitcherAvgWalks",
        "Away_PitcherAvgStrikeouts",
        "Away_PitcherAvgHomeRuns",
        "Away_PitcherWHIP",
        "Away_PitcherK9",
        "Away_PitcherBB9",
        "Away_PitcherRecentERA",

        # Pitcher differences

        "PitcherStarts_Diff",
        "PitcherAvgIP_Diff",
        "PitcherAvgER_Diff",
        "PitcherAvgHits_Diff",
        "PitcherAvgWalks_Diff",
        "PitcherAvgStrikeouts_Diff",
        "PitcherAvgHomeRuns_Diff",
        "PitcherWHIP_Diff",
        "PitcherK9_Diff",
        "PitcherBB9_Diff",
        "PitcherRecentERA_Diff",

        # Target

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

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print("")

    print(
        "========================================"
    )

    print(
        "FEATURE ENGINEERING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Games: {len(feature_rows)}"
    )

    print(
        f"Features: {len(fieldnames)}"
    )

    print(
        f"Pitcher features added: "
        f"{len(fieldnames) - 37}"
    )

    print(
        f"Games with both historical "
        f"pitchers: {both_pitchers_count}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    # -----------------------------------------------------
    # SEASON COUNTS
    # -----------------------------------------------------

    season_counts = {}

    for row in feature_rows:

        season = row["Season"]

        season_counts[season] = (
            season_counts.get(
                season,
                0
            ) + 1
        )

    print(
        "\nGames by season:"
    )

    for season in sorted(
        season_counts
    ):

        print(
            f"  {season}: "
            f"{season_counts[season]}"
        )

    # -----------------------------------------------------
    # DOUBLEHEADER CHECK
    # -----------------------------------------------------

    print(
        "\nDoubleheader check:"
    )

    doubleheaders = {}

    for row in feature_rows:

        key = (
            row["Season"],
            row["Date"],
            row["Home"],
            row["Away"]
        )

        doubleheaders[key] = (
            doubleheaders.get(key, 0) + 1
        )

    doubleheader_groups = {
        key: count
        for key, count in doubleheaders.items()
        if count > 1
    }

    print(
        f"Doubleheader groups: "
        f"{len(doubleheader_groups)}"
    )

    print(
        f"Games contained in doubleheader "
        f"groups: "
        f"{sum(doubleheader_groups.values())}"
    )

    # -----------------------------------------------------
    # FIRST FIVE
    # -----------------------------------------------------

    print(
        "\nFirst 5 games:"
    )

    for row in feature_rows[:5]:

        print(row)


if __name__ == "__main__":

    main()