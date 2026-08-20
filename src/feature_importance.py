import csv
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# =========================================================
# FILE
# =========================================================

INPUT_FILE = Path(
    "data/processed/mlb_multi_season_features.csv"
)


# =========================================================
# FEATURES
# =========================================================

FEATURES = [

    # Overall home team performance

    "Home_WinPct",
    "Home_AvgRunsScored",
    "Home_AvgRunsAllowed",
    "Home_RecentWinPct",
    "Home_RunDifferential",
    "Home_RecentRunDifferential",

    # Overall away team performance

    "Away_WinPct",
    "Away_AvgRunsScored",
    "Away_AvgRunsAllowed",
    "Away_RecentWinPct",
    "Away_RunDifferential",
    "Away_RecentRunDifferential",

    # Home / road splits

    "Home_HomeWinPct",
    "Home_HomeAvgRunsScored",
    "Home_HomeAvgRunsAllowed",
    "Home_HomeRecentWinPct",

    "Away_RoadWinPct",
    "Away_RoadAvgRunsScored",
    "Away_RoadAvgRunsAllowed",
    "Away_RoadRecentWinPct",

    # Overall matchup differences

    "WinPct_Diff",
    "AvgRunsScored_Diff",
    "AvgRunsAllowed_Diff",
    "RecentWinPct_Diff",
    "RunDifferential_Diff",
    "RecentRunDifferential_Diff",

    # Home / road matchup differences

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
    "PitcherRecentERA_Diff"
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print(
        "Loading feature data...",
        flush=True
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Games loaded: {len(df)}",
        flush=True
    )

    return df


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "========================================"
    )

    print(
        "MLB FEATURE IMPORTANCE ANALYSIS"
    )

    print(
        "RANDOM FOREST + PERMUTATION IMPORTANCE"
    )

    print(
        "========================================"
    )

    df = load_data()

    # -----------------------------------------------------
    # SPLIT BY SEASON
    # -----------------------------------------------------

    train = df[
        df["Season"] == 2022
    ]

    test = df[
        df["Season"] == 2024
    ]

    X_train = train[FEATURES]

    y_train = train["HomeWon"]

    X_test = test[FEATURES]

    y_test = test["HomeWon"]

    print("")

    print(
        f"Training games: {len(train)}"
    )

    print(
        f"Test games:     {len(test)}"
    )

    print(
        f"Features:        {len(FEATURES)}"
    )

    # -----------------------------------------------------
    # TRAIN RANDOM FOREST
    # -----------------------------------------------------

    print("")
    print(
        "Training Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest trained."
    )

    # -----------------------------------------------------
    # BUILT-IN FEATURE IMPORTANCE
    # -----------------------------------------------------

    print("")
    print(
        "Calculating built-in feature importance..."
    )

    importance = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("")
    print(
        "========================================"
    )

    print(
        "TOP 20 BUILT-IN FEATURE IMPORTANCE"
    )

    print(
        "========================================"
    )

    print("")

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # PERMUTATION IMPORTANCE
    # -----------------------------------------------------

    print("")
    print(
        "Calculating permutation importance..."
    )

    permutation = permutation_importance(

        model,

        X_test,

        y_test,

        n_repeats=10,

        random_state=42,

        n_jobs=-1
    )

    permutation_df = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": permutation.importances_mean,

        "Std": permutation.importances_std

    })

    permutation_df = permutation_df.sort_values(
        "Importance",
        ascending=False
    )

    print("")
    print(
        "========================================"
    )

    print(
        "TOP 20 PERMUTATION IMPORTANCE"
    )

    print(
        "========================================"
    )

    print("")

    print(
        permutation_df.head(20).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # PITCHER FEATURE SUMMARY
    # -----------------------------------------------------

    pitcher_features = [

        feature

        for feature in FEATURES

        if "Pitcher" in feature
    ]

    pitcher_importance = permutation_df[
        permutation_df["Feature"].isin(
            pitcher_features
        )
    ]

    print("")
    print(
        "========================================"
    )

    print(
        "PITCHER FEATURE IMPORTANCE"
    )

    print(
        "========================================"
    )

    print("")

    print(
        pitcher_importance.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    output_dir = Path(
        "data/processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_file = (
        output_dir /
        "feature_importance.csv"
    )

    permutation_file = (
        output_dir /
        "permutation_importance.csv"
    )

    importance.to_csv(
        importance_file,
        index=False
    )

    permutation_df.to_csv(
        permutation_file,
        index=False
    )

    print("")
    print(
        f"Built-in importance saved to: "
        f"{importance_file}"
    )

    print(
        f"Permutation importance saved to: "
        f"{permutation_file}"
    )

    print("")
    print(
        "========================================"
    )

    print(
        "FEATURE ANALYSIS COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()