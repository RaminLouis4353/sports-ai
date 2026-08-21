import csv
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    log_loss
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# FILE
# ---------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/mlb_multi_season_features.csv"
)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

FEATURES = [

    # -----------------------------------------------------
    # TEAM FEATURES
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # PITCHER FEATURES
    # -----------------------------------------------------

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


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    print(
        "Loading feature data...",
        flush=True
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    print(
        f"Games loaded: {len(rows)}",
        flush=True
    )

    return rows


# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------

def prepare_data(rows):

    X = []

    y = []

    dates = []

    seasons = []

    for row in rows:

        X.append([
            float(row[feature])
            for feature in FEATURES
        ])

        y.append(
            int(row["HomeWon"])
        )

        dates.append(
            row["Date"]
        )

        seasons.append(
            int(row["Season"])
        )

    return X, y, dates, seasons


# ---------------------------------------------------------
# CREATE MODELS
# ---------------------------------------------------------

def create_models():

    return {

        "Logistic Regression":

            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000
                    )
                )
            ]),

        "Random Forest":

            RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":

            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=10,
                random_state=42
            )
    }


# ---------------------------------------------------------
# EVALUATE PERIOD
# ---------------------------------------------------------

def evaluate_period(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("")
    print(
        f"  {name}"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    loss = log_loss(
        y_test,
        probabilities
    )

    print(
        f"    Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"    Log Loss: "
        f"{loss:.4f}"
    )

    return accuracy, loss


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "========================================"
    )

    print(
        "MLB WALK-FORWARD VALIDATION"
    )

    print(
        "========================================"
    )

    X, y, dates, seasons = prepare_data(
        load_data()
    )

    print("")
    print(
        f"Features: {len(FEATURES)}"
    )

    # -----------------------------------------------------
    # PERIODS
    # -----------------------------------------------------
    #
    # Each test period happens AFTER the training period.
    #
    # Period 1:
    # Train 2022
    # Test 2023
    #
    # Period 2:
    # Train 2022-2023
    # Test 2024
    #
    # Period 3:
    # Train first 80% of data
    # Test final 20%
    # -----------------------------------------------------

    periods = []

    # 2022 -> 2023

    train_2022 = [
        i for i, season in enumerate(seasons)
        if season == 2022
    ]

    test_2023 = [
        i for i, season in enumerate(seasons)
        if season == 2023
    ]

    periods.append(
        (
            "2022 -> 2023",
            train_2022,
            test_2023
        )
    )

    # 2022-2023 -> 2024

    train_22_23 = [
        i for i, season in enumerate(seasons)
        if season in [2022, 2023]
    ]

    test_2024 = [
        i for i, season in enumerate(seasons)
        if season == 2024
    ]

    periods.append(
        (
            "2022-2023 -> 2024",
            train_22_23,
            test_2024
        )
    )

    # Existing 80/20 split

    split_index = int(
        len(X) * 0.80
    )

    train_80 = list(
        range(0, split_index)
    )

    test_20 = list(
        range(split_index, len(X))
    )

    periods.append(
        (
            "80/20 Chronological",
            train_80,
            test_20
        )
    )

    # -----------------------------------------------------
    # RESULTS STORAGE
    # -----------------------------------------------------

    results = {

        "Logistic Regression": [],

        "Random Forest": [],

        "Gradient Boosting": []
    }

    # -----------------------------------------------------
    # RUN VALIDATION
    # -----------------------------------------------------

    for period_name, train_indices, test_indices in periods:

        print("")
        print(
            "========================================"
        )

        print(
            period_name
        )

        print(
            "========================================"
        )

        X_train = [
            X[i]
            for i in train_indices
        ]

        y_train = [
            y[i]
            for i in train_indices
        ]

        X_test = [
            X[i]
            for i in test_indices
        ]

        y_test = [
            y[i]
            for i in test_indices
        ]

        print(
            f"Training games: {len(X_train)}"
        )

        print(
            f"Testing games: {len(X_test)}"
        )

        models = create_models()

        for model_name, model in models.items():

            accuracy, loss = evaluate_period(
                model_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test
            )

            results[model_name].append(
                (
                    accuracy,
                    loss
                )
            )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("")
    print(
        "========================================"
    )

    print(
        "WALK-FORWARD SUMMARY"
    )

    print(
        "========================================"
    )

    for model_name in results:

        model_results = results[
            model_name
        ]

        average_accuracy = sum(
            result[0]
            for result in model_results
        ) / len(model_results)

        average_log_loss = sum(
            result[1]
            for result in model_results
        ) / len(model_results)

        print("")
        print(
            model_name
        )

        print(
            f"  Average Accuracy: "
            f"{average_accuracy * 100:.2f}%"
        )

        print(
            f"  Average Log Loss: "
            f"{average_log_loss:.4f}"
        )

    # -----------------------------------------------------
    # DETAILED TABLE
    # -----------------------------------------------------

    print("")
    print(
        "========================================"
    )

    print(
        "DETAILED RESULTS"
    )

    print(
        "========================================"
    )

    print("")

    print(
        f"{'Model':<23}"
        f"{'Period':<25}"
        f"{'Accuracy':>12}"
        f"{'Log Loss':>12}"
    )

    print(
        "-" * 72
    )

    for model_name in results:

        for index, result in enumerate(
            results[model_name]
        ):

            period_name = periods[
                index
            ][0]

            accuracy = result[0]

            loss = result[1]

            print(
                f"{model_name:<23}"
                f"{period_name:<25}"
                f"{accuracy * 100:>11.2f}%"
                f"{loss:>12.4f}"
            )

    print("")
    print(
        "========================================"
    )

    print(
        "WALK-FORWARD VALIDATION COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()