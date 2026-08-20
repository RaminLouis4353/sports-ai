import csv
from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
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
    "HomeRoadRecentWinPct_Diff"
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

    for row in rows:

        X.append([
            float(row[feature])
            for feature in FEATURES
        ])

        y.append(
            int(row["HomeWon"])
        )

    return X, y


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("")
    print("----------------------------------------")
    print(f"Training: {name}")
    print("----------------------------------------")

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
        f"Accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Log Loss: {loss:.4f}"
    )

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "log_loss": loss
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "========================================"
    )

    print(
        "MLB MODEL COMPARISON"
    )

    print(
        "========================================"
    )

    rows = load_data()

    X, y = prepare_data(
        rows
    )

    # -----------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # -----------------------------------------------------

    split_index = int(
        len(X) * 0.80
    )

    X_train = X[:split_index]

    X_test = X[split_index:]

    y_train = y[:split_index]

    y_test = y[split_index:]

    print("")

    print(
        f"Training games: {len(X_train)}"
    )

    print(
        f"Testing games: {len(X_test)}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    models = [

        (
            "Logistic Regression",

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
            ])
        ),

        (
            "Random Forest",

            RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        ),

        (
            "Gradient Boosting",

            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=10,
                random_state=42
            )
        ),

        (
            "HistGradientBoosting",

            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.03,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ]

    # -----------------------------------------------------
    # RUN MODELS
    # -----------------------------------------------------

    results = []

    for name, model in models:

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(
            result
        )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print("")

    print(
        "========================================"
    )

    print(
        "MODEL COMPARISON RESULTS"
    )

    print(
        "========================================"
    )

    print("")

    print(
        f"{'Model':<25}"
        f"{'Accuracy':>12}"
        f"{'Log Loss':>12}"
    )

    print(
        "-" * 49
    )

    for result in results:

        print(
            f"{result['name']:<25}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['log_loss']:>12.4f}"
        )

    # -----------------------------------------------------
    # BEST MODELS
    # -----------------------------------------------------

    best_accuracy = max(
        results,
        key=lambda x: x["accuracy"]
    )

    best_log_loss = min(
        results,
        key=lambda x: x["log_loss"]
    )

    print("")

    print(
        "Best Accuracy:"
    )

    print(
        f"  {best_accuracy['name']} "
        f"({best_accuracy['accuracy'] * 100:.2f}%)"
    )

    print("")

    print(
        "Best Log Loss:"
    )

    print(
        f"  {best_log_loss['name']} "
        f"({best_log_loss['log_loss']:.4f})"
    )

    # -----------------------------------------------------
    # SAVE BEST MODEL BY LOG LOSS
    # -----------------------------------------------------

    output_dir = Path(
        "models"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_file = (
        output_dir /
        "mlb_best_model.pkl"
    )

    joblib.dump(
        best_log_loss["model"],
        model_file
    )

    print("")

    print(
        f"Best model saved to: {model_file}"
    )

    print("")

    print(
        "========================================"
    )

    print(
        "MODEL COMPARISON COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()