"""
MLB Sports Analytics Project

Evaluate the trained MLB prediction model using
chronological historical data.

This script:
- Loads the multi-season feature dataset
- Loads the trained logistic regression model
- Uses the final 20% of games as the test period
- Calculates accuracy and log loss
- Calculates performance at different confidence levels
- Shows results by season
- Shows the strongest and weakest predictions
"""

import csv
from pathlib import Path

import joblib

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/mlb_multi_season_features.csv"
)

MODEL_FILE = Path(
    "models/mlb_logistic_model.pkl"
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
# PREPARE FEATURES
# ---------------------------------------------------------

def prepare_features(rows):

    X = []

    y = []

    for row in rows:

        features = []

        for feature in FEATURES:

            features.append(
                float(row[feature])
            )

        X.append(features)

        y.append(
            int(row["HomeWon"])
        )

    return X, y


# ---------------------------------------------------------
# CONFIDENCE EVALUATION
# ---------------------------------------------------------

def evaluate_confidence(
    y_true,
    probabilities,
    threshold
):

    correct = 0

    total = 0

    for actual, probability in zip(
        y_true,
        probabilities
    ):

        confidence = max(
            probability,
            1.0 - probability
        )

        if confidence >= threshold:

            prediction = (
                1
                if probability >= 0.5
                else 0
            )

            if prediction == actual:

                correct += 1

            total += 1

    if total == 0:

        return 0, 0.0

    accuracy = correct / total

    return total, accuracy


# ---------------------------------------------------------
# SEASON EVALUATION
# ---------------------------------------------------------

def evaluate_by_season(
    rows,
    predictions,
    probabilities
):

    seasons = sorted(
        set(
            int(row["Season"])
            for row in rows
        )
    )

    print("")
    print(
        "=============================="
    )
    print(
        "RESULTS BY SEASON"
    )
    print(
        "=============================="
    )

    for season in seasons:

        indexes = [

            index

            for index, row in enumerate(rows)

            if int(row["Season"]) == season
        ]

        if not indexes:

            continue

        actual = [

            int(rows[index]["HomeWon"])

            for index in indexes
        ]

        season_predictions = [

            predictions[index]

            for index in indexes
        ]

        season_probabilities = [

            probabilities[index]

            for index in indexes
        ]

        accuracy = accuracy_score(
            actual,
            season_predictions
        )

        loss = log_loss(
            actual,
            season_probabilities
        )

        print("")
        print(
            f"Season {season}"
        )

        print(
            f"Games: {len(indexes)}"
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Accuracy %: {accuracy * 100:.2f}%"
        )

        print(
            f"Log Loss: {loss:.4f}"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "========================================"
    )

    print(
        "MLB MODEL EVALUATION"
    )

    print(
        "========================================"
    )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print(
        "\nLoading trained model...",
        flush=True
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model loaded.",
        flush=True
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    rows = load_data()

    X, y = prepare_features(
        rows
    )

    # -----------------------------------------------------
    # CHRONOLOGICAL TEST SET
    # -----------------------------------------------------

    split_index = int(
        len(X) * 0.80
    )

    X_test = X[
        split_index:
    ]

    y_test = y[
        split_index:
    ]

    test_rows = rows[
        split_index:
    ]

    print("")

    print(
        f"Training portion: {split_index}"
    )

    print(
        f"Testing portion: {len(X_test)}"
    )

    print(
        f"Test start date: "
        f"{test_rows[0]['Date']}"
    )

    print(
        f"Test end date: "
        f"{test_rows[-1]['Date']}"
    )

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    print(
        "\nGenerating predictions...",
        flush=True
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print(
        "Predictions generated.",
        flush=True
    )

    # -----------------------------------------------------
    # BASIC RESULTS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    loss = log_loss(
        y_test,
        probabilities
    )

    print("")
    print(
        "=============================="
    )

    print(
        "OVERALL RESULTS"
    )

    print(
        "=============================="
    )

    print(
        f"Games tested: {len(y_test)}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy %: {accuracy * 100:.2f}%"
    )

    print(
        f"Log Loss: {loss:.4f}"
    )

    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    print("")
    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Away Win",
                "Home Win"
            ]
        )
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # -----------------------------------------------------
    # CONFIDENCE LEVELS
    # -----------------------------------------------------

    print("")
    print(
        "=============================="
    )

    print(
        "CONFIDENCE ANALYSIS"
    )

    print(
        "=============================="
    )

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70
    ]

    for threshold in thresholds:

        total, confidence_accuracy = (
            evaluate_confidence(
                y_test,
                probabilities,
                threshold
            )
        )

        print("")

        print(
            f"Confidence >= "
            f"{threshold:.0%}"
        )

        print(
            f"Games: {total}"
        )

        if total > 0:

            print(
                f"Accuracy: "
                f"{confidence_accuracy * 100:.2f}%"
            )

        else:

            print(
                "Accuracy: N/A"
            )

    # -----------------------------------------------------
    # SEASON RESULTS
    # -----------------------------------------------------

    evaluate_by_season(
        test_rows,
        predictions,
        probabilities
    )

    # -----------------------------------------------------
    # STRONGEST PREDICTIONS
    # -----------------------------------------------------

    prediction_data = []

    for index in range(
        len(test_rows)
    ):

        probability = probabilities[index]

        prediction = predictions[index]

        actual = y_test[index]

        confidence = max(
            probability,
            1.0 - probability
        )

        prediction_data.append({

            "row":
                test_rows[index],

            "probability":
                probability,

            "prediction":
                prediction,

            "actual":
                actual,

            "confidence":
                confidence
        })

    strongest = sorted(
        prediction_data,
        key=lambda x: x["confidence"],
        reverse=True
    )

    print("")
    print(
        "=============================="
    )

    print(
        "10 STRONGEST PREDICTIONS"
    )

    print(
        "=============================="
    )

    for item in strongest[:10]:

        row = item["row"]

        probability = item["probability"]

        prediction = item["prediction"]

        actual = item["actual"]

        confidence = item["confidence"]

        predicted_team = (
            row["Home"]
            if prediction == 1
            else row["Away"]
        )

        actual_team = (
            row["Home"]
            if actual == 1
            else row["Away"]
        )

        print("")

        print(
            f"{row['Date']} | "
            f"{row['Away']} @ {row['Home']}"
        )

        print(
            f"Prediction: "
            f"{predicted_team}"
        )

        print(
            f"Home Win Probability: "
            f"{probability:.3f}"
        )

        print(
            f"Confidence: "
            f"{confidence:.3f}"
        )

        print(
            f"Actual Winner: "
            f"{actual_team}"
        )

        print(
            f"Correct: "
            f"{prediction == actual}"
        )

    # -----------------------------------------------------
    # END
    # -----------------------------------------------------

    print("")
    print(
        "========================================"
    )

    print(
        "MODEL EVALUATION COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()