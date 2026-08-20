"""
MLB Sports Analytics Project

Train a chronological Logistic Regression model.

Training:
    2022

Validation:
    2023

Final Test:
    2024

IMPORTANT:
    Data is split chronologically by season.
    No random train/test split is used.

Input:
    data/processed/mlb_multi_season_features.csv

Output:
    models/mlb_logistic_model.pkl
"""


import csv
from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


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
# SEASONS
# ---------------------------------------------------------

TRAIN_SEASON = 2022

VALIDATION_SEASON = 2023

TEST_SEASON = 2024


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

FEATURES = [

    # -----------------------------------------------------
    # HOME TEAM
    # -----------------------------------------------------

    "Home_WinPct",
    "Home_AvgRunsScored",
    "Home_AvgRunsAllowed",
    "Home_RecentWinPct",
    "Home_RunDifferential",
    "Home_RecentRunDifferential",

    # -----------------------------------------------------
    # AWAY TEAM
    # -----------------------------------------------------

    "Away_WinPct",
    "Away_AvgRunsScored",
    "Away_AvgRunsAllowed",
    "Away_RecentWinPct",
    "Away_RunDifferential",
    "Away_RecentRunDifferential",

    # -----------------------------------------------------
    # HOME SPLITS
    # -----------------------------------------------------

    "Home_HomeWinPct",
    "Home_HomeAvgRunsScored",
    "Home_HomeAvgRunsAllowed",
    "Home_HomeRecentWinPct",

    # -----------------------------------------------------
    # AWAY SPLITS
    # -----------------------------------------------------

    "Away_RoadWinPct",
    "Away_RoadAvgRunsScored",
    "Away_RoadAvgRunsAllowed",
    "Away_RoadRecentWinPct",

    # -----------------------------------------------------
    # TEAM DIFFERENCES
    # -----------------------------------------------------

    "WinPct_Diff",
    "AvgRunsScored_Diff",
    "AvgRunsAllowed_Diff",
    "RecentWinPct_Diff",
    "RunDifferential_Diff",
    "RecentRunDifferential_Diff",

    # -----------------------------------------------------
    # HOME / ROAD DIFFERENCES
    # -----------------------------------------------------

    "HomeRoadWinPct_Diff",
    "HomeRoadRunsScored_Diff",
    "HomeRoadRunsAllowed_Diff",
    "HomeRoadRecentWinPct_Diff",

    # -----------------------------------------------------
    # HOME STARTING PITCHER
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

    # -----------------------------------------------------
    # AWAY STARTING PITCHER
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # PITCHER DIFFERENCES
    # -----------------------------------------------------

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
        encoding="utf-8-sig",
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
# GET SEASON DATA
# ---------------------------------------------------------

def get_season_rows(
    rows,
    season
):

    return [
        row
        for row in rows
        if int(float(row["Season"])) == season
    ]


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model(
    model,
    X,
    y,
    season_name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    loss = log_loss(
        y,
        probabilities
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    brier = brier_score_loss(
        y,
        probabilities
    )

    print("")

    print(
        "========================================"
    )

    print(
        f"{season_name} RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Accuracy %: {accuracy * 100:.2f}%"
    )

    print(
        f"Log Loss:  {loss:.4f}"
    )

    print(
        f"ROC-AUC:   {auc:.4f}"
    )

    print(
        f"Brier:     {brier:.4f}"
    )

    print("")

    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Away Win",
                "Home Win"
            ]
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    return {
        "accuracy": accuracy,
        "log_loss": loss,
        "roc_auc": auc,
        "brier": brier
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "========================================"
    )

    print(
        "MLB LOGISTIC REGRESSION"
    )

    print(
        "CHRONOLOGICAL SEASON SPLIT"
    )

    print(
        "========================================"
    )

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    rows = load_data()

    # -----------------------------------------------------
    # GET SEASONS
    # -----------------------------------------------------

    train_rows = get_season_rows(
        rows,
        TRAIN_SEASON
    )

    validation_rows = get_season_rows(
        rows,
        VALIDATION_SEASON
    )

    test_rows = get_season_rows(
        rows,
        TEST_SEASON
    )

    print("")

    print(
        f"Training season:   {TRAIN_SEASON}"
    )

    print(
        f"Validation season: {VALIDATION_SEASON}"
    )

    print(
        f"Test season:       {TEST_SEASON}"
    )

    print("")

    print(
        f"Training games:    {len(train_rows)}"
    )

    print(
        f"Validation games:  {len(validation_rows)}"
    )

    print(
        f"Test games:        {len(test_rows)}"
    )

    # -----------------------------------------------------
    # PREPARE
    # -----------------------------------------------------

    X_train, y_train = prepare_data(
        train_rows
    )

    X_validation, y_validation = prepare_data(
        validation_rows
    )

    X_test, y_test = prepare_data(
        test_rows
    )

    # -----------------------------------------------------
    # CHECK
    # -----------------------------------------------------

    print("")

    print(
        f"Model features: {len(FEATURES)}"
    )

    print("")

    print(
        "Training target distribution:"
    )

    print(
        f"  Away wins: {y_train.count(0)}"
    )

    print(
        f"  Home wins: {y_train.count(1)}"
    )

    print("")

    print(
        "Validation target distribution:"
    )

    print(
        f"  Away wins: {y_validation.count(0)}"
    )

    print(
        f"  Home wins: {y_validation.count(1)}"
    )

    print("")

    print(
        "Test target distribution:"
    )

    print(
        f"  Away wins: {y_test.count(0)}"
    )

    print(
        f"  Home wins: {y_test.count(1)}"
    )

    # -----------------------------------------------------
    # BUILD MODEL
    # -----------------------------------------------------

    model = Pipeline([

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

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print("")

    print(
        "Training model...",
        flush=True
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model trained."
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        "2023 VALIDATION"
    )

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "2024 FINAL TEST"
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "train_season": TRAIN_SEASON,
            "validation_season": VALIDATION_SEASON,
            "test_season": TEST_SEASON
        },
        MODEL_FILE
    )

    print("")

    print(
        "========================================"
    )

    print(
        "MODEL TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Model saved to: {MODEL_FILE}"
    )

    print("")

    print(
        "Final 2024 metrics:"
    )

    print(
        f"  Accuracy: {test_results['accuracy']:.4f}"
    )

    print(
        f"  Log Loss: {test_results['log_loss']:.4f}"
    )

    print(
        f"  ROC-AUC:  {test_results['roc_auc']:.4f}"
    )

    print(
        f"  Brier:    {test_results['brier']:.4f}"
    )


if __name__ == "__main__":

    main()