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
    log_loss,
    roc_auc_score,
    brier_score_loss
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================================================
# FILES
# =========================================================

INPUT_FILE = Path(
    "data/processed/mlb_multi_season_features.csv"
)

MODEL_FILE = Path(
    "models/mlb_best_model.pkl"
)


# =========================================================
# FEATURES
# =========================================================

FEATURES = [

    # -----------------------------------------------------
    # Overall home team performance
    # -----------------------------------------------------

    "Home_WinPct",
    "Home_AvgRunsScored",
    "Home_AvgRunsAllowed",
    "Home_RecentWinPct",
    "Home_RunDifferential",
    "Home_RecentRunDifferential",

    # -----------------------------------------------------
    # Overall away team performance
    # -----------------------------------------------------

    "Away_WinPct",
    "Away_AvgRunsScored",
    "Away_AvgRunsAllowed",
    "Away_RecentWinPct",
    "Away_RunDifferential",
    "Away_RecentRunDifferential",

    # -----------------------------------------------------
    # Home / road splits
    # -----------------------------------------------------

    "Home_HomeWinPct",
    "Home_HomeAvgRunsScored",
    "Home_HomeAvgRunsAllowed",
    "Home_HomeRecentWinPct",

    "Away_RoadWinPct",
    "Away_RoadAvgRunsScored",
    "Away_RoadAvgRunsAllowed",
    "Away_RoadRecentWinPct",

    # -----------------------------------------------------
    # Overall matchup differences
    # -----------------------------------------------------

    "WinPct_Diff",
    "AvgRunsScored_Diff",
    "AvgRunsAllowed_Diff",
    "RecentWinPct_Diff",
    "RunDifferential_Diff",
    "RecentRunDifferential_Diff",

    # -----------------------------------------------------
    # Home / road matchup differences
    # -----------------------------------------------------

    "HomeRoadWinPct_Diff",
    "HomeRoadRunsScored_Diff",
    "HomeRoadRunsAllowed_Diff",
    "HomeRoadRecentWinPct_Diff",

    # -----------------------------------------------------
    # Home starting pitcher
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
    # Away starting pitcher
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
    # Pitcher matchup differences
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


# =========================================================
# LOAD DATA
# =========================================================

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


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(rows):

    X = []
    y = []
    seasons = []

    for row in rows:

        X.append([
            float(row[feature])
            for feature in FEATURES
        ])

        y.append(
            int(row["HomeWon"])
        )

        seasons.append(
            int(row["Season"])
        )

    return X, y, seasons


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_name
):

    print("")
    print("----------------------------------------")
    print(f"{name} - {dataset_name}")
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

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    print(
        f"Accuracy:  {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Log Loss:  {loss:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"Brier:     {brier:.4f}"
    )

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "log_loss": loss,
        "roc_auc": roc_auc,
        "brier": brier
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "========================================"
    )

    print(
        "MLB MODEL COMPARISON"
    )

    print(
        "CHRONOLOGICAL SEASON SPLIT"
    )

    print(
        "WITH PITCHER FEATURES"
    )

    print(
        "========================================"
    )

    rows = load_data()

    X, y, seasons = prepare_data(
        rows
    )

    # =====================================================
    # SEASON SPLIT
    # =====================================================

    X_train = []
    y_train = []

    X_validation = []
    y_validation = []

    X_test = []
    y_test = []

    for i, season in enumerate(seasons):

        if season == 2022:

            X_train.append(
                X[i]
            )

            y_train.append(
                y[i]
            )

        elif season == 2023:

            X_validation.append(
                X[i]
            )

            y_validation.append(
                y[i]
            )

        elif season == 2024:

            X_test.append(
                X[i]
            )

            y_test.append(
                y[i]
            )

    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    print("")

    print(
        "Training season:   2022"
    )

    print(
        "Validation season: 2023"
    )

    print(
        "Test season:       2024"
    )

    print("")

    print(
        f"Training games:    {len(X_train)}"
    )

    print(
        f"Validation games:  {len(X_validation)}"
    )

    print(
        f"Test games:        {len(X_test)}"
    )

    print("")

    print(
        f"Model features:    {len(FEATURES)}"
    )

    # =====================================================
    # MODELS
    # =====================================================

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

    # =====================================================
    # VALIDATION
    # =====================================================

    print("")
    print(
        "========================================"
    )

    print(
        "2023 VALIDATION"
    )

    print(
        "========================================"
    )

    validation_results = []

    for name, model in models:

        result = evaluate_model(
            name,
            model,
            X_train,
            X_validation,
            y_train,
            y_validation,
            "2023 Validation"
        )

        validation_results.append(
            result
        )

    # =====================================================
    # VALIDATION SUMMARY
    # =====================================================

    print("")
    print(
        "========================================"
    )

    print(
        "2023 VALIDATION SUMMARY"
    )

    print(
        "========================================"
    )

    print("")

    print(
        f"{'Model':<25}"
        f"{'Accuracy':>12}"
        f"{'Log Loss':>12}"
        f"{'ROC-AUC':>12}"
        f"{'Brier':>12}"
    )

    print(
        "-" * 73
    )

    for result in validation_results:

        print(
            f"{result['name']:<25}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['log_loss']:>12.4f}"
            f"{result['roc_auc']:>12.4f}"
            f"{result['brier']:>12.4f}"
        )

    # =====================================================
    # SELECT BEST MODEL
    # =====================================================

    best_validation = min(
        validation_results,
        key=lambda x: x["log_loss"]
    )

    print("")

    print(
        "Best validation model by Log Loss:"
    )

    print(
        f"  {best_validation['name']}"
    )

    print(
        f"  Log Loss: {best_validation['log_loss']:.4f}"
    )

    # =====================================================
    # FINAL TEST
    # =====================================================

    print("")
    print(
        "========================================"
    )

    print(
        "2024 FINAL TEST"
    )

    print(
        "========================================"
    )

    final_results = []

    for name, model in models:

        result = evaluate_model(
            name,
            model,
            X_train + X_validation,
            X_test,
            y_train + y_validation,
            y_test,
            "2024 Final Test"
        )

        final_results.append(
            result
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("")
    print(
        "========================================"
    )

    print(
        "2024 FINAL MODEL COMPARISON"
    )

    print(
        "========================================"
    )

    print("")

    print(
        f"{'Model':<25}"
        f"{'Accuracy':>12}"
        f"{'Log Loss':>12}"
        f"{'ROC-AUC':>12}"
        f"{'Brier':>12}"
    )

    print(
        "-" * 73
    )

    for result in final_results:

        print(
            f"{result['name']:<25}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['log_loss']:>12.4f}"
            f"{result['roc_auc']:>12.4f}"
            f"{result['brier']:>12.4f}"
        )

    # =====================================================
    # BEST FINAL MODELS
    # =====================================================

    best_accuracy = max(
        final_results,
        key=lambda x: x["accuracy"]
    )

    best_log_loss = min(
        final_results,
        key=lambda x: x["log_loss"]
    )

    best_roc_auc = max(
        final_results,
        key=lambda x: x["roc_auc"]
    )

    best_brier = min(
        final_results,
        key=lambda x: x["brier"]
    )

    print("")

    print(
        "Best 2024 Accuracy:"
    )

    print(
        f"  {best_accuracy['name']}"
        f" ({best_accuracy['accuracy'] * 100:.2f}%)"
    )

    print("")

    print(
        "Best 2024 Log Loss:"
    )

    print(
        f"  {best_log_loss['name']}"
        f" ({best_log_loss['log_loss']:.4f})"
    )

    print("")

    print(
        "Best 2024 ROC-AUC:"
    )

    print(
        f"  {best_roc_auc['name']}"
        f" ({best_roc_auc['roc_auc']:.4f})"
    )

    print("")

    print(
        "Best 2024 Brier:"
    )

    print(
        f"  {best_brier['name']}"
        f" ({best_brier['brier']:.4f})"
    )

    # =====================================================
    # SAVE BEST FINAL MODEL
    # =====================================================

    output_dir = Path(
        "models"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_log_loss["model"],
        MODEL_FILE
    )

    print("")

    print(
        f"Best model saved to: {MODEL_FILE}"
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