import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================================================
# FILE
# =========================================================

INPUT_FILE = Path(
    "data/processed/mlb_multi_season_features.csv"
)


# =========================================================
# TEAM FEATURES
# =========================================================

TEAM_FEATURES = [

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


# =========================================================
# PITCHER FEATURES
# =========================================================

PITCHER_FEATURES = [

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


# =========================================================
# MODEL
# =========================================================

def build_model():

    return Pipeline([

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


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    name,
    features,
    train,
    validation,
    test
):

    print("")
    print("----------------------------------------")
    print(f"{name}")
    print("----------------------------------------")

    X_train = train[features]
    y_train = train["HomeWon"]

    X_validation = validation[features]
    y_validation = validation["HomeWon"]

    X_test = test[features]
    y_test = test["HomeWon"]

    model = build_model()

    print(
        f"Features: {len(features)}"
    )

    print(
        "Training..."
    )

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_log_loss = log_loss(
        y_validation,
        validation_probabilities
    )

    validation_auc = roc_auc_score(
        y_validation,
        validation_probabilities
    )

    validation_brier = brier_score_loss(
        y_validation,
        validation_probabilities
    )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_log_loss = log_loss(
        y_test,
        test_probabilities
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    test_brier = brier_score_loss(
        y_test,
        test_probabilities
    )

    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print("")
    print("2023 Validation:")
    print(
        f"  Accuracy:  {validation_accuracy:.4f} "
        f"({validation_accuracy * 100:.2f}%)"
    )
    print(
        f"  Log Loss:  {validation_log_loss:.4f}"
    )
    print(
        f"  ROC-AUC:   {validation_auc:.4f}"
    )
    print(
        f"  Brier:     {validation_brier:.4f}"
    )

    print("")
    print("2024 Final Test:")
    print(
        f"  Accuracy:  {test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )
    print(
        f"  Log Loss:  {test_log_loss:.4f}"
    )
    print(
        f"  ROC-AUC:   {test_auc:.4f}"
    )
    print(
        f"  Brier:     {test_brier:.4f}"
    )

    return {
        "Model": name,
        "Features": len(features),

        "Validation Accuracy": validation_accuracy,
        "Validation Log Loss": validation_log_loss,
        "Validation ROC-AUC": validation_auc,
        "Validation Brier": validation_brier,

        "Test Accuracy": test_accuracy,
        "Test Log Loss": test_log_loss,
        "Test ROC-AUC": test_auc,
        "Test Brier": test_brier
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "========================================"
    )

    print(
        "MLB FEATURE ABLATION STUDY"
    )

    print(
        "TEAM vs PITCHER vs COMBINED"
    )

    print(
        "========================================"
    )

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    print("")
    print(
        "Loading feature data..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Games loaded: {len(df)}"
    )

    # -----------------------------------------------------
    # SEASON SPLIT
    # -----------------------------------------------------

    train = df[
        df["Season"] == 2022
    ].copy()

    validation = df[
        df["Season"] == 2023
    ].copy()

    test = df[
        df["Season"] == 2024
    ].copy()

    print("")
    print(
        f"Training games:    {len(train)}"
    )

    print(
        f"Validation games:  {len(validation)}"
    )

    print(
        f"Test games:        {len(test)}"
    )

    # -----------------------------------------------------
    # FEATURE SETS
    # -----------------------------------------------------

    combined_features = (
        TEAM_FEATURES +
        PITCHER_FEATURES
    )

    # -----------------------------------------------------
    # RUN EXPERIMENTS
    # -----------------------------------------------------

    results = []

    results.append(
        evaluate(
            "TEAM FEATURES ONLY",
            TEAM_FEATURES,
            train,
            validation,
            test
        )
    )

    results.append(
        evaluate(
            "PITCHER FEATURES ONLY",
            PITCHER_FEATURES,
            train,
            validation,
            test
        )
    )

    results.append(
        evaluate(
            "TEAM + PITCHER FEATURES",
            combined_features,
            train,
            validation,
            test
        )
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("")
    print(
        "========================================"
    )

    print(
        "ABLATION STUDY RESULTS"
    )

    print(
        "========================================"
    )

    print("")

    display_columns = [

        "Model",
        "Features",

        "Validation Accuracy",
        "Validation Log Loss",
        "Validation ROC-AUC",
        "Validation Brier",

        "Test Accuracy",
        "Test Log Loss",
        "Test ROC-AUC",
        "Test Brier"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "Validation Accuracy":
                    "{:.4f}".format,

                "Validation Log Loss":
                    "{:.4f}".format,

                "Validation ROC-AUC":
                    "{:.4f}".format,

                "Validation Brier":
                    "{:.4f}".format,

                "Test Accuracy":
                    "{:.4f}".format,

                "Test Log Loss":
                    "{:.4f}".format,

                "Test ROC-AUC":
                    "{:.4f}".format,

                "Test Brier":
                    "{:.4f}".format
            }
        )
    )

    # -----------------------------------------------------
    # BEST MODELS
    # -----------------------------------------------------

    best_accuracy = results_df.loc[
        results_df["Test Accuracy"].idxmax()
    ]

    best_log_loss = results_df.loc[
        results_df["Test Log Loss"].idxmin()
    ]

    best_auc = results_df.loc[
        results_df["Test ROC-AUC"].idxmax()
    ]

    best_brier = results_df.loc[
        results_df["Test Brier"].idxmin()
    ]

    print("")
    print(
        "========================================"
    )

    print(
        "BEST 2024 RESULTS"
    )

    print(
        "========================================"
    )

    print("")

    print(
        f"Best Accuracy: "
        f"{best_accuracy['Model']} "
        f"({best_accuracy['Test Accuracy'] * 100:.2f}%)"
    )

    print(
        f"Best Log Loss: "
        f"{best_log_loss['Model']} "
        f"({best_log_loss['Test Log Loss']:.4f})"
    )

    print(
        f"Best ROC-AUC: "
        f"{best_auc['Model']} "
        f"({best_auc['Test ROC-AUC']:.4f})"
    )

    print(
        f"Best Brier: "
        f"{best_brier['Model']} "
        f"({best_brier['Test Brier']:.4f})"
    )

    # -----------------------------------------------------
    # PITCHER IMPACT
    # -----------------------------------------------------

    team_result = results_df[
        results_df["Model"] ==
        "TEAM FEATURES ONLY"
    ].iloc[0]

    combined_result = results_df[
        results_df["Model"] ==
        "TEAM + PITCHER FEATURES"
    ].iloc[0]

    print("")
    print(
        "========================================"
    )

    print(
        "PITCHER FEATURE IMPACT"
    )

    print(
        "========================================"
    )

    print("")

    accuracy_change = (
        combined_result["Test Accuracy"]
        -
        team_result["Test Accuracy"]
    )

    log_loss_change = (
        combined_result["Test Log Loss"]
        -
        team_result["Test Log Loss"]
    )

    auc_change = (
        combined_result["Test ROC-AUC"]
        -
        team_result["Test ROC-AUC"]
    )

    brier_change = (
        combined_result["Test Brier"]
        -
        team_result["Test Brier"]
    )

    print(
        f"Accuracy change: "
        f"{accuracy_change:+.4f} "
        f"({accuracy_change * 100:+.2f} percentage points)"
    )

    print(
        f"Log Loss change: "
        f"{log_loss_change:+.4f}"
    )

    print(
        f"ROC-AUC change: "
        f"{auc_change:+.4f}"
    )

    print(
        f"Brier change: "
        f"{brier_change:+.4f}"
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    output_file = Path(
        "data/processed/feature_ablation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("")
    print(
        f"Results saved to: {output_file}"
    )

    print("")
    print(
        "========================================"
    )

    print(
        "ABLATION STUDY COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()