import csv
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


INPUT_FILE = Path("data/processed/mlb_2024_features.csv")
MODEL_FILE = Path("models/mlb_logistic_model.pkl")

FEATURES = [
    "Home_WinPct",
    "Home_AvgRunsScored",
    "Home_AvgRunsAllowed",
    "Home_RecentWinPct",
    "Away_WinPct",
    "Away_AvgRunsScored",
    "Away_AvgRunsAllowed",
    "Away_RecentWinPct"
]


def load_data():

    print("Loading feature data...", flush=True)

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


def prepare_data(rows):

    X = []
    y = []

    for row in rows:

        features = []

        for feature in FEATURES:
            features.append(float(row[feature]))

        X.append(features)

        y.append(int(row["HomeWon"]))

    return X, y


def main():

    rows = load_data()

    X, y = prepare_data(rows)

    # Use the first 80% for training
    # and the final 20% for testing.
    #
    # This keeps the model evaluation
    # chronological instead of randomly
    # mixing past and future games.

    split_index = int(len(X) * 0.80)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    print("")
    print(f"Training games: {len(X_train)}")
    print(f"Testing games: {len(X_test)}")

    # -----------------------------------------
    # BUILD MODEL
    # -----------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    print("")
    print("Training model...", flush=True)

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model trained.",
        flush=True
    )

    # -----------------------------------------
    # PREDICTIONS
    # -----------------------------------------

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # -----------------------------------------
    # EVALUATION
    # -----------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    loss = log_loss(
        y_test,
        probabilities
    )

    print("")
    print("==============================")
    print("MODEL RESULTS")
    print("==============================")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy %: {accuracy * 100:.2f}%"
    )

    print(
        f"Log Loss: {loss:.4f}"
    )

    print("")
    print("Classification Report:")

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

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # -----------------------------------------
    # SAVE MODEL
    # -----------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("")
    print(
        f"Model saved to: {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()