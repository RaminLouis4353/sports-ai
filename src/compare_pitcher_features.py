import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

print("=" * 50)
print("PITCHER FEATURE VALUE TEST")
print("=" * 50)

df = pd.read_csv("data/processed/mlb_multi_season_features.csv")

exclude = ["Season", "Date", "Home", "Away", "GameNumber", "MatchKey", "HomeWon"]

all_features = [c for c in df.columns if c not in exclude]
pitcher_features = [c for c in all_features if "Pitcher" in c]
team_features = [c for c in all_features if "Pitcher" not in c]

train = df[df.Season == 2022]
test = df[df.Season == 2024]

y_train = train["HomeWon"]
y_test = test["HomeWon"]


def train_and_test(features, name):
    print()
    print("-" * 50)
    print(name)
    print("-" * 50)
    print("Features:", len(features))

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(train[features], y_train)

    pred = model.predict(test[features])
    prob = model.predict_proba(test[features])[:, 1]

    accuracy = accuracy_score(y_test, pred)
    logloss = log_loss(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    brier = brier_score_loss(y_test, prob)

    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Log Loss:  {logloss:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Brier:     {brier:.4f}")

    return accuracy, logloss, auc, brier


print()
print("Training on 2022")
print("Testing on 2024")
print("2023 intentionally skipped for this experiment.")

team_results = train_and_test(
    team_features,
    "MODEL A: TEAM FEATURES ONLY"
)

all_results = train_and_test(
    all_features,
    "MODEL B: TEAM + PITCHER FEATURES"
)

print()
print("=" * 50)
print("COMPARISON")
print("=" * 50)

print(f"Team-only accuracy:        {team_results[0]:.4f}")
print(f"Team + pitcher accuracy:   {all_results[0]:.4f}")
print()

print(f"Team-only ROC-AUC:         {team_results[2]:.4f}")
print(f"Team + pitcher ROC-AUC:    {all_results[2]:.4f}")
print()

print(f"Team-only Log Loss:        {team_results[1]:.4f}")
print(f"Team + pitcher Log Loss:   {all_results[1]:.4f}")
print()

print(f"Team-only Brier:           {team_results[3]:.4f}")
print(f"Team + pitcher Brier:      {all_results[3]:.4f}")

print()
print("=" * 50)
print("PITCHER FEATURE TEST COMPLETE")
print("=" * 50)