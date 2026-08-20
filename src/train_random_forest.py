import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix
)

print("=" * 40)
print("MLB RANDOM FOREST")
print("CHRONOLOGICAL SEASON SPLIT")
print("=" * 40)

# Load data
print("Loading feature data...")
df = pd.read_csv("data/processed/mlb_multi_season_features.csv")

print(f"Games loaded: {len(df)}")

# Features to exclude
exclude = [
    "Season",
    "Date",
    "Home",
    "Away",
    "GameNumber",
    "MatchKey",
    "HomeWon"
]

features = [c for c in df.columns if c not in exclude]

X = df[features]
y = df["HomeWon"]

# Chronological split
train = df[df["Season"] == 2022]
validation = df[df["Season"] == 2023]
test = df[df["Season"] == 2024]

X_train = train[features]
y_train = train["HomeWon"]

X_val = validation[features]
y_val = validation["HomeWon"]

X_test = test[features]
y_test = test["HomeWon"]

print()
print("Training season:   2022")
print("Validation season: 2023")
print("Test season:       2024")

print()
print(f"Training games:    {len(train)}")
print(f"Validation games:  {len(validation)}")
print(f"Test games:        {len(test)}")

print()
print(f"Model features: {len(features)}")

# Train model
print()
print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight=None
)

model.fit(X_train, y_train)

print("Model trained.")

# Validation
print()
print("=" * 40)
print("2023 VALIDATION RESULTS")
print("=" * 40)

val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]

print(f"Accuracy:  {accuracy_score(y_val, val_pred):.4f}")
print(f"Accuracy %: {accuracy_score(y_val, val_pred) * 100:.2f}%")
print(f"Log Loss:  {log_loss(y_val, val_prob):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, val_prob):.4f}")
print(f"Brier:     {brier_score_loss(y_val, val_prob):.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_val,
        val_pred,
        target_names=["Away Win", "Home Win"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))

# Final test
print()
print("=" * 40)
print("2024 FINAL TEST RESULTS")
print("=" * 40)

test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, test_pred)
logloss = log_loss(y_test, test_prob)
auc = roc_auc_score(y_test, test_prob)
brier = brier_score_loss(y_test, test_prob)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Accuracy %: {accuracy * 100:.2f}%")
print(f"Log Loss:  {logloss:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"Brier:     {brier:.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Away Win", "Home Win"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print()
print("=" * 40)
print("TOP 15 FEATURES")
print("=" * 40)

print(importance.head(15).to_string(index=False))

# Save model
joblib.dump(model, "models/mlb_random_forest_model.pkl")

print()
print("=" * 40)
print("RANDOM FOREST TRAINING COMPLETE")
print("=" * 40)

print("Model saved to: models/mlb_random_forest_model.pkl")

print()
print("Final 2024 metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Log Loss: {logloss:.4f}")
print(f"  ROC-AUC:  {auc:.4f}")
print(f"  Brier:    {brier:.4f}")