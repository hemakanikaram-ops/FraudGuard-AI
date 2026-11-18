
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

project_path = Path(__file__).resolve().parents[1]

df = pd.read_csv(project_path / "data" / "creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# --------------------------------------------------
# 4. Scale Time and Amount
# --------------------------------------------------

columns_to_scale = ["Time", "Amount"]

scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[columns_to_scale] = scaler.fit_transform(
    X_train[columns_to_scale]
)

X_test[columns_to_scale] = scaler.transform(
    X_test[columns_to_scale]
)

print("\nFeature scaling completed!")


# --------------------------------------------------
# 5. Create Random Forest model
# --------------------------------------------------

rf = RandomForestClassifier(
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 6. Define hyperparameter combinations
# --------------------------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}


# --------------------------------------------------
# 7. Grid Search with Cross Validation
# --------------------------------------------------

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="average_precision",
    cv=3,
    n_jobs=-1,
    verbose=1
)


# --------------------------------------------------
# 8. Train and tune the model
# --------------------------------------------------

print("\nStarting Hyperparameter Tuning...")

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-validation Average Precision:")
print(grid_search.best_score_)


# --------------------------------------------------
# 9. Evaluate the best model
# --------------------------------------------------

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)

probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

test_ap = average_precision_score(y_test, probabilities)

print("\nTest Average Precision:", test_ap)


# --------------------------------------------------
# 10. Save model and scaler using Joblib
# --------------------------------------------------

model_path = project_path / "models"

model_path.mkdir(exist_ok=True)

# Save trained Random Forest model
joblib.dump(
    best_model,
    model_path / "fraud_detection_model.pkl"
)

# Save the fitted scaler
joblib.dump(
    scaler,
    model_path / "scaler.pkl"
)

print("\nModel and scaler saved successfully!")

print("Model location:", model_path / "fraud_detection_model.pkl")
print("Scaler location:", model_path / "scaler.pkl")
