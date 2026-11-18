
import pandas as pd
from pathlib import Path

# Machine Learning tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


# ==========================================
# 1. PROJECT PATH AND DATASET LOADING
# ==========================================

project_path = Path(__file__).resolve().parents[1]

file_path = project_path / "data" / "creditcard.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)

y = df["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 4. FEATURE SCALING
# ==========================================

X_train_scaled = X_train.copy()

X_test_scaled = X_test.copy()

columns_to_scale = ["Time", "Amount"]

scaler = StandardScaler()

X_train_scaled[columns_to_scale] = scaler.fit_transform(
    X_train[columns_to_scale]
)

X_test_scaled[columns_to_scale] = scaler.transform(
    X_test[columns_to_scale]
)

print("\nFeature scaling completed!")


# ==========================================
# 5. MODEL STORAGE
# ==========================================

models_results = {}


# ==========================================
# 6. LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred, zero_division=0))

models_results["Logistic Regression"] = (lr_pred, lr_prob)


# ==========================================
# 7. RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)

rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred, zero_division=0))

models_results["Random Forest"] = (rf_pred, rf_prob)


# ==========================================
# 8. XGBOOST
# ==========================================

print("\nTraining XGBoost...")

scale_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)

xgb_pred = xgb_model.predict(X_test_scaled)

xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

print("\nXGBoost Report:")
print(classification_report(y_test, xgb_pred, zero_division=0))

models_results["XGBoost"] = (xgb_pred, xgb_prob)


# ==========================================
# 9. MODEL PERFORMANCE COMPARISON
# ==========================================

comparison = []

for model_name, (predictions, probabilities) in models_results.items():

    comparison.append({

        "Model": model_name,

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Average Precision": average_precision_score(
            y_test,
            probabilities
        )
    })


# Convert results into DataFrame

comparison_df = pd.DataFrame(comparison)

print("\n===================================")
print("     MODEL PERFORMANCE COMPARISON")
print("===================================")

print(comparison_df.to_string(index=False))


# ==========================================
# 10. SAVE COMPARISON REPORT
# ==========================================

docs_path = project_path / "docs"

docs_path.mkdir(exist_ok=True)

report_path = docs_path / "model_comparison.csv"

comparison_df.to_csv(
    report_path,
    index=False
)

print("\nComparison report saved successfully!")

print("Report location:", report_path)

print("\nAll three models completed successfully!")
