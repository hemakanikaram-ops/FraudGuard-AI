
import pandas as pd
import joblib

from pathlib import Path

# 1. Project directory
project_path = Path(__file__).resolve().parents[1]

# 2. Load saved model and scaler
model = joblib.load(
    project_path / "models" / "fraud_detection_model.pkl"
)

scaler = joblib.load(
    project_path / "models" / "scaler.pkl"
)

print("Model and scaler loaded successfully!")

# 3. Load one sample transaction from dataset
df = pd.read_csv(
    project_path / "data" / "creditcard.csv"
)

# Select one transaction without its actual Class label
transaction = df.drop("Class", axis=1).iloc[[0]].copy()

# 4. Scale Time and Amount
columns_to_scale = ["Time", "Amount"]

transaction[columns_to_scale] = scaler.transform(
    transaction[columns_to_scale]
)

# 5. Predict fraud probability
fraud_probability = model.predict_proba(transaction)[0][1]

# 6. Predict class
prediction = model.predict(transaction)[0]

# 7. Display result
print("\nTransaction Prediction:")

print("Fraud Probability:", round(fraud_probability * 100, 2), "%")

if prediction == 1:
    print("Result: Potential Fraud")
else:
    print("Result: Genuine Transaction")