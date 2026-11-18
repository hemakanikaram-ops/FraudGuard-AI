
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from pathlib import Path
import pandas as pd
import joblib

from backend.database import engine, get_db, Base
from backend.models import PredictionRecord


# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="FraudGuard AI",
    description="AI-Powered Credit Card Fraud Detection API",
    version="1.0"
)


# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_PATH / "models"


# --------------------------------------------------
# 3. Load saved ML model and scaler
# --------------------------------------------------

try:

    model = joblib.load(
        MODEL_PATH / "fraud_detection_model.pkl"
    )

    scaler = joblib.load(
        MODEL_PATH / "scaler.pkl"
    )

    print("ML model loaded successfully!")
    print("Scaler loaded successfully!")

except FileNotFoundError as error:

    raise RuntimeError(
        f"Saved model or scaler file not found: {error}"
    )


# --------------------------------------------------
# 4. Create database tables
# --------------------------------------------------

Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")


# --------------------------------------------------
# 5. Define transaction input structure
# --------------------------------------------------

class Transaction(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# --------------------------------------------------
# 6. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to FraudGuard AI API",
        "status": "Running"
    }


# --------------------------------------------------
# 7. Fraud prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_fraud(
    transaction: Transaction,
    db: Session = Depends(get_db)
):

    try:

        # Convert input into DataFrame
        transaction_data = pd.DataFrame(
            [transaction.model_dump()]
        )

        # Maintain original feature order
        feature_columns = [
            "Time",
            *[f"V{i}" for i in range(1, 29)],
            "Amount"
        ]

        transaction_data = transaction_data[
            feature_columns
        ]

        # Scale Time and Amount
        columns_to_scale = ["Time", "Amount"]

        transaction_data[columns_to_scale] = (
            scaler.transform(
                transaction_data[columns_to_scale]
            )
        )

        # Predict fraud probability
        probability = float(
            model.predict_proba(
                transaction_data
            )[0][1]
        )

        # Predict transaction class
        prediction = int(
            model.predict(transaction_data)[0]
        )

        # Generate result
        result = (
            "Potential Fraud"
            if prediction == 1
            else "Genuine Transaction"
        )

        # --------------------------------------------------
        # 8. Save prediction into SQLite database
        # --------------------------------------------------

        record = PredictionRecord(

            transaction_time=transaction.Time,

            amount=transaction.Amount,

            fraud_probability=probability,

            prediction=prediction,

            result=result,

            transaction_features=transaction.model_dump()
        )

        db.add(record)

        db.commit()

        db.refresh(record)

        # --------------------------------------------------
        # 9. Return API response
        # --------------------------------------------------

        return {

            "transaction_id": record.id,

            "fraud_probability": round(
                probability, 4
            ),

            "prediction": prediction,

            "result": result,

            "message": "Prediction saved successfully"

        }

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

# --------------------------------------------------
# 10. Get all transaction history
# --------------------------------------------------

@app.get("/transactions")
def get_transaction_history(
    db: Session = Depends(get_db)
):

    try:

        records = (
            db.query(PredictionRecord)
            .order_by(PredictionRecord.id.desc())
            .all()
        )

        return [
            {
                "transaction_id": record.id,
                "transaction_time": record.transaction_time,
                "amount": record.amount,
                "fraud_probability": record.fraud_probability,
                "prediction": record.prediction,
                "result": record.result,
                "created_at": record.created_at
            }
            for record in records
        ]

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# 11. Get a specific transaction by ID
# --------------------------------------------------

@app.get("/transactions/{transaction_id}")
def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db)
):

    record = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.id == transaction_id)
        .first()
    )

    if record is None:

        raise HTTPException(
            status_code=404,
            detail="Transaction not found"
        )

    return {
        "transaction_id": record.id,
        "transaction_time": record.transaction_time,
        "amount": record.amount,
        "fraud_probability": record.fraud_probability,
        "prediction": record.prediction,
        "result": record.result,
        "created_at": record.created_at
    }
