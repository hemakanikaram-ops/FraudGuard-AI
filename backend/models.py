
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from datetime import datetime

from backend.database import Base


class PredictionRecord(Base):

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    transaction_time = Column(Float)

    amount = Column(Float)

    fraud_probability = Column(Float)

    prediction = Column(Integer)

    result = Column(String)

    transaction_features = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
