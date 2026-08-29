"""
main.py - FastAPI Backend
Loan Approval Prediction System

Responsibilities:
- Load the trained model from model.pkl
- Accept prediction requests through POST /predict
- Save requests/results to SQLite
- Return prediction history through GET /history
"""

from datetime import datetime
from pathlib import Path
import sqlite3
import tempfile

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Vercel serverless ke liye writable temporary directory
DB_PATH = Path(tempfile.gettempdir()) / "loan_predictions.db"

app = FastAPI(
    title="Loan Approval Prediction API",
    description="FastAPI backend for the Loan Approval Prediction System",
    version="1.0.0",
)


# ---------- Request schema ----------
class LoanApplication(BaseModel):
    ApplicantIncome: float = Field(ge=0)
    CoapplicantIncome: float = Field(ge=0)
    LoanAmount: float = Field(gt=0)
    Loan_Amount_Term: int = Field(gt=0)
    Credit_History: float = Field(ge=0, le=1)
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    Property_Area: str


# ---------- Model ----------
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl was not found. Run `python train_model.py` first."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]

except Exception as exc:
    model = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None


# ---------- Database ----------
def get_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    """Create the prediction history table if it does not exist."""
    try:
        with get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    applicant_income REAL NOT NULL,
                    coapplicant_income REAL NOT NULL,
                    loan_amount REAL NOT NULL,
                    loan_amount_term INTEGER NOT NULL,
                    credit_history REAL NOT NULL,
                    gender TEXT NOT NULL,
                    married TEXT NOT NULL,
                    dependents TEXT NOT NULL,
                    education TEXT NOT NULL,
                    self_employed TEXT NOT NULL,
                    property_area TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    probability REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()
    except Exception as exc:
        print(f"Database init warning: {exc}")


@app.on_event("startup")
def startup_event() -> None:
    """Initialize the database when FastAPI starts."""
    init_database()


# ---------- API endpoints ----------
@app.get("/")
def root():
    return {
        "message": "Loan Approval Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Simple backend health check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(application: LoanApplication):
    """Predict loan approval, save the request/result, and return JSON."""
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model could not be loaded: {MODEL_LOAD_ERROR}",
        )

    try:
        input_data = pd.DataFrame([application.model_dump()])

        prediction_code = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)
        approved_index = classes.index("Y")
        approval_probability = float(probabilities[approved_index])
        rejection_probability = 1.0 - approval_probability

        prediction = "Approved" if prediction_code == "Y" else "Rejected"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record_id = None

        # Safe DB insertion block (app won't crash if environment is read-only)
        try:
            init_database()
            with get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO predictions_history (
                        applicant_income,
                        coapplicant_income,
                        loan_amount,
                        loan_amount_term,
                        credit_history,
                        gender,
                        married,
                        dependents,
                        education,
                        self_employed,
                        property_area,
                        prediction,
                        probability,
                        timestamp
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        application.ApplicantIncome,
                        application.CoapplicantIncome,
                        application.LoanAmount,
                        application.Loan_Amount_Term,
                        application.Credit_History,
                        application.Gender,
                        application.Married,
                        application.Dependents,
                        application.Education,
                        application.Self_Employed,
                        application.Property_Area,
                        prediction,
                        approval_probability,
                        timestamp,
                    ),
                )
                conn.commit()
                record_id = cursor.lastrowid
        except Exception as db_exc:
            print(f"Skipping database write due to environment restrictions: {db_exc}")

        return {
            "id": record_id,
            "prediction": prediction,
            "approval_probability": approval_probability,
            "rejection_probability": rejection_probability,
            "timestamp": timestamp,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.get("/history")
def history():
    """Return all prediction records, newest first."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    applicant_income AS ApplicantIncome,
                    coapplicant_income AS CoapplicantIncome,
                    loan_amount AS LoanAmount,
                    loan_amount_term AS Loan_Amount_Term,
                    credit_history AS Credit_History,
                    gender AS Gender,
                    married AS Married,
                    dependents AS Dependents,
                    education AS Education,
                    self_employed AS Self_Employed,
                    property_area AS Property_Area,
                    prediction AS Prediction,
                    probability AS Approval_Probability,
                    timestamp AS Timestamp
                FROM predictions_history
                ORDER BY id DESC
                """
            ).fetchall()

        return {
            "count": len(rows),
            "records": [dict(row) for row in rows],
        }

    except Exception as exc:
        return {
            "count": 0,
            "records": [],
        }
