"""
main.py - FastAPI Backend with PostgreSQL (Supabase) & JWT Auth
Loan Approval Prediction System
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
from passlib.context import CryptContext
from jose import JWTError, jwt

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Environment Variable se Supabase Database URL fetch karein
DATABASE_URL = os.environ.get("DATABASE_URL")

# Security Configurations
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "super-secret-key-change-this-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 Hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(title="Loan Approval API with Auth & Supabase")

# ---------- Model Loading ----------
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl was not found.")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    MODEL_LOAD_ERROR = None
except Exception as exc:
    model = None
    MODEL_LOAD_ERROR = str(exc)

# ---------- Schemas ----------
class UserRegister(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

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

# ---------- Database Helpers ----------
def get_db():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not set in environment variables.")
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

def init_db():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            # Users Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # History Table (Linked to User ID)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    applicant_income REAL NOT NULL,
                    coapplicant_income REAL NOT NULL,
                    loan_amount REAL NOT NULL,
                    loan_amount_term INTEGER NOT NULL,
                    credit_history REAL NOT NULL,
                    gender VARCHAR(20) NOT NULL,
                    married VARCHAR(20) NOT NULL,
                    dependents VARCHAR(20) NOT NULL,
                    education VARCHAR(20) NOT NULL,
                    self_employed VARCHAR(20) NOT NULL,
                    property_area VARCHAR(20) NOT NULL,
                    prediction VARCHAR(20) NOT NULL,
                    probability REAL NOT NULL,
                    timestamp VARCHAR(50) NOT NULL
                );
            """)
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Init Exception: {e}")

@app.on_event("startup")
def startup():
    init_db()

# ---------- Auth Utilities ----------
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s;", (username,))
        user = cur.fetchone()
    conn.close()

    if user is None:
        raise credentials_exception
    return user

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "Loan Approval API with Supabase & Auth is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/signup")
def signup(user_data: UserRegister):
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE username = %s;", (user_data.username,))
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="Username already exists.")
        
        hashed_pwd = get_password_hash(user_data.password)
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id;",
            (user_data.username, hashed_pwd)
        )
        conn.commit()
    conn.close()
    return {"message": "User registered successfully"}

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s;", (form_data.username,))
        user = cur.fetchone()
    conn.close()

    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer", "username": user["username"]}

@app.post("/predict")
def predict(application: LoanApplication, current_user: dict = Depends(get_current_user)):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model error: {MODEL_LOAD_ERROR}")

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

        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions_history (
                    user_id, applicant_income, coapplicant_income, loan_amount,
                    loan_amount_term, credit_history, gender, married,
                    dependents, education, self_employed, property_area,
                    prediction, probability, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                current_user["id"], application.ApplicantIncome, application.CoapplicantIncome,
                application.LoanAmount, application.Loan_Amount_Term, application.Credit_History,
                application.Gender, application.Married, application.Dependents,
                application.Education, application.Self_Employed, application.Property_Area,
                prediction, approval_probability, timestamp
            ))
            record_id = cur.fetchone()["id"]
            conn.commit()
        conn.close()

        return {
            "id": record_id,
            "prediction": prediction,
            "approval_probability": approval_probability,
            "rejection_probability": rejection_probability,
            "timestamp": timestamp,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

@app.get("/history")
def history(current_user: dict = Depends(get_current_user)):
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id, applicant_income AS "ApplicantIncome", coapplicant_income AS "CoapplicantIncome",
                    loan_amount AS "LoanAmount", loan_amount_term AS "Loan_Amount_Term",
                    credit_history AS "Credit_History", gender AS "Gender", married AS "Married",
                    dependents AS "Dependents", education AS "Education", self_employed AS "Self_Employed",
                    property_area AS "Property_Area", prediction AS "Prediction",
                    probability AS "Approval_Probability", timestamp AS "Timestamp"
                FROM predictions_history
                WHERE user_id = %s
                ORDER BY id DESC;
            """, (current_user["id"],))
            rows = cur.fetchall()
        conn.close()

        return {"count": len(rows), "records": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch history: {exc}")
