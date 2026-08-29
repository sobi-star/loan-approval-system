"""
main.py - FastAPI Backend with Supabase REST Client & Auth
Loan Approval Prediction System
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
from supabase import create_client, Client

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Supabase Credentials (Direct HTTPS REST API - Serverless Safe)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://vdqdmxgcxnatgxlyutxr.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZkcWRteGdjeG5hdGd4bHl1dHhyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA1NjQ5MzcsImV4cCI6MjA1NjE0MDkzN30.YOUR_SUPABASE_KEY_HERE")

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "super-secret-key-change-this-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(title="Loan Approval API with Auth & Supabase")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Supabase Client Setup ----------
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

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

    supabase = get_supabase()
    res = supabase.table("users").select("*").eq("username", username).execute()
    if not res.data:
        raise credentials_exception
    return res.data[0]

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "Loan Approval API with Supabase REST is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/signup")
def signup(user_data: UserRegister):
    try:
        supabase = get_supabase()
        
        existing_user = supabase.table("users").select("id").eq("username", user_data.username).execute()
        if existing_user.data:
            raise HTTPException(status_code=400, detail="Username already exists.")

        hashed_pwd = get_password_hash(user_data.password)
        res = supabase.table("users").insert({
            "username": user_data.username,
            "password_hash": hashed_pwd
        }).execute()

        return {"message": "User registered successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        supabase = get_supabase()
        res = supabase.table("users").select("*").eq("username", form_data.username).execute()
        
        if not res.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = res.data[0]
        if not verify_password(form_data.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token = create_access_token(data={"sub": user["username"]})
        return {"access_token": access_token, "token_type": "bearer", "username": user["username"]}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

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

        supabase = get_supabase()
        res = supabase.table("predictions_history").insert({
            "user_id": current_user["id"],
            "applicant_income": application.ApplicantIncome,
            "coapplicant_income": application.CoapplicantIncome,
            "loan_amount": application.LoanAmount,
            "loan_amount_term": application.Loan_Amount_Term,
            "credit_history": application.Credit_History,
            "gender": application.Gender,
            "married": application.Married,
            "dependents": application.Dependents,
            "education": application.Education,
            "self_employed": application.Self_Employed,
            "property_area": application.Property_Area,
            "prediction": prediction,
            "probability": approval_probability,
            "timestamp": timestamp
        }).execute()

        # FIXED: Safely retrieve the newly inserted row ID regardless of supabase-py response version behavior
        record_id = None
        if res.data and len(res.data) > 0:
            record_id = res.data[0].get("id")

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
        supabase = get_supabase()
        res = supabase.table("predictions_history").select("*").eq("user_id", current_user["id"]).order("id", desc=True).execute()

        formatted_records = []
        for r in res.data:
            formatted_records.append({
                "id": r.get("id"),
                "ApplicantIncome": r.get("applicant_income"),
                "CoapplicantIncome": r.get("coapplicant_income"),
                "LoanAmount": r.get("loan_amount"),
                "Loan_Amount_Term": r.get("loan_amount_term"),
                "Credit_History": r.get("credit_history"),
                "Gender": r.get("gender"),
                "Married": r.get("married"),
                "Dependents": r.get("dependents"),
                "Education": r.get("education"),
                "Self_Employed": r.get("self_employed"),
                "Property_Area": r.get("property_area"),
                "Prediction": r.get("prediction"),
                "Approval_Probability": r.get("probability"),
                "Timestamp": r.get("timestamp")
            })

        return {"count": len(formatted_records), "records": formatted_records}
    posts_exc = except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch history: {exc}")
