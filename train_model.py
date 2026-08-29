"""
train_model.py
Loan Approval Prediction System
Generates a reproducible synthetic dataset, preprocesses it, trains a model,
evaluates it, and saves the complete pipeline to model.pkl.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
RANDOM_STATE = 42


def generate_dataset(n_samples: int = 1500) -> pd.DataFrame:
    """Generate a realistic synthetic loan-approval dataset."""
    rng = np.random.default_rng(RANDOM_STATE)

    gender = rng.choice(["Male", "Female"], n_samples, p=[0.80, 0.20])
    married = rng.choice(["Yes", "No"], n_samples, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], n_samples, p=[0.55, 0.18, 0.17, 0.10])
    education = rng.choice(["Graduate", "Not Graduate"], n_samples, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], n_samples, p=[0.14, 0.86])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n_samples, p=[0.35, 0.40, 0.25])

    applicant_income = np.clip(rng.lognormal(mean=8.15, sigma=0.55, size=n_samples), 1500, 25000).round().astype(int)
    coapplicant_income = np.where(
        married == "Yes",
        np.clip(rng.lognormal(mean=7.0, sigma=0.8, size=n_samples) - 800, 0, 12000),
        0,
    ).round().astype(int)

    loan_amount = np.clip(
        applicant_income * rng.uniform(0.10, 0.28, n_samples)
        + coapplicant_income * rng.uniform(0.03, 0.12, n_samples)
        + rng.normal(0, 55, n_samples),
        30,
        700,
    ).round()

    loan_term = rng.choice([120, 180, 240, 300, 360, 480], n_samples,
                            p=[0.03, 0.05, 0.08, 0.08, 0.70, 0.06])
    credit_history = rng.choice([0.0, 1.0], n_samples, p=[0.18, 0.82])

    # Create approval probability from sensible business-like relationships.
    income_total = applicant_income + coapplicant_income
    debt_burden = loan_amount / np.maximum(income_total / 1000, 1)
    score = (
        2.7 * credit_history
        + 0.000035 * income_total
        + 0.25 * (education == "Graduate")
        + 0.18 * (property_area == "Semiurban")
        - 0.28 * (property_area == "Rural")
        - 0.0018 * debt_burden
        - 0.18 * (self_employed == "Yes")
        + rng.normal(0, 0.65, n_samples)
    )

    probability = 1 / (1 + np.exp(-(score - 1.8)))
    loan_status = np.where(rng.random(n_samples) < probability, "Y", "N")

    df = pd.DataFrame({
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    })

    # Intentionally add a small amount of missing data to test preprocessing.
    missing_cols = ["Gender", "Married", "Dependents", "LoanAmount", "Credit_History"]
    for col in missing_cols:
        mask = rng.random(n_samples) < 0.025
        df.loc[mask, col] = np.nan

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing + classification pipeline."""
    numeric_features = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
    categorical_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def main() -> None:
    print("Generating dataset...")
    df = generate_dataset()

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(X_train)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index("Y")]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score((y_test == "Y").astype(int), probabilities)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    artifact = {
        "model": pipeline,
        "feature_names": list(X.columns),
        "classes": list(pipeline.classes_),
        "model_name": "Logistic Regression",
        "random_state": RANDOM_STATE,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise
