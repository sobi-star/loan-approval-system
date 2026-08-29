# Loan Approval Prediction System
## Streamlit + FastAPI + SQLite

This version uses a proper Frontend-Backend architecture:

```text
Streamlit (app.py)
       |
       | HTTP POST /predict
       | HTTP GET /history
       v
FastAPI (main.py)
       |
       +---- model.pkl
       |
       +---- loan_predictions.db
```

## Files

```text
loan_approval_prediction_system/
├── main.py
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

`loan_predictions.db` is created automatically by the FastAPI backend when it starts.

## 1. Install dependencies

Open PowerShell in this project folder:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not recognized, use your Python executable:

```powershell
& "C:\Users\Sohaib\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install -r requirements.txt
```

FastAPI and Uvicorn can also be installed directly:

```powershell
python -m pip install fastapi "uvicorn[standard]"
```

## 2. Start the FastAPI backend

Open PowerShell in the project folder:

```powershell
python -m uvicorn main:app --reload
```

Or with the full Python path:

```powershell
& "C:\Users\Sohaib\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m uvicorn main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## 3. Start the Streamlit frontend

Open a SECOND PowerShell window in the same project folder:

```powershell
python -m streamlit run app.py
```

Or:

```powershell
& "C:\Users\Sohaib\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run app.py
```

The frontend normally opens at:

```text
http://localhost:8501
```

## Important

Run the backend FIRST, then the Streamlit frontend.

Every prediction follows this flow:

1. User enters data in Streamlit.
2. Streamlit sends JSON to `POST /predict`.
3. FastAPI loads the model and generates the prediction.
4. FastAPI saves input + result + timestamp to SQLite.
5. FastAPI returns JSON to Streamlit.
6. Streamlit displays the result.
7. Prediction history is fetched from `GET /history`.

The model is never loaded directly by `app.py`, and Streamlit never accesses SQLite directly.
