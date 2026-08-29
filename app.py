"""
app.py - Streamlit Frontend with Authentication
Loan Approval Prediction System
"""

import requests
import pandas as pd
import streamlit as st

API_URL = "https://loan-approval-system-lemon.vercel.app"

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="wide",
)

# ---------- Session State Initialization ----------
if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.25rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1rem;
    }
    .approved {
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #166534;
    }
    .rejected {
        background: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- API helpers ----------
def get_auth_headers() -> dict:
    """Return bearer token headers if logged in."""
    if st.session_state["token"]:
        return {"Authorization": f"Bearer {st.session_state['token']}"}
    return {}

def predict_via_api(payload: dict) -> dict:
    """Send application data to FastAPI /predict."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers=get_auth_headers(),
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("Cannot connect to FastAPI backend.") from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The FastAPI backend took too long to respond.") from exc

    except requests.exceptions.HTTPError as exc:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Backend error: {detail}") from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc


def get_history_from_api() -> pd.DataFrame:
    """Fetch user prediction history from FastAPI /history."""
    try:
        response = requests.get(
            f"{API_URL}/history",
            headers=get_auth_headers(),
            timeout=15,
        )
        response.raise_for_status()

        data = response.json()
        records = data.get("records", [])

        if not records:
            return pd.DataFrame()

        history_df = pd.DataFrame(records)

        if "Approval_Probability" in history_df.columns:
            history_df["Approval_Probability"] = (
                history_df["Approval_Probability"] * 100
            ).round(2).astype(str) + "%"

        return history_df

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("Cannot connect to FastAPI backend.") from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The FastAPI history request timed out.") from exc

    except requests.exceptions.HTTPError as exc:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Backend error: {detail}") from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"History request failed: {exc}") from exc


# ---------- Header ----------
st.markdown(
    '<div class="main-title">💳 Loan Approval Prediction System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Streamlit Frontend + FastAPI Backend + Supabase PostgreSQL</div>',
    unsafe_allow_html=True,
)

# ---------- Authentication Guard ----------
if not st.session_state["token"]:
    st.info("🔐 Please Log In or Create an Account to access prediction features.")
    
    auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with auth_tab1:
        st.subheader("Login to Your Account")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In", type="primary", use_container_width=True):
            try:
                res = requests.post(
                    f"{API_URL}/login", 
                    data={"username": login_user, "password": login_pass}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["token"] = data["access_token"]
                    st.session_state["username"] = data["username"]
                    st.success(f"Welcome back, {data['username']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            except Exception as e:
                st.error(f"Error connecting to server: {e}")

    with auth_tab2:
        st.subheader("Create a New Account")
        signup_user = st.text_input("Choose Username", key="signup_user")
        signup_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        if st.button("Sign Up", use_container_width=True):
            try:
                res = requests.post(
                    f"{API_URL}/signup", 
                    json={"username": signup_user, "password": signup_pass}
                )
                if res.status_code == 200:
                    st.success("Account created successfully! Please switch to Login tab to log in.")
                else:
                    st.error(res.json().get("detail", "Sign up failed."))
            except Exception as e:
                st.error(f"Error connecting to server: {e}")

else:
    # ---------- Main App Content (Logged In) ----------
    st.sidebar.markdown(f"👤 **Logged in as:** `{st.session_state['username']}`")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state["token"] = None
        st.session_state["username"] = None
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Applicant Information")

    with st.sidebar.form("loan_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000,
            step=500,
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=0,
            step=500,
        )
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1,
            value=150,
            step=10,
        )
        loan_term = st.selectbox(
            "Loan Amount Term (months)",
            [120, 180, 240, 300, 360, 480],
            index=4,
        )
        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x: "Good (1)" if x == 1.0 else "Poor (0)",
        )

        submitted = st.form_submit_button(
            "🔍 Predict Loan Approval Status",
            use_container_width=True,
        )

    # ---------- Dashboard ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("Applicant Income", f"{applicant_income:,.0f}")
    col2.metric("Loan Amount", f"{loan_amount:,.0f}")
    col3.metric(
        "Credit History",
        "Good" if credit_history == 1.0 else "Poor",
    )

    st.divider()
    st.subheader("Prediction")

    if submitted:
        payload = {
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
        }

        try:
            result = predict_via_api(payload)

            prediction = result["prediction"]
            approval_probability = float(result["approval_probability"])
            rejection_probability = float(result["rejection_probability"])

            if prediction == "Approved":
                st.markdown(
                    f"""
                    <div class="result-box approved">
                        <h2>✅ Loan Approved</h2>
                        <p>Estimated approval probability:
                        <strong>{approval_probability:.1%}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box rejected">
                        <h2>❌ Loan Rejected</h2>
                        <p>Estimated approval probability:
                        <strong>{approval_probability:.1%}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("Probability Breakdown")
            p1, p2 = st.columns(2)
            p1.metric("Approved", f"{approval_probability:.1%}")
            p2.metric("Rejected", f"{rejection_probability:.1%}")

            st.progress(
                approval_probability,
                text="Approval probability",
            )

            st.success(
                f"Prediction saved successfully. Record ID: {result['id']} "
                f"| {result['timestamp']}"
            )

            with st.expander("View submitted application"):
                st.dataframe(
                    pd.DataFrame([payload]),
                    use_container_width=True,
                    hide_index=True,
                )

        except RuntimeError as exc:
            st.error(str(exc))

    # ---------- Prediction History Sidebar Expander ----------
    with st.sidebar.expander("📊 View Prediction History"):
        try:
            history_df = get_history_from_api()

            if history_df.empty:
                st.info("No predictions saved yet for this account.")
            else:
                st.caption(f"Total predictions: {len(history_df)}")
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True,
                )

        except RuntimeError as exc:
            st.warning(str(exc))

    # ---------- Full History Main View ----------
    with st.expander("📋 Full Prediction History"):
        try:
            history_df = get_history_from_api()

            if history_df.empty:
                st.info("No predictions saved yet for this account.")
            else:
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True,
                )

        except RuntimeError as exc:
            st.error(str(exc))

st.caption(
    "Frontend: Streamlit | Backend: FastAPI | Database: Supabase (PostgreSQL) | "
    "Model: Scikit-Learn"
)
