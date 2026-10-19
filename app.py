import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "customer_churn_bundle.joblib"

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found: customer_churn_bundle.joblib. "
            "Place the saved model bundle beside app.py."
        )
    return joblib.load(MODEL_PATH)

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a customer is likely to leave the company "
    "using a trained machine learning model."
)

try:
    bundle = load_model_bundle()
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.sidebar.header("About this project")
st.sidebar.write(
    "This application uses a trained Random Forest pipeline "
    "to estimate customer churn risk."
)
st.sidebar.write(f"Classification threshold: {threshold:.2f}")
st.sidebar.caption(
    "Predictions are estimates, not guarantees of customer behavior."
)

st.header("Enter Customer Details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        if phone_service == "No":
            multiple_lines = "No phone service"
            st.info("Multiple Lines: No phone service")
        else:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        if internet_service == "No":
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
            st.info("Internet-dependent services set to 'No internet service'.")
        else:
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=float(monthly_charges * tenure), step=1.0)

    submitted = st.form_submit_button("Predict Customer Churn", type="primary", use_container_width=True)

if submitted:
    average_monthly_spend = total_charges / tenure if tenure > 0 else total_charges
    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 36:
        tenure_group = "Medium"
    else:
        tenure_group = "Long-term"

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "AverageMonthlySpend": average_monthly_spend,
        "TenureGroup": tenure_group
    }])

    st.subheader("Prediction Results")
    try:
        churn_probability = float(model.predict_proba(customer)[0, 1])
        prediction = int(churn_probability >= threshold)
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Estimated Churn Probability", f"{churn_probability:.1%}")
            st.progress(min(max(churn_probability, 0.0), 1.0))
        with result_col2:
            if prediction == 1:
                st.error("⚠️ Prediction: Likely to Churn")
                st.write("Consider reviewing the customer's experience and whether a retention offer is appropriate.")
            else:
                st.success("✅ Prediction: Likely to Stay")
                st.write("The model did not flag this customer as likely to churn at the selected threshold.")
        with st.expander("View submitted customer information"):
            st.dataframe(customer, use_container_width=True)
        st.caption(f"Classification threshold used: {threshold:.2f}.")
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption("Customer Churn Prediction | Machine Learning Portfolio Project")
