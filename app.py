import streamlit as st
import pandas as pd
import joblib

# ---- Page Setup ----
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

# ---- Load Model ----
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---- Title ----
st.title("📊 Customer Churn Prediction System")
st.write(" Enter Customer details to predict whether they will churn or not.")

st.divider()

# ---- Input Form ----
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

st.divider()

# ---- Predict Button ----
if st.button("🔍 Predict Churn", use_container_width=True):

    new_customer = {
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
        "TotalCharges": total_charges
    }

    new_df = pd.DataFrame([new_customer])
    categorical_cols = new_df.select_dtypes(include="object").columns.tolist()
    new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)

    for col in model_columns:
        if col not in new_df_encoded.columns:
            new_df_encoded[col] = 0
    new_df_encoded = new_df_encoded[model_columns]

    prediction = model.predict(new_df_encoded)[0]
    probability = model.predict_proba(new_df_encoded)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn! (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ This customer is likely to stay. (Risk: {probability*100:.2f}%)")

    st.progress(probability)