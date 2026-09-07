import pandas as pd
import joblib

# ---- Step 1: Load saved model and columns ----
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---- Step 2: Define a new customer's data (you can change these values) ----
new_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 425.0
}

# ---- Step 3: Convert to DataFrame ----
new_df = pd.DataFrame([new_customer])

# ---- Step 4: Apply the same One-Hot Encoding used during training ----
categorical_cols = new_df.select_dtypes(include="object").columns.tolist()
new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)

# ---- Step 5: Add any missing columns (set to 0) ----
for col in model_columns:
    if col not in new_df_encoded.columns:
        new_df_encoded[col] = 0

# ---- Step 6: Keep column order consistent with training data ----
new_df_encoded = new_df_encoded[model_columns]

# ---- Step 7: Make prediction ----
prediction = model.predict(new_df_encoded)[0]
probability = model.predict_proba(new_df_encoded)[0][1]

# ---- Step 8: Display result ----
print("========== PREDICTION RESULT ==========")
if prediction == 1:
    print("⚠️  This customer is likely to CHURN")
else:
    print("✅ This customer is likely to STAY")

print(f"Churn Probability: {probability*100:.2f}%")