import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---- Step 1: Load data ----
df = pd.read_csv("churn_final.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---- Step 2: Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Step 3: Train the final model (Random Forest) ----
final_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
final_model.fit(X_train, y_train)

# ---- Step 4: Save the trained model to a file ----
joblib.dump(final_model, "churn_model.pkl")

# ---- Step 5: Save column names (needed for future predictions) ----
model_columns = X.columns.tolist()
joblib.dump(model_columns, "model_columns.pkl")

print("✅ Model saved successfully as 'churn_model.pkl'")
print("✅ Column info saved as 'model_columns.pkl'")
print(f"Total features used: {len(model_columns)}")