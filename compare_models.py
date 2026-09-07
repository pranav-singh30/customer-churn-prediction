import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("churn_final.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Model 1: Logistic Regression ----
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("========== LOGISTIC REGRESSION ==========")
print("Accuracy:", f"{accuracy_score(y_test, log_pred)*100:.2f}%")
print(classification_report(y_test, log_pred, target_names=["No Churn", "Churn"]))

# ---- Model 2: Random Forest ----
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",   # Helps handle imbalanced classes
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n========== RANDOM FOREST ==========")
print("Accuracy:", f"{accuracy_score(y_test, rf_pred)*100:.2f}%")
print(classification_report(y_test, rf_pred, target_names=["No Churn", "Churn"]))