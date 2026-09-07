import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv("churn_final.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ---- Accuracy ----
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", f"{accuracy*100:.2f}%")

# ---- Confusion Matrix ----
print("\n----- Confusion Matrix -----")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("""
Format:
[[True Negative,  False Positive],
 [False Negative, True Positive]]
""")

# ---- Detailed classification report ----
print("----- Classification Report -----")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))