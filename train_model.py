import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---- Step 1: Load final dataset ----
df = pd.read_csv("churn_final.csv")

# ---- Step 2: Separate features (X) and target (y) ----
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---- Step 3: Split data into training and testing sets ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# ---- Step 4: Train the model ----
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---- Step 5: Predict on test data ----
y_pred = model.predict(X_test)

# ---- Step 6: Evaluate accuracy ----
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Trained Successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%")