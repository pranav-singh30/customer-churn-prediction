import pandas as pd

# Load the raw dataset
df = pd.read_csv("churn_data.csv")

# ---- Step 1: Drop customerID column (not useful for prediction) ----
df.drop("customerID", axis=1, inplace=True)

# ---- Step 2: Convert TotalCharges to numeric ----
# Some rows have blank spaces (" ") instead of numbers, which become NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# ---- Step 3: Check missing values ----
print("----- Missing Values (before fixing) -----")
print(df.isnull().sum())

# ---- Step 4: Fill missing TotalCharges with median ----
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ---- Step 5: Confirm no missing values remain ----
print("\n----- Missing Values (after fixing) -----")
print(df.isnull().sum())

# ---- Step 6: Convert target column 'Churn' from Yes/No to 1/0 ----
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---- Step 7: Save cleaned data to a new CSV file ----
df.to_csv("churn_cleaned.csv", index=False)

print("\n✅ Data cleaning complete! 'churn_cleaned.csv' file created.")
print("Final shape:", df.shape)