import pandas as pd

# Load the dataset
df = pd.read_csv("churn_data.csv")

# Display first 5 rows
print("----- First 5 Rows -----")
print(df.head())

# Dataset shape (rows, columns)
print("\n----- Dataset Shape -----")
print(df.shape)

# All column names
print("\n----- Column Names -----")
print(df.columns.tolist())

# Data types and missing values info
print("\n----- Dataset Info -----")
print(df.info())