import pandas as pd

# Load cleaned data
df = pd.read_csv("churn_cleaned.csv")

# ---- Step 1: Identify text (categorical) columns ----
print("----- Categorical Columns -----")
categorical_cols = df.select_dtypes(include="object").columns.tolist()
print(categorical_cols)

# ---- Step 2: One-Hot Encoding (convert text into 0/1 columns) ----
# drop_first=True avoids redundant duplicate columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ---- Step 3: Convert boolean columns to integers (0/1) ----
df_encoded = df_encoded.astype(int, errors="ignore")

# ---- Step 4: Compare shape before and after encoding ----
print("\n----- Shape Before Encoding -----")
print(df.shape)
print("----- Shape After Encoding -----")
print(df_encoded.shape)

# ---- Step 5: Display final columns ----
print("\n----- Final Columns -----")
print(df_encoded.columns.tolist())

# ---- Step 6: Save the feature-engineered dataset ----
df_encoded.to_csv("churn_final.csv", index=False)

print("\n✅ Feature engineering complete! 'churn_final.csv' file created.")