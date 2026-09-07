import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("churn_cleaned.csv")

# Set plot style
sns.set_style("whitegrid")

# ---- Graph 1: Churn distribution ----
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution (0 = No, 1 = Yes)")
plt.savefig("graph1_churn_distribution.png")
plt.close()

# ---- Graph 2: Contract type vs Churn ----
plt.figure(figsize=(7, 5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.savefig("graph2_contract_vs_churn.png")
plt.close()

# ---- Graph 3: Monthly Charges vs Churn ----
plt.figure(figsize=(7, 5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("graph3_monthlycharges_vs_churn.png")
plt.close()

# ---- Graph 4: Tenure vs Churn ----
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
plt.title("Tenure vs Churn")
plt.savefig("graph4_tenure_vs_churn.png")
plt.close()

print("✅ 4 graphs created! Check the PNG files in your folder:")
print("1. graph1_churn_distribution.png")
print("2. graph2_contract_vs_churn.png")
print("3. graph3_monthlycharges_vs_churn.png")
print("4. graph4_tenure_vs_churn.png")