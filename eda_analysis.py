import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Database Connection
DB_URI = "postgresql+psycopg2://postgres:your_password@localhost:5432/fluz_ds"
engine = create_engine(DB_URI)

# Load Data
print("Loading analytics dataset...")
df = pd.read_sql("SELECT * FROM analytics_dataset", engine)
print(f"Dataset shape: {df.shape}")
print(df.head())

# --- Quick Data Health Check ---
print("\nNull Summary:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

# --- Distribution of Spending ---
plt.figure(figsize=(8,5))
sns.histplot(df["total_spent"], bins=50, kde=True)
plt.title("Distribution of Total Spending per Customer")
plt.xlabel("Total Spent")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- Relationship: Rating vs Spend ---
plt.figure(figsize=(7,5))
sns.scatterplot(x="avg_merchant_rating", y="total_spent", data=df)
plt.title("Merchant Rating vs Total Spend")
plt.xlabel("Average Merchant Rating")
plt.ylabel("Total Spent")
plt.tight_layout()
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.tight_layout()
plt.show()

# --- Spending Outliers ---
q1 = df["total_spent"].quantile(0.25)
q3 = df["total_spent"].quantile(0.75)
iqr = q3 - q1
outliers = df[df["total_spent"] > q3 + 1.5 * iqr]
print(f"Potential high-spend outliers: {len(outliers)} customers")
