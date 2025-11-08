import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Database Connection ---
DB_URI = "postgresql+psycopg2://postgres:your_password@localhost:5432/fluz_ds"
engine = create_engine(DB_URI)

print("Loading analytics dataset...")
df = pd.read_sql("SELECT * FROM analytics_dataset", engine)
print(f"Initial shape: {df.shape}")

# --- Handle Missing Values ---
df = df.fillna(0)

# --- Feature Engineering ---
print("Engineering customer features...")

df["spend_log"] = np.log1p(df["total_spent"])          # log transform spend
df["price_spread"] = df["avg_item_price"] / df["avg_item_price"].mean()  # relative price index
df["high_spender_flag"] = (df["total_spent"] > df["total_spent"].quantile(0.75)).astype(int)
df["spend_to_price_ratio"] = df["total_spent"] / (df["avg_item_price"] + 1e-5)

features = ["spend_log", "price_spread", "spend_to_price_ratio", "high_spender_flag"]

# --- Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# --- Determine Optimal Number of Clusters ---
print("Running silhouette optimization...")
sil_scores = []
K = range(2, 8)
for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    print(f"k={k}, silhouette={sil:.3f}")

plt.figure(figsize=(7,4))
plt.plot(K, sil_scores, marker="o")
plt.title("Silhouette Scores vs. Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# --- Final Model (using best k) ---
best_k = K[np.argmax(sil_scores)]
print(f"\nBest number of clusters: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# --- Cluster Profiling ---
profile = (
    df.groupby("cluster")
    .agg({
        "total_spent": "mean",
        "avg_item_price": "mean",
        "spend_to_price_ratio": "mean",
        "high_spender_flag": "mean"
    })
    .reset_index()
    .rename(columns={
        "total_spent": "avg_spent",
        "avg_item_price": "avg_item_price",
        "spend_to_price_ratio": "avg_spend_to_price_ratio",
        "high_spender_flag": "pct_high_spenders"
    })
)

print("\nCluster Profiles:\n", profile)

# --- Visualization ---
plt.figure(figsize=(7,5))
sns.boxplot(x="cluster", y="total_spent", data=df)
plt.title("Spending Distribution per Cluster")
plt.tight_layout()
plt.show()

# --- Save Cluster Assignments ---
df.to_sql("customer_segments", engine, if_exists="replace", index=False)
print("\n✅ Customer segmentation complete. Saved as 'customer_segments' table in PostgreSQL.")
