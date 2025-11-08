import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection (same as db_connection.py)
DB_URI = "postgresql+psycopg2://postgres:your_password@localhost:5432/fluz_ds"
engine = create_engine(DB_URI)

# --- Load Data ---
print("Loading data from PostgreSQL...")
transactions = pd.read_sql("SELECT * FROM transactions", engine)
social_edges = pd.read_sql("SELECT * FROM social_edges", engine)
merchants = pd.read_sql("SELECT * FROM merchants", engine)

print(f"Transactions: {len(transactions)} | Social edges: {len(social_edges)} | Merchants: {len(merchants)}")

# --- Clean Transactions ---
print("\nCleaning transactions...")
transactions = transactions.drop_duplicates(subset=["invoiceno", "stockcode"])
transactions = transactions[transactions["quantity"] > 0]
transactions = transactions[transactions["unitprice"] > 0]
transactions["total_price"] = transactions["quantity"] * transactions["unitprice"]

# Clean Customer IDs (convert to numeric)
transactions["customerid"] = pd.to_numeric(transactions["customerid"], errors="coerce")
transactions = transactions.dropna(subset=["customerid"])

# --- Clean Social Edges ---
print("Cleaning social edges...")
social_edges = social_edges.drop_duplicates()
social_edges.columns = ["source_user", "target_user"]

# --- Clean Merchants ---
print("Cleaning merchants...")
merchants = merchants.drop_duplicates(subset=["business_id"])
merchants = merchants.dropna(subset=["name", "stars"])

# --- Create Link Keys ---
print("Creating synthetic link keys...")
# Simulate linking between customers and social nodes (for demo)
unique_customers = transactions["customerid"].unique()
unique_nodes = pd.unique(social_edges["source_user"])
link_df = pd.DataFrame({
    "customerid": unique_customers[:len(unique_nodes)],
    "social_node": unique_nodes[:len(unique_customers)]
})

# --- Merge Datasets ---
print("Merging datasets...")
transactions = transactions.merge(link_df, on="customerid", how="left")
merged = transactions.merge(merchants, left_on="country", right_on="state", how="left")

# --- Feature Engineering ---
print("Engineering features...")
agg = (
    merged.groupby("customerid")
    .agg({
        "total_price": "sum",
        "unitprice": "mean",
        "stars": "mean",
        "review_count": "mean"
    })
    .reset_index()
    .rename(columns={
        "total_price": "total_spent",
        "unitprice": "avg_item_price",
        "stars": "avg_merchant_rating",
        "review_count": "avg_review_count"
    })
)

# --- Store Final Dataset ---
print("Saving to PostgreSQL table: analytics_dataset...")
agg.to_sql("analytics_dataset", engine, if_exists="replace", index=False)

print("✅ Data cleaning and integration completed successfully!")
