import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import joblib

# --- Connect to PostgreSQL ---
DB_URI = "postgresql+psycopg2://postgres:your_password@localhost:5432/fluz_ds"
engine = create_engine(DB_URI)

# --- Load model and data ---
print("Loading best model and data...")
model = joblib.load("best_cashback_model.pkl")
df = pd.read_sql("SELECT * FROM customer_segments", engine)
df = df.fillna(0)

# --- Prepare features ---
features = ["total_spent", "avg_item_price", "spend_to_price_ratio", "high_spender_flag"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SHAP Explainer ---
print("Computing SHAP values...")
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

# --- Global Feature Importance ---
print("Generating SHAP summary plot...")
plt.figure(figsize=(8,5))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.close()

# --- Detailed Feature Distribution Plot ---
print("Generating SHAP detailed summary...")
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_detailed_summary.png")
plt.close()

# --- SHAP Values DataFrame ---
shap_df = pd.DataFrame(shap_values.values, columns=features)
shap_mean = shap_df.abs().mean().reset_index()
shap_mean.columns = ["feature", "mean_abs_shap"]
shap_mean = shap_mean.sort_values(by="mean_abs_shap", ascending=False)

# --- Save Results ---
shap_mean.to_sql("cashback_shap_importance", engine, if_exists="replace", index=False)
print("✅ SHAP results saved to PostgreSQL as 'cashback_shap_importance'.")
print("Feature importance plots saved as PNGs.")
