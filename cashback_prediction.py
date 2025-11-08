import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Connect to Database ---
DB_URI = "postgresql+psycopg2://postgres:your_password@localhost:5432/fluz_ds"
engine = create_engine(DB_URI)

print("Loading customer segments dataset...")
df = pd.read_sql("SELECT * FROM customer_segments", engine)

# --- Prepare Dataset ---
print(f"Initial shape: {df.shape}")
df = df.fillna(0)

# Engineer target proxy: Cashback Rate (synthetic proxy)
# Assume merchants with higher avg_spent and lower avg_item_price give higher cashback
df["cashback_rate"] = (
    (df["total_spent"] / df["avg_item_price"].replace(0, np.nan)).fillna(0)
)
df["cashback_rate"] = np.log1p(df["cashback_rate"])  # smooth scaling

# --- Feature Selection ---
features = ["total_spent", "avg_item_price", "spend_to_price_ratio", "high_spender_flag"]
target = "cashback_rate"

X = df[features]
y = df[target]

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Initialize Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
}

results = []

# --- Train and Evaluate ---
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append((name, rmse, r2))
    print(f"{name} → RMSE: {rmse:.4f}, R²: {r2:.4f}")

# --- Results Comparison ---
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"]).sort_values(by="R2", ascending=False)
print("\nModel Performance:\n", results_df)

plt.figure(figsize=(7,4))
sns.barplot(x="R2", y="Model", data=results_df, palette="viridis")
plt.title("Model Comparison: Cashback Prediction")
plt.xlabel("R² Score")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# --- Save Best Model ---
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
joblib.dump(best_model, "best_cashback_model.pkl")
print(f"✅ Best model saved: {best_model_name}")

# --- Feature Importance (for Tree Models) ---
if hasattr(best_model, "feature_importances_"):
    importance = pd.DataFrame({
        "feature": features,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(6,4))
    sns.barplot(x="importance", y="feature", data=importance, palette="mako")
    plt.title(f"{best_model_name} Feature Importance")
    plt.tight_layout()
    plt.show()

    importance.to_sql("cashback_feature_importance", engine, if_exists="replace", index=False)
    print("Feature importances saved to PostgreSQL.")
