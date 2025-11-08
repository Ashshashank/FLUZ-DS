Fluz Cashback Optimization & Customer Segmentation
End-to-End FinTech Data Science Project | Python · PostgreSQL · Tableau

Project Overview

This project simulates a real-world cashback optimization and customer segmentation system for a fintech platform like Fluz.
It demonstrates how data science can drive personalized cashback tiers, merchant-level optimization, and user segmentation to increase engagement and profitability.

The solution integrates multi-source data ingestion, ETL pipelines, unsupervised clustering, supervised regression modeling, and model explainability, capped with interactive Tableau dashboards for business storytelling.

Tech Stack

| Layer                         | Tools & Libraries                            |
| ----------------------------- | -------------------------------------------- |
| **Data Ingestion & ETL**      | Python (pandas, numpy, SQLAlchemy, psycopg2) |
| **Database**                  | PostgreSQL (local instance)                  |
| **EDA & Feature Engineering** | pandas, seaborn, matplotlib                  |
| **Modeling & ML**             | scikit-learn, XGBoost                        |
| **Explainability**            | SHAP                                         |
| **Visualization & BI**        | Tableau Public                               |
| **Environment**               | macOS · Python 3.11                          |


1. Data Ingestion & Integration
Datasets Used
| Source                                                                             | Description                       | Purpose                                  |
| ---------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------- |
| [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) | Transaction-level e-commerce data | Core transactional & spend metrics       |
| [Stanford SNAP Facebook Dataset](https://snap.stanford.edu/data/ego-facebook.html) | Social connections (edges)        | Proxy for referral/social engagement     |
| [Yelp Business Dataset (Kaggle)](https://www.kaggle.com/datasets/yelp-dataset)     | Merchant reviews and ratings      | Proxy for merchant value and CRM metrics |

Each dataset was loaded into PostgreSQL using SQLAlchemy:
retail.to_sql('transactions', engine, if_exists='replace', index=False)

Output:

397,924 retail transactions
88,234 social connections
150,346 merchant records

2. Exploratory Data Analysis (EDA)

Key stats from eda_analysis.py:
Dataset shape: (4338, 5)
Null Summary:
avg_merchant_rating    4338
avg_review_count       4338

Highlights:

Total 4,338 unique customers.

Average spend: $2,038.84

425 high-spend outliers detected.

Data normalization and missing value handling applied.

Tools: pandas, matplotlib, seaborn
Output: analytics_summary.png

3. Feature Engineering & Customer Segmentation

Goal: Identify unique spending personas for cashback targeting.

Approach:

Aggregated customer-level metrics (total_spent, avg_item_price, spend-to-price ratio).

Applied KMeans Clustering with silhouette optimization.

Output:
Best number of clusters: 4
Cluster Profiles:
    cluster      avg_spent  avg_item_price
0        0     564.27         3.92
1        1  114046.83         2.47
2        2    4852.18         4.27
3        3    2033.10      2033.10

High-value and value-seeking clusters clearly separated.
Saved to PostgreSQL as customer_segments.

4. Cashback Prediction Model

Goal: Predict optimal cashback rate per customer.

Models Tested:
| Model             | RMSE       | R²         |
| ----------------- | ---------- | ---------- |
| Linear Regression | 0.9001     | 0.5598     |
| Decision Tree     | 0.0382     | 0.9992     |
| Random Forest     | **0.0114** | **0.9999** |
| XGBoost           | 0.0490     | 0.9987     |
Random Forest Regressor selected as final model.
Feature Importance:

total_spent
avg_item_price
spend_to_price_ratio

Outputs:

cashback_feature_importance.csv
Trained model pickle file (optional)

🔍 5. Model Explainability

Using SHAP (SHapley Additive Explanations) to understand feature influence.

✅ Output:
Computing SHAP values...
Feature importances saved to PostgreSQL as 'cashback_shap_importance'.

Generated:

shap_summary_plot.png
shap_feature_importance.png

Key finding:

Cashback prediction is primarily influenced by total spend and spend-to-price efficiency, confirming that high-value, consistent shoppers drive ROI.

📊 6. Tableau Public Dashboard

Interactive Story: “Fluz Cashback Optimization Narrative”

Dashboards:

Customer Segmentation & Spend Behavior
Predicted Cashback Distribution
Model Explainability & Feature Drivers
Executive Summary / Recommendations

Story Slides:
| Slide | Description                                   |
| ----- | --------------------------------------------- |
| 1     | Overview of Fluz cashback data pipeline       |
| 2     | Visual segmentation of customer personas      |
| 3     | Predicted cashback tiers by customer behavior |
| 4     | Feature importance & SHAP explainability      |
| 5     | Strategic recommendations for marketing       |

Published on Tableau Public

Key Business Insights

High Spenders consistently earn better cashback—Fluz should introduce tiered cashback levels.
Low-Price Consistency merchants drive long-term user engagement.
Customer Segmentation enables personalized marketing campaigns.
Model Explainability builds trust with transparent cashback logic.


How to Reproduce

# 1. Clone repository
git clone https://github.com/<your-username>/FLUZ-DS.git
cd FLUZ-DS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run ETL pipeline
python3 etl_ingest.py

# 4. Perform EDA
python3 eda_analysis.py

# 5. Run segmentation
python3 customer_segmentation.py

# 6. Train cashback model
python3 cashback_prediction.py

# 7. Explain model
python3 model_explainability.py

Then connect Tableau to the CSV outputs in /data/.
