import pandas as pd
from db_connection import get_engine

# Connect to PostgreSQL
engine = get_engine()

# ========== 1. Load UCI Online Retail Data ==========
print("Loading Online Retail dataset...")
retail = pd.read_excel("data/Online Retail.xlsx")
retail.columns = [c.lower().replace(' ', '_') for c in retail.columns]

# Basic Cleaning
if 'CustomerID' in retail.columns:
    retail = retail.dropna(subset=['CustomerID'])
elif 'customerid' in retail.columns:
    retail = retail.dropna(subset=['customerid'])
retail = retail[~retail['invoiceno'].astype(str).str.startswith('C')]  # Remove cancellations
# Normalize column name variations
date_col = [c for c in retail.columns if 'date' in c.lower()][0]
retail[date_col] = pd.to_datetime(retail[date_col])
retail.rename(columns={date_col: 'invoice_date'}, inplace=True)

retail.to_sql('transactions', engine, if_exists='replace', index=False)
print(f"Loaded {len(retail)} retail transactions.")

# ========== 2. Load Social Network Data ==========
print("Loading Facebook social network data...")
social = pd.read_csv("data/facebook_combined.txt", sep=' ', names=['user_source', 'user_target'])
social.to_sql('social_edges', engine, if_exists='replace', index=False)
print(f"Loaded {len(social)} social edges.")

# ========== 3. Load Yelp Merchant Data ==========
print("Loading Yelp business data...")
import json
records = [json.loads(line) for line in open("data/yelp_academic_dataset_business.json")]
yelp = pd.DataFrame(records)[['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'is_open']]
yelp.columns = yelp.columns.str.lower()

yelp.to_sql('merchants', engine, if_exists='replace', index=False)
print(f"Loaded {len(yelp)} merchant records.")

print("✅ All datasets ingested successfully!")
