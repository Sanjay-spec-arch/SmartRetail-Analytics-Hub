import os
import json
import boto3
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

load_dotenv()

# ── AWS client setup ──────────────────────────────────────────
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET = os.getenv("S3_BUCKET_NAME")
TODAY  = datetime.today().strftime("%Y-%m-%d")


# ── Helper: upload a dataframe as CSV to S3 ───────────────────
def upload_df_to_s3(df, s3_key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=csv_buffer.getvalue()
    )
    print(f"  ✅ Uploaded → s3://{BUCKET}/{s3_key}  ({len(df)} rows)")


# ── Helper: upload a dict as JSON to S3 ───────────────────────
def upload_json_to_s3(data, s3_key):
    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=json.dumps(data, indent=2)
    )
    print(f"  ✅ Uploaded → s3://{BUCKET}/{s3_key}")


# ── 1. Fetch products from Fake Store API ─────────────────────
def fetch_products():
    print("\n[1/3] Fetching products from Fake Store API...")
    response = requests.get("https://fakestoreapi.com/products", timeout=10)
    products = response.json()

    df = pd.DataFrame(products)
    df = df[["id", "title", "price", "category", "rating"]]
    df["rating_rate"]  = df["rating"].apply(lambda x: x["rate"])
    df["rating_count"] = df["rating"].apply(lambda x: x["count"])
    df.drop(columns=["rating"], inplace=True)
    df.rename(columns={"id": "product_id", "title": "product_name"}, inplace=True)

    upload_df_to_s3(df, f"raw/products/{TODAY}/products.csv")
    upload_json_to_s3(products, f"raw/products/{TODAY}/products_raw.json")
    return df


# ── 2. Fetch users from Fake Store API ────────────────────────
def fetch_users():
    print("\n[2/3] Fetching users from Fake Store API...")
    response = requests.get("https://fakestoreapi.com/users", timeout=10)
    users = response.json()

    df = pd.DataFrame(users)
    df["full_name"] = df["name"].apply(
        lambda x: f"{x['firstname']} {x['lastname']}"
    )
    df["city"] = df["address"].apply(lambda x: x["city"])
    df = df[["id", "full_name", "email", "city", "phone"]]
    df.rename(columns={"id": "user_id"}, inplace=True)

    upload_df_to_s3(df, f"raw/users/{TODAY}/users.csv")
    return df


# ── 3. Generate synthetic sales transactions ──────────────────
def generate_transactions(products_df, users_df):
    print("\n[3/3] Generating synthetic sales transactions...")

    random.seed(42)
    records = []
    start_date = datetime.today() - timedelta(days=90)

    product_ids = products_df["product_id"].tolist()
    product_prices = dict(
        zip(products_df["product_id"], products_df["price"])
    )
    product_categories = dict(
        zip(products_df["product_id"], products_df["category"])
    )
    user_ids = users_df["user_id"].tolist()

    statuses  = ["completed", "completed", "completed", "returned", "pending"]
    channels  = ["web", "mobile", "web", "mobile", "in-store"]

    for i in range(1, 1001):
        product_id = random.choice(product_ids)
        quantity   = random.randint(1, 5)
        unit_price = round(product_prices[product_id], 2)
        discount   = round(random.uniform(0, 0.25), 2)
        total      = round(quantity * unit_price * (1 - discount), 2)
        order_date = start_date + timedelta(days=random.randint(0, 90))

        records.append({
            "transaction_id": f"TXN{i:05d}",
            "user_id":        random.choice(user_ids),
            "product_id":     product_id,
            "category":       product_categories[product_id],
            "quantity":       quantity,
            "unit_price":     unit_price,
            "discount":       discount,
            "total_amount":   total,
            "order_date":     order_date.strftime("%Y-%m-%d"),
            "channel":        random.choice(channels),
            "status":         random.choice(statuses),
        })

    df = pd.DataFrame(records)
    upload_df_to_s3(df, f"raw/transactions/{TODAY}/transactions.csv")
    return df


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SmartRetail — Data Ingestion Pipeline")
    print("=" * 50)

    products_df = fetch_products()
    users_df    = fetch_users()
    txn_df      = generate_transactions(products_df, users_df)

    print("\n" + "=" * 50)
    print("  Pipeline complete! Summary:")
    print(f"  Products loaded    : {len(products_df)} rows")
    print(f"  Users loaded       : {len(users_df)} rows")
    print(f"  Transactions gen.  : {len(txn_df)} rows")
    print(f"  S3 bucket          : {BUCKET}")
    print(f"  Date partition     : {TODAY}")
    print("=" * 50)