import os
import boto3
import pandas as pd
import numpy as np
import time
import json
from io import StringIO
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

# ── AWS clients ───────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

athena = boto3.client(
    "athena",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET          = os.getenv("S3_BUCKET_NAME")
ATHENA_RESULTS  = f"s3://{os.getenv('ATHENA_RESULTS_BUCKET')}/"
ATHENA_DB       = os.getenv("ATHENA_DATABASE")


# ── Run Athena query and return DataFrame ─────────────────────
def run_athena_query(sql):
    print(f"  Running Athena query...")
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        ResultConfiguration={"OutputLocation": ATHENA_RESULTS}
    )
    query_id = response["QueryExecutionId"]

    # Wait for query to complete
    while True:
        status = athena.get_query_execution(
            QueryExecutionId=query_id
        )["QueryExecution"]["Status"]["State"]
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(1)

    if status != "SUCCEEDED":
        raise Exception(f"Athena query failed with status: {status}")

    # Fetch results
    results = athena.get_query_results(QueryExecutionId=query_id)
    rows    = results["ResultSet"]["Rows"]
    headers = [c["VarCharValue"] for c in rows[0]["Data"]]
    data    = [
        [c.get("VarCharValue", "") for c in row["Data"]]
        for row in rows[1:]
    ]
    return pd.DataFrame(data, columns=headers)


# ── Upload DataFrame as CSV to S3 ─────────────────────────────
def upload_to_s3(df, key):
    buf = StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f"  ✅ Saved → s3://{BUCKET}/{key}  ({len(df)} rows)")


# ── 1. Pull daily sales data from Athena ──────────────────────
def fetch_daily_sales():
    print("\n[1/4] Fetching daily sales from Athena...")
    sql = """
        SELECT
            order_date,
            COUNT(*)                    AS total_orders,
            ROUND(SUM(total_amount), 2) AS daily_revenue,
            ROUND(AVG(total_amount), 2) AS avg_order_value,
            SUM(quantity)               AS units_sold
        FROM smartretail_db.transactions
        WHERE status = 'completed'
        GROUP BY order_date
        ORDER BY order_date
    """
    df = run_athena_query(sql)
    df["total_orders"]    = df["total_orders"].astype(int)
    df["daily_revenue"]   = df["daily_revenue"].astype(float)
    df["avg_order_value"] = df["avg_order_value"].astype(float)
    df["units_sold"]      = df["units_sold"].astype(int)
    print(f"  Fetched {len(df)} days of sales data")
    return df


# ── 2. Run Isolation Forest anomaly detection ─────────────────
def detect_anomalies(df):
    print("\n[2/4] Running anomaly detection (Isolation Forest)...")

    features = ["total_orders", "daily_revenue",
                "avg_order_value", "units_sold"]
    X = df[features].values

    # Normalize features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    # contamination=0.1 means we expect ~10% of days to be anomalous
    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    df["anomaly_score"] = model.fit_predict(X_scaled)
    df["is_anomaly"]    = df["anomaly_score"].apply(
        lambda x: "YES" if x == -1 else "NO"
    )

    # Anomaly score as percentage (how anomalous is each day)
    raw_scores          = model.decision_function(X_scaled)
    df["anomaly_pct"]   = np.round(
        (1 - (raw_scores - raw_scores.min()) /
         (raw_scores.max() - raw_scores.min())) * 100, 1
    )

    anomalies = df[df["is_anomaly"] == "YES"]
    print(f"  Found {len(anomalies)} anomalous days "
          f"out of {len(df)} total days")
    return df


# ── 3. Generate summary statistics ────────────────────────────
def generate_summary(df):
    print("\n[3/4] Generating summary statistics...")
    anomalies    = df[df["is_anomaly"] == "YES"]
    normal       = df[df["is_anomaly"] == "NO"]

    summary = {
        "total_days_analyzed":  int(len(df)),
        "anomalous_days":       int(len(anomalies)),
        "normal_days":          int(len(normal)),
        "avg_normal_revenue":   round(
            float(normal["daily_revenue"].mean()), 2),
        "avg_anomaly_revenue":  round(
            float(anomalies["daily_revenue"].mean()), 2),
        "max_revenue_day":      df.loc[
            df["daily_revenue"].idxmax(), "order_date"],
        "min_revenue_day":      df.loc[
            df["daily_revenue"].idxmin(), "order_date"],
        "anomaly_dates":        anomalies["order_date"].tolist()
    }

    for k, v in summary.items():
        print(f"    {k}: {v}")
    return summary


# ── 4. Save results to S3 ─────────────────────────────────────
def save_results(df, summary):
    print("\n[4/4] Saving results to S3...")

    # Full results with anomaly flags
    upload_to_s3(df, "processed/anomalies/daily_sales_with_anomalies.csv")

    # Anomalies only
    anomalies_df = df[df["is_anomaly"] == "YES"].copy()
    upload_to_s3(
        anomalies_df,
        "processed/anomalies/anomalous_days_only.csv"
    )

    # Summary JSON
    s3.put_object(
        Bucket=BUCKET,
        Key="processed/anomalies/summary.json",
        Body=json.dumps(summary, indent=2)
    )
    print(f"  ✅ Saved → s3://{BUCKET}/processed/anomalies/summary.json")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SmartRetail — Anomaly Detection Pipeline")
    print("=" * 55)

    daily_df = fetch_daily_sales()
    result_df = detect_anomalies(daily_df)
    summary = generate_summary(result_df)
    save_results(result_df, summary)

    print("\n" + "=" * 55)
    print("  Anomaly detection complete!")
    print(f"  Anomalous days flagged : "
          f"{summary['anomalous_days']} / {summary['total_days_analyzed']}")
    print(f"  Avg normal revenue     : ${summary['avg_normal_revenue']}")
    print(f"  Avg anomaly revenue    : ${summary['avg_anomaly_revenue']}")
    print("=" * 55)