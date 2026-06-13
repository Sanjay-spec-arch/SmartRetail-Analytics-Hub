import os
import boto3
import pandas as pd
import json
import time
from io import StringIO
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Clients ───────────────────────────────────────────────────
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

openai_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BUCKET         = os.getenv("S3_BUCKET_NAME")
ATHENA_RESULTS = f"s3://{os.getenv('ATHENA_RESULTS_BUCKET')}/"
ATHENA_DB      = os.getenv("ATHENA_DATABASE")


# ── Run Athena query ──────────────────────────────────────────
def run_athena_query(sql):
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        ResultConfiguration={"OutputLocation": ATHENA_RESULTS}
    )
    query_id = response["QueryExecutionId"]
    while True:
        status = athena.get_query_execution(
            QueryExecutionId=query_id
        )["QueryExecution"]["Status"]["State"]
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(1)
    if status != "SUCCEEDED":
        raise Exception(f"Athena query failed: {status}")
    results  = athena.get_query_results(QueryExecutionId=query_id)
    rows     = results["ResultSet"]["Rows"]
    headers  = [c["VarCharValue"] for c in rows[0]["Data"]]
    data     = [
        [c.get("VarCharValue", "") for c in row["Data"]]
        for row in rows[1:]
    ]
    return pd.DataFrame(data, columns=headers)


# ── Load anomaly summary from S3 ──────────────────────────────
def load_anomaly_summary():
    obj = s3.get_object(
        Bucket=BUCKET,
        Key="processed/anomalies/summary.json"
    )
    return json.loads(obj["Body"].read().decode("utf-8"))


# ── Load anomalous days CSV from S3 ──────────────────────────
def load_anomalies_df():
    obj = s3.get_object(
        Bucket=BUCKET,
        Key="processed/anomalies/daily_sales_with_anomalies.csv"
    )
    return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))


# ── Fetch supporting stats from Athena ───────────────────────
def fetch_supporting_stats():
    print("  Fetching category revenue...")
    category_sql = """
        SELECT
            category,
            ROUND(SUM(total_amount), 2) AS revenue,
            COUNT(*) AS orders
        FROM smartretail_db.transactions
        WHERE status = 'completed'
        GROUP BY category
        ORDER BY revenue DESC
    """
    category_df = run_athena_query(category_sql)

    print("  Fetching channel stats...")
    channel_sql = """
        SELECT
            channel,
            COUNT(*) AS orders,
            ROUND(SUM(total_amount), 2) AS revenue
        FROM smartretail_db.transactions
        GROUP BY channel
        ORDER BY revenue DESC
    """
    channel_df = run_athena_query(channel_sql)

    print("  Fetching return rate...")
    status_sql = """
        SELECT
            status,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
        FROM smartretail_db.transactions
        GROUP BY status
    """
    status_df = run_athena_query(status_sql)

    return category_df, channel_df, status_df


# ── Generate insight using LLM ────────────────────────────────
def generate_insight(prompt, context):
    response = openai_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior retail data analyst. "
                    "Analyze the provided sales data and generate "
                    "clear, concise, and actionable business insights. "
                    "Use specific numbers from the data. "
                    "Write in plain English for a business manager. "
                    "Keep each insight to 3-4 sentences."
                )
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nData:\n{context}"
            }
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# ── Main insight generation ───────────────────────────────────
def generate_all_insights():
    print("\n[1/5] Loading anomaly data from S3...")
    summary      = load_anomaly_summary()
    anomalies_df = load_anomalies_df()

    print("\n[2/5] Fetching supporting stats from Athena...")
    category_df, channel_df, status_df = fetch_supporting_stats()

    print("\n[3/5] Generating LLM insights...")
    insights = {}

    # Insight 1: Overall performance
    print("  → Generating overall performance insight...")
    overall_context = f"""
    Total days analyzed: {summary['total_days_analyzed']}
    Average normal daily revenue: ${summary['avg_normal_revenue']}
    Average anomalous day revenue: ${summary['avg_anomaly_revenue']}
    Number of anomalous days: {summary['anomalous_days']}
    Highest revenue day: {summary['max_revenue_day']}
    Lowest revenue day: {summary['min_revenue_day']}
    """
    insights["overall_performance"] = generate_insight(
        "Provide an overall business performance summary for this retail store.",
        overall_context
    )

    # Insight 2: Anomaly explanation
    print("  → Generating anomaly insight...")
    anomaly_days = anomalies_df[
        anomalies_df["is_anomaly"] == "YES"
    ][["order_date", "daily_revenue", "total_orders"]].to_string(index=False)
    insights["anomaly_explanation"] = generate_insight(
        "Explain the sales anomalies detected and what actions the business should take.",
        f"Anomalous days:\n{anomaly_days}\nNormal avg revenue: ${summary['avg_normal_revenue']}"
    )

    # Insight 3: Category performance
    print("  → Generating category insight...")
    insights["category_performance"] = generate_insight(
        "Analyze which product categories are performing best and worst, and give recommendations.",
        category_df.to_string(index=False)
    )

    # Insight 4: Channel analysis
    print("  → Generating channel insight...")
    insights["channel_analysis"] = generate_insight(
        "Analyze sales channel performance (mobile, web, in-store) and recommend where to invest.",
        channel_df.to_string(index=False)
    )

    # Insight 5: Risk & return analysis
    print("  → Generating risk insight...")
    insights["risk_analysis"] = generate_insight(
        "Analyze the order status breakdown and identify business risks from returns or pending orders.",
        status_df.to_string(index=False)
    )

    return insights


# ── Save insights to S3 ───────────────────────────────────────
def save_insights(insights):
    print("\n[4/5] Saving insights to S3...")
    s3.put_object(
        Bucket=BUCKET,
        Key="processed/insights/llm_insights.json",
        Body=json.dumps(insights, indent=2)
    )
    print(f"  ✅ Saved → s3://{BUCKET}/processed/insights/llm_insights.json")


# ── Print insights nicely ─────────────────────────────────────
def print_insights(insights):
    print("\n[5/5] Generated insights:")
    titles = {
        "overall_performance":  "📊 Overall Performance",
        "anomaly_explanation":  "⚠️  Anomaly Analysis",
        "category_performance": "🛍️  Category Performance",
        "channel_analysis":     "📱 Channel Analysis",
        "risk_analysis":        "🔴 Risk Analysis"
    }
    for key, title in titles.items():
        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")
        print(f"  {insights[key]}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SmartRetail — LLM Insight Generation")
    print("=" * 55)

    insights = generate_all_insights()
    save_insights(insights)
    print_insights(insights)

    print(f"\n{'='*55}")
    print("  All insights generated and saved to S3!")
    print(f"{'='*55}")