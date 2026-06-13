import os
import boto3
import pandas as pd
import json
import time
from io import StringIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="SmartRetail Analytics API",
    description="Data engineering pipeline API for retail analytics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AWS Clients ───────────────────────────────────────────────
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

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BUCKET         = os.getenv("S3_BUCKET_NAME")
ATHENA_RESULTS = f"s3://{os.getenv('ATHENA_RESULTS_BUCKET')}/"
ATHENA_DB      = os.getenv("ATHENA_DATABASE")


# ── Helper: Run Athena Query ──────────────────────────────────
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
    results = athena.get_query_results(QueryExecutionId=query_id)
    rows    = results["ResultSet"]["Rows"]
    headers = [c["VarCharValue"] for c in rows[0]["Data"]]
    data    = [
        [c.get("VarCharValue", "") for c in row["Data"]]
        for row in rows[1:]
    ]
    return pd.DataFrame(data, columns=headers).to_dict(orient="records")


# ── Helper: Load JSON from S3 ─────────────────────────────────
def load_json_from_s3(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


# ── Helper: Load CSV from S3 ──────────────────────────────────
def load_csv_from_s3(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df  = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    return df.to_dict(orient="records")


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

# ── 1. Health check ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "project": "SmartRetail Analytics Hub",
        "version": "1.0.0"
    }


# ── 2. Daily sales with anomaly flags ────────────────────────
@app.get("/sales/daily")
def get_daily_sales():
    try:
        data = load_csv_from_s3(
            "processed/anomalies/daily_sales_with_anomalies.csv"
        )
        return {"status": "success", "count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Anomalous days only ────────────────────────────────────
@app.get("/sales/anomalies")
def get_anomalies():
    try:
        data = load_csv_from_s3(
            "processed/anomalies/anomalous_days_only.csv"
        )
        return {"status": "success", "count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. Revenue by category ────────────────────────────────────
@app.get("/sales/by-category")
def get_sales_by_category():
    try:
        data = run_athena_query("""
            SELECT
                category,
                COUNT(*) AS total_orders,
                ROUND(SUM(total_amount), 2) AS total_revenue,
                ROUND(AVG(total_amount), 2) AS avg_order_value
            FROM smartretail_db.transactions
            WHERE status = 'completed'
            GROUP BY category
            ORDER BY total_revenue DESC
        """)
        return {"status": "success", "count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. Revenue by channel ─────────────────────────────────────
@app.get("/sales/by-channel")
def get_sales_by_channel():
    try:
        data = run_athena_query("""
            SELECT
                channel,
                COUNT(*) AS orders,
                ROUND(SUM(total_amount), 2) AS revenue
            FROM smartretail_db.transactions
            GROUP BY channel
            ORDER BY revenue DESC
        """)
        return {"status": "success", "count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Top 5 products ─────────────────────────────────────────
@app.get("/products/top")
def get_top_products():
    try:
        data = run_athena_query("""
            SELECT
                t.product_id,
                p.product_name,
                p.category,
                SUM(t.quantity) AS units_sold,
                ROUND(SUM(t.total_amount), 2) AS revenue
            FROM smartretail_db.transactions t
            JOIN smartretail_db.products p
                ON t.product_id = p.product_id
            WHERE t.status = 'completed'
            GROUP BY t.product_id, p.product_name, p.category
            ORDER BY revenue DESC
            LIMIT 5
        """)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 7. Monthly revenue trend ──────────────────────────────────
@app.get("/sales/monthly-trend")
def get_monthly_trend():
    try:
        data = run_athena_query("""
            SELECT
                SUBSTR(order_date, 1, 7) AS month,
                COUNT(*) AS orders,
                ROUND(SUM(total_amount), 2) AS revenue
            FROM smartretail_db.transactions
            WHERE status = 'completed'
            GROUP BY SUBSTR(order_date, 1, 7)
            ORDER BY month
        """)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 8. LLM insights ───────────────────────────────────────────
@app.get("/insights")
def get_insights():
    try:
        data = load_json_from_s3("processed/insights/llm_insights.json")
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 9. Anomaly summary ────────────────────────────────────────
@app.get("/insights/anomaly-summary")
def get_anomaly_summary():
    try:
        data = load_json_from_s3("processed/anomalies/summary.json")
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 10. AI Chat — ask anything about your data ───────────────
@app.get("/chat")
def chat_with_data(question: str):
    try:
        # Load context from S3
        summary  = load_json_from_s3("processed/anomalies/summary.json")
        insights = load_json_from_s3("processed/insights/llm_insights.json")

        context = f"""
        Business Summary:
        - Total days analyzed: {summary['total_days_analyzed']}
        - Average normal daily revenue: ${summary['avg_normal_revenue']}
        - Anomalous days detected: {summary['anomalous_days']}
        - Best revenue day: {summary['max_revenue_day']}
        - Worst revenue day: {summary['min_revenue_day']}

        Overall performance: {insights['overall_performance']}
        Channel analysis: {insights['channel_analysis']}
        Category performance: {insights['category_performance']}
        """

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a retail business analyst assistant. "
                        "Answer questions using only the provided data context. "
                        "Be concise and use specific numbers."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            max_tokens=300,
            temperature=0.5
        )
        answer = response.choices[0].message.content.strip()
        return {"status": "success", "question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))