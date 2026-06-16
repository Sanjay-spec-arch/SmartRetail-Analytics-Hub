# 🛒 SmartRetail Analytics Hub

An end-to-end **Data Engineering project** built on AWS that ingests
retail data, runs SQL analytics, detects anomalies using AI, generates
LLM insights, and serves everything through a live interactive dashboard.

---

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud Storage | AWS S3 |
| SQL Analytics | AWS Athena |
| AI Detection | scikit-learn Isolation Forest |
| LLM Insights | Groq LLaMA 3.3 70B (Free) |
| REST API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.12 |
| IaC | AWS IAM, S3 Bucket Policies |

---

## ✨ Features

- ✅ Ingests real product & user data from Fake Store API
- ✅ Generates 1000 synthetic retail transactions
- ✅ Uploads raw data to AWS S3 with date partitioning
- ✅ Queries data using SQL on AWS Athena (serverless)
- ✅ Detects revenue anomalies using Isolation Forest ML model
- ✅ Auto-generates 5 business insights using Groq LLaMA 3.3
- ✅ Serves 10 REST API endpoints via FastAPI
- ✅ Interactive dashboard with 5 pages in Streamlit
- ✅ AI Chat — ask business questions in plain English

---

## 📁 Project Structure
---

## 🚀 Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Sanjay-spec-arch/SmartRetail-Analytics-Hub.git
cd SmartRetail-Analytics-Hub

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
```

Create a `.env` file with:
AWS_ACCESS_KEY_ID=your_key

AWS_SECRET_ACCESS_KEY=your_secret

AWS_REGION=ap-south-1

S3_BUCKET_NAME=your_bucket

ATHENA_RESULTS_BUCKET=your_athena_bucket

ATHENA_DATABASE=smartretail_db

GROQ_API_KEY=your_groq_key

```bash
# 5. Run the pipeline
python ingestion\fetch_and_upload.py
python ai_processing\anomaly_detection.py
python ai_processing\llm_insights.py

# 6. Start API (keep this running)
uvicorn api.main:app --reload --port 8000

# 7. Start Dashboard (new terminal)
streamlit run dashboard\app.py
```

---

## 📊 Dashboard Preview

| Page | Description |
|------|-------------|
| Overview | KPI cards + 90-day revenue chart |
| Sales Trends | Anomaly detection visualization |
| Categories & Channels | Revenue breakdown charts |
| AI Insights | LLM-generated business summaries |
| AI Chat | Ask questions in plain English |

---

## 👨‍💻 Author

**Sanjay** — B.Tech CSE (Data Science), SRM Institute of Science and Technology
