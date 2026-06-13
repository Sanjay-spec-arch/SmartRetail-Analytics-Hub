import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SmartRetail Analytics Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

API = "http://127.0.0.1:8000"

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.85;
        margin: 0;
    }
    .insight-box {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #333;
    }
    .anomaly-badge {
        background: #ff4b4b;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .normal-badge {
        background: #00c853;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: fetch from API ────────────────────────────────────
@st.cache_data(ttl=300)
def fetch(endpoint):
    try:
        r = requests.get(f"{API}{endpoint}", timeout=30)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shopping-cart.png",
    width=80
)
st.sidebar.title("SmartRetail Hub")
st.sidebar.markdown("*Data Engineering Project*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview",
     "📈 Sales Trends",
     "🛍️ Categories & Channels",
     "🤖 AI Insights",
     "💬 AI Chat"]
)

st.sidebar.divider()
st.sidebar.caption("Built with AWS · FastAPI · Streamlit · Groq LLM")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 SmartRetail Analytics Hub")
    st.markdown("Real-time retail intelligence powered by AWS & AI")
    st.divider()

    # Fetch data
    daily   = fetch("/sales/daily")
    summary = fetch("/insights/anomaly-summary")

    if daily and summary:
        df = pd.DataFrame(daily["data"])
        df["daily_revenue"]   = df["daily_revenue"].astype(float)
        df["total_orders"]    = df["total_orders"].astype(int)
        df["units_sold"]      = df["units_sold"].astype(int)
        s  = summary["data"]

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        total_revenue = df["daily_revenue"].sum()
        total_orders  = df["total_orders"].sum()
        anomaly_days  = int(s["anomalous_days"])
        avg_daily     = float(s["avg_normal_revenue"])

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value'>${total_revenue:,.0f}</p>
                <p class='metric-label'>Total Revenue (90 days)</p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value'>{total_orders:,}</p>
                <p class='metric-label'>Total Orders</p>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value'>${avg_daily:,.0f}</p>
                <p class='metric-label'>Avg Daily Revenue</p>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-value'>{anomaly_days}</p>
                <p class='metric-label'>Anomalous Days Detected</p>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Mini revenue chart
        st.subheader("Revenue Last 90 Days")
        df_sorted = df.sort_values("order_date")
        colors    = df_sorted["is_anomaly"].map(
            {"YES": "#ff4b4b", "NO": "#667eea"}
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_sorted["order_date"],
            y=df_sorted["daily_revenue"],
            marker_color=colors,
            name="Daily Revenue"
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Red bars = anomalous days detected by AI  |  🔵 Blue = normal days")

        # Best & worst days
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 Best Day: {s['max_revenue_day']}")
        with col2:
            st.error(f"⚠️ Worst Day: {s['min_revenue_day']}")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — SALES TRENDS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Sales Trends":
    st.title("📈 Sales Trends & Anomaly Detection")
    st.markdown("AI-powered anomaly detection using Isolation Forest")
    st.divider()

    daily = fetch("/sales/daily")
    if daily:
        df = pd.DataFrame(daily["data"])
        df["daily_revenue"]   = df["daily_revenue"].astype(float)
        df["total_orders"]    = df["total_orders"].astype(int)
        df["avg_order_value"] = df["avg_order_value"].astype(float)
        df                    = df.sort_values("order_date")

        # Revenue trend with anomalies
        st.subheader("Daily Revenue with Anomaly Highlights")
        fig = go.Figure()
        normal   = df[df["is_anomaly"] == "NO"]
        anomalies = df[df["is_anomaly"] == "YES"]

        fig.add_trace(go.Scatter(
            x=normal["order_date"],
            y=normal["daily_revenue"],
            mode="lines+markers",
            name="Normal Days",
            line=dict(color="#667eea", width=2),
            marker=dict(size=5)
        ))
        fig.add_trace(go.Scatter(
            x=anomalies["order_date"],
            y=anomalies["daily_revenue"],
            mode="markers",
            name="Anomalous Days",
            marker=dict(color="#ff4b4b", size=12, symbol="x")
        ))
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Orders trend
        st.subheader("Daily Orders Trend")
        fig2 = px.area(
            df, x="order_date", y="total_orders",
            color_discrete_sequence=["#764ba2"]
        )
        fig2.update_layout(
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Anomaly table
        st.subheader("⚠️ Anomalous Days Detail")
        anom_df = anomalies[
            ["order_date", "daily_revenue",
             "total_orders", "anomaly_pct"]
        ].copy()
        anom_df.columns = [
            "Date", "Revenue ($)", "Orders", "Anomaly Score (%)"
        ]
        st.dataframe(anom_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CATEGORIES & CHANNELS
# ══════════════════════════════════════════════════════════════
elif page == "🛍️ Categories & Channels":
    st.title("🛍️ Category & Channel Performance")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        cat = fetch("/sales/by-category")
        if cat:
            df_cat = pd.DataFrame(cat["data"])
            df_cat["total_revenue"] = df_cat["total_revenue"].astype(float)
            df_cat["total_orders"]  = df_cat["total_orders"].astype(int)

            st.subheader("Revenue by Category")
            fig = px.bar(
                df_cat,
                x="total_revenue", y="category",
                orientation="h",
                color="total_revenue",
                color_continuous_scale="Purples",
                text="total_revenue"
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Orders by Category")
            fig2 = px.pie(
                df_cat,
                values="total_orders",
                names="category",
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        ch = fetch("/sales/by-channel")
        if ch:
            df_ch = pd.DataFrame(ch["data"])
            df_ch["revenue"] = df_ch["revenue"].astype(float)
            df_ch["orders"]  = df_ch["orders"].astype(int)

            st.subheader("Revenue by Channel")
            fig3 = px.bar(
                df_ch,
                x="channel", y="revenue",
                color="channel",
                color_discrete_map={
                    "mobile":   "#667eea",
                    "web":      "#764ba2",
                    "in-store": "#f093fb"
                },
                text="revenue"
            )
            fig3.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig3.update_layout(
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Top Products")
            top = fetch("/products/top")
            if top:
                df_top = pd.DataFrame(top["data"])
                df_top["revenue"] = df_top["revenue"].astype(float)
                for i, row in df_top.iterrows():
                    st.markdown(
                        f"**{i+1}. {row['product_name'][:35]}...**  "
                        f"— ${row['revenue']:,.2f}"
                    )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "🤖 AI Insights":
    st.title("🤖 AI-Generated Business Insights")
    st.markdown("Powered by Groq LLaMA 3.3 — insights generated from your real sales data")
    st.divider()

    insights = fetch("/insights")
    if insights:
        data = insights["data"]
        sections = {
            "overall_performance":  ("📊", "Overall Performance"),
            "anomaly_explanation":  ("⚠️", "Anomaly Analysis"),
            "category_performance": ("🛍️", "Category Performance"),
            "channel_analysis":     ("📱", "Channel Analysis"),
            "risk_analysis":        ("🔴", "Risk Analysis")
        }
        for key, (icon, title) in sections.items():
            st.subheader(f"{icon} {title}")
            st.markdown(
                f"<div class='insight-box'>{data.get(key, 'N/A')}</div>",
                unsafe_allow_html=True
            )
            st.write("")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — AI CHAT
# ══════════════════════════════════════════════════════════════
elif page == "💬 AI Chat":
    st.title("💬 Chat with Your Data")
    st.markdown("Ask any business question about your retail data in plain English")
    st.divider()

    # Suggested questions
    st.subheader("Try asking:")
    cols = st.columns(3)
    suggestions = [
        "Which channel has the highest revenue?",
        "How many anomalous days were detected?",
        "What is the average daily revenue?",
        "Which day had the best sales?",
        "What are the main business risks?",
        "Which category performs best?"
    ]
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, use_container_width=True):
                st.session_state["chat_input"] = s

    st.divider()

    # Chat input
    question = st.text_input(
        "Your question:",
        value=st.session_state.get("chat_input", ""),
        placeholder="e.g. What was our best performing sales channel?"
    )

    if st.button("Ask AI", type="primary") and question:
        with st.spinner("Thinking..."):
            result = fetch(f"/chat?question={question.replace(' ', '+')}")
            if result:
                st.markdown("**Your question:**")
                st.info(question)
                st.markdown("**AI Answer:**")
                st.success(result["answer"])