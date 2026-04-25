import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ================= CONFIG =================
st.set_page_config(page_title="AI Enterprise BI Platform", layout="wide")

st.title("🤖 AI Enterprise BI Platform")

# ================= SIDEBAR NAV =================
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "📊 EDA", "📈 Visualizations", "🔮 Prediction", "🤖 AI Analyst"]
)

file = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# ================= PIPELINE ANIMATION =================
def smooth_pipeline():
    steps = [
        "📥 Loading Data",
        "🧹 Cleaning Data",
        "🔍 Running Analysis",
        "📊 Building KPIs",
        "🤖 Generating Insights"
    ]

    container = st.sidebar.container()
    progress = container.progress(0)

    for i, step in enumerate(steps):
        container.markdown(f"### {step}")
        progress.progress((i + 1) * 20)
        time.sleep(0.25)

# ================= CLEAN DATA =================
def clean(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ================= KPI ENGINE =================
def kpi_engine(df):
    kpis = {}
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Avg {col}"] = df[col].mean()

    return kpis

# ================= AI ANALYST ENGINE =================
def ai_engine(df, query):

    query = query.lower()
    cols = df.columns
    num_cols = df.select_dtypes(include="number").columns

    # detect date
    date_col = None
    for c in cols:
        if "date" in c.lower():
            date_col = c

    # detect metric
    metric = None
    for c in num_cols:
        if any(k in c.lower() for k in ["sales", "amount", "revenue", "profit"]):
            metric = c

    if metric is None and len(num_cols) > 0:
        metric = num_cols[0]

    # ================= DATE ANALYSIS =================
    if "date" in query:

        if date_col is None:
            return {"type": "text", "answer": "❌ No date column found"}

        grouped = df.groupby(date_col)[metric].sum().reset_index()

        best = grouped.loc[grouped[metric].idxmax()]

        return {
            "type": "chart",
            "df": grouped,
            "x": date_col,
            "y": metric,
            "answer": f"📅 Best {date_col}: {best[date_col]} → {best[metric]:.2f}"
        }

    # ================= TOP ANALYSIS =================
    if "top" in query or "highest" in query:

        cat_cols = [c for c in cols if df[c].dtype == "object"]

        if len(cat_cols) == 0:
            return {"type": "text", "answer": "❌ No categorical column found"}

        group_col = cat_cols[0]

        grouped = df.groupby(group_col)[metric].sum().sort_values(ascending=False).head(5)

        return {
            "type": "chart",
            "df": grouped.reset_index(),
            "x": group_col,
            "y": metric,
            "answer": "🏆 Top analysis generated"
        }

    # ================= GROUP BY =================
    if "by" in query:

        for c in cols:
            if c.lower() in query:

                grouped = df.groupby(c)[metric].sum().reset_index()

                return {
                    "type": "chart",
                    "df": grouped,
                    "x": c,
                    "y": metric,
                    "answer": f"📊 Grouped by {c}"
                }

    # ================= FALLBACK =================
    return {
        "type": "text",
        "answer": f"📊 Total {metric}: {df[metric].sum():.2f} | Avg: {df[metric].mean():.2f}"
    }

# ================= MAIN =================
if file is not None:

    smooth_pipeline()

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("📥 Data Loaded")

    df = clean(df)

    numeric_cols = df.select_dtypes(include="number").columns

    # ================= DASHBOARD =================
    if page == "🏠 Dashboard":

        st.subheader("📊 Executive AI Dashboard")

        kpis = kpi_engine(df)

        # KPI CARDS
        st.markdown("### 🎯 KPIs")

        cols = st.columns(min(4, len(kpis)))

        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.2f}")

        # MAIN DATA TABLE (IMPORTANT)
        st.markdown("### 📋 Full Sales Dataset")

        st.dataframe(df, use_container_width=True)

        # KPI TABLE
        st.markdown("### 📊 KPI Summary Table")

        st.dataframe(pd.DataFrame({
            "KPI": list(kpis.keys()),
            "Value": list(kpis.values())
        }), use_container_width=True)

        # TREND CHARTS
        st.markdown("### 📈 Trends")

        for col in numeric_cols[:3]:
            st.plotly_chart(px.line(df, y=col, title=f"{col} Trend"),
                            use_container_width=True)

        # INSIGHTS
        st.markdown("### 🧠 Insights")

        for col in numeric_cols[:3]:
            avg = df[col].mean()
            st.info(f"📊 {col} average is {avg:.2f}")

    # ================= EDA =================
    elif page == "📊 EDA":

        st.subheader("📊 Data Exploration")

        st.dataframe(df.head(), use_container_width=True)
        st.dataframe(df.describe())

    # ================= VISUALIZATION =================
    elif page == "📈 Visualizations":

        st.subheader("📊 Visualization Engine")

        chart = st.selectbox("Chart Type",
                             ["Bar Chart", "Pie Chart", "Line Chart", "Histogram"])

        x = st.selectbox("X Axis", df.columns)

        y = None
        if chart in ["Bar Chart", "Line Chart"]:
            y = st.selectbox("Y Axis", numeric_cols)

        if chart == "Bar Chart":
            fig = px.bar(df, x=x, y=y)
        elif chart == "Pie Chart":
            fig = px.pie(df, names=x)
        elif chart == "Line Chart":
            fig = px.line(df, x=x, y=y)
        else:
            fig = px.histogram(df, x=x)

        st.plotly_chart(fig, use_container_width=True)

    # ================= PREDICTION =================
    elif page == "🔮 Prediction":

        st.subheader("🔮 Simple Forecasting")

        if len(numeric_cols) > 0:

            col = st.selectbox("Select Column", numeric_cols)

            df["Forecast"] = df[col].rolling(3).mean()

            st.plotly_chart(px.line(df, y=[col, "Forecast"]),
                            use_container_width=True)

    # ================= AI ANALYST =================
    elif page == "🤖 AI Analyst":

        st.subheader("🤖 AI Data Analyst")

        query = st.text_input("Ask anything about your data")

        if query:

            output = ai_engine(df, query)

            st.success(output["answer"])

            if output["type"] == "chart":
                st.plotly_chart(
                    px.bar(output["df"], x=output["x"], y=output["y"]),
                    use_container_width=True
                )

else:
    st.info("📂 Upload file to start AI Analytics Platform")
