import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ================= CONFIG =================
st.set_page_config(page_title="AI BI Copilot", layout="wide")
st.title("🤖 AI BI Copilot - Smart Data Analyst")

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "📊 EDA", "📈 Visualizations", "🤖 AI Analyst"]
)

file = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# ================= PIPELINE =================
def smooth_pipeline():
    steps = [
        "📥 Uploading Data",
        "🧹 Cleaning Dataset",
        "🔍 Running Analysis",
        "📊 Preparing Visuals",
        "🤖 AI Ready"
    ]

    container = st.sidebar.container()
    progress = container.progress(0)

    for i, step in enumerate(steps):
        container.markdown(f"### {step}")
        progress.progress((i + 1) * 20)
        time.sleep(0.2)

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

# ================= VISUAL =================
def plot_chart(df, chart_type, x, y=None):

    if chart_type == "Bar Chart":
        return px.bar(df, x=x, y=y)

    if chart_type == "Pie Chart":
        return px.pie(df, names=x)

    if chart_type == "Line Chart":
        return px.line(df, x=x, y=y)

    if chart_type == "Histogram":
        return px.histogram(df, x=x)

    return None

# ================= AI ENGINE (FIXED & RELIABLE) =================
def ai_engine(df, query):

    query = query.lower()

    cols = df.columns
    num_cols = df.select_dtypes(include="number").columns

    # ---------------- detect date column ----------------
    date_col = None
    for c in cols:
        if "date" in c.lower():
            date_col = c

    # ---------------- detect metric ----------------
    metric = None
    keywords = ["sales", "amount", "revenue", "profit", "price", "total"]

    for k in keywords:
        for c in num_cols:
            if k in c.lower():
                metric = c
                break

    if metric is None and len(num_cols) > 0:
        metric = num_cols[0]

    # ================= CASE 1: DATE ANALYSIS =================
    if "date" in query or "time" in query:

        if date_col is None:
            return {"type": "text", "answer": "❌ No date column found"}

        grouped = df.groupby(date_col)[metric].sum().reset_index()

        best = grouped.loc[grouped[metric].idxmax()]

        return {
            "type": "chart",
            "df": grouped,
            "x": date_col,
            "y": metric,
            "answer": f"📅 Best {date_col}: {best[date_col]} with {best[metric]:.2f}"
        }

    # ================= CASE 2: TOP / HIGHEST =================
    if "top" in query or "highest" in query or "max" in query:

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
            "answer": f"🏆 Top {group_col} by {metric}"
        }

    # ================= CASE 3: GROUP BY =================
    if "by" in query:

        for c in cols:
            if c.lower() in query:

                grouped = df.groupby(c)[metric].sum().reset_index()

                return {
                    "type": "chart",
                    "df": grouped,
                    "x": c,
                    "y": metric,
                    "answer": f"📊 {metric} grouped by {c}"
                }

    # ================= DEFAULT =================
    total = df[metric].sum()
    avg = df[metric].mean()

    return {
        "type": "text",
        "answer": f"📊 Total {metric}: {total:.2f} | Avg: {avg:.2f}"
    }

# ================= MAIN APP =================
if file is not None:

    smooth_pipeline()

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("📥 Data Loaded Successfully")

    df = clean(df)

    numeric = df.select_dtypes(include="number").columns

    # ================= DASHBOARD =================
    if page == "🏠 Dashboard":

        st.subheader("📊 KPI Dashboard")

        kpis = kpi_engine(df)

        cols = st.columns(min(4, len(kpis)))

        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.2f}")

    # ================= EDA =================
    elif page == "📊 EDA":

        st.subheader("📊 Exploratory Data Analysis")

        tab1, tab2, tab3 = st.tabs(["Preview", "Missing Values", "Stats"])

        with tab1:
            st.dataframe(df.head(), use_container_width=True)

        with tab2:
            st.dataframe(df.isnull().sum())

        with tab3:
            st.dataframe(df.describe())

    # ================= VISUAL =================
    elif page == "📈 Visualizations":

        st.subheader("📊 Visualization Engine")

        chart_type = st.selectbox("Chart Type",
                                   ["Bar Chart", "Pie Chart", "Line Chart", "Histogram"])

        x_axis = st.selectbox("X Axis", df.columns)

        y_axis = None
        if chart_type in ["Bar Chart", "Line Chart"]:
            y_axis = st.selectbox("Y Axis", numeric)

        fig = plot_chart(df, chart_type, x_axis, y_axis)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ================= AI ANALYST =================
    elif page == "🤖 AI Analyst":

        st.subheader("🤖 Smart AI Data Analyst")

        query = st.text_input("Ask anything (e.g. which date has highest sales, top products, sales by region)")

        if query:

            output = ai_engine(df, query)

            st.markdown("### 🧠 AI Result")
            st.success(output["answer"])

            if output["type"] == "chart":
                fig = px.bar(output["df"], x=output["x"], y=output["y"],
                             title="AI Generated Insight")

                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📂 Upload CSV or Excel file to start")
