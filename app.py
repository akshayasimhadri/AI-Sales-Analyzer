import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import re

# ================= CONFIG =================
st.set_page_config(page_title="AI BI Copilot", layout="wide")

st.title("🤖 AI BI Copilot - Unlimited KPI + AI Analyst Engine")

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "📊 EDA", "📈 Visualizations", "🔮 Prediction", "🤖 AI Analyst"]
)

file = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# ================= ANIMATED PIPELINE =================
def smooth_pipeline():
    steps = [
        "📥 Uploading Data",
        "🧹 Cleaning Dataset",
        "🔍 Running EDA",
        "📊 Generating Visuals",
        "🔮 Prediction Engine"
    ]

    container = st.sidebar.container()
    progress = container.progress(0)

    for i, step in enumerate(steps):
        container.markdown(f"### {step}")
        progress.progress((i + 1) * 20)
        time.sleep(0.3)

# ================= CLEANING =================
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
    num = df.select_dtypes(include="number").columns

    for col in num:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Avg {col}"] = df[col].mean()

    return kpis

# ================= VISUAL ENGINE =================
def plot_chart(df, chart_type, x, y=None):

    if chart_type == "Bar Chart":
        return px.bar(df, x=x, y=y)

    elif chart_type == "Pie Chart":
        return px.pie(df, names=x)

    elif chart_type == "Line Chart":
        return px.line(df, x=x, y=y)

    elif chart_type == "Histogram":
        return px.histogram(df, x=x)

    return None

# ================= AI KPI ENGINE (UNLIMITED) =================
def ai_kpi_engine(df, query):

    query = query.lower()
    cols = df.columns
    num_cols = df.select_dtypes(include="number").columns

    result_df = df.copy()

    # ================= DETECT DATE =================
    date_col = None
    for c in cols:
        if "date" in c.lower():
            date_col = c

    # ================= DETECT METRIC =================
    metric = None
    for c in num_cols:
        if any(k in c.lower() for k in ["sales", "amount", "revenue", "profit"]):
            metric = c

    if metric is None and len(num_cols) > 0:
        metric = num_cols[0]

    # ================= TOP / MAX =================
    if "top" in query or "highest" in query or "max" in query:

        if date_col and "date" in query:

            grouped = result_df.groupby(date_col)[metric].sum().sort_values(ascending=False)

            return {
                "type": "single",
                "answer": f"Best {date_col}: {grouped.index[0]}",
                "value": float(grouped.iloc[0])
            }

        cat_cols = [c for c in cols if df[c].dtype == "object"]

        if len(cat_cols) > 0:
            c = cat_cols[0]

            grouped = result_df.groupby(c)[metric].sum().sort_values(ascending=False).head(5)

            return {
                "type": "chart",
                "df": grouped.reset_index(),
                "x": c,
                "y": metric,
                "answer": "Top analysis generated"
            }

    # ================= GROUP BY =================
    if "by" in query:

        for c in cols:
            if c.lower() in query:

                grouped = result_df.groupby(c)[metric].sum().reset_index()

                return {
                    "type": "chart",
                    "df": grouped,
                    "x": c,
                    "y": metric,
                    "answer": f"Grouped by {c}"
                }

    # ================= TIME ANALYSIS =================
    if date_col:

        grouped = result_df.groupby(date_col)[metric].sum().reset_index()

        return {
            "type": "chart",
            "df": grouped,
            "x": date_col,
            "y": metric,
            "answer": "Time-based analysis generated"
        }

    return {
        "type": "text",
        "answer": "I understood the query but need clearer column mapping"
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

        tab1, tab2, tab3 = st.tabs(["Preview", "Missing Values", "Statistics"])

        with tab1:
            st.dataframe(df.head(), use_container_width=True)

        with tab2:
            st.dataframe(df.isnull().sum())

        with tab3:
            st.dataframe(df.describe())

    # ================= VISUALIZATION =================
    elif page == "📈 Visualizations":

        st.subheader("📊 Visualization Engine")

        chart_type = st.selectbox(
            "Choose Chart Type",
            ["Bar Chart", "Pie Chart", "Line Chart", "Histogram"]
        )

        x_axis = st.selectbox("X Axis", df.columns)

        y_axis = None
        if chart_type in ["Bar Chart", "Line Chart"]:
            y_axis = st.selectbox("Y Axis", numeric)

        fig = plot_chart(df, chart_type, x_axis, y_axis)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ================= PREDICTION =================
    elif page == "🔮 Prediction":

        st.subheader("🔮 Prediction Engine")

        if len(numeric) > 0:

            target = st.selectbox("Select Target Column", numeric)

            df["Prediction"] = df[target].rolling(3).mean()

            fig = px.line(df, y=[target, "Prediction"], title="Trend Prediction")

            st.plotly_chart(fig, use_container_width=True)

    # ================= AI ANALYST =================
    elif page == "🤖 AI Analyst":

        st.subheader("🤖 Unlimited AI KPI Analyst")

        query = st.text_input("Ask anything (e.g. which date has highest sales, sales by region, top products)")

        if query:

            output = ai_kpi_engine(df, query)

            st.markdown(output["answer"])

            if output["type"] == "single":
                st.metric("Result", output["value"])

            elif output["type"] == "chart":
                fig = px.bar(output["df"], x=output["x"], y=output["y"])
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📂 Upload dataset to start AI BI Copilot")
