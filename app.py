import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Sales Analyst", layout="wide")

st.title("🤖 AI Sales Data Analyst Agent")
st.write("Upload CSV/Excel → Auto KPIs, Charts, Insights")

# ---------------- UPLOAD ----------------
file = st.file_uploader("📂 Upload Dataset", type=["csv", "xlsx"])

# ---------------- CLEAN FUNCTION ----------------
def clean(df):
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ---------------- KPI ENGINE ----------------
def generate_kpis(df):
    numeric_cols = df.select_dtypes(include="number").columns
    kpis = {}

    for col in numeric_cols:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Average {col}"] = df[col].mean()
        kpis[f"Max {col}"] = df[col].max()

    return kpis

# ---------------- CHART ENGINE ----------------
def generate_charts(df):
    charts = []

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # 1. Histograms
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append(fig)

    # 2. Category vs Numeric (if possible)
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat = categorical_cols[0]
        num = numeric_cols[0]

        grouped = df.groupby(cat)[num].sum().reset_index()
        fig = px.bar(grouped, x=cat, y=num, title=f"{cat} vs {num}")
        charts.append(fig)

    # 3. Correlation heatmap
    if len(numeric_cols) > 1:
        fig = px.imshow(df[numeric_cols].corr(), text_auto=True, title="Correlation Heatmap")
        charts.append(fig)

    return charts

# ---------------- INSIGHTS ----------------
def generate_insights(df):
    insights = []

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        if df[col].mean() > df[col].median():
            insights.append(f"📈 {col} is positively skewed (high values dominate)")
        else:
            insights.append(f"📊 {col} is stable or evenly distributed")

    return insights

# ---------------- MAIN APP ----------------
if file is not None:

    # 1️⃣ LOAD DATA FIRST
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📄 Raw Data")
    st.dataframe(df.head(), use_container_width=True)

    # 2️⃣ CLEAN DATA
    df = clean(df)

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df.head(), use_container_width=True)

    # 3️⃣ KPIs
    st.subheader("📊 Auto KPIs")

    kpis = generate_kpis(df)

    cols = st.columns(min(5, len(kpis)))

    for i, (k, v) in enumerate(kpis.items()):
        with cols[i % len(cols)]:
            st.metric(k, f"{v:.2f}" if isinstance(v, float) else v)

    # 4️⃣ CHARTS
    st.subheader("📈 Auto Visualizations")

    charts = generate_charts(df)

    for chart in charts:
        st.plotly_chart(chart, use_container_width=True)

    # 5️⃣ INSIGHTS
    st.subheader("🧠 Insights")

    insights = generate_insights(df)

    for i in insights:
        st.write("✔", i)

else:
    st.info("📂 Upload a CSV or Excel file to start analysis")
