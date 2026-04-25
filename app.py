import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- LOAD ----------------
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# ---------------- CLEAN ----------------
def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ---------------- AUTO KPIs ----------------
def auto_kpis(df):
    numeric_cols = df.select_dtypes(include="number").columns

    kpis = []

    for col in numeric_cols:
        kpis.append({
            "name": f"Total {col}",
            "value": df[col].sum()
        })
        kpis.append({
            "name": f"Avg {col}",
            "value": df[col].mean()
        })
        kpis.append({
            "name": f"Max {col}",
            "value": df[col].max()
        })

    return kpis

# ---------------- AUTO VISUALS ----------------
def auto_visuals(df):
    charts = []

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # 1. Numeric distributions
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append(fig)

    # 2. Category vs numeric (top 1 auto pairing)
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat = categorical_cols[0]
        num = numeric_cols[0]

        grouped = df.groupby(cat)[num].sum().reset_index()

        fig = px.bar(grouped, x=cat, y=num, title=f"{cat} vs {num}")
        charts.append(fig)

    # 3. Correlation heatmap
    if len(numeric_cols) > 1:
        fig = px.imshow(df[numeric_cols].corr(), text_auto=True,
                        title="Correlation Heatmap")
        charts.append(fig)

    return charts

# ---------------- AUTO INSIGHTS ----------------
def auto_insights(df):
    numeric_cols = df.select_dtypes(include="number").columns

    insights = []

    for col in numeric_cols:
        growth = df[col].pct_change().mean()

        if growth > 0:
            insights.append(f"📈 {col} shows positive growth trend")
        else:
            insights.append(f"📉 {col} shows decline or instability")

    return insights

# ---------------- DASHBOARD UI ----------------
def render_dashboard(df):

    st.title("📊 Auto BI Dashboard")

    df = clean_data(df)

    # ---------------- KPIs ----------------
    st.subheader("📌 Auto Generated KPIs")

    kpis = auto_kpis(df)

    cols = st.columns(min(5, len(kpis)))

    for i, kpi in enumerate(kpis):
        with cols[i % len(cols)]:
            st.metric(kpi["name"], f"{kpi['value']:.2f}")

    st.divider()

    # ---------------- VISUALS ----------------
    st.subheader("📈 Auto Visualizations")

    charts = auto_visuals(df)

    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    st.divider()

    # ---------------- INSIGHTS ----------------
    st.subheader("🧠 Auto Insights")

    insights = auto_insights(df)

    for i in insights:
        st.write(i)

# ---------------- MAIN ----------------
file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if file is not None:
    df = load_data(file)
    render_dashboard(df)
else:
    st.info("Upload a dataset to generate automatic KPIs and visualizations")
