import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("🤖 AI Data Analyst Agent")
st.write("Upload CSV/Excel and get automatic insights, KPIs, and dashboards")

# ---------------- API KEY ----------------
api_key = st.text_input("Enter OpenAI API Key", type="password")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# ---------------- SIDEBAR PIPELINE ----------------
st.sidebar.title("🤖 Processing Pipeline")

steps = [
    "📂 File Loaded",
    "🔍 Understanding Data",
    "🧹 Cleaning Data",
    "📊 KPI Calculation",
    "📈 Trend Analysis",
    "🧠 AI Insights",
    "📊 Visualization Ready"
]

status = st.sidebar.empty()
progress = st.sidebar.progress(0)

def update_step(i):
    progress.progress((i + 1) / len(steps))
    status.markdown(f"### {steps[i]}")
    time.sleep(0.3)

# ---------------- LOAD DATA ----------------
def load_data(file):
    if file.name.endswith("csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# ---------------- CLEAN DATA ----------------
def clean_data(df):
    df = df.copy()

    # remove duplicates
    df = df.drop_duplicates()

    # fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    return df

# ---------------- KPI ENGINE ----------------
def generate_kpis(df):
    kpis = {}

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Average {col}"] = df[col].mean()
        kpis[f"Max {col}"] = df[col].max()

    return kpis

# ---------------- TREND ANALYSIS ----------------
def trend_analysis(df):
    trends = {}

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        trends[col] = df[col].pct_change().mean()

    return trends

# ---------------- AI INSIGHTS ----------------
def generate_insights(df, api_key):
    openai.api_key = api_key

    prompt = f"""
You are a senior data analyst.

Dataset columns:
{df.columns.tolist()}

Sample data:
{df.head(10).to_string()}

Provide:
1. Key insights
2. Trends
3. Business recommendations
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# ---------------- CHARTS ----------------
def create_charts(df):
    charts = []

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols[:2]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append(fig)

    return charts

# ---------------- MAIN APP ----------------
if file is not None:

    update_step(0)
    df = load_data(file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    update_step(1)

    st.subheader("🧠 Data Understanding")
    st.write(df.info())

    update_step(2)

    df = clean_data(df)

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df.head())

    update_step(3)

    kpis = generate_kpis(df)

    st.subheader("📊 KPIs")
    for k, v in kpis.items():
        st.metric(label=k, value=f"{v:.2f}" if isinstance(v, float) else v)

    update_step(4)

    trends = trend_analysis(df)

    st.subheader("📈 Trend Analysis")
    st.json(trends)

    update_step(5)

    if api_key:
        insights = generate_insights(df, api_key)
        st.subheader("🧠 AI Insights")
        st.success(insights)
    else:
        st.warning("Enter API key for AI insights")

    update_step(6)

    st.subheader("📊 Visualizations")

    charts = create_charts(df)
    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    # ---------------- FINAL REPORT ----------------
    st.subheader("📄 Auto Report")

    st.markdown(f"""
### Executive Summary

- Rows: {df.shape[0]}
- Columns: {df.shape[1]}

### Key KPIs
{list(kpis.items())[:5]}

### Trend Summary
{trends}
""")

    st.sidebar.success("✅ Analysis Complete")

else:
    st.info("Upload a file to start analysis")
