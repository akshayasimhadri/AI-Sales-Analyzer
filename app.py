import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("🤖 AI Data Analyst Agent (Stable Version)")
st.write("Upload CSV/Excel → Auto KPIs, Insights, Dashboards")

# ---------------- API KEY ----------------
api_key = st.text_input("Enter OpenAI API Key", type="password")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# ---------------- SIDEBAR PIPELINE ----------------
st.sidebar.title("🤖 Processing Pipeline")

steps = [
    "📂 File Loaded",
    "🔍 Column Detection",
    "🧹 Data Cleaning",
    "📊 KPI Engine",
    "📈 Trend Engine",
    "🧠 AI Insights",
    "📊 Visualization Ready"
]

status = st.sidebar.empty()
progress = st.sidebar.progress(0)

def step(i):
    progress.progress((i + 1) / len(steps))
    status.markdown(f"### {steps[i]}")
    time.sleep(0.3)

# ---------------- LOAD FILE ----------------
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# ---------------- COLUMN DETECTION ----------------
def detect_columns(df):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "string"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    return numeric, categorical, datetime_cols

# ---------------- CLEAN DATA (FIXED) ----------------
def clean_data(df):
    df = df.copy()

    # remove duplicates
    df = df.drop_duplicates()

    for col in df.columns:

        # STRING columns
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")

        # NUMERIC columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

        # DATETIME columns
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].ffill()   # ✅ FIXED (no deprecated method)

    return df

# ---------------- KPI ENGINE ----------------
def generate_kpis(df, numeric_cols):
    kpis = {}

    for col in numeric_cols:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Average {col}"] = df[col].mean()
        kpis[f"Median {col}"] = df[col].median()
        kpis[f"Max {col}"] = df[col].max()

    return kpis

# ---------------- TREND ENGINE ----------------
def trend_engine(df, numeric_cols):
    trends = {}

    for col in numeric_cols:
        trends[col] = df[col].pct_change().mean()

    return trends

# ---------------- GROUP INSIGHTS ----------------
def grouping_engine(df, categorical_cols, numeric_cols):
    results = {}

    for cat in categorical_cols:
        for num in numeric_cols:
            try:
                results[f"{cat} → {num}"] = (
                    df.groupby(cat)[num]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )
            except:
                pass

    return results

# ---------------- AI INSIGHTS ----------------
def generate_insights(df, api_key):
    openai.api_key = api_key

    prompt = f"""
You are a senior data analyst.

Dataset columns: {df.columns.tolist()}
Shape: {df.shape}

Sample:
{df.head(10).to_string()}

Give:
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
def create_charts(df, numeric_cols):
    charts = []

    for col in numeric_cols[:2]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append(fig)

    return charts

# ---------------- MAIN APP ----------------
if file is not None:

    step(0)
    df = load_file(file)

    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    step(1)

    numeric_cols, categorical_cols, datetime_cols = detect_columns(df)

    st.write("Numeric:", numeric_cols)
    st.write("Categorical:", categorical_cols)
    st.write("Datetime:", datetime_cols)

    step(2)

    df = clean_data(df)

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df.head())

    step(3)

    kpis = generate_kpis(df, numeric_cols)

    st.subheader("📊 KPIs")
    for k, v in kpis.items():
        st.metric(k, f"{v:.2f}" if isinstance(v, float) else v)

    step(4)

    trends = trend_engine(df, numeric_cols)

    st.subheader("📈 Trends")
    st.json(trends)

    grouping = grouping_engine(df, categorical_cols, numeric_cols)

    st.subheader("🧑‍🤝‍🧑 Top Group Insights")
    for k, v in list(grouping.items())[:3]:
        st.write(k)
        st.write(v)

    step(5)

    if api_key:
        insights = generate_insights(df, api_key)
        st.subheader("🧠 AI Insights")
        st.success(insights)

    step(6)

    st.subheader("📊 Charts")
    charts = create_charts(df, numeric_cols)

    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    st.sidebar.success("✅ Analysis Complete")

else:
    st.info("Upload a CSV or Excel file to start analysis")
