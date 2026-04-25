import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("🤖 Production-Grade AI Data Analyst Agent")

api_key = st.text_input("Enter OpenAI API Key", type="password")

file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

# ---------------- SIDEBAR PIPELINE ----------------
st.sidebar.title("🤖 Processing Pipeline")

steps = [
    "📂 File Loaded",
    "🔍 Column Detection",
    "🧹 Data Cleaning",
    "📊 KPI Engine",
    "📈 Trend Engine",
    "🧠 Insight Engine",
    "📊 Dashboard Ready"
]

status = st.sidebar.empty()
progress = st.sidebar.progress(0)

def step(i):
    progress.progress((i + 1) / len(steps))
    status.markdown(f"### {steps[i]}")
    time.sleep(0.3)

# ---------------- LOAD DATA ----------------
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# ---------------- COLUMN CLASSIFICATION ----------------
def detect_columns(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    return numeric_cols, datetime_cols, categorical_cols

# ---------------- CLEANING ----------------
def clean_data(df):
    df = df.copy()

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df

# ---------------- KPI ENGINE ----------------
def generate_kpis(df, numeric_cols):
    kpis = {}

    for col in numeric_cols:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Average {col}"] = df[col].mean()
        kpis[f"Max {col}"] = df[col].max()

    return kpis

# ---------------- TREND ENGINE ----------------
def trend_engine(df, numeric_cols):
    trends = {}

    for col in numeric_cols:
        trends[col] = df[col].pct_change().mean()

    return trends

# ---------------- SMART GROUPING ----------------
def smart_grouping(df, categorical_cols, numeric_cols):
    results = {}

    for cat in categorical_cols:
        for num in numeric_cols:
            try:
                grouped = df.groupby(cat)[num].sum().sort_values(ascending=False)
                results[f"{cat} vs {num}"] = grouped.head(5)
            except:
                pass

    return results

# ---------------- INSIGHTS (AI) ----------------
def generate_insights(df, api_key):
    openai.api_key = api_key

    prompt = f"""
You are a senior data analyst.

Dataset summary:
Columns: {df.columns.tolist()}
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

# ---------------- VISUALIZATION ----------------
def create_charts(df, numeric_cols):
    charts = []

    for col in numeric_cols[:2]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append(fig)

    return charts

# ---------------- MAIN ----------------
if file is not None:

    step(0)
    df = load_file(file)

    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    step(1)

    numeric_cols, datetime_cols, categorical_cols = detect_columns(df)

    st.subheader("🧠 Column Detection")
    st.write("Numeric:", numeric_cols)
    st.write("Datetime:", datetime_cols)
    st.write("Categorical:", categorical_cols)

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

    grouping = smart_grouping(df, categorical_cols, numeric_cols)

    st.subheader("🧑‍🤝‍🧑 Business Breakdown")
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
    st.info("Upload file to start analysis")
