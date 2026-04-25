import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Autonomous AI Data Analyst", layout="wide")

st.title("🤖 Autonomous AI Data Analyst Agent v2")
st.write("Upload data + define goal → AI builds full analysis automatically")

# ---------------- API KEY ----------------
api_key = st.text_input("Enter OpenAI API Key", type="password")

# ---------------- USER GOAL ----------------
user_goal = st.text_area(
    "🎯 What is your business goal?",
    placeholder="e.g., Increase sales, reduce churn, analyze product performance"
)

# ---------------- FILE ----------------
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# ---------------- PIPELINE UI ----------------
st.sidebar.title("🤖 Agent Pipeline")

steps = [
    "📂 Input Loaded",
    "🧠 Understanding Goal",
    "🔍 Profiling Data",
    "🧹 Cleaning Data",
    "📊 KPI Selection (AI)",
    "📈 Analysis Engine",
    "🧠 Insight Generation",
    "📊 Dashboard Build"
]

status = st.sidebar.empty()
progress = st.sidebar.progress(0)

def step(i):
    progress.progress((i + 1) / len(steps))
    status.markdown(f"### {steps[i]}")
    time.sleep(0.3)

# ---------------- LOAD DATA ----------------
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# ---------------- CLEAN DATA ----------------
def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ---------------- AUTO KPI SELECTION (BASED ON GOAL) ----------------
def select_kpis(df, goal):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    kpi_map = {}

    goal = goal.lower()

    for col in numeric_cols:
        if "sales" in goal or "revenue" in goal:
            kpi_map[col] = ["sum", "mean", "max"]

        elif "churn" in goal:
            kpi_map[col] = ["mean", "count"]

        else:
            kpi_map[col] = ["sum", "mean"]

    return kpi_map

# ---------------- KPI ENGINE ----------------
def compute_kpis(df, kpi_map):
    kpis = {}

    for col, ops in kpi_map.items():
        for op in ops:
            if op == "sum":
                kpis[f"Total {col}"] = df[col].sum()
            elif op == "mean":
                kpis[f"Average {col}"] = df[col].mean()
            elif op == "max":
                kpis[f"Max {col}"] = df[col].max()
            elif op == "count":
                kpis[f"Count {col}"] = df[col].count()

    return kpis

# ---------------- TREND ENGINE ----------------
def trend_engine(df):
    numeric_cols = df.select_dtypes(include="number").columns
    trends = {}

    for col in numeric_cols:
        trends[col] = df[col].pct_change().mean()

    return trends

# ---------------- GROUP ANALYSIS ----------------
def grouping_engine(df):
    cat = df.select_dtypes(include=["object"]).columns
    num = df.select_dtypes(include="number").columns

    results = {}

    for c in cat:
        for n in num:
            try:
                results[f"{c} vs {n}"] = df.groupby(c)[n].sum().sort_values(ascending=False).head(5)
            except:
                pass

    return results

# ---------------- AI INSIGHTS ----------------
def generate_insights(df, goal, kpis, api_key):
    openai.api_key = api_key

    prompt = f"""
You are a senior data analyst AI.

Business Goal:
{goal}

KPIs:
{kpis}

Dataset:
Columns: {df.columns.tolist()}
Shape: {df.shape}

Sample:
{df.head(10).to_string()}

Generate:
1. Key insights
2. Why it is happening
3. Business impact
4. Action recommendations
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# ---------------- VISUALS ----------------
def create_charts(df):
    charts = []

    num = df.select_dtypes(include="number").columns

    for col in num[:2]:
        charts.append(px.histogram(df, x=col, title=f"{col} Distribution"))

    return charts

# ---------------- MAIN ----------------
if file is not None:

    step(0)
    df = load_data(file)

    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    step(1)

    if not user_goal:
        st.warning("Please enter business goal to continue")
        st.stop()

    step(2)

    st.subheader("🧠 Goal Understanding")
    st.info(user_goal)

    step(3)

    df = clean_data(df)

    step(4)

    kpi_map = select_kpis(df, user_goal)
    kpis = compute_kpis(df, kpi_map)

    st.subheader("📊 AI-Selected KPIs")
    for k, v in kpis.items():
        st.metric(k, f"{v:.2f}" if isinstance(v, float) else v)

    step(5)

    trends = trend_engine(df)

    st.subheader("📈 Trends")
    st.json(trends)

    grouping = grouping_engine(df)

    st.subheader("🧑‍🤝‍🧑 Business Drivers")
    for k, v in list(grouping.items())[:3]:
        st.write(k)
        st.write(v)

    step(6)

    if api_key:
        insights = generate_insights(df, user_goal, kpis, api_key)

        st.subheader("🧠 AI Business Insights")
        st.success(insights)

    step(7)

    st.subheader("📊 Dashboard")

    charts = create_charts(df)

    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    st.sidebar.success("✅ Autonomous Analysis Complete")

else:
    st.info("Upload a dataset to begin")
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
