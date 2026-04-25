import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ================= CONFIG =================
st.set_page_config(page_title="AI BI Copilot", layout="wide")

st.title("🤖 AI ANALYZER (Animated Pipeline Edition)")

# ================= SIDEBAR NAV =================
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "📊 EDA", "📈 Visualizations", "🔮 Prediction"]
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

        time.sleep(0.4)

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

# ================= MAIN =================
if file is not None:

    # STEP 1: PIPELINE START
    smooth_pipeline()

    # LOAD DATA
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("📥 Data Loaded")

    # CLEAN
    df = clean(df)
    st.success("🧹 Data Cleaned")

    numeric = df.select_dtypes(include="number").columns
    categorical = df.select_dtypes(include=["object"]).columns

    # ================= DASHBOARD =================
    if page == "🏠 Dashboard":

        st.subheader("📊 KPI Dashboard")

        kpis = kpi_engine(df)

        cols = st.columns(min(4, len(kpis)))

        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.2f}")

        st.markdown("---")

        st.subheader("📈 Quick Insights")

        if len(numeric) > 0:
            st.write("📌 Highest value column:", numeric[0])

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

        st.markdown("### 📊 Auto Charts")

        for col in numeric[:3]:
            st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

    # ================= PREDICTION =================
    elif page == "🔮 Prediction":

        st.subheader("🔮 Prediction Engine")

        if len(numeric) > 0:

            target = st.selectbox("Select Target Column", numeric)

            df["Prediction"] = df[target].rolling(3).mean()

            fig = px.line(df, y=[target, "Prediction"], title="Trend Prediction")

            st.plotly_chart(fig, use_container_width=True)

            st.info("Basic AI prediction using rolling average model")

else:
    st.info("📂 Upload dataset to start AI pipeline")
