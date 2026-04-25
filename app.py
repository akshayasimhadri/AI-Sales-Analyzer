import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(page_title="AI BI Copilot Pro", layout="wide")

st.title("🤖 AI BI Copilot Pro (Self-Validating System)")
st.write("Upload data → Auto KPIs + Validation + Custom Visuals")

# ================= FILE =================
file = st.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# ================= CLEANING =================
def clean(df):
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ================= DATA QUALITY SCORE =================
def data_quality_score(df):
    score = 100

    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1] + 1)

    score -= missing_ratio * 50
    score -= df.duplicated().sum() * 0.1

    return max(0, min(100, score))

# ================= KPI ENGINE =================
def kpi_engine(df):
    kpis = {}
    num = df.select_dtypes(include="number").columns

    for col in num:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Avg {col}"] = df[col].mean()
        kpis[f"Max {col}"] = df[col].max()

    return kpis

# ================= KPI VALIDATION =================
def validate_kpis(df, kpis):
    issues = []

    num = df.select_dtypes(include="number").columns

    for col in num:
        actual = df[col].sum()
        calc = kpis.get(f"Total {col}", None)

        if calc is not None and abs(actual - calc) > 1e-6:
            issues.append(f"⚠ KPI mismatch in {col}")

    return issues

# ================= VISUAL ENGINE =================
def plot_chart(df, chart_type, x, y=None):

    if chart_type == "Bar Chart":
        return px.bar(df, x=x, y=y if y else x)

    elif chart_type == "Pie Chart":
        return px.pie(df, names=x)

    elif chart_type == "Line Chart":
        return px.line(df, x=x, y=y if y else x)

    elif chart_type == "Histogram":
        return px.histogram(df, x=x)

    return None

# ================= MAIN =================
if file is not None:

    # LOAD
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("📥 Data Loaded")

    # CLEAN
    df = clean(df)
    st.success("🧹 Data Cleaned")

    # QUALITY SCORE
    score = data_quality_score(df)

    st.subheader("🧪 Data Quality Score")
    st.metric("Score", f"{score:.2f}/100")

    if score < 50:
        st.warning("⚠ Low quality dataset detected")

    # ================= KPIs =================
    st.subheader("📊 KPIs")

    kpis = kpi_engine(df)

    cols = st.columns(min(4, len(kpis)))

    for i, (k, v) in enumerate(kpis.items()):
        with cols[i % len(cols)]:
            st.metric(k, f"{v:.2f}")

    # VALIDATION
    issues = validate_kpis(df, kpis)

    if issues:
        st.subheader("⚠ KPI Validation Issues")
        for i in issues:
            st.write(i)
    else:
        st.success("✔ All KPIs validated successfully")

    # ================= VISUALIZATION =================
    st.subheader("📈 Custom Visualization Engine")

    numeric = df.select_dtypes(include="number").columns
    categorical = df.select_dtypes(include=["object"]).columns

    chart_type = st.selectbox(
        "Choose Chart Type",
        ["Bar Chart", "Pie Chart", "Line Chart", "Histogram"]
    )

    if len(numeric) > 0:

        x_axis = st.selectbox("X Axis", df.columns)

        y_axis = None
        if chart_type in ["Bar Chart", "Line Chart"]:
            y_axis = st.selectbox("Y Axis (optional)", numeric)

        fig = plot_chart(df, chart_type, x_axis, y_axis)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ================= DATA PREVIEW =================
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

else:
    st.info("📂 Upload dataset to start AI BI analysis")
