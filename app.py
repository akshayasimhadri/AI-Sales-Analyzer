import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(page_title="Zero-Error BI Agent", layout="wide")

st.title("🤖 Zero-Error Autonomous BI Agent")
st.write("Upload any dataset → AI will auto-fix + analyze it safely")

# ================= UPLOAD =================
file = st.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# ================= SAFE LOAD =================
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"File load error: {e}")
        return None

# ================= SELF-HEALING CLEANER =================
def clean_data(df):
    df = df.copy()

    # 1. Remove duplicates safely
    df = df.drop_duplicates()

    for col in df.columns:

        # TRY convert numeric if possible
        df[col] = pd.to_numeric(df[col], errors="ignore")

        # NUMERIC
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

        # DATE FIX (SAFE)
        elif "date" in col.lower() or "time" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].ffill()

        # STRING / OTHER
        else:
            df[col] = df[col].astype(str).fillna("Unknown")

    return df

# ================= KPI ENGINE =================
def generate_kpis(df):
    kpis = {}
    numeric = df.select_dtypes(include="number").columns

    for col in numeric:
        try:
            kpis[f"Total {col}"] = float(df[col].sum())
            kpis[f"Avg {col}"] = float(df[col].mean())
            kpis[f"Max {col}"] = float(df[col].max())
        except:
            continue

    return kpis

# ================= VISUAL ENGINE =================
def generate_charts(df):
    charts = []

    numeric = df.select_dtypes(include="number").columns
    categorical = df.select_dtypes(include=["object"]).columns

    # Safe histogram
    for col in numeric:
        try:
            charts.append(px.histogram(df, x=col, title=f"{col} Distribution"))
        except:
            pass

    # Safe category chart
    if len(categorical) > 0 and len(numeric) > 0:
        try:
            cat = categorical[0]
            num = numeric[0]

            grouped = df.groupby(cat)[num].sum().reset_index()
            charts.append(px.bar(grouped, x=cat, y=num, title=f"{cat} vs {num}"))
        except:
            pass

    # Correlation heatmap
    if len(numeric) > 1:
        try:
            charts.append(px.imshow(df[numeric].corr(), text_auto=True))
        except:
            pass

    return charts

# ================= INSIGHTS =================
def generate_insights(df):
    insights = []
    numeric = df.select_dtypes(include="number").columns

    for col in numeric:
        try:
            mean = df[col].mean()
            median = df[col].median()

            if mean > median:
                insights.append(f"📈 {col} shows growth bias (high values dominate)")
            else:
                insights.append(f"📊 {col} is stable or balanced")
        except:
            continue

    if len(numeric) == 0:
        insights.append("⚠ No numeric data found for analysis")

    return insights

# ================= MAIN APP =================
if file is not None:

    df = load_data(file)

    if df is None:
        st.stop()

    st.subheader("📄 Raw Data")
    st.dataframe(df.head(), width="stretch")

    # CLEAN (SELF HEALING)
    df = clean_data(df)

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df.head(), width="stretch")

    # KPI
    st.subheader("📊 Auto KPIs")
    kpis = generate_kpis(df)

    if kpis:
        cols = st.columns(min(5, len(kpis)))

        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.2f}")
    else:
        st.warning("No KPIs generated")

    # CHARTS
    st.subheader("📈 Auto Visualizations")
    charts = generate_charts(df)

    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    # INSIGHTS
    st.subheader("🧠 Insights")
    insights = generate_insights(df)

    for i in insights:
        st.write("✔", i)

else:
    st.info("Upload dataset to begin AI analysis")
