import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(page_title="AI BI Analyst", layout="wide")

st.title("🤖 AI Autonomous BI Analyst")
st.write("Upload CSV/Excel → Auto KPIs, Charts, Insights")

# ================= UPLOAD =================
file = st.file_uploader("📂 Upload Dataset", type=["csv", "xlsx"])

# ================= SAFE LOADER =================
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"File loading error: {e}")
        return None

# ================= ROBUST CLEANING =================
def clean_data(df):
    df = df.copy()

    # remove duplicates
    df = df.drop_duplicates()

    for col in df.columns:

        # TRY convert to numeric safely
        try:
            converted = pd.to_numeric(df[col], errors="coerce")

            # if at least some values become numeric → treat as numeric
            if converted.notna().sum() > 0:
                df[col] = converted
                df[col] = df[col].fillna(df[col].median())
                continue
        except:
            pass

        # DATE handling
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].ffill()
            except:
                df[col] = df[col].fillna("Unknown")
            continue

        # categorical fallback
        df[col] = df[col].astype(str).fillna("Unknown")

    return df

# ================= KPI ENGINE =================
def generate_kpis(df):
    kpis = {}
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
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

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # numeric distributions
    for col in numeric_cols:
        try:
            charts.append(px.histogram(df, x=col, title=f"{col} Distribution"))
        except:
            pass

    # category vs numeric
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        try:
            cat = categorical_cols[0]
            num = numeric_cols[0]

            grouped = df.groupby(cat)[num].sum().reset_index()
            charts.append(px.bar(grouped, x=cat, y=num, title=f"{cat} vs {num}"))
        except:
            pass

    # correlation heatmap
    if len(numeric_cols) > 1:
        try:
            charts.append(px.imshow(df[numeric_cols].corr(), text_auto=True))
        except:
            pass

    return charts

# ================= INSIGHTS =================
def generate_insights(df):
    insights = []
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        return ["⚠ No numeric data found for insights"]

    for col in numeric_cols:
        try:
            if df[col].mean() > df[col].median():
                insights.append(f"📈 {col} shows growth tendency")
            else:
                insights.append(f"📊 {col} is stable")
        except:
            continue

    return insights

# ================= MAIN APP =================
if file is not None:

    df = load_data(file)

    if df is None:
        st.stop()

    st.subheader("📄 Raw Data")
    st.dataframe(df.head(), width="stretch")

    # CLEAN
    df = clean_data(df)

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df.head(), width="stretch")

    # KPIs
    st.subheader("📊 KPIs")
    kpis = generate_kpis(df)

    if kpis:
        cols = st.columns(min(5, len(kpis)))

        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.2f}")
    else:
        st.warning("No KPIs generated")

    # CHARTS
    st.subheader("📈 Visualizations")
    charts = generate_charts(df)

    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    # INSIGHTS
    st.subheader("🧠 Insights")
    insights = generate_insights(df)

    for i in insights:
        st.write("✔", i)

else:
    st.info("Upload a file to start analysis")
