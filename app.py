import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(page_title="AI BI Analyst", layout="wide")

st.title("🤖 AI Data Analyst Agent (BI System)")
st.write("Upload dataset → Get KPIs, insights, dashboards, reports")

# ================= INPUT =================
file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

user_goal = st.text_area(
    "🎯 Enter Business Goal (optional)",
    placeholder="e.g. Increase sales, analyze region performance"
)

# ================= DATA LOADING =================
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# ================= CLEANING PIPELINE =================
def clean_data(df):
    df = df.copy()

    # remove duplicates
    df = df.drop_duplicates()

    for col in df.columns:

        # numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

        # datetime fix attempt
        elif "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].fillna(method="ffill")

        # categorical
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ================= KPI ENGINE =================
def kpi_engine(df):
    numeric = df.select_dtypes(include="number").columns

    kpis = {}

    for col in numeric:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Average {col}"] = df[col].mean()
        kpis[f"Max {col}"] = df[col].max()

    return kpis

# ================= AGGREGATION ENGINE =================
def aggregation_engine(df):
    numeric = df.select_dtypes(include="number").columns
    categorical = df.select_dtypes(include=["object"]).columns

    results = {}

    if len(categorical) > 0 and len(numeric) > 0:
        cat = categorical[0]
        num = numeric[0]

        results["top_group"] = df.groupby(cat)[num].sum().sort_values(ascending=False).head(5)

    return results

# ================= TREND ENGINE =================
def trend_engine(df):
    numeric = df.select_dtypes(include="number").columns

    trends = {}

    for col in numeric:
        trends[col] = df[col].pct_change().mean()

    return trends

# ================= INSIGHT ENGINE =================
def insight_engine(df):
    insights = []

    numeric = df.select_dtypes(include="number").columns

    for col in numeric:
        mean = df[col].mean()
        median = df[col].median()

        if mean > median:
            insights.append(f"📈 {col} shows strong upward bias (possible growth)")
        else:
            insights.append(f"📊 {col} is stable or evenly distributed")

    if len(numeric) > 1:
        insights.append("🔗 Multiple numeric variables detected → correlation possible")

    return insights

# ================= VISUAL ENGINE =================
def visualization_engine(df):
    charts = []

    numeric = df.select_dtypes(include="number").columns
    categorical = df.select_dtypes(include=["object"]).columns

    # histograms
    for col in numeric:
        charts.append(px.histogram(df, x=col, title=f"{col} Distribution"))

    # bar chart
    if len(categorical) > 0 and len(numeric) > 0:
        cat = categorical[0]
        num = numeric[0]

        grouped = df.groupby(cat)[num].sum().reset_index()
        charts.append(px.bar(grouped, x=cat, y=num, title=f"{cat} vs {num}"))

    # correlation
    if len(numeric) > 1:
        charts.append(px.imshow(df[numeric].corr(), text_auto=True, title="Correlation Heatmap"))

    return charts

# ================= SUMMARY REPORT =================
def summary_report(df, kpis, insights, trends):
    report = f"""
    📊 DATA SUMMARY REPORT

    Rows: {df.shape[0]}
    Columns: {df.shape[1]}

    --- KPIs ---
    {kpis}

    --- INSIGHTS ---
    {insights}

    --- TRENDS ---
    {trends}
    """
    return report

# ================= MAIN APP =================
if file is not None:

    # LOAD
    df = load_data(file)

    st.subheader("📄 Raw Data")
    st.dataframe(df.head(), use_container_width=True)

    # CLEAN
    df = clean_data(df)

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df.head(), use_container_width=True)

    # KPIs
    st.subheader("📊 KPIs")
    kpis = kpi_engine(df)

    cols = st.columns(min(5, len(kpis)))

    for i, (k, v) in enumerate(kpis.items()):
        with cols[i % len(cols)]:
            st.metric(k, f"{v:.2f}" if isinstance(v, float) else v)

    # AGGREGATION
    st.subheader("📦 Aggregations")
    agg = aggregation_engine(df)

    for k, v in agg.items():
        st.write(k)
        st.write(v)

    # TRENDS
    st.subheader("📈 Trends")
    trends = trend_engine(df)
    st.json(trends)

    # INSIGHTS
    st.subheader("🧠 Insights")
    insights = insight_engine(df)

    for i in insights:
        st.write("✔", i)

    # VISUALS
    st.subheader("📊 Visual Dashboard")
    charts = visualization_engine(df)

    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    # REPORT
    st.subheader("📄 Summary Report")

    report = summary_report(df, kpis, insights, trends)
    st.text_area("Auto Report", report, height=300)

else:
    st.info("Upload dataset to start AI analysis")
