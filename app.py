import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Sales Analyzer", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f172a,#020617);
    color: white;
}
.stButton button {
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("🚀 AI Dashboard")

pages = ["Dashboard","EDA","Visualizations","Prediction"]

page = st.sidebar.radio("Navigate", pages)

file = st.sidebar.file_uploader("Upload Data", type=["csv","xlsx"])
kpi_file = st.sidebar.file_uploader("Upload KPI File", type=["csv"])

# ---------- DB ----------
conn = sqlite3.connect(":memory:")
df = None
kpi_df = None

# ---------- LOAD DATA ----------
if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df.columns = df.columns.str.strip()
    df.to_sql("sales_data", conn, index=False, if_exists="replace")

if kpi_file:
    kpi_df = pd.read_csv(kpi_file)

# ---------- SQL ENGINE ----------
def generate_sql(row):
    m = str(row["metric"]).lower()
    c = row["dimension"]

    if m == "sum":
        return f"SELECT SUM({c}) as value FROM sales_data"
    elif m == "avg":
        return f"SELECT AVG({c}) as value FROM sales_data"
    elif m == "count":
        return f"SELECT COUNT({c}) as value FROM sales_data"
    elif m == "top":
        return f"""
        SELECT {c}, COUNT(*) as value
        FROM sales_data
        GROUP BY {c}
        ORDER BY value DESC
        LIMIT 1
        """

# ---------- AUTO VISUAL ----------
def auto_visualize(df):
    st.subheader("📊 Auto Visualizations")

    numeric = df.select_dtypes(include=["int64","float64"]).columns
    categorical = df.select_dtypes(include=["object"]).columns

    for col in numeric[:3]:
        st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

    for col in categorical[:3]:
        temp = df[col].value_counts().reset_index()
        temp.columns = [col, "count"]

        st.plotly_chart(px.bar(temp, x=col, y="count"), use_container_width=True)
        st.plotly_chart(px.pie(temp, names=col, values="count"), use_container_width=True)

    if len(numeric) > 1:
        st.plotly_chart(px.imshow(df[numeric].corr(), text_auto=True), use_container_width=True)

# ---------- SMART BUILDER ----------
def smart_builder(df):
    st.subheader("🎛️ Build Your Own Chart")

    x = st.selectbox("X Axis", df.columns)
    y = st.selectbox("Y Axis", df.columns)

    chart = st.selectbox("Chart Type", ["Bar","Line","Scatter","Box"])

    if chart == "Bar":
        fig = px.bar(df, x=x, y=y)
    elif chart == "Line":
        fig = px.line(df, x=x, y=y)
    elif chart == "Scatter":
        fig = px.scatter(df, x=x, y=y)
    else:
        fig = px.box(df, x=x, y=y)

    st.plotly_chart(fig, use_container_width=True)

# ================= DASHBOARD =================
if page == "Dashboard":
    st.title("📊 Dashboard")

    if df is not None:
        c1,c2,c3 = st.columns(3)

        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Total Sales", int(df["Amount"].sum()) if "Amount" in df.columns else "N/A")

        st.dataframe(df.head())

        # KPI
        if kpi_df is not None:
            st.subheader("🤖 KPI Results")

            for _, row in kpi_df.iterrows():
                sql = generate_sql(row)

                try:
                    res = pd.read_sql(sql, conn)

                    st.markdown(f"### {row['kpi_name']}")
                    st.code(sql)
                    st.dataframe(res)

                    auto_visualize(res)

                except Exception as e:
                    st.error(str(e))

        # Auto Visuals
        auto_visualize(df)

        # Smart Builder
        smart_builder(df)

    else:
        st.warning("Upload a dataset")

# ================= EDA =================
elif page == "EDA":
    st.title("🔍 EDA")

    if df is not None:
        st.dataframe(df.describe())
        st.dataframe(df.isnull().sum())

# ================= VISUAL =================
elif page == "Visualizations":
    st.title("📈 Visualizations")

    if df is not None:
        col = st.selectbox("Column", df.columns)

        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col)
        else:
            temp = df[col].value_counts().reset_index()
            temp.columns = [col,"count"]
            fig = px.bar(temp, x=col, y="count")

        st.plotly_chart(fig, use_container_width=True)

# ================= PREDICTION =================
elif page == "Prediction":
    st.title("🤖 Prediction")

    if df is not None and "Amount" in df.columns:
        nums = df.select_dtypes(include=["int64","float64"]).columns

        if len(nums)>=2:
            X_col = st.selectbox("Feature", nums)
            X = df[[X_col]]
            y = df["Amount"]

            model = LinearRegression()
            model.fit(X,y)

            st.metric("R²", f"{r2_score(y, model.predict(X)):.2f}")

            val = st.slider("Input", int(df[X_col].min()), int(df[X_col].max()), int(df[X_col].mean()))

            if st.button("Predict"):
                pred = model.predict([[val]])
                st.success(f"₹ {int(pred[0]):,}")
