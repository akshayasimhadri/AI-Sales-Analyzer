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
    border-radius: 8px;
}
h1, h2, h3 {
    color: #e2e8f0;
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

# ---------- LOAD ----------
if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df.columns = df.columns.str.strip()
    df.to_sql("sales_data", conn, index=False, if_exists="replace")

if kpi_file:
    kpi_df = pd.read_csv(kpi_file)

# ---------- 🔧 FIX: SAFE COLUMN MAPPER ----------
def safe_col(df, col):
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    key = col.lower().replace(" ", "")
    return cols.get(key, col)

# ---------- SQL ----------
def generate_sql(row):
    m = str(row["metric"]).lower()
    
    c = safe_col(df, row["dimension"])
    c = f'"{c}"'   # ✅ FIX: handles spaces safely for SQLite  # ✅ FIXED HERE

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

    for i, col in enumerate(numeric[:3]):
        st.markdown(f"### 📈 {col} Distribution")
        if st.button(f"🔄 Reset {col}", key=f"num_{i}"):
            st.rerun()
        st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

    for i, col in enumerate(categorical[:3]):
        st.markdown(f"### 📊 {col} Count")
        if st.button(f"🔄 Reset {col}", key=f"cat_{i}"):
            st.rerun()

        temp = df[col].value_counts().reset_index()
        temp.columns = [col, "count"]

        st.plotly_chart(px.bar(temp, x=col, y="count", text="count"), use_container_width=True)
        st.plotly_chart(px.pie(temp, names=col, values="count"), use_container_width=True)

    if len(numeric) > 1:
        st.markdown("### 🔥 Correlation Heatmap")
        if st.button("🔄 Reset Correlation"):
            st.rerun()
        st.plotly_chart(px.imshow(df[numeric].corr(), text_auto=True), use_container_width=True)

# ---------- SMART BUILDER ----------
def smart_builder(df):
    st.subheader("🎛️ Build Your Own Chart")

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X Axis", df.columns)
    y = c2.selectbox("Y Axis", df.columns)
    chart = c3.selectbox("Chart Type", ["Bar","Line","Scatter","Box"])

    if st.button("🔄 Reset Chart"):
        st.rerun()

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
    st.title("📊 AI Sales Dashboard")

    if df is not None:
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Total Sales", int(df["Amount"].sum()) if "Amount" in df.columns else "N/A")

        st.divider()
        st.dataframe(df.head())

        if kpi_df is not None:
            st.divider()
            st.subheader("🤖 KPI Insights")

            for _, row in kpi_df.iterrows():
                sql = generate_sql(row)

                try:
                    res = pd.read_sql(sql, conn)

                    st.markdown(f"### {row['kpi_name']}")
                    st.code(sql)

                    if res.shape[1] == 2:
                        x = res.columns[0]
                        y = res.columns[1]

                        if st.button(f"🔄 Reset {row['kpi_name']}", key=row['kpi_name']):
                            st.rerun()

                        st.plotly_chart(px.bar(res, x=x, y=y, text=y), use_container_width=True)
                    else:
                        st.metric(row['kpi_name'], res.iloc[0,0])

                except Exception as e:
                    st.error(str(e))

        st.divider()

        auto_visualize(df)

        st.divider()

        smart_builder(df)

    else:
        st.warning("Upload a dataset")

elif page == "EDA":
    st.title("🔍 EDA")

    if df is not None:
        st.dataframe(df.describe())
        st.dataframe(df.isnull().sum())

elif page == "Visualizations":
    st.title("📈 Visualizations")

    if df is not None:
        col = st.selectbox("Column", df.columns)

        if st.button("🔄 Reset Visualization"):
            st.rerun()

        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col)
        else:
            temp = df[col].value_counts().reset_index()
            temp.columns = [col,"count"]
            fig = px.bar(temp, x=col, y="count")

        st.plotly_chart(fig, use_container_width=True)

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

            st.metric("R² Score", f"{r2_score(y, model.predict(X)):.2f}")

            val = st.slider("Input Value", int(df[X_col].min()), int(df[X_col].max()), int(df[X_col].mean()))

            if st.button("Predict"):
                pred = model.predict([[val]])
                st.success(f"Predicted Sales: ₹ {int(pred[0]):,}")
