import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sqlite3
import time
import requests
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
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#0f172a);
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("🚀 AI Dashboard")

pages = ["Dashboard","EDA","Visualizations","Prediction","APIs"]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

page = st.sidebar.radio("Navigate", pages, index=pages.index(st.session_state.page))

file = st.sidebar.file_uploader("Upload Data", type=["csv","xlsx"])
kpi_file = st.sidebar.file_uploader("Upload KPI File", type=["csv"])

# ---------- SESSION ----------
if "pipeline_step" not in st.session_state:
    st.session_state.pipeline_step = 0

if "logs" not in st.session_state:
    st.session_state.logs = []

# ---------- DB ----------
conn = sqlite3.connect(":memory:")

df = None
kpi_df = None

# ---------- LOAD CSV ----------
if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df.columns = df.columns.str.strip()
    df.to_sql("sales_data", conn, index=False, if_exists="replace")

# ---------- LOAD KPI ----------
if kpi_file:
    kpi_df = pd.read_csv(kpi_file)

# ---------- LOAD API DATA ----------
if "df" in st.session_state:
    df = st.session_state["df"]

# ---------- PIPELINE ----------
def pipeline_sidebar(step):
    st.sidebar.markdown("## ⚙️ Data Pipeline")
    st.sidebar.markdown("---")

    steps = [
        "📂 Data Ingestion",
        "🧠 SQL Processing",
        "🔍 Python EDA",
        "📊 Dashboard",
        "🤖 Prediction"
    ]

    for i, s in enumerate(steps):
        if i < step:
            st.sidebar.success(s)
        elif i == step:
            st.sidebar.warning(f"{s} 🔄")
        else:
            st.sidebar.info(s)

# ---------- SQL ----------
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

# ---------- PIPELINE RUN ----------
def run_pipeline():
    log_box = st.empty()
    progress = st.progress(0)

    steps = [
        "📂 Loading Data...",
        "🧠 Running SQL...",
        "🔍 Performing EDA...",
        "📊 Building Dashboard...",
        "🤖 Training Model..."
    ]

    for i, msg in enumerate(steps):
        st.session_state.logs.append(msg)
        log_box.code("\n".join(st.session_state.logs))
        progress.progress((i+1)/len(steps))
        pipeline_sidebar(i)
        time.sleep(1)

    st.session_state.pipeline_step = 5
    st.success("✅ Pipeline Completed")

# ---------- RESET ----------
if file or kpi_file:
    st.session_state.pipeline_step = 0
    st.session_state.logs = []

# ---------- AUTO RUN ----------
if df is not None and kpi_df is not None and st.session_state.pipeline_step == 0:
    st.session_state.page = "Dashboard"
    run_pipeline()

# ================= DASHBOARD =================
if page == "Dashboard":
    st.title("📊 Dashboard")

    if df is not None:
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Total Sales", int(df["Amount"].sum()) if "Amount" in df.columns else "N/A")

        st.dataframe(df.head())

        if kpi_df is not None:
            st.subheader("🤖 KPI Results with SQL")

            results = []

            for _, row in kpi_df.iterrows():
                sql = generate_sql(row)

                try:
                    res = pd.read_sql(sql, conn)
                    val = res.iloc[0,0] if res.shape[1]==1 else res.to_dict("records")[0]

                    results.append({
                        "KPI Name": row["kpi_name"],
                        "SQL Query": sql.strip(),
                        "Result": val
                    })
                except Exception as e:
                    results.append({
                        "KPI Name": row["kpi_name"],
                        "SQL Query": sql,
                        "Result": str(e)
                    })

            st.dataframe(pd.DataFrame(results))

        if "Amount" in df.columns:
            st.plotly_chart(px.histogram(df, x="Amount"), use_container_width=True)

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

# ================= APIs =================
elif page == "APIs":
    st.title("🌐 API Integration")

    api_options = {
        "Users": "https://jsonplaceholder.typicode.com/users",
        "Posts": "https://jsonplaceholder.typicode.com/posts",
        "Products": "https://dummyjson.com/products"
    }

    selected = st.selectbox("Select API", list(api_options.keys()))
    url = api_options[selected]

    if st.button("Fetch API Data"):
        try:
            res = requests.get(url)
            data = res.json()

            if isinstance(data, dict) and "products" in data:
                df_api = pd.DataFrame(data["products"])
            else:
                df_api = pd.DataFrame(data)

            st.success("API Loaded")
            st.dataframe(df_api.head())

            df_api.columns = df_api.columns.str.strip()
            df_api.to_sql("sales_data", conn, index=False, if_exists="replace")

            st.session_state["df"] = df_api

            st.info("Go to Dashboard to analyze")

        except Exception as e:
            st.error(str(e))
