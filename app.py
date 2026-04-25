import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sqlite3
import time
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

pages = ["Dashboard","EDA","Visualizations","Prediction"]

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

# ---------- LOAD ----------
conn = sqlite3.connect(":memory:")
df = None
kpi_df = None

if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df.columns = df.columns.str.strip()
    df.to_sql("sales_data", conn, index=False, if_exists="replace")

if kpi_file:
    kpi_df = pd.read_csv(kpi_file)

# ---------- PIPELINE SIDEBAR ----------
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

# ---------- KPI SQL ENGINE ----------
def generate_dynamic_sql(row):
    metric = str(row.get("metric","")).lower()
    col = row.get("dimension","")
    group = row.get("group_by","")
    flt = row.get("filter","")
    order = row.get("order_by","")
    limit = row.get("limit","")

    if metric == "sum":
        select = f"SUM({col}) as value"
    elif metric == "avg":
        select = f"AVG({col}) as value"
    elif metric == "count":
        select = f"COUNT({col}) as value"
    elif metric == "min":
        select = f"MIN({col}) as value"
    elif metric == "max":
        select = f"MAX({col}) as value"
    elif metric == "top":
        select = f"{col}, COUNT(*) as value"
        group = col
        order = "value DESC"
        limit = 1
    else:
        return None

    sql = f"SELECT {group + ',' if group else ''} {select} FROM sales_data"

    if flt:
        sql += f" WHERE {flt}"
    if group:
        sql += f" GROUP BY {group}"
    if order:
        sql += f" ORDER BY {order}"
    if limit:
        sql += f" LIMIT {limit}"

    return sql

# ---------- VISUALIZE KPI ----------
def visualize_kpi(df_result, kpi_name):
    if df_result.shape[1] >= 2:
        x = df_result.columns[0]
        y = df_result.columns[1]

        chart = st.selectbox(
            f"{kpi_name} Chart Type",
            ["Bar","Pie","Line"],
            key=kpi_name
        )

        if chart == "Bar":
            fig = px.bar(df_result, x=x, y=y)
        elif chart == "Pie":
            fig = px.pie(df_result, names=x, values=y)
        else:
            fig = px.line(df_result, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

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

# ---------- AUTO RESET ----------
if file or kpi_file:
    st.session_state.pipeline_step = 0
    st.session_state.logs = []

# ---------- AUTO RUN ----------
if df is not None and kpi_df is not None and st.session_state.pipeline_step == 0:
    st.session_state.page = "Dashboard"
    run_pipeline()

# ---------- DASHBOARD ----------
if page == "Dashboard":
    st.title("📊 Dashboard")

    if df is not None:
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Total Sales", int(df["Amount"].sum()) if "Amount" in df.columns else "N/A")

        st.dataframe(df.head())

        if kpi_df is not None:
            st.subheader("🤖 KPI Results + Visuals")

            for _, row in kpi_df.iterrows():
                kpi_name = row["kpi_name"]
                sql = generate_dynamic_sql(row)

                if sql:
                    try:
                        res = pd.read_sql(sql, conn)

                        st.markdown(f"### {kpi_name}")
                        st.code(sql)

                        st.dataframe(res)

                        visualize_kpi(res, kpi_name)

                    except Exception as e:
                        st.error(f"{kpi_name}: {e}")

# ---------- EDA ----------
elif page == "EDA":
    st.title("🔍 EDA")

    if df is not None:
        st.write(df.shape)
        st.dataframe(df.dtypes)

        st.subheader("Missing")
        st.dataframe(df.isnull().sum())

        st.subheader("Stats")
        st.dataframe(df.describe())

        num = df.select_dtypes(include=["int64","float64"])

        if len(num.columns)>0:
            col = st.selectbox("Column", num.columns)
            st.plotly_chart(px.histogram(df,x=col), use_container_width=True)
            st.plotly_chart(px.box(df,y=col), use_container_width=True)

        if len(num.columns)>1:
            st.plotly_chart(px.imshow(num.corr(), text_auto=True), use_container_width=True)

# ---------- VISUAL ----------
elif page == "Visualizations":
    st.title("📈 Visualizations")

    if df is not None:
        col = st.selectbox("Column", df.columns)

        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df,x=col)
        else:
            tmp = df[col].value_counts().reset_index()
            tmp.columns=[col,"count"]
            fig = px.bar(tmp,x=col,y="count")

        st.plotly_chart(fig, use_container_width=True, config={"doubleClick":"reset"})

# ---------- PREDICTION ----------
elif page == "Prediction":
    st.title("🤖 Prediction")

    if df is not None and "Amount" in df.columns:
        num = df.select_dtypes(include=["int64","float64"]).columns

        if len(num)>=2:
            X_col = st.selectbox("Feature", num)
            X = df[[X_col]]
            y = df["Amount"]

            model = LinearRegression()
            model.fit(X,y)

            st.metric("R²", f"{r2_score(y, model.predict(X)):.2f}")

            val = st.slider("Input", int(df[X_col].min()), int(df[X_col].max()), int(df[X_col].mean()))

            if st.button("Predict"):
                pred = model.predict([[val]])
                st.success(f"₹ {int(pred[0]):,}")

            xr = np.linspace(df[X_col].min(), df[X_col].max(),100)
            yr = model.predict(xr.reshape(-1,1))

            fig = px.scatter(df, x=X_col, y="Amount")
            fig.add_scatter(x=xr, y=yr, mode="lines")

            st.plotly_chart(fig, use_container_width=True)
