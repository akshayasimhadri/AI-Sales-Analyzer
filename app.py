import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sqlite3

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Sales Analyzer", layout="wide")

# ---------- UI ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}
.stButton button {
    width: 100%;
    border-radius: 10px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("🚀 AI Dashboard")

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Visualizations",
    "Prediction"
])

file = st.sidebar.file_uploader("Upload Data File", type=["csv","xlsx"])
kpi_file = st.sidebar.file_uploader("Upload KPI File", type=["csv"])

# ---------- DB ----------
conn = sqlite3.connect(":memory:")

# ---------- LOAD DATA ----------
df = None
kpi_df = None

if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df.columns = df.columns.str.strip()
    df.to_sql("sales_data", conn, index=False, if_exists="replace")

if kpi_file:
    kpi_df = pd.read_csv(kpi_file)

# ---------- KPI SQL GENERATOR ----------
def generate_sql_from_kpi(row):
    metric = str(row["metric"]).lower()
    column = row["dimension"]
    name = row["kpi_name"]

    if metric == "sum":
        sql = f"SELECT SUM({column}) AS value FROM sales_data"

    elif metric == "avg":
        sql = f"SELECT AVG({column}) AS value FROM sales_data"

    elif metric == "count":
        sql = f"SELECT COUNT({column}) AS value FROM sales_data"

    elif metric == "top":
        sql = f"""
        SELECT {column}, COUNT(*) as value
        FROM sales_data
        GROUP BY {column}
        ORDER BY value DESC
        LIMIT 1
        """

    else:
        sql = None

    return name, sql

# ================== DASHBOARD ==================
if page == "Dashboard":
    st.title("📊 Dashboard")

    if df is not None:
        st.success("✅ Data Loaded Successfully")

        # Basic KPIs
        st.subheader("📊 Basic Metrics")
        c1, c2, c3 = st.columns(3)

        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))

        if "Amount" in df.columns:
            c3.metric("Total Sales", int(df["Amount"].sum()))
        else:
            c3.metric("Total Sales", "N/A")

        st.dataframe(df.head())

        # ---------- KPI AUTO SQL WITH QUERY COLUMN ----------
        if kpi_df is not None:
            st.subheader("🤖 KPI Results with SQL Proof")

            results = []

            for _, row in kpi_df.iterrows():
                kpi_name, sql = generate_sql_from_kpi(row)

                if sql:
                    try:
                        result_df = pd.read_sql(sql, conn)

                        # Extract value
                        if result_df.shape[1] == 1:
                            value = result_df.iloc[0, 0]
                        else:
                            value = result_df.to_dict(orient="records")[0]

                        results.append({
                            "KPI Name": kpi_name,
                            "SQL Query": sql.strip(),
                            "Result": value
                        })

                    except Exception as e:
                        results.append({
                            "KPI Name": kpi_name,
                            "SQL Query": sql,
                            "Result": f"Error: {e}"
                        })

            final_df = pd.DataFrame(results)

            st.dataframe(final_df)

            # Download
            csv = final_df.to_csv(index=False).encode()
            st.download_button("📥 Download KPI Results", csv, "kpi_results.csv")

        # ---------- CHART ----------
        if "Amount" in df.columns:
            st.subheader("📈 Sales Distribution")
            fig = px.histogram(df, x="Amount")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Upload dataset")

# ================== VISUALIZATION ==================
elif page == "Visualizations":
    st.title("📈 Visualizations")

    if df is not None:
        col = st.selectbox("Select Column", df.columns)

        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col)
        else:
            temp = df[col].value_counts().reset_index()
            temp.columns = [col, "count"]
            fig = px.bar(temp, x=col, y="count")

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Upload dataset")

# ================== PREDICTION ==================
elif page == "Prediction":
    st.title("🤖 Sales Prediction")

    if df is not None and "Amount" in df.columns:

        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()

        if len(numeric_cols) >= 2:
            X_col = st.selectbox("Select Input Feature", numeric_cols)
            y_col = "Amount"

            X = df[[X_col]]
            y = df[y_col]

            model = LinearRegression()
            model.fit(X, y)

            # Accuracy
            y_pred_train = model.predict(X)
            score = r2_score(y, y_pred_train)

            st.metric("Model Accuracy (R²)", f"{score:.2f}")

            val = st.slider(
                "Input Value",
                int(df[X_col].min()),
                int(df[X_col].max()),
                int(df[X_col].mean())
            )

            if st.button("Predict"):
                pred = model.predict([[val]])
                st.success(f"💰 Predicted Sales: ₹ {int(pred[0]):,}")

                st.info(f"""
📊 Insight:
For **{val}**, expected sales ≈ **₹ {int(pred[0]):,}**
based on historical trend.
""")

            # Graph
            st.subheader("📈 Regression Trend")

            x_range = np.linspace(df[X_col].min(), df[X_col].max(), 100)
            y_range = model.predict(x_range.reshape(-1,1))

            fig = px.scatter(df, x=X_col, y=y_col)
            fig.add_scatter(x=x_range, y=y_range, mode='lines', name="Trend")

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Not enough numeric columns")

    else:
        st.warning("Upload dataset with 'Amount'")
