import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ================= CONFIG =================
st.set_page_config(page_title="AI Analyzer", layout="wide")

st.title("🤖 AI Sales Analyzer - Executive Analytics Engine")

# ================= SIDEBAR NAV =================
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "📊 EDA", "📈 Visualizations", "🔮 Prediction", "🤖 AI Analyst"]
)

file = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# ================= PIPELINE =================
def pipeline():
    steps = [
        "📥 Loading Data",
        "🧹 Cleaning Data",
        "🔍 Processing Dataset",
        "📊 Building KPIs",
        "🤖 AI Engine Ready"
    ]

    box = st.sidebar.container()
    prog = box.progress(0)

    for i, s in enumerate(steps):
        box.markdown(f"### {s}")
        prog.progress((i + 1) * 20)
        time.sleep(0.2)

# ================= CLEAN DATA =================
def clean(df):
    df = df.drop_duplicates()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

        else:
            df[col] = df[col].astype(str).fillna("Unknown")

    return df

# ================= KPI ENGINE =================
def kpi_engine(df):
    kpis = {}
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Avg {col}"] = df[col].mean()

    return kpis

# ================= ML ENGINE =================
def ml_engine(df, target):

    df = df.copy()
    num = df.select_dtypes(include="number")

    X = num.drop(columns=[target])
    y = num[target]

    if X.shape[1] == 0:
        X = pd.DataFrame(np.arange(len(y)), columns=["index"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = -1

    for m in models.values():
        try:
            score = cross_val_score(m, X_train, y_train, cv=3).mean()
            if score > best_score:
                best_score = score
                best_model = m
        except:
            pass

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)

    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    accuracy = max(0, 100 - (rmse / (y.mean() + 1e-6)) * 100)

    return best_model, accuracy, y_test, pred

# ================= RUN =================
if file is not None:

    pipeline()

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df = clean(df)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower()]

    # ================= DASHBOARD =================
    if page == "🏠 Dashboard":

        st.subheader("📊 Executive Dashboard")

        kpis = kpi_engine(df)

        cols = st.columns(min(4, len(kpis)))

        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.2f}")

        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 KPI Table")
        st.dataframe(pd.DataFrame({
            "KPI": list(kpis.keys()),
            "Value": list(kpis.values())
        }))

        st.markdown("### 📈 Trends")

        chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Pie"])

        for col in num_cols[:3]:

            st.markdown(f"#### {col}")

            if chart_type == "Line":
                fig = px.line(df, y=col)

            elif chart_type == "Bar":
                fig = px.bar(df, y=col)

            else:
                fig = px.pie(df, names=col)

            st.plotly_chart(fig, use_container_width=True)

    # ================= EDA =================
    elif page == "📊 EDA":

        st.subheader("Data Exploration")

        st.dataframe(df.head(), use_container_width=True)
        st.dataframe(df.describe())

    # ================= VISUALIZATION (FIXED AXIS SYSTEM) =================
    elif page == "📈 Visualizations":

        st.subheader("📊 Visualization Builder")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Pie Chart", "Histogram"]
        )

        cols = df.columns.tolist()

        x_axis = st.selectbox("Select X-Axis", cols, index=0)

        y_axis = None
        if chart_type in ["Bar Chart", "Line Chart"]:
            y_axis = st.selectbox("Select Y-Axis",
                                  num_cols if len(num_cols) > 0 else cols)

        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis)

        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis)

        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_axis)

        else:
            fig = px.histogram(df, x=x_axis)

        st.plotly_chart(fig, use_container_width=True)

        # INDEX MODE
        st.markdown("### ⚡ Index Based View")

        if st.checkbox("Use Row Index") and len(num_cols) > 0:

            metric = st.selectbox("Select Metric", num_cols)

            temp = df.copy()
            temp["index"] = temp.index

            fig2 = px.line(temp, x="index", y=metric)

            st.plotly_chart(fig2, use_container_width=True)

    # ================= PREDICTION =================
    elif page == "🔮 Prediction":

        st.subheader("ML Prediction Engine")

        if len(num_cols) > 0:

            target = st.selectbox("Select Target", num_cols)

            model, acc, y_test, pred = ml_engine(df, target)

            st.metric("Accuracy", f"{acc:.2f}%")

            if acc > 75:
                st.success("High Confidence 🟢")
            elif acc > 50:
                st.warning("Medium Confidence 🟡")
            else:
                st.error("Low Confidence 🔴")

            result = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": pred
            })

            st.plotly_chart(px.line(result), use_container_width=True)

    # ================= AI ANALYST (FIXED) =================
    elif page == "🤖 AI Analyst":

        st.subheader("🤖 Smart AI Analyst")

        query = st.text_input("Ask (e.g. lowest sales by date, max revenue, average sales)")

        if query:

            q = query.lower()

            metric = num_cols[0] if len(num_cols) > 0 else None

            if "date" in q:

                if len(date_cols) == 0:
                    st.error("No date column found")
                else:
                    dcol = date_cols[0]

                    temp = df.copy()
                    temp[dcol] = pd.to_datetime(temp[dcol], errors="coerce")

                    grouped = temp.groupby(dcol)[metric].sum().reset_index()

                    if "low" in q or "least" in q:

                        row = grouped.loc[grouped[metric].idxmin()]
                        st.success(f"📉 Lowest on {row[dcol].date()} = {row[metric]:.2f}")

                    elif "high" in q or "max" in q:

                        row = grouped.loc[grouped[metric].idxmax()]
                        st.success(f"📈 Highest on {row[dcol].date()} = {row[metric]:.2f}")

                    st.plotly_chart(px.line(grouped, x=dcol, y=metric),
                                     use_container_width=True)

            elif "average" in q:
                st.success(f"Average {metric}: {df[metric].mean():.2f}")

            elif "max" in q:
                st.success(f"Max {metric}: {df[metric].max():.2f}")

            elif "min" in q:
                st.success(f"Min {metric}: {df[metric].min():.2f}")

            else:
                st.info("Try: lowest sales by date / max revenue / average sales")

else:
    st.info("📂 Upload dataset to start AI BI system")
