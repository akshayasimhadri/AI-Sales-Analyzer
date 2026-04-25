import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ================= CONFIG =================
st.set_page_config(page_title="AI BI Copilot", layout="wide")

st.title("🤖 AI BI Copilot - Executive Analytics Platform")

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "📊 EDA", "📈 Visualizations", "🔮 Prediction", "🤖 AI Analyst"]
)

file = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# ================= PIPELINE =================
def smooth_pipeline():
    steps = [
        "📥 Loading Data",
        "🧹 Cleaning Data",
        "🔍 Processing",
        "📊 Generating KPIs",
        "🤖 AI Ready"
    ]

    box = st.sidebar.container()
    prog = box.progress(0)

    for i, s in enumerate(steps):
        box.markdown(f"### {s}")
        prog.progress((i + 1) * 20)
        time.sleep(0.2)

# ================= CLEAN DATA (FIXED) =================
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
    num = df.select_dtypes(include="number").columns

    for c in num:
        kpis[f"Total {c}"] = df[c].sum()
        kpis[f"Avg {c}"] = df[c].mean()

    return kpis

# ================= ML ENGINE =================
def ml_engine(df, target):

    df = df.copy()
    num = df.select_dtypes(include="number")

    if target not in num.columns:
        return None, None, None, None

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
            continue

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)

    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    accuracy = max(0, 100 - (rmse / (y.mean() + 1e-6)) * 100)

    return best_model, accuracy, y_test, pred

# ================= MAIN =================
if file is not None:

    smooth_pipeline()

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df = clean(df)

    num_cols = df.select_dtypes(include="number").columns
    date_cols = [c for c in df.columns if "date" in c.lower()]

    # ================= DASHBOARD =================
    if page == "🏠 Dashboard":

        st.subheader("📊 Executive Dashboard")

        kpis = kpi_engine(df)

        cols = st.columns(min(4, len(kpis)))

        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.2f}")

        st.markdown("### 📋 Data Table")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 KPI Table")
        st.dataframe(pd.DataFrame({
            "KPI": list(kpis.keys()),
            "Value": list(kpis.values())
        }))

        st.markdown("### 📈 Trends")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Line", "Bar", "Pie"]
        )

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

        st.subheader("Data Overview")

        st.dataframe(df.head(), use_container_width=True)
        st.dataframe(df.describe())

    # ================= VISUALIZATION =================
    elif page == "📈 Visualizations":

        col = st.selectbox("Select Column", df.columns)

        if col in num_cols:
            st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

    # ================= PREDICTION =================
    elif page == "🔮 Prediction":

        st.subheader("ML Prediction Engine")

        if len(num_cols) > 0:

            target = st.selectbox("Target Column", num_cols)

            model, acc, y_test, pred = ml_engine(df, target)

            if model:

                st.metric("Accuracy", f"{acc:.2f}%")

                if acc > 75:
                    st.success("High Prediction Confidence 🟢")
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

            # ================= DATE LOGIC FIX =================
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
    st.info("📂 Upload a dataset to start analysis")
