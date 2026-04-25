import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ================= CONFIG =================
st.set_page_config(page_title="AI Enterprise BI Platform", layout="wide")
st.title("🤖 AI Enterprise BI Platform (Pro Edition)")

# ================= NAV =================
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
        "🔍 Analyzing Structure",
        "📊 Building KPIs",
        "🤖 AI Engine Ready"
    ]

    container = st.sidebar.container()
    progress = container.progress(0)

    for i, step in enumerate(steps):
        container.markdown(f"### {step}")
        progress.progress((i + 1) * 20)
        time.sleep(0.2)

# ================= CLEAN =================
def clean(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ================= KPI ENGINE =================
def kpi_engine(df):
    kpis = {}
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:
        kpis[f"Total {col}"] = df[col].sum()
        kpis[f"Avg {col}"] = df[col].mean()

    return kpis

# ================= AI ANALYST =================
def ai_engine(df, query):

    query = query.lower()
    cols = df.columns
    num_cols = df.select_dtypes(include="number").columns

    date_col = None
    for c in cols:
        if "date" in c.lower():
            date_col = c

    metric = None
    for c in num_cols:
        if any(k in c.lower() for k in ["sales", "amount", "revenue", "profit"]):
            metric = c

    if metric is None and len(num_cols) > 0:
        metric = num_cols[0]

    if "date" in query:

        if date_col is None:
            return {"type": "text", "answer": "❌ No date column found"}

        grouped = df.groupby(date_col)[metric].sum().reset_index()

        best = grouped.loc[grouped[metric].idxmax()]

        return {
            "type": "chart",
            "df": grouped,
            "x": date_col,
            "y": metric,
            "answer": f"📅 Best {date_col}: {best[date_col]}"
        }

    if "top" in query:

        cat_cols = [c for c in cols if df[c].dtype == "object"]

        if len(cat_cols) == 0:
            return {"type": "text", "answer": "❌ No categorical column found"}

        group_col = cat_cols[0]

        grouped = df.groupby(group_col)[metric].sum().sort_values(ascending=False).head(5)

        return {
            "type": "chart",
            "df": grouped.reset_index(),
            "x": group_col,
            "y": metric,
            "answer": "🏆 Top analysis generated"
        }

    return {
        "type": "text",
        "answer": f"📊 Total {metric}: {df[metric].sum():.2f}"
    }

# ================= ML PREDICTION ENGINE (NEW UPGRADE) =================
def ml_prediction(df, target):

    df = df.copy()

    # only numeric columns
    num_df = df.select_dtypes(include="number")

    if target not in num_df.columns:
        return None, None, None, None

    X = num_df.drop(columns=[target])
    y = num_df[target]

    if len(X.columns) == 0:
        X = np.arange(len(y)).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=3)
            score = scores.mean()

            if score > best_score:
                best_score = score
                best_model = model
        except:
            continue

    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)

    rmse = np.sqrt(np.mean((y_test - preds) ** 2))

    accuracy = max(0, 100 - (rmse / (y.mean() + 1e-6)) * 100)

    return best_model, accuracy, y_test, preds

# ================= MAIN =================
if file is not None:

    smooth_pipeline()

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("📥 Data Loaded")

    df = clean(df)

    num_cols = df.select_dtypes(include="number").columns

    # ================= DASHBOARD =================
    if page == "🏠 Dashboard":

        st.subheader("📊 Executive Dashboard")

        kpis = kpi_engine(df)

        cols = st.columns(min(4, len(kpis)))

        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.2f}")

        st.markdown("### 📋 Full Dataset")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 KPI Table")

        st.dataframe(pd.DataFrame({
            "KPI": list(kpis.keys()),
            "Value": list(kpis.values())
        }))

    # ================= EDA =================
    elif page == "📊 EDA":

        st.subheader("📊 Data Analysis")

        st.dataframe(df.head())

        st.dataframe(df.describe())

    # ================= VISUAL =================
    elif page == "📈 Visualizations":

        st.subheader("📊 Charts")

        col = st.selectbox("Select Column", df.columns)

        if col in num_cols:
            st.plotly_chart(px.histogram(df, x=col))

    # ================= PREDICTION (NEW ENTERPRISE VERSION) =================
    elif page == "🔮 Prediction":

        st.subheader("🔮 AI ML Prediction Engine")

        if len(num_cols) > 0:

            target = st.selectbox("Select Target Column", num_cols)

            model, accuracy, y_test, preds = ml_prediction(df, target)

            if model is not None:

                st.markdown("### 📊 Model Performance")

                st.metric("Accuracy (%)", f"{accuracy:.2f}")

                if accuracy > 80:
                    st.success("🟢 High Confidence Model")
                elif accuracy > 50:
                    st.warning("🟡 Medium Confidence Model")
                else:
                    st.error("🔴 Low Confidence Model")

                result_df = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": preds
                })

                st.markdown("### 📈 Actual vs Predicted")

                st.plotly_chart(px.line(result_df))

    # ================= AI ANALYST =================
    elif page == "🤖 AI Analyst":

        st.subheader("🤖 Smart Data Analyst")

        query = st.text_input("Ask anything")

        if query:
            output = ai_engine(df, query)

            st.success(output["answer"])

            if output["type"] == "chart":
                st.plotly_chart(px.bar(output["df"], x=output["x"], y=output["y"]))

else:
    st.info("📂 Upload dataset to start")
