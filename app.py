import streamlit as st
import pandas as pd
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(page_title="AI BI Copilot", layout="wide")

# ================= CUSTOM UI STYLE =================
st.markdown("""
<style>
.big-title {
    font-size: 38px;
    font-weight: 700;
    color: #1f77b4;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
    color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}
.metric {
    font-size: 20px;
    font-weight: bold;
}
.small {
    font-size: 13px;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="big-title">🤖 AI BI Copilot Dashboard</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "📊 EDA", "📈 Visual Analytics", "🔮 Prediction"]
)

file = st.sidebar.file_uploader("📂 Upload Dataset", type=["csv", "xlsx"])

# ================= PIPELINE VISUAL =================
def pipeline(step):
    steps = ["Upload", "Clean", "EDA", "Visualization", "Prediction"]

    st.sidebar.markdown("### ⚙ Pipeline")

    for i, s in enumerate(steps):
        if i < step:
            st.sidebar.success("✔ " + s)
        elif i == step:
            st.sidebar.info("🔵 " + s)
        else:
            st.sidebar.write("⬜ " + s)

# ================= CLEAN DATA =================
def clean(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ================= LOAD =================
if file is not None:

    pipeline(0)

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("Dataset Loaded")
    pipeline(1)

    df = clean(df)
    st.success("Data Cleaned")
    pipeline(2)

    numeric = df.select_dtypes(include="number").columns
    categorical = df.select_dtypes(include=["object"]).columns

    # ================= DASHBOARD =================
    if page == "🏠 Dashboard":

        st.markdown("## 📊 Business KPIs")

        cols = st.columns(min(4, len(numeric)))

        for i, col in enumerate(numeric[:4]):

            with cols[i % len(cols)]:

                st.markdown(f"""
                <div class="card">
                    <div class="metric">📌 {col}</div>
                    <div>Total: {df[col].sum():.2f}</div>
                    <div class="small">Avg: {df[col].mean():.2f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("## 📈 Quick Overview Charts")

        c1, c2 = st.columns(2)

        if len(numeric) > 0:
            with c1:
                fig = px.histogram(df, x=numeric[0], title="Distribution")
                st.plotly_chart(fig, use_container_width=True)

        if len(numeric) > 1:
            with c2:
                fig = px.scatter(df, x=numeric[0], y=numeric[1], title="Relation")
                st.plotly_chart(fig, use_container_width=True)

    # ================= EDA =================
    elif page == "📊 EDA":

        st.markdown("## 🔍 Exploratory Data Analysis")

        st.write("Shape:", df.shape)

        tab1, tab2, tab3 = st.tabs(["Missing Values", "Statistics", "Data Preview"])

        with tab1:
            st.dataframe(df.isnull().sum())

        with tab2:
            st.dataframe(df.describe())

        with tab3:
            st.dataframe(df.head())

        pipeline(3)

    # ================= VISUALIZATION =================
    elif page == "📈 Visual Analytics":

        st.markdown("## 📊 Advanced Visual Dashboard")

        pipeline(3)

        for col in numeric:
            fig = px.histogram(df, x=col, title=f"{col}")
            st.plotly_chart(fig, use_container_width=True)

        if len(numeric) > 1:
            fig = px.imshow(df[numeric].corr(), text_auto=True, title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

    # ================= PREDICTION =================
    elif page == "🔮 Prediction":

        st.markdown("## 🔮 AI Prediction Module")

        pipeline(4)

        target = st.selectbox("Select Target Column", numeric)

        if target:

            df["Prediction"] = df[target].rolling(3).mean()

            fig = px.line(df, y=[target, "Prediction"], title="Trend Prediction")

            st.plotly_chart(fig, use_container_width=True)

            st.info("Simple AI baseline prediction (rolling trend model)")

else:
    st.info("📂 Upload dataset to activate AI BI dashboard")
