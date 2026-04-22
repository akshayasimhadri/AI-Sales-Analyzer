import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Sales Analyzer", layout="wide")

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- STYLING ----------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
}
.stButton button {
    width: 100%;
    border-radius: 8px;
}
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("🚀 AI Dashboard")

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Visualizations",
    "AI Chat",
    "Prediction"
])

file = st.sidebar.file_uploader("Upload CSV", type=["csv","xlsx"])

# ---------- LOAD DATA ----------
if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# ---------- FREE AI FUNCTION ----------
def ask_ai(query, df):
    query = query.lower()

    try:
        # Smart insights
        if "total" in query:
            return f"💰 Total Sales: ₹ {df['Amount'].sum()}"

        elif "average" in query:
            return f"📊 Average Sales: ₹ {df['Amount'].mean():.2f}"

        elif "top product" in query:
            p = df.groupby("Product")["Amount"].sum().idxmax()
            return f"🏆 Top Product: {p}"

        elif "top country" in query:
            c = df.groupby("Country")["Amount"].sum().idxmax()
            return f"🌍 Top Country: {c}"

        elif "insight" in query:
            return f"""
📊 Key Insights:
- Total Sales: ₹ {df['Amount'].sum()}
- Avg Sales: ₹ {df['Amount'].mean():.2f}
- Top Product: {df.groupby("Product")["Amount"].sum().idxmax()}
- Top Country: {df.groupby("Country")["Amount"].sum().idxmax()}
"""

        elif "trend" in query:
            return "📈 Sales trend shows variation over time. Use charts for better visualization."

        else:
            return "🤖 Try asking: total sales, top product, insights, average, or country."

    except Exception as e:
        return str(e)

# ================== PAGES ==================

# ---------- DASHBOARD ----------
if page == "Dashboard":
    st.title("📊 Dashboard")

    if file:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Sales", int(df["Amount"].sum()))
        col2.metric("Records", len(df))
        col3.metric("Products", df["Product"].nunique())

        st.dataframe(df.head())
    else:
        st.warning("Upload file from sidebar")

# ---------- VISUALIZATION ----------
elif page == "Visualizations":
    st.title("📈 Visualizations")

    if file:
        col = st.selectbox("Select Column", df.columns)
        fig = px.bar(df, x=col, color=col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload file")

# ---------- AI CHAT ----------
elif page == "AI Chat":
    st.title("💬 AI Chat (Free Mode)")

    if file:
        # Show history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask about your data")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            answer = ask_ai(query, df)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

    else:
        st.warning("Upload file first")

# ---------- PREDICTION ----------
elif page == "Prediction":
    st.title("🤖 Sales Prediction")

    if file and "Boxes Shipped" in df.columns:
        X = df[["Boxes Shipped"]]
        y = df["Amount"]

        model = LinearRegression()
        model.fit(X, y)

        val = st.slider("Boxes Shipped", 1, 500, 100)

        if st.button("Predict"):
            pred = model.predict([[val]])
            st.success(f"₹ {int(pred[0])}")
    else:
        st.warning("Upload correct dataset")