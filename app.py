import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import sqlite3

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Sales Analyzer", layout="wide")

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- STYLING ----------
st.markdown("""
<style>
body { background-color: #0e1117; }
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

# ---------- LOAD ----------
df = None

def load_data(file):
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

# ---------- AUTO COLUMN DETECTION ----------
def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}

    def find(keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return None

    return {
        "amount": find(["amount","sales","revenue","price","total"]),
        "product": find(["product","item","product name"]),
        "country": find(["country","region"]),
        "date": find(["date","order date"]),
        "quantity": find(["quantity","qty","boxes shipped","units"])
    }

# ---------- LOAD DATA ----------
mapping = None
conn = sqlite3.connect(":memory:")

if file:
    df = load_data(file)
    df.columns = df.columns.str.strip()
    mapping = detect_columns(df)

    # Load into SQL
    df.to_sql("sales_data", conn, index=False, if_exists="replace")

    if mapping["date"]:
        df[mapping["date"]] = pd.to_datetime(df[mapping["date"]], errors="coerce")

# ---------- AI FUNCTION ----------
def ask_ai(query, df, mapping):
    query = query.lower()

    if df is None:
        return "Upload dataset first"

    amt = mapping["amount"]
    prod = mapping["product"]
    country = mapping["country"]

    if not amt:
        return "No sales column detected"

    if "total" in query:
        return f"💰 Total Sales: ₹ {df[amt].sum():,.0f}"

    elif "average" in query:
        return f"📊 Average: ₹ {df[amt].mean():.2f}"

    elif "top product" in query and prod:
        return f"🏆 Top Product: {df.groupby(prod)[amt].sum().idxmax()}"

    elif "top country" in query and country:
        return f"🌍 Top Country: {df.groupby(country)[amt].sum().idxmax()}"

    else:
        return "Try: total, average, top product, top country"

# ================== PAGES ==================

# ---------- DASHBOARD ----------
if page == "Dashboard":
    st.title("📊 Dashboard")

    if df is not None:
        amt = mapping["amount"]

        # KPI
        st.subheader("📊 Business KPIs")
        c1, c2, c3 = st.columns(3)

        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Sales", int(df[amt].sum()) if amt else 0)

        st.dataframe(df.head())

        # BI Charts
        if mapping["date"] and amt:
            trend = df.groupby(mapping["date"])[amt].sum().reset_index()
            fig = px.line(trend, x=mapping["date"], y=amt, title="Sales Trend")
            st.plotly_chart(fig, use_container_width=True)

        if mapping["country"] and amt:
            region = df.groupby(mapping["country"])[amt].sum().reset_index()
            fig = px.bar(region, x=mapping["country"], y=amt, title="Sales by Country")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Upload file")

# ---------- VISUALIZATION ----------
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
        st.warning("Upload file")

# ---------- AI CHAT ----------
elif page == "AI Chat":
    st.title("💬 AI Chat")

    if df is not None:

        for msg in st.session_state.messages[-20:]:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask about your data")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            answer = ask_ai(query, df, mapping)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

        # SQL Section
        st.subheader("🧠 Run SQL Query")

        sql_query = st.text_area("Write SQL (SELECT * FROM sales_data LIMIT 5)")

        if st.button("Run Query"):
            try:
                result = pd.read_sql(sql_query, conn)
                st.dataframe(result)
            except Exception as e:
                st.error(e)

    else:
        st.warning("Upload file first")

# ---------- PREDICTION ----------
elif page == "Prediction":
    st.title("🤖 Prediction")

    if df is not None:
        qty = mapping["quantity"]
        amt = mapping["amount"]

        if qty and amt:
            X = df[[qty]]
            y = df[amt]

            model = LinearRegression()
            model.fit(X, y)

            val = st.slider("Input Value", 1, 500, 100)

            if st.button("Predict"):
                pred = model.predict([[val]])
                st.success(f"₹ {int(pred[0]):,}")

        else:
            st.warning("No suitable columns found")

    else:
        st.warning("Upload dataset")
