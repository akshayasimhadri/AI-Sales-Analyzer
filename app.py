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

# ---------- MODERN UI STYLING ----------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}

/* Buttons */
.stButton button {
    width: 100%;
    border-radius: 10px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    font-weight: 600;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
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
conn = sqlite3.connect(":memory:")

def load_data(file):
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

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

mapping = None

if file:
    df = load_data(file)
    df.columns = df.columns.str.strip()
    mapping = detect_columns(df)

    df.to_sql("sales_data", conn, index=False, if_exists="replace")

    if mapping["date"]:
        df[mapping["date"]] = pd.to_datetime(df[mapping["date"]], errors="coerce")

# ---------- AI ----------
def ask_ai(query):
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

        st.subheader("📊 Business KPIs")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Rows", len(df))
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Columns", len(df.columns))
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Sales", int(df[amt].sum()) if amt else 0)
            st.markdown('</div>', unsafe_allow_html=True)

        st.dataframe(df.head())

        if mapping["date"] and amt:
            trend = df.groupby(mapping["date"])[amt].sum().reset_index()
            fig = px.line(trend, x=mapping["date"], y=amt, title="Sales Trend")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if mapping["country"] and amt:
            region = df.groupby(mapping["country"])[amt].sum().reset_index()
            fig = px.bar(region, x=mapping["country"], y=amt, title="Sales by Country")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Upload file")

# ---------- AI CHAT ----------
elif page == "AI Chat":
    st.title("💬 AI Chat")
    st.markdown("### 🤖 AI Assistant")

    if df is not None:

        for msg in st.session_state.messages[-20:]:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask about your data")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            answer = ask_ai(query)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

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
