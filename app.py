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

# ---------- MODERN UI ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
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
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 12px;
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
        "product": find(["product","item"]),
        "country": find(["country","region"]),
        "date": find(["date","order date"]),
        "quantity": find(["quantity","qty","boxes shipped","units"])
    }

# ---------- KPI ENGINE ----------
def generate_kpis(df, mapping):
    insights = {}
    amt = mapping["amount"]
    prod = mapping["product"]
    country = mapping["country"]
    date = mapping["date"]

    if not amt:
        return {}

    insights["total"] = df[amt].sum()
    insights["average"] = df[amt].mean()

    if prod:
        insights["top_product"] = df.groupby(prod)[amt].sum().idxmax()

    if country:
        insights["top_country"] = df.groupby(country)[amt].sum().idxmax()

    if date:
        insights["trend"] = df.groupby(date)[amt].sum().reset_index()

    return insights

# ---------- AUTO SQL ----------
def generate_sql(query, mapping):
    q = query.lower()
    amt = mapping["amount"]
    prod = mapping["product"]
    country = mapping["country"]

    if not amt:
        return None

    if "total" in q:
        return f"SELECT SUM({amt}) AS total_sales FROM sales_data"

    elif "average" in q:
        return f"SELECT AVG({amt}) AS avg_sales FROM sales_data"

    elif "top product" in q and prod:
        return f"SELECT {prod}, SUM({amt}) FROM sales_data GROUP BY {prod} ORDER BY SUM({amt}) DESC LIMIT 1"

    elif "top country" in q and country:
        return f"SELECT {country}, SUM({amt}) FROM sales_data GROUP BY {country} ORDER BY SUM({amt}) DESC LIMIT 1"

    return None

# ---------- LOAD DATA ----------
mapping = None

if file:
    df = load_data(file)
    df.columns = df.columns.str.strip()
    mapping = detect_columns(df)

    df.to_sql("sales_data", conn, index=False, if_exists="replace")

    if mapping["date"]:
        df[mapping["date"]] = pd.to_datetime(df[mapping["date"]], errors="coerce")

# ================== PAGES ==================

# ---------- DASHBOARD ----------
if page == "Dashboard":
    st.title("📊 Dashboard")

    if df is not None:
        kpis = generate_kpis(df, mapping)

        st.success("🤖 AI automatically analyzed your dataset")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sales", f"₹ {kpis.get('total',0):,.0f}")
        c2.metric("Average", f"₹ {kpis.get('average',0):.2f}")
        c3.metric("Records", len(df))

        st.markdown(f"""
🏆 **Top Product:** {kpis.get('top_product','N/A')}  
🌍 **Top Country:** {kpis.get('top_country','N/A')}
""")

        if "trend" in kpis:
            fig = px.line(kpis["trend"], x=mapping["date"], y=mapping["amount"], title="Sales Trend")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.head())

    else:
        st.warning("Upload dataset")

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
    st.title("💬 AI Data Assistant")

    if df is not None:

        for msg in st.session_state.messages[-20:]:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask about your data")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            sql = generate_sql(query, mapping)

            if sql:
                result = pd.read_sql(sql, conn)
                st.chat_message("assistant").write("📊 Result:")
                st.dataframe(result)
            else:
                st.chat_message("assistant").write("🤖 Try: total, average, top product")

            st.session_state.messages.append({"role": "assistant", "content": "Done"})

    else:
        st.warning("Upload dataset first")

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

            val = st.slider("Input Value", int(df[qty].min()), int(df[qty].max()), int(df[qty].mean()))

            if st.button("Predict"):
                pred = model.predict([[val]])
                st.success(f"₹ {int(pred[0]):,}")
        else:
            st.warning("No suitable columns found")

    else:
        st.warning("Upload dataset")
