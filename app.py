import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ================= CONFIG =================
st.set_page_config(page_title="GPT BI Analyst", layout="wide")

st.title("🤖 GPT-Style AI BI Analyst (Natural Language Data Agent)")

# ================= UPLOAD =================
file = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# ================= LOAD DATA =================
if file is not None:

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("Data Loaded")

    # ================= CLEAN =================
    def clean(df):
        df = df.drop_duplicates()

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown")

        return df

    df = clean(df)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ================= GPT-LIKE ANALYST ENGINE =================
    def ai_analyst(df, query):

        query = query.lower()
        cols = df.columns
        num_cols = df.select_dtypes(include="number").columns

        # detect date column
        date_col = None
        for c in cols:
            if "date" in c.lower():
                date_col = c

        # detect main metric
        metric = None
        for c in num_cols:
            if any(k in c.lower() for k in ["sales", "amount", "revenue", "profit"]):
                metric = c

        if metric is None and len(num_cols) > 0:
            metric = num_cols[0]

        # ================= CASE 1: DATE ANALYSIS =================
        if "date" in query and ("more" in query or "highest" in query or "max" in query):

            if date_col is None:
                return None, "❌ No date column found in dataset"

            result = df.groupby(date_col)[metric].sum().reset_index()

            best = result.loc[result[metric].idxmax()]

            fig = px.bar(result, x=date_col, y=metric,
                         title=f"{metric} by {date_col}")

            answer = f"""
🤖 AI INSIGHT:

✔ Best {date_col}: {best[date_col]}
✔ Highest {metric}: {best[metric]:.2f}
"""

            return fig, answer

        # ================= CASE 2: GROUP BY =================
        if "by" in query:

            for c in cols:
                if c.lower() in query:

                    result = df.groupby(c)[metric].sum().reset_index()

                    fig = px.bar(result, x=c, y=metric,
                                 title=f"{metric} by {c}")

                    return fig, f"🤖 Showing {metric} grouped by {c}"

        # ================= CASE 3: TOP N =================
        if "top" in query:

            result = df.sort_values(metric, ascending=False).head(5)

            fig = px.bar(result, x=result.columns[0], y=metric,
                         title=f"Top 5 {metric}")

            return fig, "🤖 Top 5 results generated"

        # ================= CASE 4: CORRELATION INSIGHT =================
        if "correlation" in query:

            corr = df[num_cols].corr()

            fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")

            return fig, "🤖 Correlation analysis completed"

        # ================= FALLBACK =================
        return None, "🤖 I understood your query but need clearer column match"

    # ================= UI =================
    st.subheader("🤖 Ask AI Analyst")

    query = st.text_input("Type your question (e.g. which date has more sales)")

    if query:

        fig, answer = ai_analyst(df, query)

        st.markdown(answer)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(answer)

else:
    st.info("📂 Upload dataset to start AI Analyst")
