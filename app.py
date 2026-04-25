import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Autonomous BI SaaS")

st.title("🤖 Autonomous BI SaaS AI Agent")

file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if file:

    # LOAD
    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

    # PROCESS
    df = clean(df)
    schema = detect_schema(df)

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 KPI Dashboard")

    kpis = auto_kpis(df, schema)

    cols = st.columns(4)
    for i, (k, v) in enumerate(kpis.items()):
        with cols[i % 4]:
            st.metric(k, f"{v:.2f}")

    st.divider()

    # ---------------- VISUALS ----------------
    st.subheader("📈 Auto Visual Analytics")

    charts = auto_visuals(df, schema)

    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    st.divider()

    # ---------------- INSIGHTS ----------------
    st.subheader("🧠 AI Insights")

    insights = generate_insights(df, schema)

    for i in insights:
        st.write("✔", i)

else:
    st.info("Upload dataset to start autonomous BI analysis")
