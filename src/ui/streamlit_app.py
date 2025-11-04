# src/ui/streamlit_app.py
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import IsolationForest
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- Ensure root project path for imports ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.agent.agent_graph import run_agent_graph

# -------------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Ad Campaign Optimization Assistant",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Ad Campaign Optimization Assistant (GenAI + LangGraph)")

# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------
DATA_PATH = os.path.join(ROOT_DIR, "data", "adtech_campaign_performance.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["CTR"] = (df["clicks"] / df["impressions"]) * 100
    df["CVR"] = (df["conversions"] / df["clicks"].replace(0, pd.NA)) * 100
    df["CPA"] = df["cost"] / df["conversions"].replace(0, pd.NA)
    return df

df = load_data()
st.sidebar.success("✅ Data Loaded Successfully")

# -------------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------------
st.sidebar.header("📊 Filters")
regions = st.sidebar.multiselect("Select Region(s)", options=df["region"].unique(), default=df["region"].unique())
devices = st.sidebar.multiselect("Select Device(s)", options=df["device"].unique(), default=df["device"].unique())
filtered_df = df[df["region"].isin(regions) & df["device"].isin(devices)]

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview", "🚨 Anomalies", "🤖 AI Recommendations", "🧮 Simulation Dashboard"
])

# -------------------------------------------------------------------------
# TAB 1: PERFORMANCE OVERVIEW
# -------------------------------------------------------------------------
with tab1:
    st.subheader("📈 Campaign Performance Overview")
    summary = (
        filtered_df.groupby(["region", "device"])
        .agg({"impressions": "sum", "clicks": "sum", "conversions": "sum", "cost": "sum"})
        .reset_index()
    )
    summary["CTR"] = (summary["clicks"] / summary["impressions"]) * 100
    summary["CVR"] = (summary["conversions"] / summary["clicks"].replace(0, pd.NA)) * 100
    summary["CPA"] = summary["cost"] / summary["conversions"].replace(0, pd.NA)
    st.dataframe(summary.style.format({"CTR": "{:.2f}", "CVR": "{:.2f}", "CPA": "{:.2f}"}))

    trend_df = filtered_df.groupby("date")[["CTR", "CVR", "cost"]].mean().reset_index()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x="date", y="CTR", data=trend_df, label="CTR (%)", ax=ax)
        sns.lineplot(x="date", y="CVR", data=trend_df, label="CVR (%)", ax=ax)
        plt.title("CTR & CVR Over Time")
        plt.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x="date", y="cost", data=trend_df, color="orange", ax=ax)
        plt.title("Average Daily Cost Over Time")
        st.pyplot(fig)

# -------------------------------------------------------------------------
# TAB 2: ANOMALY DETECTION
# -------------------------------------------------------------------------
with tab2:
    st.subheader("🚨 Anomaly Detection (CTR-based)")
    numeric_cols = ["CTR", "CVR", "CPA"]
    filtered_df[numeric_cols] = filtered_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    filtered_df[numeric_cols] = filtered_df[numeric_cols].fillna(filtered_df[numeric_cols].median())

    model = IsolationForest(contamination=0.03, random_state=42)
    filtered_df["AnomalyFlag"] = model.fit_predict(filtered_df[numeric_cols])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=filtered_df, x="CTR", y="CVR", hue="AnomalyFlag", palette={1: "blue", -1: "red"}, ax=ax)
    plt.title("Anomaly Detection: CTR vs CVR")
    st.pyplot(fig)
    st.write(f"🔍 **Detected {(filtered_df['AnomalyFlag'] == -1).sum()} anomalous records.**")

# -------------------------------------------------------------------------
# TAB 3: AI OPTIMIZATION RECOMMENDATIONS
# -------------------------------------------------------------------------
with tab3:
    st.subheader("🤖 AI Optimization Recommendations")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    prompt = ChatPromptTemplate.from_template("""
    You are an AI marketing optimization assistant.
    Given the campaign summary data below:
    {campaign_data}

    Please provide:
    1. Key performance insights
    2. Optimization strategies (budget, targeting, bidding)
    3. A/B testing ideas
    4. Warnings for poor-performing segments
    """)

    chain = prompt | llm
    if st.button("🔍 Generate AI Recommendations"):
        sample = summary.head(10).to_string(index=False)
        with st.spinner("Generating optimization insights..."):
            response = chain.invoke({"campaign_data": sample})
        st.markdown(response.content)

    st.divider()
    if st.button("🚀 Run Full LangGraph Agent Workflow"):
        with st.spinner("Running agent analysis..."):
            try:
                report = run_agent_graph()
                st.success("✅ Agent completed analysis")
                st.markdown(report)
            except Exception as e:
                st.error(f"❌ Agent execution failed: {str(e)}")

# -------------------------------------------------------------------------
# TAB 4: SIMULATION DASHBOARD
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# TAB 4: SIMULATION DASHBOARD
# -------------------------------------------------------------------------
with tab4:
    st.subheader("🧮 Scenario Simulation Dashboard")

    sim_type = st.selectbox("Choose Simulation Type", ["Budget Change", "Bidding Strategy", "Seasonality Forecast"])

    # ---------- Budget Change Simulation ----------
    if sim_type == "Budget Change":
        pct_change = st.slider("💰 Budget Change (%)", -50, 50, 10)
        sim_df = filtered_df.copy()

        # Simulate cost and conversions
        sim_df["Sim_Cost"] = sim_df["cost"] * (1 + pct_change / 100)
        sim_df["Sim_Conversions"] = sim_df["conversions"] * (1 + (pct_change / 120))
        sim_df["Sim_CPA"] = sim_df["Sim_Cost"] / sim_df["Sim_Conversions"]

        st.markdown(f"### 📊 Simulated {pct_change}% Budget Change Impact")
        comparison = sim_df.groupby("region")[["cost", "Sim_Cost", "CPA", "Sim_CPA"]].mean().reset_index()
        comparison["CPA_Change(%)"] = ((comparison["Sim_CPA"] - comparison["CPA"]) / comparison["CPA"]) * 100

        # 📊 Comparison Table
        st.write("#### 🔍 Comparison Table (Before vs After)")
        st.dataframe(
            comparison.style.format({"cost": "{:,.0f}", "Sim_Cost": "{:,.0f}", "CPA": "{:.2f}", "Sim_CPA": "{:.2f}", "CPA_Change(%)": "{:+.2f}"})
            .background_gradient(subset=["CPA_Change(%)"], cmap="RdYlGn_r")
        )

        # 📈 ROI Curve (Budget vs CPA)
        st.write("#### 📈 ROI Sensitivity Curve")
        roi_df = []
        for change in range(-50, 55, 5):
            cost = filtered_df["cost"].sum() * (1 + change / 100)
            conversions = filtered_df["conversions"].sum() * (1 + change / 120)
            roi_df.append({"Budget Change (%)": change, "Simulated CPA": cost / conversions})
        roi_df = pd.DataFrame(roi_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x="Budget Change (%)", y="Simulated CPA", data=roi_df, marker="o", ax=ax)
        plt.title("ROI Curve: Budget vs CPA")
        st.pyplot(fig)

        # 🧠 LLM (Agent) Analysis
        st.markdown("#### 🤖 AI Insight on Simulation Results")
        if st.button("🧠 Analyze Simulation with Agent"):
            with st.spinner("Running AI analysis on simulated data..."):
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate

                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
                prompt = ChatPromptTemplate.from_template("""
                You are an AI marketing analyst.
                Given this simulation of campaign budget vs CPA:
                {comparison_data}

                Please provide:
                1. Key ROI insights.
                2. At what budget % the campaign becomes cost-inefficient.
                3. Recommendations to optimize budget allocation.
                """)
                chain = prompt | llm
                sample = comparison.to_string(index=False)
                result = chain.invoke({"comparison_data": sample})
                st.markdown(result.content)

    # ---------- Bidding Strategy Simulation ----------
    elif sim_type == "Bidding Strategy":
        strategies = ["CPC", "CPM", "CPA"]
        results = []
        for s in strategies:
            temp = filtered_df.copy()
            if s == "CPC":
                temp["clicks"] *= 1.1
            elif s == "CPM":
                temp["impressions"] *= 1.2
            elif s == "CPA":
                temp["conversions"] *= 1.15
            cost_eff = (temp["conversions"].sum() / temp["cost"].sum()) * 1000
            results.append({"Strategy": s, "Cost Efficiency": cost_eff})
        res_df = pd.DataFrame(results)

        st.write("#### 📊 Bidding Strategy Comparison")
        st.bar_chart(res_df.set_index("Strategy"))
        best_strategy = res_df.loc[res_df["Cost Efficiency"].idxmax(), "Strategy"]
        st.success(f"💡 Recommended Strategy: **{best_strategy}** (Highest Cost Efficiency)")

        # 🧠 Ask AI for Interpretation
        if st.button("🧠 Interpret Bidding Results"):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            prompt = ChatPromptTemplate.from_template("""
            You are a digital ads strategist. Analyze the following bidding performance:
            {bidding_data}
            Provide short insights on which bidding model is optimal and why.
            """)
            chain = prompt | llm
            sample = res_df.to_string(index=False)
            result = chain.invoke({"bidding_data": sample})
            st.markdown(result.content)

    # ---------- Seasonality Forecast ----------
    elif sim_type == "Seasonality Forecast":
        season_factor = st.slider("☀️ Seasonal Impact (%)", -30, 50, 10)
        df_season = filtered_df.copy()
        df_season["Forecast_Conversions"] = df_season["conversions"] * (1 + season_factor / 100)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x="date", y="Forecast_Conversions", data=df_season, color="green", ax=ax)
        plt.title(f"Forecasted Conversions with {season_factor}% Seasonal Impact")
        st.pyplot(fig)

        st.info(f"🌦️ Simulated a {season_factor}% seasonal impact on conversions.")
        if st.button("🧠 Analyze Seasonal Forecast"):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
            prompt = ChatPromptTemplate.from_template("""
            You are a marketing forecasting AI.
            Based on the following seasonality simulation, summarize:
            - Expected performance changes
            - Risk of over/under-spending
            - Suggested seasonal tactics
            Data:
            {forecast_data}
            """)
            chain = prompt | llm
            result = chain.invoke({"forecast_data": df_season.head(10).to_string(index=False)})
            st.markdown(result.content)
