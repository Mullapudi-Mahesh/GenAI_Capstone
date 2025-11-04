import os
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.trend_detection import detect_trends
from src.anomaly_detection import detect_anomalies
from src.scenario_simulation import simulate_budget_change

# Configuration
os.environ["OPENAI_API_KEY"] = ""

# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, max_tokens=700)

# Define the agent’s master prompt

agent_prompt = PromptTemplate(
    input_variables=["user_goal", "data_summary", "trend_summary", "anomaly_summary", "simulation_summary"],
    template="""
You are an intelligent Ad Campaign Optimization Agent working for a global AdTech platform.

Your goal: autonomously analyze performance, detect anomalies, simulate changes, and generate optimization insights.

User Goal: {user_goal}

Here are summaries from various analytical modules:
 Trend Summary:
{trend_summary}

 Anomaly Summary:
{anomaly_summary}

 Simulation Summary:
{simulation_summary}

Based on this information:
1. Generate a concise performance summary.
2. Highlight top performing and poor performing segments.
3. Suggest optimization strategies.
4. Propose potential next analytical actions.

Respond in a clear, structured, bullet-pointed format.
"""
)

#  Main Agent Function

def run_agent(user_goal: str, data_path = os.path.join(os.path.dirname(__file__), "../../data/adtech_campaign_performance.csv")
):
    # Step 1: Load dataset
    df = pd.read_csv(data_path, parse_dates=["date"])

    # Step 2: Run trend detection
    trend_summary = detect_trends(df)

    # Step 3: Run anomaly detection
    anomaly_summary = detect_anomalies(df)

    # Step 4: Run a sample scenario simulation
    simulation_summary = simulate_budget_change(df, budget_increase_pct=10)

    # Step 5: Generate AI-driven insights
    chain = LLMChain(prompt=agent_prompt, llm=llm)

    result = chain.run(
        user_goal=user_goal,
        data_summary="Aggregated campaign data across multiple regions, devices, and bid strategies.",
        trend_summary=trend_summary,
        anomaly_summary=anomaly_summary,
        simulation_summary=simulation_summary,
    )

    print("\n AI Agent Analysis & Optimization Output:\n")
    print(result)


#  test

if __name__ == "__main__":
    run_agent("Optimize budget allocation and identify poor-performing segments.")
