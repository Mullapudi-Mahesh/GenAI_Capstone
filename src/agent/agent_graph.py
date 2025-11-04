"""
LangGraph-based Agent Orchestration for Ad Campaign Optimization
Includes:
- Agentic workflow using LangGraph
- Structured governance via Guardrails (Pydantic validation)
- Observability & tracing via LangSmith
"""

import os
import sys
from typing import TypedDict
import pandas as pd

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

# --- Add project root to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

#  Governance & Observability Config
os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY",
    "#place OPEN API KEY HERE"
)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AdTech Optimization Assistant"

# --- Local imports ---
from src.trend_detection import detect_trends
from src.anomaly_detection import detect_anomalies
from src.scenario_simulation import simulate_budget_change

# --- Governance: Guardrails AI ---
from guardrails import Guard
from pydantic import BaseModel, Field

# ---------- Structured AI Output Schema ----------
class OptimizationReport(BaseModel):
    performance_summary: str = Field(..., description="Overview of top/bottom segments")
    recommendations: str = Field(..., description="Budget, targeting, and bid optimizations")
    ab_test_ideas: str = Field(..., description="A/B test ideas and hypotheses")
    next_actions: str = Field(..., description="Follow-up analytics steps or monitoring plans")

# Create Guard object for validation
guard = Guard.from_pydantic(output_class=OptimizationReport)


# ---------- Agent State ----------
class AgentState(TypedDict, total=False):
    user_goal: str
    data_path: str
    df: pd.DataFrame
    trend_summary: str
    anomaly_summary: str
    simulation_summary: str
    final_report: str


# ---------- Helper: Resolve data path ----------
def resolve_data_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "data", "adtech_campaign_performance.csv")


# ---------- Agent Steps ----------
@traceable(name="load_data_step")
def load_data(state: AgentState) -> AgentState:
    data_path = state.get("data_path") or resolve_data_path()
    df = pd.read_csv(data_path, parse_dates=["date"])
    return {**state, "data_path": data_path, "df": df}


@traceable(name="trend_detection_step")
def run_trends(state: AgentState) -> AgentState:
    trend_summary = detect_trends(state["df"])
    return {**state, "trend_summary": trend_summary}


@traceable(name="anomaly_detection_step")
def run_anomalies(state: AgentState) -> AgentState:
    anomaly_summary = detect_anomalies(state["df"])
    return {**state, "anomaly_summary": anomaly_summary}


@traceable(name="simulation_step")
def run_simulation(state: AgentState) -> AgentState:
    simulation_summary = simulate_budget_change(state["df"], budget_increase_pct=10)
    return {**state, "simulation_summary": simulation_summary}


@traceable(name="llm_summary_step")
def summarize_with_llm(state: AgentState) -> AgentState:
    """Summarizes all results into a final structured AI report with governance & validation."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set. Please export it before running.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, max_tokens=800)

    prompt = PromptTemplate(
    input_variables=["user_goal", "trend_summary", "anomaly_summary", "simulation_summary"],
    template=(
        "You are an intelligent AdTech optimization assistant.\n"
        "User Goal: {user_goal}\n\n"
        " Trend Summary:\n{trend_summary}\n\n"
        " Anomaly Summary:\n{anomaly_summary}\n\n"
        " Simulation Summary:\n{simulation_summary}\n\n"
        "Please return a JSON object strictly following this schema:\n"
        "{{'Key insights (CTR, CVR, CPA)':..., 'performance_summary': ..., 'recommendations': ..., 'ab_test_ideas': ..., 'next_actions': ...}}\n"
    ),
)


    chain = prompt | llm
    result = chain.invoke({
        "user_goal": state.get("user_goal", "Optimize performance"),
        "trend_summary": state.get("trend_summary", "N/A"),
        "anomaly_summary": state.get("anomaly_summary", "N/A"),
        "simulation_summary": state.get("simulation_summary", "N/A"),
    })

    text = getattr(result, "content", None) or getattr(result, "text", None) or str(result)

    # Governance Layer: Validate and Correct Output
    try:
        validated = guard.parse(llm_output=text)
        final_text = (
            f"### Performance Summary\n{validated.validated_output.performance_summary}\n\n"
            f"### Recommendations\n{validated.validated_output.recommendations}\n\n"
            f"### A/B Test Ideas\n{validated.validated_output.ab_test_ideas}\n\n"
            f"### Next Actions\n{validated.validated_output.next_actions}"
        )
    except Exception as e:
        final_text = f" Validation failed, returning raw output:\n\n{text}\n\nError: {e}"

    return {**state, "final_report": final_text}


# ---------- Router ----------
def router(state: AgentState) -> str:
    if "trend_summary" not in state:
        return "trends"
    if "anomaly_summary" not in state:
        return "anomalies"
    if "simulation_summary" not in state:
        return "simulate"
    return "summarize"


# ---------- Build Graph ----------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load", load_data)
    graph.add_node("trends", run_trends)
    graph.add_node("anomalies", run_anomalies)
    graph.add_node("simulate", run_simulation)
    graph.add_node("summarize", summarize_with_llm)

    graph.set_entry_point("load")

    for node in ["load", "trends", "anomalies", "simulate"]:
        graph.add_conditional_edges(node, router, {
            "trends": "trends",
            "anomalies": "anomalies",
            "simulate": "simulate",
            "summarize": "summarize",
        })

    graph.add_edge("summarize", END)
    return graph.compile()


# ---------- Exported Function ----------
@traceable(name="run_agent_graph")
def run_agent_graph():
    """Run the full LangGraph pipeline and return validated report text."""
    app = build_graph()
    init_state: AgentState = {
        "user_goal": "Optimize budget allocation and identify poor-performing segments."
    }
    final_state = app.invoke(init_state)
    return final_state.get("final_report", " No report generated.")


# ---------- CLI Entry ----------
if __name__ == "__main__":
    print("\n Running LangGraph Agent for AdTech Optimization...\n")
    report = run_agent_graph()
    print("\n Final Governed Report:\n")
    print(report)
