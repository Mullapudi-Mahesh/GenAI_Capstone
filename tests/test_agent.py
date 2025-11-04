import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.agent.agent_graph import run_agent_graph

def test_agent_graph_runs_successfully():
    report = run_agent_graph()
    assert isinstance(report, str)
    assert len(report) > 50, "Agent output too short"
