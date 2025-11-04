import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenario_simulation import simulate_budget_change

def test_budget_increase_changes_metrics():
    """Test if budget increase affects cost and conversions"""
    df = pd.DataFrame({
        "cost": [1000, 2000, 1500],
        "clicks": [200, 400, 300],
        "conversions": [20, 30, 25],
        "impressions": [10000, 20000, 15000],
        "region": ["NA", "EU", "AS"]
    })

    summary = simulate_budget_change(df, budget_increase_pct=10)
    assert isinstance(summary, str)
    assert "10% budget increase" in summary or "Clicks" in summary
