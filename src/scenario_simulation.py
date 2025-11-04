# src/scenario_simulation.py

import pandas as pd

def simulate_budget_change(df: pd.DataFrame, budget_increase_pct: float = 10) -> str:
    """
    Simulates a scenario where campaign budgets (costs) increase by a percentage
    and estimates the effect on clicks and conversions.
    """

    df_sim = df.copy()
    increase_factor = 1 + (budget_increase_pct / 100)

    # Assume performance metrics scale linearly (simplified assumption)
    df_sim["cost_new"] = df_sim["cost"] * increase_factor
    df_sim["clicks_new"] = (df_sim["clicks"] * increase_factor).astype(int)
    df_sim["conversions_new"] = (df_sim["conversions"] * increase_factor).astype(int)

    # Aggregate by region to show simulated impact
    summary = (
        df_sim.groupby("region")[["clicks_new", "conversions_new", "cost_new"]]
        .sum()
        .reset_index()
    )

    summary["CTR_est"] = (summary["clicks_new"] / df["impressions"].sum()) * 100
    summary["CVR_est"] = (summary["conversions_new"] / summary["clicks_new"]) * 100
    summary["CPA_est"] = summary["cost_new"] / summary["conversions_new"]

    # Build a summary string for the LLM
    summary_text = "Simulated 10% budget increase impact (aggregated by region):\n\n"
    for _, row in summary.iterrows():
        summary_text += (
            f"- {row['region']}: "
            f"Clicks={int(row['clicks_new'])}, "
            f"Conversions={int(row['conversions_new'])}, "
            f"CPA=${row['CPA_est']:.2f}\n"
        )

    return summary_text
