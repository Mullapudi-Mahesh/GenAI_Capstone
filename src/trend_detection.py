import pandas as pd

def detect_trends(df: pd.DataFrame) -> str:
    """Detects key campaign trends over time and returns a summary string."""
    
    # Calculate CTR, CVR
    df["CTR"] = (df["clicks"] / df["impressions"]) * 100
    df["CVR"] = (df["conversions"] / df["clicks"].replace(0, pd.NA)) * 100
    df["CPA"] = df["cost"] / df["conversions"].replace(0, pd.NA)

    # Aggregate by region
    trend_summary = (
        df.groupby("region")[["CTR", "CVR", "CPA"]]
        .mean()
        .round(2)
        .sort_values(by="CTR", ascending=False)
        .reset_index()
    )

    summary_text = "Average campaign metrics by region:\n"
    for _, row in trend_summary.iterrows():
        summary_text += f"- {row['region']}: CTR={row['CTR']}%, CVR={row['CVR']}%, CPA=${row['CPA']}\n"

    return summary_text.strip()
