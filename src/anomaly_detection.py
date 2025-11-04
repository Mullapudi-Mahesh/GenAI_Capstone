import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame) -> str:
    """Detects campaign anomalies based on CTR and CVR using Isolation Forest."""
    
    df["CTR"] = (df["clicks"] / df["impressions"]) * 100
    df["CVR"] = (df["conversions"] / df["clicks"].replace(0, pd.NA)) * 100

    features = df[["CTR", "CVR"]].fillna(0)
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(features)

    anomalies = df[df["anomaly"] == -1]
    if anomalies.empty:
        return "No major anomalies detected in campaign performance."

    region_counts = anomalies["region"].value_counts().to_dict()
    summary = "Detected performance anomalies in the following regions:\n"
    for region, count in region_counts.items():
        summary += f"- {region}: {count} anomaly records\n"

    return summary.strip()
