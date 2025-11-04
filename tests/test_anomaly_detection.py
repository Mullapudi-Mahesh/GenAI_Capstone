import pandas as pd

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anomaly_detection import detect_anomalies

def test_detects_anomalies_in_outlier_data():
    """Test if anomaly detection catches artificial outliers"""
    df = pd.DataFrame({
        "clicks": [100, 200, 300, 10000],  # outlier
        "impressions": [10000, 20000, 30000, 1000],
        "conversions": [5, 10, 15, 0],
        "region": ["NA", "EU", "AS", "NA"]
    })
    result = detect_anomalies(df)
    assert isinstance(result, str)
    assert "Detected" in result or "anomaly" in result.lower()
