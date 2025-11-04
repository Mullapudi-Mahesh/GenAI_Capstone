import pandas as pd
from src.trend_detection import detect_trends

def test_trend_detection_returns_string():
    """Ensure detect_trends() returns readable output"""
    data = {
        "date": pd.date_range("2024-01-01", periods=10),
        "impressions": [1000]*10,
        "clicks": [50, 55, 53, 58, 60, 61, 59, 65, 67, 70],
        "conversions": [5, 6, 5, 7, 8, 7, 8, 9, 9, 10],
        "cost": [100, 110, 120, 115, 125, 130, 135, 140, 150, 155],
        "region": ["NA"]*10,
        "device": ["Mobile"]*10
    }
    df = pd.DataFrame(data)
    result = detect_trends(df)
    assert isinstance(result, str)
    assert any(metric in result for metric in ["CTR", "CVR", "trend"]), "Missing trend info"
