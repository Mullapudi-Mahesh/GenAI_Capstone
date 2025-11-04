import os
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/adtech_campaign_performance.csv")

def test_data_file_exists():
    """Ensure dataset file is present"""
    assert os.path.exists(DATA_PATH), " Data file not found!"

def test_data_loads_correctly():
    """Ensure CSV loads properly into a DataFrame"""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    assert not df.empty, " DataFrame is empty!"
    assert all(col in df.columns for col in ["impressions", "clicks", "conversions", "cost"]), "Missing key columns"

def test_metric_computation():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["CTR"] = (df["clicks"] / df["impressions"]) * 100
    df["CVR"] = (df["conversions"] / df["clicks"].replace(0, pd.NA)) * 100
    df["CPA"] = (df["cost"] / df["conversions"].replace(0, pd.NA)).fillna(0)
    
    assert df["CTR"].between(0, 100).all(), "CTR out of range"
    assert df["CVR"].between(0, 100).all(), "CVR out of range"
    assert (df["CPA"] >= 0).all(), "CPA should be non-negative"

