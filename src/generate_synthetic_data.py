import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

num_campaigns = 50
days = pd.date_range('2024-01-01', '2024-03-31')
devices = ['Mobile', 'Desktop', 'Tablet']
regions = ['North America', 'Europe', 'Asia', 'South America']
bid_strategies = ['CPC', 'CPM', 'CPA']

data = []
for campaign_id in [f'C{str(i).zfill(3)}' for i in range(1, num_campaigns + 1)]:
    for date in days:
        impressions = np.random.randint(5000, 200000)
        ctr = np.random.uniform(0.005, 0.02)
        clicks = int(impressions * ctr)
        cvr = np.random.uniform(0.02, 0.10)
        conversions = int(clicks * cvr)
        cost_per_click = np.random.uniform(1, 5)
        cost = clicks * cost_per_click
        bid_strategy = np.random.choice(bid_strategies)
        device = np.random.choice(devices)
        region = np.random.choice(regions)

        data.append([
            campaign_id, date, impressions, clicks, conversions,
            round(cost, 2), bid_strategy, device, region
        ])

df = pd.DataFrame(data, columns=[
    'campaign_id', 'date', 'impressions', 'clicks', 'conversions',
    'cost', 'bid_strategy', 'device', 'region'
])

df.to_csv('../data/adtech_campaign_performance.csv', index=False)
print("Synthetic AdTech campaign data CSV created successfully at: data/adtech_campaign_performance.csv")
