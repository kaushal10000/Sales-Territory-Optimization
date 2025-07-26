import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

# Define segments, industries, regions, and categories
segments = ['SMB', 'MidMarket', 'Enterprise']
industries = ['Retail', 'Finance', 'Healthcare', 'Technology', 'Manufacturing']
regions = ['North', 'South', 'East', 'West']
products = ['Analytics', 'CRM', 'ERP']

# Generate base data
data = pd.DataFrame({
    'CustomerID': range(1, n+1),
    'CustomerSegment': np.random.choice(segments, n, p=[0.5, 0.3, 0.2]),
    'Industry': np.random.choice(industries, n),
    'Region': np.random.choice(regions, n),
    'ProductCategory': np.random.choice(products, n),
    'DealSize': np.round(np.random.normal(loc=25000, scale=8000, size=n)).clip(3000, 100000),
    'SalesRep': np.random.choice(['Rep1', 'Rep2', 'Rep3', 'Rep4'], n),
    'PreviousInteractions': np.random.poisson(lam=4, size=n).clip(0, 15),
    'LeadScore': np.round(np.random.beta(a=2, b=5, size=n), 2),
    'CreatedDate': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 180, n), unit='d')
})

# Generate ClosedDate and Duration
data['SalesCycleDays'] = np.random.randint(10, 60, n)
data['ClosedDate'] = data['CreatedDate'] + pd.to_timedelta(data['SalesCycleDays'], unit='d')

# Win rate logic based on region & segment
region_win_rate = {'North': 0.35, 'South': 0.25, 'East': 0.30, 'West': 0.40}
segment_modifier = {'SMB': 0.90, 'MidMarket': 1.00, 'Enterprise': 1.10}

win_prob = []
for _, row in data.iterrows():
    base = region_win_rate[row['Region']]
    modifier = segment_modifier[row['CustomerSegment']]
    score = base * modifier + (row['LeadScore'] - 0.5) * 0.3  # boost based on lead quality
    win_prob.append(score)

data['WinProb'] = np.clip(win_prob, 0, 1)
data['Win'] = np.random.binomial(1, data['WinProb'])

# Profit margin based on product type
product_margins = {'Analytics': (0.25, 0.45), 'CRM': (0.20, 0.35), 'ERP': (0.15, 0.30)}

margins = []
for _, row in data.iterrows():
    if row['Win'] == 1:
        min_m, max_m = product_margins[row['ProductCategory']]
        margin = np.round(np.random.uniform(min_m, max_m), 2)
    else:
        margin = 0
    margins.append(margin)

data['ProfitMargin'] = margins
data['Profit'] = np.round(data['DealSize'] * data['ProfitMargin'], 2)

# Final formatting
data = data.drop(columns=['WinProb'])
data = data.sort_values(by='CreatedDate').reset_index(drop=True)

# Save to CSV
data.to_csv("synthetic_sales_data.csv", index=False)
print("Saved as synthetic_sales_data.csv")
