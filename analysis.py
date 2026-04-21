import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

print("Step 1: Libraries loaded!")

random.seed(42)
np.random.seed(42)

regions    = ['North', 'South', 'East', 'West']
categories = ['Electronics', 'Furniture', 'Clothing', 'Food & Bev']
products   = {
    'Electronics': ['MacBook Pro', 'iPhone 15', 'Samsung TV', 'iPad Air', 'Dell Monitor'],
    'Furniture':   ['Office Desk', 'Standing Desk', 'Office Chair', 'Bookshelf', 'Sofa'],
    'Clothing':    ['Winter Jacket', 'Running Shoes', 'Formal Shirt', 'Jeans', 'Sneakers'],
    'Food & Bev':  ['Coffee Maker', 'Air Fryer', 'Water Bottle', 'Blender', 'Tea Set']
}
segments   = ['Consumer', 'Corporate', 'Home Office']
rows       = []
start_date = datetime(2022, 1, 1)

for i in range(5000):
    cat         = random.choice(categories)
    prod        = random.choice(products[cat])
    region      = random.choice(regions)
    segment     = random.choice(segments)
    order_date  = start_date + timedelta(days=random.randint(0, 730))
    ship_date   = order_date + timedelta(days=random.randint(1, 7))
    quantity    = random.randint(1, 10)
    unit_price  = round(random.uniform(20, 1500), 2)
    discount    = round(random.choice([0, 0.05, 0.1, 0.15, 0.2]), 2)
    revenue     = round(unit_price * quantity * (1 - discount), 2)
    profit      = round(revenue * random.uniform(0.1, 0.45), 2)
    customer_id = f"CUST-{random.randint(1000, 3000)}"
    order_id    = f"ORD-{i+1000}"
    rows.append([order_id, customer_id, order_date, ship_date,
                 segment, region, cat, prod,
                 quantity, unit_price, discount, revenue, profit])

df = pd.DataFrame(rows, columns=[
    'order_id','customer_id','order_date','ship_date',
    'segment','region','category','product_name',
    'quantity','unit_price','discount','revenue','profit'
])

df.to_csv('../data/superstore.csv', index=False)
print(f"Step 2: Dataset created! Shape: {df.shape}")

df['order_date']   = pd.to_datetime(df['order_date'])
df['ship_date']    = pd.to_datetime(df['ship_date'])
df['year']         = df['order_date'].dt.year
df['month']        = df['order_date'].dt.month
df['quarter']      = df['order_date'].dt.quarter
df['days_to_ship'] = (df['ship_date'] - df['order_date']).dt.days
print("Step 3: Data cleaned!")

total_revenue = round(df['revenue'].sum(), 2)
total_orders  = df['order_id'].nunique()
aov           = round(total_revenue / total_orders, 2)
gross_margin  = round(df['profit'].sum() / df['revenue'].sum() * 100, 2)

print("\n" + "="*40)
print(f"  Total Revenue  : ${total_revenue:,.2f}")
print(f"  Total Orders   : {total_orders:,}")
print(f"  Avg Order Value: ${aov:,.2f}")
print(f"  Gross Margin   : {gross_margin}%")
print("="*40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sales Dashboard - FY 2022-2023', fontsize=16, fontweight='bold')

monthly = df.groupby(['year','month'])['revenue'].sum().reset_index()
monthly['period'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
axes[0,0].plot(monthly['period'], monthly['revenue'], marker='o', color='#378ADD', linewidth=2)
axes[0,0].set_title('Monthly Revenue Trend')
axes[0,0].tick_params(axis='x', rotation=90)

by_cat = df.groupby('category')['revenue'].sum().reset_index()
axes[0,1].bar(by_cat['category'], by_cat['revenue'],
              color=['#378ADD','#1D9E75','#EF9F27','#D4537E'])
axes[0,1].set_title('Revenue by Category')

top_products = df.groupby('product_name')['revenue'].sum().sort_values(ascending=True).tail(10)
axes[1,0].barh(top_products.index, top_products.values, color='#1D9E75')
axes[1,0].set_title('Top 10 Products by Revenue')

by_region = df.groupby('region')['revenue'].sum().reset_index()
axes[1,1].bar(by_region['region'], by_region['revenue'], color='#EF9F27')
axes[1,1].set_title('Revenue by Region')

plt.tight_layout()
plt.savefig('../data/sales_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nStep 4: Charts saved!")
print("Project 1 COMPLETE!")