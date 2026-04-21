import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# -- Load Data -----------------------------------------------------
df = pd.read_csv('../data/superstore.csv')
df['order_date'] = pd.to_datetime(df['order_date'])
print("Data loaded! Shape:", df.shape)

# -- RFM Calculation -----------------------------------------------
snapshot_date = df['order_date'].max() + pd.Timedelta(days=1)
print("Snapshot date:", snapshot_date)

rfm = df.groupby('customer_id').agg(
    recency   = ('order_date', lambda x: (snapshot_date - x.max()).days),
    frequency = ('order_id',   'nunique'),
    monetary  = ('revenue',    'sum')
).reset_index()

rfm['monetary'] = rfm['monetary'].round(2)
print("\nRFM Table Sample:")
print(rfm.head())

# -- RFM Scoring (1-5) ---------------------------------------------
rfm['r_score'] = pd.qcut(rfm['recency'],   q=5, labels=[5,4,3,2,1]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'],  q=5, labels=[1,2,3,4,5]).astype(int)

rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
rfm['total_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']

print("\nRFM Scores Sample:")
print(rfm[['customer_id','recency','frequency','monetary','r_score','f_score','m_score','total_score']].head(10))

# -- Customer Segmentation -----------------------------------------
def segment_customer(row):
    r = row['r_score']
    f = row['f_score']
    m = row['m_score']
    score = row['total_score']

    if r >= 4 and f >= 4 and m >= 4:
        return 'High Value'
    elif r >= 3 and f >= 3:
        return 'Loyal'
    elif r >= 4 and f <= 2:
        return 'New Customer'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Churned'
    else:
        return 'Potential'

rfm['segment'] = rfm.apply(segment_customer, axis=1)

# -- Segment Summary -----------------------------------------------
summary = rfm.groupby('segment').agg(
    customers = ('customer_id', 'count'),
    avg_recency   = ('recency',   'mean'),
    avg_frequency = ('frequency', 'mean'),
    avg_monetary  = ('monetary',  'mean'),
    total_revenue = ('monetary',  'sum')
).round(2).reset_index()

summary['pct_customers'] = (summary['customers'] / summary['customers'].sum() * 100).round(1)
summary = summary.sort_values('total_revenue', ascending=False)

print("\n" + "="*60)
print("CUSTOMER SEGMENTATION RESULTS")
print("="*60)
print(summary.to_string(index=False))
print("="*60)

# -- Save RFM Table ------------------------------------------------
rfm.to_csv('../data/rfm_segments.csv', index=False)
print("\nRFM data saved to data/rfm_segments.csv")

# -- Charts --------------------------------------------------------
colors = {
    'High Value':   '#1D9E75',
    'Loyal':        '#378ADD',
    'New Customer': '#EF9F27',
    'At Risk':      '#E24B4A',
    'Churned':      '#888780',
    'Potential':    '#D4537E'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Segmentation - RFM Analysis', fontsize=16, fontweight='bold')

# Chart 1 - Customer count by segment
seg_colors = [colors.get(s, '#888780') for s in summary['segment']]
axes[0,0].bar(summary['segment'], summary['customers'], color=seg_colors)
axes[0,0].set_title('Customers by Segment')
axes[0,0].set_xlabel('Segment')
axes[0,0].set_ylabel('Number of Customers')
axes[0,0].tick_params(axis='x', rotation=30)

# Chart 2 - Revenue by segment
axes[0,1].bar(summary['segment'], summary['total_revenue'], color=seg_colors)
axes[0,1].set_title('Total Revenue by Segment')
axes[0,1].set_xlabel('Segment')
axes[0,1].set_ylabel('Revenue ($)')
axes[0,1].tick_params(axis='x', rotation=30)

# Chart 3 - RFM Scatter (Recency vs Monetary)
scatter_colors = [colors.get(s, '#888780') for s in rfm['segment']]
axes[1,0].scatter(rfm['recency'], rfm['monetary'],
                  c=scatter_colors, alpha=0.6, s=40)
axes[1,0].set_title('Recency vs Monetary Value')
axes[1,0].set_xlabel('Recency (days)')
axes[1,0].set_ylabel('Monetary Value ($)')
legend_patches = [mpatches.Patch(color=v, label=k) for k,v in colors.items()]
axes[1,0].legend(handles=legend_patches, fontsize=8)

# Chart 4 - Avg Monetary by Segment
axes[1,1].barh(summary['segment'], summary['avg_monetary'], color=seg_colors)
axes[1,1].set_title('Avg Revenue per Customer by Segment')
axes[1,1].set_xlabel('Avg Revenue ($)')

plt.tight_layout()
plt.savefig('../data/rfm_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nStep: Charts saved to data/rfm_dashboard.png")

# -- Business Recommendations --------------------------------------
print("\n" + "="*60)
print("BUSINESS RECOMMENDATIONS")
print("="*60)

high_value = summary[summary['segment']=='High Value']
at_risk    = summary[summary['segment']=='At Risk']
churned    = summary[summary['segment']=='Churned']

if len(high_value) > 0:
    print(f"\n1. HIGH VALUE ({int(high_value['customers'].values[0])} customers)")
    print("   -> Launch VIP loyalty program")
    print("   -> Offer exclusive early access to new products")
    print("   -> Assign dedicated account managers")

if len(at_risk) > 0:
    print(f"\n2. AT RISK ({int(at_risk['customers'].values[0])} customers)")
    print("   -> Send win-back email campaign immediately")
    print("   -> Offer 15% discount coupon")
    print("   -> Conduct survey to understand drop-off reason")

if len(churned) > 0:
    print(f"\n3. CHURNED ({int(churned['customers'].values[0])} customers)")
    print("   -> Last chance reactivation campaign")
    print("   -> Offer biggest discount (20-25%)")
    print("   -> If no response in 30 days - remove from active list")

print("\nProject 2 COMPLETE!")
print("="*60)