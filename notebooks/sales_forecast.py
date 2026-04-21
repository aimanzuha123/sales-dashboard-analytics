import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../data/superstore.csv')
df['order_date'] = pd.to_datetime(df['order_date'])

monthly = df.groupby(
    pd.Grouper(key='order_date', freq='ME')
)['revenue'].sum().reset_index()

monthly['month_num'] = range(1, len(monthly)+1)

X = monthly[['month_num']]
y = monthly['revenue']

model = LinearRegression()
model.fit(X, y)

future_months = pd.DataFrame({
    'month_num': range(len(monthly)+1, len(monthly)+7)
})

future_sales = model.predict(future_months)

last_date = monthly['order_date'].max()

future_dates = pd.date_range(
    start=last_date,
    periods=7,
    freq='ME'
)[1:]

forecast = pd.DataFrame({
    'order_date': future_dates,
    'forecast_sales': future_sales.round(2)
})

print(forecast)

plt.figure(figsize=(12,6))
plt.plot(monthly['order_date'], monthly['revenue'], marker='o', label='Actual')
plt.plot(forecast['order_date'], forecast['forecast_sales'], marker='o', label='Forecast')

plt.title("Sales Forecast")
plt.legend()
plt.tight_layout()
plt.savefig('../data/sales_forecast.png')
plt.show()

forecast.to_csv('../data/sales_forecast.csv', index=False)

print("PROJECT 3 COMPLETE")