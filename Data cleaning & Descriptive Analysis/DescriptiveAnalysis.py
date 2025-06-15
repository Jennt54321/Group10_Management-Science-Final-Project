import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('cleaned_output.csv', parse_dates=['InvoiceDate'])
#建立新欄位"YearMonth"區分每月交易紀錄
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
#找出總銷售量最高的前三個商品，並過濾其他商品選項僅保留此三項
top_products = df.groupby('StockCode')['Quantity'].sum().nlargest(3).index.tolist()
filtered_df = df[df['StockCode'].isin(top_products)]

#1. 商品每月銷售狀況
#商品每月銷售量：sum Quantity
monthly_sales = (filtered_df.groupby(['YearMonth', 'StockCode'])['Quantity'].sum().reset_index())
#繪製圖表
plt.figure(figsize=(12, 6))
for stock in top_products:
    stock_data = monthly_sales[monthly_sales['StockCode'] == stock]
    plt.plot(stock_data['YearMonth'], stock_data['Quantity'], marker='o', label=f'{stock}')
plt.title('Comparison of monthly sales trends of products (taking 84077, 22197, 85099B as example)')
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.tight_layout()
#顯示x軸的每個刻度
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.show()

#2. 商品每月收益
#商品每月收益：sum TotalPrice
monthly_profit = (filtered_df.groupby(['YearMonth', 'StockCode'])['TotalPrice'].sum().reset_index())
#繪製圖表
plt.figure(figsize=(12, 6))
for stock in top_products:
    stock_data = monthly_profit[monthly_profit['StockCode'] == stock]
    plt.plot(stock_data['YearMonth'], stock_data['TotalPrice'], marker='o', label=f'{stock}')
plt.title('Comparison of monthly profit trends of products (taking 84077, 22197, 85099B as example)')
plt.xlabel('Month')
plt.ylabel('TotalPrice')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.show()
