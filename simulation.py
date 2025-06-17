import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sim_times = 100  # 蒙地卡羅模擬次數
price_elasticity = -2  # 價格彈性

df = pd.read_csv('cleaned_output_S1.csv')

# 退貨率對營收損耗影響並多次模擬
sales_df = df[df['IsReturn'] == 0]
base_sales = sales_df['TotalPrice'].sum()
return_rates = np.arange(0, 0.31, 0.05)
total_sales_mean = []
total_sales_std = []
for rate in return_rates:
    sim_results = []
    for _ in range(sim_times):
        n_return = int(len(sales_df) * rate)
        simulated = sales_df.copy()
        simulated.loc[simulated.sample(n=n_return).index, 'IsReturn'] = 1
        simulated['NetSales'] = simulated.apply(lambda row: -row['TotalPrice'] if row['IsReturn']==1 else row['TotalPrice'], axis=1)
        sim_results.append(simulated['NetSales'].sum())
    total_sales_mean.append(np.mean(sim_results))
    total_sales_std.append(np.std(sim_results))

plt.figure()
plt.errorbar(return_rates * 100, total_sales_mean, yerr=total_sales_std, fmt='o-', capsize=5)
plt.xlabel('Return Rate (%)')
plt.ylabel('Net Sales')
plt.title('Impact of Return Rate on Net Sales (Monte Carlo)')
plt.grid()
plt.tight_layout()
plt.savefig('part3_return_rate.png')
plt.close()

# 不同月份價格對銷量影響
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month
discounts = [1, 0.95, 0.9, 0.85]
monthly_quantity = {d: [] for d in discounts}
for d in discounts:
    discount_rate = 1 - d
    for m in range(1, 13):
        base_qty = df[df['Month'] == m]['Quantity'].sum()
        # 價格彈性
        adjusted_qty = base_qty * (1 + price_elasticity * discount_rate)
        monthly_quantity[d].append(adjusted_qty)

plt.figure()
for d in discounts:
    plt.plot(range(1, 13), monthly_quantity[d], label=f'Discount {int(d*100)}%')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold (with Price Elasticity)')
plt.title(f'Quantity Sold by Month under Different Discounts\n(Price Elasticity={price_elasticity})')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('part3_discount_month.png')
plt.close()

# 商品價格折扣對收益變化
discounts_curve = np.arange(0.7, 1.01, 0.05)
revenues = []
for d in discounts_curve:
    discount_rate = 1 - d
    # 價格彈性
    adjusted_qty = df['Quantity'] * (1 + price_elasticity * discount_rate)
    discounted_price = df['UnitPrice'] * d
    discounted_sales = discounted_price * adjusted_qty
    revenues.append(discounted_sales.sum())

plt.figure()
plt.plot(discounts_curve * 100, revenues, marker='o')
plt.xlabel('Discount (%)')
plt.ylabel('Total Revenue (with Price Elasticity)')
plt.title(f'Revenue vs. Discount Rate\n(Price Elasticity={price_elasticity})')
plt.grid()
plt.tight_layout()
plt.savefig('part3_revenue_curve.png')
plt.close()
