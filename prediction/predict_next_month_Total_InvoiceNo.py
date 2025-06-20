import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Step1: 載入資料並處理
df = pd.read_csv("cleaned_output.csv")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M') #建立InvoiceMonth 儲存月份的欄位

start_date = pd.to_datetime('2010-12-01')
end_date = pd.to_datetime('2011-12-31 23:59:59')
df_all = df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] <= end_date)] #取日期介於 2010/12-2011/12 的資料集
df_all = df_all[~df_all['InvoiceNo'].astype(str).str.startswith('C')] #移除退貨資料

#Step2: 畫出 Acutal data 的趨勢
monthly_orders = df_all.groupby('InvoiceMonth')['InvoiceNo'].nunique()
monthly_orders.index = monthly_orders.index.to_timestamp()
monthly_orders.plot(figsize=(10, 5), title="Monthly Number of Orders")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.ylim(bottom=0)
plt.grid(True)
plt.show()

#Step3: #開始使用 ExponentialSmoothing & ARIMA 做rollaing expanding window
all_months = monthly_orders.index 
start_idx = all_months.get_loc('2011-04-01')
end_idx = all_months.get_loc('2011-12-01')

ex_forecast_values = []
ar_forecast_values = []
actual_values = []

for i in range(start_idx,end_idx+1):
    train_data = monthly_orders[:i]
    actual_data = monthly_orders.iloc[i]
    print("Rolling train length:", len(train＿data))
    
    #ExponentialSoothing
    ex_model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
    ex_fit = ex_model.fit()
    ex_forecast = ex_fit.forecast(steps=1)
    ex_forecast_values.append(ex_forecast.values[0])

    # SARIMAX(p,d,q)(P,D,Q,s)
    ar_model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    ar_fit = ar_model.fit(disp=False)
    ar_forecast = ar_fit.forecast(steps=1)
    ar_forecast_values.append(ar_forecast.values[0])
    
    actual_values.append(actual_data)

#Step4: 製作 actual vs forecast 對照表
forecast_months = all_months[start_idx:end_idx+1]

results_df =  pd.DataFrame({
    'ExponentialSmoothing Forecast' : ex_forecast_values,
    'ARIMA Forecast' : ar_forecast_values,
    'Actual' : actual_values
},index = forecast_months)

results_df.plot(figsize=(12, 5), marker='o')
plt.title("Rolling Forecast vs Actual (2011/04–2011/12)")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.grid(True)
plt.show()

#Step5: 分別計算 MAPE
mape_expo = (abs(results_df['Actual'] - results_df['ExponentialSmoothing Forecast']) / results_df['Actual']).mean() * 100
mape_arima = (abs(results_df['Actual'] - results_df['ARIMA Forecast']) / results_df['Actual']).mean() * 100


print(f"Exponential Smoothing Average MAPE: {mape_expo:.2f}%")
print(f"ARIMA Average MAPE: {mape_arima:.2f}%")

# 去除 2011 年 12 月的資料
results_df_excl_dec = results_df.drop(pd.Timestamp('2011-12-01'))

# 重新計算 MAPE（不包含 12 月）
mape_expo_excl_dec = (abs(results_df_excl_dec['Actual'] - results_df_excl_dec['ExponentialSmoothing Forecast']) / results_df_excl_dec['Actual']).mean() * 100
mape_arima_excl_dec = (abs(results_df_excl_dec['Actual'] - results_df_excl_dec['ARIMA Forecast']) / results_df_excl_dec['Actual']).mean() * 100

print(f"[不含12月] Exponential Smoothing MAPE: {mape_expo_excl_dec:.2f}%")
print(f"[不含12月] ARIMA MAPE: {mape_arima_excl_dec:.2f}%")



