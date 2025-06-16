import pandas as pd
import numpy as np
import seaborn as sns
from patsy import dmatrix
from statsmodels.api import OLS
import matplotlib.pyplot as plt

#Step1: 整理資料集
df = pd.read_csv("cleaned_output.csv")
df = df[~df['IsReturn']]
top_product = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).index[0] #找出最熱賣的商品
df_top = df[df['Description'] == top_product] #篩選出只有 top_product 的資料集)
q_low, q_high = np.percentile(df_top['Quantity'], [2.5, 97.5]) 
df_product = df_top[(df_top['Quantity'] >= q_low) & (df_top['Quantity'] <= q_high)] #移除 Quantity 的極端值，只保留中間 95% 的數量資料

#Step 2: log(Quantity + 1) 轉換目標變數，因線性回歸模型假設誤差項服從常態分布，透過對數轉換讓目標變數更符合線性模型的基本假設
df_product['log_quantity'] = np.log1p(df_product['Quantity']) 

#Step 3: 準備 spline 變數
X_raw = df_product['UnitPrice']
y = df_product['log_quantity']
k1, k2 = np.percentile(X_raw, [30, 70])
if k1 == k2:
    k1, k2 = np.percentile(X_raw, [25, 75])  # 或進一步拉大間距
spline_formula = f"bs(UnitPrice, knots=({k1}, {k2}), degree=1, include_intercept=True)" # 產生 spline 設計矩陣
X_spline = dmatrix(spline_formula, data=df_product, return_type='dataframe')
model = OLS(y, X_spline).fit() #建模
print(model.summary())

#Step 4: 建立預測曲線
X_pred = np.linspace(X_raw.min(), X_raw.max(), 100)
X_pred_df = pd.DataFrame({'UnitPrice': X_pred})
X_pred_design = dmatrix(spline_formula, data=X_pred_df, return_type='dataframe')
y_pred_log = model.predict(X_pred_design)
y_pred = np.expm1(y_pred_log)  # 還原回原本單位（expm1 = exp(x) - 1）

plt.figure(figsize=(10, 6))
plt.scatter(X_raw, df_product['Quantity'], alpha=0.3, label='Actual', s=10)
plt.plot(X_pred, y_pred, color='red', label='Spline Prediction')
plt.title(f"Two-knot Spline Regression\n{top_product}: Quantity vs UnitPrice")
plt.xlabel("UnitPrice")
plt.ylabel("Quantity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 補充分析：Quantity 以及處理後的 log_quantity 的離散趨勢
# Quantity
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_product['Quantity'], bins=50, kde=True)
plt.title('Quantity Distribution')

# log_quantity
plt.subplot(1, 2, 2)
sns.histplot(df_product['log_quantity'], bins=50, kde=True)
plt.title('Log-Transformed Quantity Distribution')
plt.show()

# 計算偏態與峰態
print("Skew (original):", df_product['Quantity'].skew())
print("Kurtosis (original):", df_product['Quantity'].kurt())
print("Skew (log):", df_product['log_quantity'].skew())
print("Kurtosis (log):", df_product['log_quantity'].kurt())

# 補充分析：實際看 Quantity 的前十大分佈
print(df_product['Quantity'].value_counts().head(10))