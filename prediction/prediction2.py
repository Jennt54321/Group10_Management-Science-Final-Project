#價格敏感度分析

import pandas as pd
import numpy as np
from patsy import dmatrix
from statsmodels.api import OLS 
import matplotlib.pyplot as plt 

#Step1: 整理資料集
df = pd.read_csv("cleaned_output.csv")
df = df[~df['IsReturn']]
top_product = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).index[0] #找出最熱賣的商品
df_top = df[df['Description'] == top_product] #篩選出只有 top_product 的資料集

#Step2: 設定 Two-knot spline
x = df_top['UnitPrice'] #單價作為自變數
y = df_top['Quantity'] #數量作為應變數
k1,k2 = np.percentile(x,[33,66])
spline_formula = f"bs(UnitPrice, knots=({k1}, {k2}), degree=1, include_intercept=True)"
X_spline = dmatrix(spline_formula, data=df_top, return_type='dataframe')

model = OLS(y, X_spline).fit() #使用 OLS 進行回歸建模
print(model.summary())

#建立新的單價區間，用來畫出 spline 曲線
X_pred = np.linspace(x.min(), x.max(), 100)               # 產生 100 個價格點
X_pred_df = pd.DataFrame({'UnitPrice': X_pred})                        # 包成 dataframe 給 patsy 使用
X_pred_design = dmatrix(spline_formula, data=X_pred_df, return_type='dataframe')  # 建立預測的設計矩陣

# 12. 使用模型進行預測
y_pred = model.predict(X_pred_design)

# 13. 繪圖：散點圖顯示原始資料 + 預測曲線
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3, label='Actual', s=10)   # 原始資料點
plt.plot(X_pred, y_pred, color='red', label='Spline Prediction')  # spline 預測線
plt.title(f"Two-knot Spline Regression\n{top_product}: Quantity vs UnitPrice")
plt.xlabel("UnitPrice")
plt.ylabel("Quantity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
