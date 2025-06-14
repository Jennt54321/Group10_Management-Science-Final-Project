import pandas as pd
import numpy as np
from patsy import dmatrix
from statsmodels.api import OLS
import matplotlib.pyplot as plt

# 載入資料
df = pd.read_csv("cleaned_output.csv")

# 選定商品（你之前圖中的商品）
product_name = "WORLD WAR 2 GLIDERS ASSTD DESIGNS"

# 篩選該商品 + 排除退貨
df_product = df[(df['Description'] == product_name) & (~df['IsReturn'])].copy()

# ========== 🧹 Step 1: 移除極端值 (Quantity) ==========
q_low, q_high = np.percentile(df_product['Quantity'], [2.5, 97.5])
df_product = df_product[(df_product['Quantity'] >= q_low) & (df_product['Quantity'] <= q_high)]

# ========== 🔢 Step 2: log 轉換目標變數 (避免大量偏態) ==========
df_product['log_quantity'] = np.log1p(df_product['Quantity'])  # log(Quantity + 1)

# ========== 🎯 Step 3: 準備 spline 變數 ==========
X_raw = df_product['UnitPrice']
y = df_product['log_quantity']

# 選取 30% 和 70% 的節點位置（避免過度靠近邊界）
k1, k2 = np.percentile(X_raw, [30, 70])

# 產生 spline 設計矩陣
spline_formula = f"bs(UnitPrice, knots=({k1}, {k2}), degree=1, include_intercept=True)"
X_spline = dmatrix(spline_formula, data=df_product, return_type='dataframe')

# ========== 📈 Step 4: 建模 ==========
model = OLS(y, X_spline).fit()
print(model.summary())

# ========== 📊 Step 5: 建立預測曲線 ==========
X_pred = np.linspace(X_raw.min(), X_raw.max(), 100)
X_pred_df = pd.DataFrame({'UnitPrice': X_pred})
X_pred_design = dmatrix(spline_formula, data=X_pred_df, return_type='dataframe')
y_pred_log = model.predict(X_pred_design)
y_pred = np.expm1(y_pred_log)  # 還原回原本單位（expm1 = exp(x) - 1）

# ========== 🖼 Step 6: 繪圖 ==========
plt.figure(figsize=(10, 6))
plt.scatter(X_raw, df_product['Quantity'], alpha=0.3, label='Actual', s=10)
plt.plot(X_pred, y_pred, color='red', label='Spline Prediction')
plt.title(f"Improved Two-knot Spline Regression\n{product_name}: Quantity vs UnitPrice")
plt.xlabel("UnitPrice")
plt.ylabel("Quantity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
