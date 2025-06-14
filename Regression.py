import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd


df = pd.read_csv("cleaned_output.csv")
print(df['UnitPrice'].describe())
df = df[['UnitPrice', 'IsReturn']] # 選擇需要欄位
df = df[df['UnitPrice'] > 0] # 過濾掉價格為 0 或負值的資料（無意義）
cutoff_95 = df['UnitPrice'].quantile(0.95)
df = df[df['UnitPrice'] <= cutoff_95]
df['IsReturn'] = df['IsReturn'].astype(int) # 將 True/False 轉換為 1/0
#print(df.sort_values(by='UnitPrice', ascending=False).head(10))


# 分割資料集
X = df[['UnitPrice']]
y = df['IsReturn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立模型並訓練
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # 預測機率

# 印出結果
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("係數 Coef:", model.coef_)
print("截距 Intercept:", model.intercept_)

# 產生價格範圍（例如：0~50）
price_range = np.linspace(df['UnitPrice'].min(), df['UnitPrice'].max(), 200).reshape(-1, 1)
return_probs = model.predict_proba(price_range)[:, 1]

# 畫圖
plt.figure(figsize=(8,5))
plt.plot(price_range, return_probs, color='red')
plt.xlabel("Unit Price")
plt.ylabel("Predicted Return Probability")
plt.title("Predicted Return Probability vs. Unit Price")
plt.grid(True)
plt.show()
