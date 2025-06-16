import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# 讀資料
df = pd.read_csv("cleaned_output.csv")
df = df[['UnitPrice', 'IsReturn']]
df = df[(df['UnitPrice'] > 0)]
df['IsReturn'] = df['IsReturn'].astype(int)
cutoff_95 = df['UnitPrice'].quantile(0.95)
df = df[df['UnitPrice'] <= cutoff_95]

# 上採樣 minority class
df_majority = df[df.IsReturn == 0]
df_minority = df[df.IsReturn == 1]
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# 訓練模型
X = df_balanced[['UnitPrice']]
y = df_balanced['IsReturn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# 評估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("係數 Coef:", model.coef_)
print("截距 Intercept:", model.intercept_)

# 視覺化
price_range = np.linspace(df['UnitPrice'].min(), df['UnitPrice'].max(), 200).reshape(-1, 1)
return_probs = model.predict_proba(price_range)[:, 1]

plt.figure(figsize=(8,5))
plt.plot(price_range, return_probs, color='red')
plt.xlabel("Unit Price")
plt.ylabel("Predicted Return Probability")
plt.title("Predicted Return Probability vs. Unit Price (with Upsampling)")
plt.grid(True)
plt.show()
