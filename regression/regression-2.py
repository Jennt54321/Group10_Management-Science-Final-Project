import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Step 1: 載入資料
df = pd.read_csv("cleaned_output.csv")
df = df[(df['UnitPrice'] > 0) & (df['Quantity'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['IsReturn'] = df['IsReturn'].astype(int)

# Step 2: 顧客維度特徵計算
customer_features = df.groupby('CustomerID').agg(
    CustomerReturnRate=('IsReturn', 'mean'),
    CustomerOrderCount=('InvoiceNo', 'nunique'),
    CustomerTotalQuantity=('Quantity', 'sum'),
    CustomerAveragePrice=('UnitPrice', 'mean'),
    LastPurchaseDate=('InvoiceDate', 'max')
).reset_index()

# Step 3: 計算 Recency
latest_date = df['InvoiceDate'].max()
customer_features['CustomerRecencyDays'] = (latest_date - customer_features['LastPurchaseDate']).dt.days
customer_features.drop(columns=['LastPurchaseDate'], inplace=True)

# Step 4: 合併回主資料集
df = df.merge(customer_features, on='CustomerID', how='left')

# Step 5: 模型資料準備
feature_cols = ['UnitPrice', 'Quantity', 'CustomerReturnRate', 'CustomerOrderCount',
                'CustomerTotalQuantity', 'CustomerAveragePrice', 'CustomerRecencyDays']
df = df.dropna(subset=feature_cols)  # 若缺失太多可先行處理

X = df[feature_cols]
y = df['IsReturn']

# Step 6: 模型建立與評估
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, model.predict(X_test)))
