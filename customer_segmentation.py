
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('cleaned_output_S1.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 計算RFM指標
now = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (now - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalPrice':'Monetary'})

# 標準化
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# KMeans 分群（預設3群）
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 可視化（Recency vs Monetary）
plt.figure(figsize=(10,6))
for c in rfm['Cluster'].unique():
    sub = rfm[rfm['Cluster']==c]
    plt.scatter(sub['Recency'], sub['Monetary'], label=f'Cluster {c}', alpha=0.7)
plt.xlabel('Recency (天數, 越小代表越近期消費)')
plt.ylabel('Monetary (總消費金額)')
plt.title('Customer Segmentation (RFM Analysis)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('part7_rfm_cluster.png')
plt.close()

# 輸出每群平均值與前10大高價值客戶
mean_table = rfm.groupby('Cluster').mean()
mean_table.to_csv('part7_cluster_summary.csv', encoding='utf-8-sig')
top_cluster = mean_table['Monetary'].idxmax()
valuable_customers = rfm[rfm['Cluster']==top_cluster].sort_values('Monetary', ascending=False)
valuable_customers.head(10).to_csv('part7_top10_valuable_customers.csv', encoding='utf-8-sig')
print("已產出分群視覺化與前10名高價值客戶列表！")
