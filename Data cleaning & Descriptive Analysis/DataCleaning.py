import pandas as pd
import os
import zipfile

zip_filename = 'OnlineRetail.zip'
csv_filename = 'OnlineRetail.csv'
output_file = 'cleaned_output.csv'

#解壓縮zip檔
if not os.path.exists(csv_filename):
    if not os.path.exists(zip_filename):
        raise FileNotFoundError(f"找不到壓縮檔：{zip_filename}")
    
    print(f"解壓縮 {zip_filename} ...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extract(csv_filename)
    print(f"已解壓出 {csv_filename}")

#Data Cleaning Process
df = pd.read_csv(csv_filename, parse_dates=['InvoiceDate'])

#1.移除含缺漏值的列（任一欄為空值即刪除）
df = df.dropna()

#2.移除異常資料（條件：UnitPrice<=0、Quantity=0、Q為極端值(在此設定為Q>20000, Q<-20000)）
df = df[
    (df['UnitPrice'] > 0) &
    (df['Quantity'] != 0) &
    (df['Quantity'].between(-20000, 20000))
].copy()

#3.建立新欄位
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Week'] = df['InvoiceDate'].dt.isocalendar().week
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['IsReturn'] = ((df['Quantity'] < 0) | (df['InvoiceNo'].astype(str).str.startswith('C'))).astype(int)

#4.儲存新資料集檔案
df.to_csv(output_file, index=False)
print(f"資料處理完成，已輸出為：{output_file}")
