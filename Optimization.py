import pandas as pd
import numpy as np
import statsmodels.api as sm
from prophet import Prophet
from gurobipy import Model, GRB


# read cleaned_output.csv
df = pd.read_csv("cleaned_output.csv")


# get rid of return
df_no_return = df[df["IsReturn"] == 0]

# get rid of not real product e.g postage
df_filtered = df_no_return[df_no_return["StockCode"] != 'POST']

# group the products
product_sales = df_filtered.groupby("StockCode").agg({
    "Description": "first",
    "UnitPrice": "mean",
    "Quantity": "sum",
    "TotalPrice": "sum"
}).reset_index()

# find top 5 products
top5 = product_sales.sort_values(by="TotalPrice", ascending=False).head(5)
print("Top 5 products by total sales:\n", top5)

# Optimization 

# product code
products = top5["StockCode"].tolist()

# UnitPrice
unit_price = dict(zip(top5["StockCode"], top5["UnitPrice"]))

# history data
quantity_sold = dict(zip(top5["StockCode"], top5["Quantity"]))

# assume cost = price*60%
cost_price = {p: unit_price[p] * 0.6 for p in products}

# profit = price - cost
profit = {p: unit_price[p] - cost_price[p] for p in products}

# create Gurobi model
m = Model("Restock_Optimization")

# decision variables：restock（integer）
x = {p: m.addVar(vtype=GRB.INTEGER, name=f"Restock_{p}") for p in products}

# objective：maximize total profit
m.setObjective(sum(profit[p] * x[p] for p in products), GRB.MAXIMIZE)

# constraint 1：budget limit（£1000）
m.addConstr(sum(cost_price[p] * x[p] for p in products) <= 1000, name="BudgetConstraint")

# constraint 2：storage limit（300）
m.addConstr(sum(x[p] for p in products) <= 300, name="StorageConstraint")

# constraint 3：do not exceed history
for p in products:
    m.addConstr(x[p] <= quantity_sold[p], name=f"MaxStock_{p}")

# optimize
m.optimize()

# print the results
print("\n 最佳補貨建議：")
for p in products:
    print(f"商品 {p}：建議補貨 {int(x[p].X)} 件，單價 £{unit_price[p]:.2f}，利潤 £{profit[p]:.2f}")