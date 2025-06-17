📊 Project Overview

This project analyzes historical transaction data from a UK-based online retailer (2010/12–2011/12), aiming to support strategic business decisions through data analysis and modeling techniques.

📂 Dataset

Source: UCI Machine Learning Repository – Online Retail
Records: 541,909 transactions
Key Fields: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

🔍 Key Analyses & Models

1. Descriptive Analysis

(1) Trends in sales volume and revenue by product and time

(2) Customer and country-level purchasing behaviors

2. Simulation

(1) Monte Carlo simulation of return rates and price discount strategies

(2) Analysis of price elasticity effects on revenue

3. Prediction

(1) Monthly order volume predictions using: Exponential Smoothing / SARIMA (better performance after outlier exclusion)

(2) Two-knot spline to model nonlinear relationship between price and quantity sold

4. Regression

Logistic regression to estimate return probability based on product price

6. Optimization

Linear programming for profit-maximizing restocking under budget and space constraints

7. Customer Segmentation

RFM-based clustering using KMeans to identify high-value customers

🛠 Tech Stack

1. Python (pandas, matplotlib, statsmodels, scikit-learn)
2. Excel (pivot tables for descriptive stats)
3. GitHub for version control

📈 Business Insights

1. High-priced products have higher return risks
2. SARIMA outperforms exponential smoothing for stable forecasts
3. Smart discounting boosts revenue but must avoid over-discounting
4. Restocking should focus on high-margin products
5. VIP customers contribute disproportionately to total revenue

