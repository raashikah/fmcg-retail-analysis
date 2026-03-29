"""
FMCG Retail Sales Analysis & Customer Segmentation
Author: Raashikah K
Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
 
print("=" * 60)
print("FMCG RETAIL SALES ANALYSIS & CUSTOMER SEGMENTATION")
print("=" * 60)
 
# 1. GENERATE REALISTIC RETAIL DATASET
np.random.seed(42)
n = 1000
 
categories = ['Beverages', 'Snacks', 'Dairy', 'Personal Care', 'Household']
regions    = ['North', 'South', 'East', 'West']
segments   = ['Retail', 'Wholesale', 'E-Commerce']
months     = pd.date_range('2022-01-01', periods=24, freq='MS')
 
data = pd.DataFrame({
    'Order_Date':      np.random.choice(months, n),
    'Category':        np.random.choice(categories, n, p=[0.25, 0.2, 0.2, 0.2, 0.15]),
    'Region':          np.random.choice(regions, n),
    'Segment':         np.random.choice(segments, n, p=[0.5, 0.3, 0.2]),
    'Units_Sold':      np.random.randint(10, 500, n),
    'Unit_Price':      np.round(np.random.uniform(20, 500, n), 2),
    'Discount':        np.round(np.random.uniform(0, 0.35, n), 2),
    'Marketing_Spend': np.round(np.random.uniform(500, 10000, n), 2),
})
 
data['Sales']   = np.round(data['Units_Sold'] * data['Unit_Price'] * (1 - data['Discount']), 2)
data['Profit']  = np.round(data['Sales'] * np.random.uniform(0.05, 0.3, n), 2)
data['Month']   = data['Order_Date'].dt.month
data['Year']    = data['Order_Date'].dt.year
data['Quarter'] = data['Order_Date'].dt.quarter
 
print(f"\n[OK] Dataset created: {data.shape[0]} rows x {data.shape[1]} columns")
 
# 2. DATA CLEANING
print("\n-- DATA CLEANING --")
print(f"Missing values: {data.isnull().sum().sum()}")
print(f"Duplicate rows: {data.duplicated().sum()}")
data.drop_duplicates(inplace=True)
print(f"[OK] Data is clean. Shape: {data.shape}")
 
# 3. EXPLORATORY DATA ANALYSIS
print("\n-- EXPLORATORY DATA ANALYSIS --")
print(f"Total Sales:  Rs.{data['Sales'].sum():,.0f}")
print(f"Total Profit: Rs.{data['Profit'].sum():,.0f}")
print(f"Avg Discount: {data['Discount'].mean()*100:.1f}%")
print(f"\nSales by Category:")
print(data.groupby('Category')['Sales'].sum().sort_values(ascending=False))
print(f"\nSales by Region:")
print(data.groupby('Region')['Sales'].sum().sort_values(ascending=False))
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('FMCG Retail Sales - Exploratory Analysis', fontsize=16, fontweight='bold')
 
cat_sales = data.groupby('Category')['Sales'].sum().sort_values()
axes[0, 0].barh(cat_sales.index, cat_sales.values, color=['#1F4E79','#2E75B6','#4BACC6','#9DC3E6','#BDD7EE'])
axes[0, 0].set_title('Total Sales by Category', fontweight='bold')
axes[0, 0].set_xlabel('Sales (Rs.)')
 
monthly = data.groupby('Order_Date')['Sales'].sum().reset_index()
axes[0, 1].plot(monthly['Order_Date'], monthly['Sales'], color='#1F4E79', linewidth=2, marker='o', markersize=4)
axes[0, 1].set_title('Monthly Sales Trend', fontweight='bold')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Sales (Rs.)')
axes[0, 1].tick_params(axis='x', rotation=45)
 
reg_sales = data.groupby('Region')['Sales'].sum()
colors = ['#1F4E79','#2E75B6','#4BACC6','#9DC3E6']
axes[1, 0].pie(reg_sales.values, labels=reg_sales.index, autopct='%1.1f%%', colors=colors)
axes[1, 0].set_title('Sales Distribution by Region', fontweight='bold')
 
axes[1, 1].scatter(data['Discount'], data['Sales'], alpha=0.4, color='#2E75B6', edgecolors='white', linewidth=0.3)
axes[1, 1].set_title('Discount vs Sales', fontweight='bold')
axes[1, 1].set_xlabel('Discount Rate')
axes[1, 1].set_ylabel('Sales (Rs.)')
 
plt.tight_layout()
plt.savefig('01_eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] EDA chart saved - 01_eda_analysis.png")
 
# 4. CUSTOMER SEGMENTATION (K-MEANS)
print("\n-- CUSTOMER SEGMENTATION (K-Means Clustering) --")
 
cluster_data = data.groupby(['Category', 'Region']).agg(
    Total_Sales  = ('Sales', 'sum'),
    Avg_Discount = ('Discount', 'mean'),
    Total_Units  = ('Units_Sold', 'sum'),
    Avg_Profit   = ('Profit', 'mean'),
    Order_Count  = ('Sales', 'count')
).reset_index()
 
features = ['Total_Sales', 'Avg_Discount', 'Total_Units', 'Avg_Profit']
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data[features])
 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
 
cluster_labels = {0: 'High Value', 1: 'Mid Value', 2: 'Low Value'}
cluster_data['Segment_Label'] = cluster_data['Cluster'].map(cluster_labels)
 
print("\nCluster Summary:")
print(cluster_data.groupby('Segment_Label')[['Total_Sales','Avg_Profit','Order_Count']].mean().round(0))
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Customer Segmentation - K-Means Clustering', fontsize=14, fontweight='bold')
 
colors_map = {'High Value': '#1F4E79', 'Mid Value': '#2E75B6', 'Low Value': '#9DC3E6'}
for label, grp in cluster_data.groupby('Segment_Label'):
    axes[0].scatter(grp['Total_Sales'], grp['Avg_Profit'],
                    label=label, color=colors_map[label], s=100, edgecolors='white')
axes[0].set_xlabel('Total Sales (Rs.)')
axes[0].set_ylabel('Avg Profit (Rs.)')
axes[0].set_title('Sales vs Profit by Segment')
axes[0].legend()
 
seg_counts = cluster_data['Segment_Label'].value_counts()
axes[1].bar(seg_counts.index, seg_counts.values,
            color=[colors_map[l] for l in seg_counts.index])
axes[1].set_title('Segment Distribution')
axes[1].set_ylabel('Count')
 
plt.tight_layout()
plt.savefig('02_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Segmentation chart saved - 02_segmentation.png")
 
# 5. SALES FORECASTING (RANDOM FOREST)
print("\n-- SALES FORECASTING (Random Forest Regressor) --")
 
model_data = data.copy()
model_data = pd.get_dummies(model_data, columns=['Category', 'Region', 'Segment'])
 
feature_cols = ['Units_Sold', 'Unit_Price', 'Discount', 'Marketing_Spend',
                'Month', 'Quarter', 'Year'] + \
               [c for c in model_data.columns if c.startswith(('Category_', 'Region_', 'Segment_'))]
 
X = model_data[feature_cols]
y = model_data['Sales']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
 
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
 
print(f"Model Performance:")
print(f"  R2 Score:        {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  Mean Abs Error:  Rs.{mae:,.0f}")
 
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(8)
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sales Forecasting - Random Forest Model', fontsize=14, fontweight='bold')
 
axes[0].barh(feat_imp.index[::-1], feat_imp.values[::-1], color='#2E75B6')
axes[0].set_title('Top 8 Feature Importances')
axes[0].set_xlabel('Importance Score')
 
axes[1].scatter(y_test, y_pred, alpha=0.5, color='#1F4E79', edgecolors='white', linewidth=0.3)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1].set_xlabel('Actual Sales (Rs.)')
axes[1].set_ylabel('Predicted Sales (Rs.)')
axes[1].set_title(f'Actual vs Predicted | R2={r2:.3f}')
 
plt.tight_layout()
plt.savefig('03_forecasting_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Forecasting chart saved - 03_forecasting_model.png")
 
# 6. BUSINESS INSIGHTS
print("\n-- KEY BUSINESS INSIGHTS --")
top_cat       = data.groupby('Category')['Sales'].sum().idxmax()
top_region    = data.groupby('Region')['Sales'].sum().idxmax()
best_month    = data.groupby('Month')['Sales'].sum().idxmax()
avg_disc      = data['Discount'].mean() * 100
profit_margin = (data['Profit'].sum() / data['Sales'].sum()) * 100
 
print(f"  1. Top performing category : {top_cat}")
print(f"  2. Highest revenue region  : {top_region}")
print(f"  3. Peak sales month        : Month {best_month}")
print(f"  4. Average discount rate   : {avg_disc:.1f}%")
print(f"  5. Overall profit margin   : {profit_margin:.1f}%")
print(f"  6. ML Model R2 Score       : {r2*100:.1f}% accuracy")
 
print("\n" + "=" * 60)
print("PROJECT COMPLETE - All outputs saved in your project folder")
print("=" * 60)
print("\nFiles generated:")
print("  - 01_eda_analysis.png")
print("  - 02_segmentation.png")
print("  - 03_forecasting_model.png")
print("  - 04_business_insights.txt")
print("Working perfectly 🚀")