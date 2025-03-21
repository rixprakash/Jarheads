"""
03_feature_engineering.py

This script creates features for SPY price prediction based on historical
SPY and VIX data. It takes the cleaned data as input and generates derived
features that will be used for model training.

Features created include:
- Lagged values (previous n days) for both SPY and VIX
- Moving averages and standard deviations over various time windows
- Relative strength indicators
- Rate of change metrics
- Volatility measures
- Day of week, month indicators (calendar effects)

Author: JARHeads Team
Date: March 2025
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Create necessary directories if they don't exist
os.makedirs('DATA/processed', exist_ok=True)
os.makedirs('OUTPUT/figures', exist_ok=True)

# Load the cleaned data
print("Loading cleaned data...")
try:
    df = pd.read_csv('DATA/processed/clean_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Set date as index for time series operations
df = df.sort_values('Date').set_index('Date')

print("\nCreating features...")

# 1. Create lagged features
print("Creating lagged features...")
# Lag SPY prices
for lag in [1, 2, 3, 5, 10, 21]:  # 1, 2, 3 days, 1 week, 2 weeks, 1 month
    df[f'SPY_lag_{lag}'] = df['Price_SPY'].shift(lag)
    df[f'VIX_lag_{lag}'] = df['Price_VIX'].shift(lag)

# 2. Create moving averages and standard deviations
print("Creating moving averages and volatility measures...")
for window in [5, 10, 21, 63]:  # 1 week, 2 weeks, 1 month, 3 months
    # Moving averages
    df[f'SPY_MA_{window}'] = df['Price_SPY'].rolling(window=window).mean()
    df[f'VIX_MA_{window}'] = df['Price_VIX'].rolling(window=window).mean()
    
    # Standard deviations (volatility)
    df[f'SPY_STD_{window}'] = df['Price_SPY'].rolling(window=window).std()
    df[f'VIX_STD_{window}'] = df['Price_VIX'].rolling(window=window).std()
    
    # Moving average convergence/divergence
    if window == 5:  # Compare 5-day MA with 21-day MA
        df['SPY_MACD'] = df['SPY_MA_5'] - df['SPY_MA_21']
        df['VIX_MACD'] = df['VIX_MA_5'] - df['VIX_MA_21']

# 3. Price momentum (rate of change)
print("Creating momentum indicators...")
for period in [1, 5, 10, 21]:
    df[f'SPY_ROC_{period}'] = df['Price_SPY'].pct_change(periods=period)
    df[f'VIX_ROC_{period}'] = df['Price_VIX'].pct_change(periods=period)

# 4. Relative indicators
print("Creating relative indicators...")
# Relative strength
df['SPY_RS_1m'] = df['Price_SPY'] / df['SPY_MA_21']
df['VIX_RS_1m'] = df['Price_VIX'] / df['VIX_MA_21']

# VIX/SPY ratio
df['VIX_SPY_ratio'] = df['Price_VIX'] / df['Price_SPY']

# 5. Calendar effects
print("Adding calendar features...")
# Reset index to access Date
df = df.reset_index()
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

# One-hot encode day of week and month for better modeling
for day in range(5):  # 0-4 (weekdays only in trading data)
    df[f'DOW_{day}'] = (df['DayOfWeek'] == day).astype(int)

for month in range(1, 13):
    df[f'Month_{month}'] = (df['Month'] == month).astype(int)

# 6. Target variable creation
print("Creating target variables...")
# Next day return (primary target)
df['target_next_day_return'] = df['Price_SPY'].shift(-1) / df['Price_SPY'] - 1

# 5-day forward return
df['target_5d_return'] = df['Price_SPY'].shift(-5) / df['Price_SPY'] - 1

# Binary direction indicators
df['target_next_day_up'] = (df['target_next_day_return'] > 0).astype(int)
df['target_5d_up'] = (df['target_5d_return'] > 0).astype(int)

# 7. Create interaction terms
print("Creating interaction features...")
# VIX change × SPY change
df['VIX_SPY_interaction'] = df['Change %_VIX'] * df['Change %_SPY']

# 8. Clean up - drop rows with NaN values from lag creation
print("Cleaning up dataset...")
df_full = df.copy()  # Keep a copy with NaNs for reference
df = df.dropna()
print(f"Rows before NaN removal: {df_full.shape[0]}, after: {df.shape[0]}")

# 9. Feature correlation analysis
print("\nAnalyzing feature correlations...")
# Correlation with target
target_corr = df.corr()['target_next_day_return'].sort_values(ascending=False)
print("\nTop features correlated with next day return:")
print(target_corr.head(10))
print("\nBottom features correlated with next day return:")
print(target_corr.tail(10))

# Create correlation heatmap for top features
plt.figure(figsize=(12, 10))
top_features = list(target_corr.head(15).index) + list(target_corr.tail(5).index)
correlation_matrix = df[top_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation of Top Features with Target Variable')
plt.tight_layout()
plt.savefig('OUTPUT/figures/feature_correlation_heatmap.png')

# 10. Save the feature-engineered dataset
print("Saving feature-engineered dataset...")
df.to_csv('DATA/processed/feature_data.csv', index=False)
print(f"Feature engineering complete! Created {df.shape[1] - len(df_full.columns)} new features.")
print(f"Feature data saved to DATA/processed/feature_data.csv with {df.shape[0]} rows and {df.shape[1]} columns.")

# 11. Create a feature summary
feature_summary = pd.DataFrame({
    'Feature': df.columns,
    'Type': df.dtypes,
    'Missing': df.isnull().sum(),
    'Unique': [df[col].nunique() for col in df.columns],
    'Target_Correlation': [df[col].corr(df['target_next_day_return']) 
                           if df[col].dtype in ['float64', 'int64'] else None 
                           for col in df.columns]
})

print("\nFeature summary (first 10 rows):")
print(feature_summary.head(10))
feature_summary.to_csv('DATA/processed/feature_summary.csv', index=False)
print("Feature summary saved to DATA/processed/feature_summary.csv")