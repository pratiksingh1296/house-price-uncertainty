# Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read CSV
df = pd.read_csv('C:/Users/Pratik/DS/house_prediction_uncertainty/data/processed/house_prices_clean.csv')
df_2 = df.copy()

cat_cols = df_2.select_dtypes(include=['object']).columns

# Categorical columns which are ordered , we will use ordinal encoding
quality_map = {
    'None': 0,
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1
}
ordinal_cols = [
    'ExterQual', 'ExterCond','KitchenQual','GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'FireplaceQu', 'HeatingQC', 'PoolQC'
]

for col in ordinal_cols:
    if col in df_2.columns:
        df_2[col] = df_2[col].map(quality_map)

# Label encoding for other categorical columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:
    if col not in ordinal_cols:
        df_2[col] = le.fit_transform(df_2[col])

# We have TotalBsmtSF , 1stFlrSF , 2ndFlrSF which are related to area of house , we can create a new feature TotalSF
df_2['TotalSF'] = df_2['TotalBsmtSF'] + df_2['1stFlrSF'] + df_2['2ndFlrSF']

# House Age feature
df_2['HouseAge'] = df_2['YrSold'] - df_2['YearBuilt']
df_2['RemodAge'] = df_2['YrSold'] - df_2['YearRemodAdd']

# Full bath and Half bath total
df_2['TotalBathrooms'] = df_2['FullBath'] + (0.5 * df_2['HalfBath']) + df_2['BsmtFullBath'] + (0.5 * df_2['BsmtHalfBath'])

df_2.to_csv('C:/Users/Pratik/DS/house_prediction_uncertainty/data/processed/house_prices_features_v1.csv', index=False)

