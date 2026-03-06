import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/Pratik/DS/house_prediction_uncertainty/data/raw/train.csv') 
df_2 = df.copy()

#Log transform target variable
df_2['SalePrice_log'] = np.log1p(df_2['SalePrice']) 

# Fill missing values in numerical columns with 0, since they likely indicate absence of that feature
num_cols = df_2.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop(['Id', 'SalePrice', 'SalePrice_log'])
df_2[num_cols] = df_2[num_cols].fillna(0)

# Fill missing values in categorical columns with 'None', indicating absence of that feature
cat_cols = df_2.select_dtypes(include=['object']).columns
df_2[cat_cols] = df_2[cat_cols].fillna('None')

df.to_csv('C:/Users/Pratik/DS/house_prediction_uncertainty/data/processed/house_prices_clean.csv')