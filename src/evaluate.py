import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

df = pd.read_csv('C:/Users/Pratik/DS/house_prediction_uncertainty/data/processed/house_prices_features_v1.csv')

X = df.drop(columns=['SalePrice', 'SalePrice_log'])
y = df['SalePrice_log']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using linear regression for baseline model 
model_LR = LinearRegression()
model_LR.fit(X_train, y_train)
y_pred = model_LR.predict(X_test)

# Ridge 
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)
y_pred_ridge_actual = np.expm1(y_pred_ridge)

# XGB
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
y_pred_xgb_actual = np.expm1(y_pred_xgb)

# Inverse transform the log predictions to get actual sale prices
y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

# Calculate RMSE & MAE on actual sale prices
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
print(f'RMSE: {rmse}, MAE: {mae}\n')

# Evaluate Ridge model
rmse_ridge = np.sqrt(mean_squared_error(y_test_actual, y_pred_ridge_actual))
mae_ridge = mean_absolute_error(y_test_actual, y_pred_ridge_actual)
print(f'Ridge RMSE: {rmse_ridge}, Ridge MAE: {mae_ridge}\n')

# Evaluate XGBoost model
rmse_xgb = np.sqrt(mean_squared_error(y_test_actual, y_pred_xgb_actual))
mae_xgb = mean_absolute_error(y_test_actual, y_pred_xgb_actual)
print(f'XGBoost RMSE: {rmse_xgb}, XGBoost MAE: {mae_xgb}\n')

# Residual analysis
residuals = y_test - y_pred
# Residual vs Predicted plot
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted SalePrice')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted SalePrice')
plt.show()
plt.savefig('C:/Users/Pratik/DS/house_prediction_uncertainty/reports/figures/residual_vs_predicted_plot.png')

# Summary of model performances
model_summary = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'XGBoost Regressor'],
    'RMSE': [rmse, rmse_ridge, rmse_xgb],
    'MAE': [mae, mae_ridge, mae_xgb]
})
print(model_summary)
model_summary.to_csv('C:/Users/Pratik/DS/house_prediction_uncertainty/reports/summary_tables/model_summary.csv')

