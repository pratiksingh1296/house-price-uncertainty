import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb

df = pd.read_csv('C:/Users/Pratik/DS/house_prediction_uncertainty/data/processed/house_prices_features_v1.csv')

X = df.drop(columns=['SalePrice', 'SalePrice_log'])
y = df['SalePrice_log']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using linear regression for baseline model 
model_LR = LinearRegression()
model_LR.fit(X_train, y_train)
y_pred = model_LR.predict(X_test)

# Inverse transform the log predictions to get actual sale prices
y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

# XBoost Regressor with Quantile Regression

# Model for 10th percentile (quantile alpha = 0.1)
model_XGB_q10 = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.1,
    n_estimators=100,
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42
    )
model_XGB_q10.fit(X_train, y_train)
y_xgb_q10_pred = model_XGB_q10.predict(X_test)
y_xgb_q10_actual = np.expm1(y_xgb_q10_pred)

# Model for Median (quantile alpha = 0.5)
model_XGB_q50 = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.5,
    n_estimators=100,
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42
    )
model_XGB_q50.fit(X_train, y_train)
y_xgb_q50_pred = model_XGB_q50.predict(X_test)
y_xgb_q50_actual = np.expm1(y_xgb_q50_pred)

# Model for 90th percentile (quantile alpha = 0.9)
model_XGB_q90 = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.9,
    n_estimators=100,
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42
    )
model_XGB_q90.fit(X_train, y_train)
y_xgb_q90_pred = model_XGB_q90.predict(X_test)
y_xgb_q90_actual = np.expm1(y_xgb_q90_pred)

# Prediction Intervals
Lower_PI = y_xgb_q10_actual
Upper_PI = y_xgb_q90_actual

coverage = np.mean((y_test_actual >= Lower_PI) & (y_test_actual <= Upper_PI))
print(f'Prediction Interval Coverage: {coverage * 100:.2f}%')
# How often the true price lies within the predicted interval.

bins = pd.qcut(y_test_actual, q=4, labels=["Low", "Mid", "High", "Luxury"])
coverage_by_bin = pd.DataFrame({
    "price_bin": bins,
    "covered": (y_test_actual >= Lower_PI) & (y_test_actual <= Upper_PI)
}).groupby("price_bin")["covered"].mean()

print(coverage_by_bin,'\n')

# Calculated Interval Width
interval_width = Upper_PI - Lower_PI
print(f'Average Prediction Interval Width: {np.mean(interval_width):.2f}')

# Uncertainty vs Price Scatter Plot
plt.scatter(y_test_actual, Upper_PI - Lower_PI, alpha=0.4)
plt.xlabel("Actual Sale Price")
plt.ylabel("Prediction Interval Width")
plt.title("Uncertainty vs Price Level")
plt.show()
plt.savefig('C:/Users/Pratik/DS/house_prediction_uncertainty/reports/figures/Uncertainty_vs_price_plot.png')

# Create a results data frame
df_results = pd.DataFrame({
    'Actual_Price': y_test_actual,
    'PredictedPrice': y_xgb_q50_actual,
    'Lower_PI': Lower_PI,
    'Upper_PI': Upper_PI,
    'Interval_Width': interval_width
})
df_results['RiskBand'] = pd.qcut(df_results['Interval_Width'], q=[0 , 0.33, 0.66, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])
df_results['Relative_Uncertainty'] = df_results['Interval_Width'] / df_results['PredictedPrice'] * 100
print(df_results.head(),'\n')

# Find homes with similar predicted prices but different uncertainty levels
df_results['PriceBucket'] = pd.qcut(df_results['PredictedPrice'], q=5)

comparison = df_results.sort_values('Interval_Width').groupby('PriceBucket').head(2)

comparison[['PredictedPrice','Lower_PI','Upper_PI', 'Interval_Width', 'RiskBand']]

df_results = df_results.reset_index(drop=True)

# Errorbar plot
sample_idx = np.random.choice(len(df_results), 50, replace=False)
left_error = np.maximum(df_results["PredictedPrice"] - df_results["Lower_PI"], 0)
right_error = np.maximum(df_results["Upper_PI"] - df_results["PredictedPrice"], 0)

plt.figure(figsize=(8, 6))
plt.errorbar(
    df_results.iloc[sample_idx]["PredictedPrice"],
    range(len(sample_idx)),
    xerr=[left_error.iloc[sample_idx], right_error.iloc[sample_idx]],
    fmt='o',
    alpha=0.6
)
plt.xlabel("Predicted Sale Price")
plt.ylabel("Sample Homes")
plt.title("Prediction Intervals for Sample Homes")
plt.grid(True)
plt.show()
plt.savefig('C:/Users/Pratik/DS/house_prediction_uncertainty/reports/figures/prediction_interval_sample_homes.png')

# Example Predictions
example_preds = df_results.sample(5, random_state=42)

example_preds[[
    "PredictedPrice",
    "Lower_PI",
    "Upper_PI",
    "Relative_Uncertainty",
    "RiskBand"
]]
print(example_preds)
example_preds.to_csv('C:/Users/Pratik/DS/house_prediction_uncertainty/reports/summary _tables/example_predictions.csv')

# Lower-than-expected coverage suggests the model underestimates uncertainty, particularly for extreme price ranges
