import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error , r2_score
import xgboost as xgb
import joblib
import os


# Load the dataset

df= pd.read_csv('data/california_housing.csv')

# Split the dataset into features and target variable
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



results = {}

# Linera Regression Baseline

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

results['Linear Regression'] = {"rmse": root_mean_squared_error(y_test, y_pred),
                                "r2": r2_score(y_test, y_pred),}

joblib.dump(lin_reg,"models/linear_regression_model.pkl")

# Random Forest Regressor with  Grid Search

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf,rf_params,cv=3,scoring='neg_mean_squared_error',n_jobs=-1,verbose=2)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
y_pred = best_rf.predict(X_test)
results["Random Forest"] = {"rmse": root_mean_squared_error(y_test, y_pred),
                            "r2": r2_score(y_test, y_pred),}

joblib.dump(best_rf,"models/random_forest_model.pkl")

# XGBoost Regressor with Randomized Search


xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

xgb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],}

xgb_random = RandomizedSearchCV(xgb_model, xgb_params, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
xgb_random.fit(X_train, y_train)

best_xgb = xgb_random.best_estimator_
y_pred = best_xgb.predict(X_test)
results["XGBoost"] = {"rmse": root_mean_squared_error(y_test, y_pred),
                      "r2": r2_score(y_test, y_pred),}
joblib.dump(best_xgb,"models/xgboost_model.pkl")


# Print the results

print("\n* Results summary:")

for model,metrics in results.items():
    print(f"{model}->RMSE:{metrics['rmse']:.3f}, R2:{metrics['r2']:.3f}")
          
#  Best model selection

best_model = min(results,key=lambda x:results[x]['rmse'])

print(f"\n best model : {best_model} -> {results[best_model]}")
