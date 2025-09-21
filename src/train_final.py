import pandas as pd
from sklearn.model_selection import train_test_split
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


# XGBoost Regressor with Randomized Search


xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)

xgb_model.fit(X_train, y_train)

#evaluate the model

y_pred = xgb_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2= r2_score(y_test, y_pred)


# Print the results

print(f" final model train->RMSE:{rmse:.3f}, R2:{r2:.3f}")
          
#  Save the model
joblib.dump(xgb_model,"models/final_model.pkl")

print(" Final model saved to models/final_model.pkl")
