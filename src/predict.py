import pandas as pd
import joblib

# Load model 

model = joblib.load("models/final_model.pkl")

# Make predictions

new_data = pd.DataFrame([
    {"MedInc": 8.3, "HouseAge": 80, "AveRooms": 10, "AveBedrms": 1.0,
     "Population": 322, "AveOccup": 2.5, "Latitude": 37.8, "Longitude": -122.4}
])

#predictions 
predictions = model.predict(new_data)
print(f" predicted median house values: {predictions[0]:.2f}")

 