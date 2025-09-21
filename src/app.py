from fastapi import FastAPI
import pandas as pd
import joblib


#load the model 
model = joblib.load("models/final_model.pkl")   

#create FASTAPI app instance
app = FastAPI(title="House Price Prediction API")

@app.get("/")
def home():
    return {"message : Welcome to the House Price Prediction API"}


@app.post("/predict")
def pridict(data:dict):
    #covert input data to dataframe
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": float(prediction[0])}