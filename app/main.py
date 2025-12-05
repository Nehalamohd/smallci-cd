# app/main.py
from fastapi import FastAPI
import pickle
from pydantic import BaseModel

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class IrisFeatures(BaseModel):
    features: list

@app.post("/predict")
def predict(data: IrisFeatures):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
