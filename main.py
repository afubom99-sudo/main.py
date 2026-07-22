from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Make sure this exact variable name is 'app'
app = FastAPI()

try:
    model = joblib.load("model.pkl")
except Exception:
    model = None

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData):
    if not model:
        return {"error": "model.pkl file not found"}
    
    output = model.predict([data.text])[0]
    return {"prediction": str(output)}
