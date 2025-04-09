from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load model, scaler, and encoders
model = joblib.load('xgboost_modelNew1.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Smart Donor Prediction API. Use /predict to get predictions."}



# Request model
class SmartDonorApp(BaseModel):
    bloodGroup: str
    occupation: str
    convTime: str
    timesDonated: str
    reaction: str
    encouragement: str
    convLocality: str
    daysSinceLastDonation: int
    prefferedFreq: int
    rateService: int
    yob: int

@app.post("/predict")
async def predict_score(features: List[SmartDonorApp]):
    input_data = pd.DataFrame([item.dict() for item in features])

    # Ensure all expected columns exist
    expected_columns = ['bloodGroup', 'occupation', 'convTime', 'timesDonated', 
                        'reaction', 'encouragement', 'convLocality', 'daysSinceLastDonation', 
                        'prefferedFreq', 'rateService', 'yob']
    
    missing_cols = [col for col in expected_columns if col not in input_data.columns]
    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}"}

    # Encode categorical columns using label encoders
    categorical_cols = ['bloodGroup', 'occupation', 'convTime', 'reaction', 'timesDonated', 'encouragement', 'convLocality']
    for col in categorical_cols:
        le = label_encoders[col]  # Load corresponding encoder
        input_data[col] = le.transform(input_data[col])

    # Scale numeric values using the saved scaler
    numeric_cols = ['daysSinceLastDonation', 'prefferedFreq', 'rateService']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Pass the correctly formatted data to the model
    predictions = model.predict(input_data)

    # Return predictions as a list
    return {"predicted_scores": predictions.tolist()}
