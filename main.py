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

# Request model
class SmartDonorApp(BaseModel):
    bloodGroup: str
    occupation: str
    yob: int
    timesDonated: str
    daysSinceLastDonation: int
    convTime: str
    prefferedFreq: int
    rateService: int
    reaction: str
    encouragement: str
    convLocality: str
   
   
   
   

@app.post("/predict")
async def predict_score(features: List[SmartDonorApp]):
    input_data = pd.DataFrame([item.dict() for item in features])

    # Ensure all expected columns exist
    expected_columns = ['bloodGroup', 'occupation', 'yob', 'timesDonated', 
                        'daysSinceLastDonation', 'convTime',  'prefferedFreq', 'rateService',
                        'reaction', 'encouragement', 'convLocality' 
                        ]
    
    missing_cols = [col for col in expected_columns if col not in input_data.columns]
    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}"}

    # Encode categorical columns using label encoders
    categorical_cols = ['bloodGroup', 'occupation', 'convTime', 'timesDonated',  'reaction', 'encouragement', 'convLocality']
    for col in categorical_cols:
        le = label_encoders[col]  # Load corresponding encoder
        input_data[col] = le.transform(input_data[col])

    # Scale numeric values using the saved scaler
    numeric_cols = ['daysSinceLastDonation', 'prefferedFreq', 'rateService', 'yob']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Pass the correctly formatted data to the model
    predictions = model.predict(input_data)

    # Return predictions as a list
    return {"predicted_scores": predictions.tolist()}
