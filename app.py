from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Enable CORS to allow Next.js frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://famcare7.vercel.app"],  # Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '')
try:
    models = joblib.load(os.path.join(MODELS_DIR, 'models.pkl'))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.pkl'))
except FileNotFoundError:
    raise Exception("Error: Model files not found. Please ensure models are trained and saved.")

input_features = [
    'Age', 'BMI', 'Exercise_Level', 'Sleep_Hours', 'Stress_Level', 'Mental_Health_Priority',
    'Smoking_Status', 'Telehealth_Preference', 'Vegetarian', 'PCOD', 'PCOS', 'Heart_Disease',
    'Diabetes', 'High_BP', 'Low_BP', 'Hypertension', 'Migraine', 'Thyroid', 'Endometriosis',
    'Osteoporosis', 'Anemia', 'Depression'
]

# Define the input model using Pydantic
class HealthInput(BaseModel):
    Age: float
    BMI: float
    Exercise_Level: float
    Sleep_Hours: float
    Stress_Level: float
    Mental_Health_Priority: float
    Smoking_Status: int
    Telehealth_Preference: int
    Vegetarian: int
    PCOD: int
    PCOS: int
    Heart_Disease: int
    Diabetes: int
    High_BP: int
    Low_BP: int
    Hypertension: int
    Migraine: int
    Thyroid: int
    Endometriosis: int
    Osteoporosis: int
    Anemia: int
    Depression: int

def minutes_to_time(minutes):
    """Convert minutes since midnight to readable time format"""
    try:
        minutes = float(minutes)
        hours = int(minutes // 60) % 24
        mins = int(minutes % 60)
        period = "AM" if hours < 12 else "PM"
        hours = hours % 12 or 12
        return f"{hours:02d}:{mins:02d} {period}"
    except (ValueError, TypeError):
        return "Invalid time"

@app.post("/predict")
async def predict(data: HealthInput):
    try:
        # Convert input to dictionary and validate
        user_input = data.dict()
        for feature, value in user_input.items():
            if feature in ['Age', 'BMI', 'Sleep_Hours']:
                val = float(value)
                if feature == 'Age' and not (18 <= val <= 100):
                    raise HTTPException(status_code=400, detail='Age must be between 18 and 100')
                elif feature == 'BMI' and not (10 <= val <= 50):
                    raise HTTPException(status_code=400, detail='BMI must be between 10 and 50')
                elif feature == 'Sleep_Hours' and not (0 <= val <= 24):
                    raise HTTPException(status_code=400, detail='Sleep Hours must be between 0 and 24')
            elif feature in ['Exercise_Level', 'Stress_Level', 'Mental_Health_Priority']:
                val = float(value)
                if not (0 <= val <= 10):
                    raise HTTPException(status_code=400, detail=f'{feature} must be between 0 and 10')

        # Prepare input for model
        input_values = [user_input[feature] for feature in input_features]
        input_df = pd.DataFrame([dict(zip(input_features, input_values))])
        processed_input = preprocessor.transform(input_df)

        # Make predictions
        predictions = {}
        for column, model in models.items():
            prediction = model.predict(processed_input)[0]
            if column in encoders:
                predictions[column] = encoders[column].inverse_transform([int(prediction)])[0]
            else:
                if column in ['Wake_Up_Time', 'Sleep_Time']:
                    predictions[column] = minutes_to_time(prediction)
                else:
                    predictions[column] = f"{prediction:.1f}"

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)