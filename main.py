import os
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import shap

app = FastAPI(title="Baby Growth Prediction API")

# 1. LOAD ASSETS WITH COMPATIBILITY FIX
# 'compile=False' prevents the mae-deserialization error on Render
model = tf.keras.models.load_model('baby_model.h5', compile=False)
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

input_features = [
    'Kcal_Burned_Asleep', 'Kcal_Burned_Awake', 'Sleep_Hours', 'Awake_Hours',
    'F-Breast/Formula', 'F-Solid Meal', 'F-Nutritious Snacks',
    'Kcal_Milk', 'Kcal_Meals', 'Kcal_Snacks',
    'Age in Days', 'Gender_F', 'Gender_M',
    'Starting Height (cm)', 'Starting Weight (kg)'
]

# 2. DATA MODELS FOR API
class DailyLog(BaseModel):
    # Logs must be a list of 7 days, each day having 15 float values
    logs: List[List[float]] 

@app.get("/")
def health_check():
    return {"status": "online", "model": "LSTM-V3-Medical"}

@app.post("/predict")
async def predict_growth(data: DailyLog):
    try:
        # Convert input to numpy array
        current_window = np.array(data.logs) # Shape (7, 15)
        
        if current_window.shape != (7, 15):
            raise ValueError(f"Expected shape (7, 15), got {current_window.shape}")

        forecast_results = {}
        temp_window = current_window.copy()
        
        # Recursive Loop for 7-Day Forecast
        acc_weight = 0
        acc_height = 0
        
        for day in range(1, 8):
            # Scale and Reshape for LSTM (Batch, Timesteps, Features)
            scaled_input = scaler_X.transform(temp_window)
            scaled_input = scaled_input.reshape(1, 7, 15)
            
            # Predict
            pred_scaled = model.predict(scaled_input, verbose=0)
            pred_real = scaler_y.inverse_transform(pred_scaled)
            
            w_gain = float(pred_real[0][0]) # in grams (per our update)
            h_gain = float(pred_real[0][1]) # in cm
            
            acc_weight += w_gain
            acc_height += h_gain
            
            # Capture specific milestones
            if day in [1, 3, 7]:
                forecast_results[f"day_{day}"] = {
                    "weight_gain_g": round(acc_weight, 2),
                    "height_gain_cm": round(acc_height, 3)
                }
            
            # UPDATE WINDOW FOR RECURSION
            # 1. Slide window up
            new_row = temp_window[-1].copy()
            # 2. Update weight/height/age for the 'simulated' next day
            new_row[14] += (w_gain / 1000) # Starting Weight (kg) - Index 14
            new_row[13] += h_gain           # Starting Height (cm) - Index 13
            new_row[10] += 1                # Age in Days - Index 10
            
            temp_window = np.vstack([temp_window[1:], new_row])

        return {
            "prediction": forecast_results,
            "unit_weight": "grams",
            "unit_height": "cm"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
