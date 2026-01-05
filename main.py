from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import shap
from typing import List

app = FastAPI()

# 1. LOAD ASSETS
model = tf.keras.models.load_model('baby_model.h5')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Define features exactly as per your training
input_features = [
    'Kcal_Burned_Asleep', 'Kcal_Burned_Awake', 'Sleep_Hours', 'Awake_Hours',
    'F-Breast/Formula', 'F-Solid Meal', 'F-Nutritious Snacks',
    'Kcal_Milk', 'Kcal_Meals', 'Kcal_Snacks',
    'Age in Days', 'Gender_F', 'Gender_M',
    'Starting Height (cm)', 'Starting Weight (kg)'
]

# 2. INITIALIZE SHAP
# Note: In production, you'd ideally use a pre-calculated background set
# For now, we assume a background slice was saved or passed
explainer = None 

def initialize_explainer(background_data):
    global explainer
    explainer = shap.DeepExplainer(model, background_data)

# 3. DATA MODELS FOR API
class DailyLog(BaseModel):
    # This matches the input_features list
    logs: List[List[float]] # Expected shape: 7 days x 15 features

@app.post("/predict")
async def predict_growth(data: DailyLog):
    try:
        # Convert input to numpy array
        current_window = np.array(data.logs) # Shape (7, 15)
        
        forecast_results = {}
        temp_window = current_window.copy()
        
        # Recursive Loop for 7 Days
        accumulated_weight_gain = 0
        accumulated_height_gain = 0
        
        for day in range(1, 8):
            # Scale and Reshape for LSTM (1, 7, 15)
            scaled_input = scaler_X.transform(temp_window)
            scaled_input = scaled_input.reshape(1, 7, 15)
            
            # Predict
            pred_scaled = model.predict(scaled_input, verbose=0)
            pred_real = scaler_y.inverse_transform(pred_scaled)
            
            w_gain = float(pred_real[0][0])
            h_gain = float(pred_real[0][1])
            
            accumulated_weight_gain += w_gain
            accumulated_height_gain += h_gain
            
            if day == 1:
                forecast_results["day_1"] = {"weight": w_gain, "height": h_gain}
            if day == 3:
                forecast_results["day_3"] = {"weight": accumulated_weight_gain, "height": accumulated_height_gain}
            if day == 7:
                forecast_results["day_7"] = {"weight": accumulated_weight_gain, "height": accumulated_height_gain}
            
            # SLIDE WINDOW: Update the features for the next recursive step
            # We update the 'Starting Weight' and 'Height' in the new row
            new_row = temp_window[-1].copy()
            new_row[-1] += (w_gain / 1000) # Weight (kg)
            new_row[-2] += h_gain         # Height (cm)
            new_row[10] += 1              # Increment Age in Days
            
            temp_window = np.vstack([temp_window[1:], new_row])

        # 4. SHAP EXPLANATION (For the most recent day)
        if explainer:
            scaled_explainer_input = scaler_X.transform(current_window).reshape(1, 7, 15)
            shap_vals = explainer.shap_values(scaled_explainer_input)
            # Impact of last day on weight gain
            last_day_impact = shap_vals[0][0, -1, :] 
            impact_dict = dict(zip(input_features, last_day_impact.tolist()))
            top_reason = sorted(impact_dict.items(), key=lambda x: x[1], reverse=True)[0]
            forecast_results["explanation"] = f"Top growth driver: {top_reason[0]}"

        return forecast_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)