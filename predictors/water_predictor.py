# predictors/water_predictor.py
import pickle
import pandas as pd
import os

# ------------------- Paths --------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ------------------- Helper --------------------
def load_pickle(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------- Load model & encoders --------------------
water_model = load_pickle("water_model.pkl")
crop_encoder = load_pickle("water_crop_encoder.pkl")
soil_encoder = load_pickle("water_soil_encoder.pkl")
stage_encoder = load_pickle("stage_encoder.pkl")

# Allowed values for input validation
allowed_crops = crop_encoder.classes_.tolist()
allowed_soils = soil_encoder.classes_.tolist()
allowed_stages = stage_encoder.classes_.tolist()

# ------------------- Prediction function --------------------
def predict_water(crop, soil_type, crop_stage, area=1.0):
    # Validate inputs
    if crop not in allowed_crops:
        raise ValueError(f"❌ Invalid Crop: {crop}. Allowed: {allowed_crops}")
    if soil_type not in allowed_soils:
        raise ValueError(f"❌ Invalid Soil_Type: {soil_type}. Allowed: {allowed_soils}")
    if crop_stage not in allowed_stages:
        raise ValueError(f"❌ Invalid Crop_Stage: {crop_stage}. Allowed: {allowed_stages}")

    # Encode inputs
    crop_enc = crop_encoder.transform([crop])[0]
    soil_enc = soil_encoder.transform([soil_type])[0]
    stage_enc = stage_encoder.transform([crop_stage])[0]

    # Predict water requirement
    input_df = pd.DataFrame([[crop_enc, soil_enc, stage_enc]],
                            columns=["Crop", "Soil Type", "Crop Stage"])
    prediction = water_model.predict(input_df)[0]

    # Multiply by area if needed
    total_water = prediction * area
    return total_water

# ------------------- Example --------------------
if __name__ == "__main__":
    water_needed = predict_water("Rice", "Clay", "Seedling", area=2.5)
    print(f"💧 Water Required for 2.5 acres: {water_needed:.2f} mm")
