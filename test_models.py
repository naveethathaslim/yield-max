# test_models.py
import pickle
import pandas as pd
import os

# 🔹 Set project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "dt")

# ------------------- Load Crop Model & Encoders -------------------
with open(os.path.join(MODEL_DIR, "crop_model.pkl"), "rb") as f:
    crop_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "soil_encoder.pkl"), "rb") as f:
    soil_encoder = pickle.load(f)
with open(os.path.join(MODEL_DIR, "crop_encoder.pkl"), "rb") as f:
    crop_encoder = pickle.load(f)

# Load sample data
sample = pd.read_csv(os.path.join(DATA_DIR, "crop_recommendation.csv")).iloc[0]

# Encode soil
soil_encoded = soil_encoder.transform([sample['Soil_Type']])[0]

# Prepare input
input_data = pd.DataFrame([[soil_encoded,
                            sample['Nitrogen'],
                            sample['Phosphorus'],
                            sample['Potassium'],
                            sample['Rainfall'],
                            sample['pH'],
                            sample['Temperature']]],
                          columns=['Soil_Type', 'Nitrogen', 'Phosphorus',
                                   'Potassium', 'Rainfall', 'pH', 'Temperature'])

# Predict
prediction = crop_model.predict(input_data)
crop_name = crop_encoder.inverse_transform([prediction[0]])[0]
print("✅ Predicted Crop:", crop_name)

# ------------------- Import other predictors -------------------
from predictors.crop_predictor import predict_crop
from predictors.fertilizer_predictor import predict_fertilizer
from predictors.water_predictor import predict_water
from predictors.insurance_predictor import predict_insurance

print("\n=== Testing Yield Max Models ===\n")

# --- Crop ---
try:
    crop_result = predict_crop("Clay", 50, 20, 30, 100, 6.5, 28)
    print(f"✅ Crop Prediction: {crop_result}")
except Exception as e:
    print(f"❌ Crop Prediction Error: {e}")

# --- Fertilizer ---
try:
    fert_result = predict_fertilizer("Rice", "Clay", "Seedling", "Nitrogen")
    print(f"✅ Fertilizer Recommendation: {fert_result}")
except Exception as e:
    print(f"❌ Fertilizer Prediction Error: {e}")

# --- Water ---
try:
    water_result = predict_water("Rice", "Clay", "Seedling")
    print(f"✅ Water Requirement: {water_result}")
except Exception as e:
    print(f"❌ Water Prediction Error: {e}")

# --- Insurance ---
try:
    ins_result = predict_insurance("Rice", "Clay")
    print(f"✅ Insurance Risk Level: {ins_result}")
except Exception as e:
    print(f"❌ Insurance Prediction Error: {e}")

print("\n=== Test Complete ===")
