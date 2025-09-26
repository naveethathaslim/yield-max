# test_models.py
import pickle
import pandas as pd

# Load models and encoders (for crop)
crop_model = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/crop_model.pkl", "rb"))
soil_encoder = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/soil_encoder.pkl", "rb"))
crop_encoder = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/crop_encoder.pkl", "rb"))

# Load sample data
sample = pd.read_csv("dt/crop_recommendation.csv").iloc[0]

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

# Optional: test fertilizer predictor
from predictors.crop_predictor import predict_crop
from predictors.fertilizer_predictor import predict_fertilizer
from predictors.water_predictor import predict_water
from predictors.insurance_predictor import predict_insurance
print("=== Testing Yield Max Models ===\n")

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