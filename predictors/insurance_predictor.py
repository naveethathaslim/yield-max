import joblib
import pandas as pd
import os

# ------------------- Project root & model folder --------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ------------------- Helper to load models --------------------
def load_model(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    return joblib.load(path)

# ------------------- Load Model & Encoders --------------------
insurance_model = load_model("insurance_model.pkl")
crop_encoder = load_model("crop_encoder.pkl")
soil_encoder = load_model("soil_encoder.pkl")

# Allowed values for input validation
allowed_crops = crop_encoder.classes_.tolist()
allowed_soils = soil_encoder.classes_.tolist()

# ------------------- Prediction Function --------------------
def predict_insurance(crop, soil):
    if crop not in allowed_crops or soil not in allowed_soils:
        raise ValueError(f"❌ Invalid input. Allowed crops: {allowed_crops}, allowed soils: {allowed_soils}")
    
    encoded_crop = crop_encoder.transform([crop])[0]
    encoded_soil = soil_encoder.transform([soil])[0]

    X = pd.DataFrame([[encoded_crop, encoded_soil]], columns=['Crop', 'Soil_Type'])
    prediction = insurance_model.predict(X)[0]
    return prediction

# ------------------- Example Usage --------------------
if __name__ == "__main__":
    result = predict_insurance("Wheat", "Clay")  # Must match allowed classes
    print(f"🔍 Predicted Risk Level: {result}")
