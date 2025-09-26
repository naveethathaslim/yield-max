import pickle
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PROJECT_ROOT, "models/insurance_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(PROJECT_ROOT, "models/crop_encoder.pkl"), "rb") as f:
    crop_encoder = pickle.load(f)

with open(os.path.join(PROJECT_ROOT, "models/soil_encoder.pkl"), "rb") as f:
    soil_encoder = pickle.load(f)

allowed_crops = crop_encoder.classes_.tolist()
allowed_soils = soil_encoder.classes_.tolist()

def predict_insurance(crop, soil):
    if crop not in allowed_crops or soil not in allowed_soils:
        print("❌ Invalid input. Use allowed crops and soils")
        return None
    encoded_crop = crop_encoder.transform([crop])[0]
    encoded_soil = soil_encoder.transform([soil])[0]
    X = pd.DataFrame([[encoded_crop, encoded_soil]], columns=['Crop', 'Soil_Type'])
    prediction = model.predict(X)[0]
    print(f"🔍 Predicted Risk Level: {prediction}")
    return prediction

if __name__ == "__main__":
    predict_insurance("Wheat", "Clay")  # must match allowed classes


