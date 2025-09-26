import pickle
import pandas as pd
import os

# 🔹 Correct project root from predictors folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and encoders
with open(os.path.join(PROJECT_ROOT, "models/crop_model.pkl"), "rb") as f:
    crop_model = pickle.load(f)

with open(os.path.join(PROJECT_ROOT, "models/soil_encoder.pkl"), "rb") as f:
    soil_encoder = pickle.load(f)

with open(os.path.join(PROJECT_ROOT, "models/crop_encoder.pkl"), "rb") as f:
    crop_encoder = pickle.load(f)

def predict_crop(soil, nitrogen, phosphorus, potassium, rainfall, ph, temperature):
    soil_encoded = soil_encoder.transform([soil])[0]
    input_data = pd.DataFrame([[soil_encoded, nitrogen, phosphorus, potassium, rainfall, ph, temperature]],
                              columns=['Soil_Type', 'Nitrogen', 'Phosphorus', 'Potassium', 'Rainfall', 'pH', 'Temperature'])
    
    # ✅ If crop_model is a numpy array, you cannot predict; it must be a proper trained model
    try:
        prediction = crop_model.predict(input_data)[0]
        crop_name = crop_encoder.inverse_transform([prediction])[0]
        return crop_name
    except AttributeError:
        print("❌ crop_model.pkl is not a trained model. Re-save it correctly using pickle.dump(model, file).")
        return None

# Example
if __name__ == "__main__":
    result = predict_crop("Clay", 50, 30, 20, 100, 6.5, 30)
    print("✅ Predicted Crop:", result)

