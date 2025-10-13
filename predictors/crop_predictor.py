# predictors/crop_predictor.py
import pickle
import pandas as pd
import os

# 🔹 Correct project root (one level above this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ------------------- Load Model & Encoders --------------------
def load_pickle(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    with open(path, "rb") as f:
        return pickle.load(f)

crop_model = load_pickle("crop_model.pkl")
soil_encoder = load_pickle("soil_encoder.pkl")
crop_encoder = load_pickle("crop_encoder.pkl")

# ------------------- Crop Prediction Function --------------------
def predict_crop(soil, nitrogen, phosphorus, potassium, rainfall, ph, temperature):
    # Encode soil
    try:
        soil_encoded = soil_encoder.transform([soil])[0]
    except ValueError:
        raise ValueError(f"❌ Soil '{soil}' not found in encoder classes: {soil_encoder.classes_}")

    # Prepare input DataFrame
    input_data = pd.DataFrame(
        [[soil_encoded, nitrogen, phosphorus, potassium, rainfall, ph, temperature]],
        columns=['Soil_Type', 'Nitrogen', 'Phosphorus', 'Potassium', 'Rainfall', 'pH', 'Temperature']
    )

    # Predict
    try:
        prediction = crop_model.predict(input_data)[0]
        crop_name = crop_encoder.inverse_transform([prediction])[0]
        return crop_name
    except AttributeError:
        raise AttributeError("❌ crop_model.pkl is not a trained model. Re-save it using pickle.dump(model, file).")

# ------------------- Example Usage --------------------
if __name__ == "__main__":
    result = predict_crop("Clay", 50, 30, 20, 100, 6.5, 30)
    print("✅ Predicted Crop:", result)
