# fertilizer_predictor.py
import pickle
import pandas as pd

# Load model and encoders
model = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fertilizer_model.pkl", "rb"))
crop_enc = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_crop_encoder.pkl", "rb"))
soil_enc = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_soil_encoder.pkl", "rb"))
stage_enc = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_stage_encoder.pkl", "rb"))
def_enc = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_def_encoder.pkl", "rb"))
fert_label_enc = pickle.load(open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_label_encoder.pkl", "rb"))

def predict_fertilizer(crop, soil_type, crop_stage, deficiency):
    # Encode inputs
    try:
        crop_encoded = crop_enc.transform([crop])[0]
        soil_encoded = soil_enc.transform([soil_type])[0]
        stage_encoded = stage_enc.transform([crop_stage])[0]
        def_encoded = def_enc.transform([deficiency])[0]
    except ValueError as e:
        print("❌ Invalid input. Check allowed values for Crop, Soil, Stage, or Deficiency.")
        return None

    # Prepare features
    features = pd.DataFrame([[crop_encoded, soil_encoded, stage_encoded, def_encoded]],
                            columns=["Crop", "Soil_Type", "Crop_Stage", "Nutrient_Deficiency"])

    # Predict
    prediction_encoded = model.predict(features)[0]
    prediction = fert_label_enc.inverse_transform([prediction_encoded])[0]
    return prediction

# Example test
if __name__ == "__main__":
    result = predict_fertilizer("Rice", "Clay", "Seedling", "Nitrogen")
    if result:
        print("🌱 Recommended Fertilizer:", result)

