<<<<<<< HEAD
# predictors/fertilizer_predictor.py
import pickle
=======
import joblib
>>>>>>> origin/main
import pandas as pd
import os

# 🔹 Project root & model folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ------------------- Load Model & Encoders --------------------
<<<<<<< HEAD
def load_pickle(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    with open(path, "rb") as f:
        return pickle.load(f)

fert_model = load_pickle("fertilizer_model.pkl")
crop_enc = load_pickle("fert_crop_encoder.pkl")
soil_enc = load_pickle("fert_soil_encoder.pkl")
stage_enc = load_pickle("fert_stage_encoder.pkl")
def_enc = load_pickle("fert_def_encoder.pkl")
fert_label_enc = load_pickle("fert_label_encoder.pkl")

# ------------------- Fertilizer Prediction Function --------------------
def predict_fertilizer(crop, soil_type, crop_stage, deficiency):
    try:
        crop_encoded = crop_enc.transform([crop])[0]
        soil_encoded = soil_enc.transform([soil_type])[0]
        stage_encoded = stage_enc.transform([crop_stage])[0]
        def_encoded = def_enc.transform([deficiency])[0]
    except ValueError as e:
        raise ValueError(f"❌ Invalid input: {e}. Check allowed values for Crop, Soil, Stage, or Deficiency.")

    features = pd.DataFrame([[crop_encoded, soil_encoded, stage_encoded, def_encoded]],
=======
def load_model(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    return joblib.load(path)

fert_model = load_model("fertilizer_model.pkl")
crop_enc = load_model("fert_crop_encoder.pkl")
soil_enc = load_model("fert_soil_encoder.pkl")
stage_enc = load_model("fert_stage_encoder.pkl")
def_enc = load_model("fert_def_encoder.pkl")
fert_label_enc = load_model("fert_label_encoder.pkl")

# ------------------- Fertilizer Prediction Function --------------------
def predict_fertilizer(crop, soil_type, crop_stage, deficiency):
    for val, enc, name in zip([crop, soil_type, crop_stage, deficiency],
                              [crop_enc, soil_enc, stage_enc, def_enc],
                              ["Crop", "Soil", "Stage", "Deficiency"]):
        if val not in enc.classes_:
            raise ValueError(f"❌ Invalid {name}: '{val}'. Allowed values: {list(enc.classes_)}")

    features = pd.DataFrame([[crop_enc.transform([crop])[0],
                              soil_enc.transform([soil_type])[0],
                              stage_enc.transform([crop_stage])[0],
                              def_enc.transform([deficiency])[0]]],
>>>>>>> origin/main
                            columns=["Crop", "Soil_Type", "Crop_Stage", "Nutrient_Deficiency"])
    
    prediction_encoded = fert_model.predict(features)[0]
    prediction = fert_label_enc.inverse_transform([prediction_encoded])[0]
    return prediction

# ------------------- Example Usage --------------------
if __name__ == "__main__":
    result = predict_fertilizer("Rice", "Clay", "Seedling", "Nitrogen")
    print("🌱 Recommended Fertilizer:", result)
