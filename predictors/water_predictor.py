import pickle
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PROJECT_ROOT, "models/water_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(PROJECT_ROOT, "models/water_label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

def predict_water(stage, soil_type, crop_stage):  # note: stage = 'Sowing', etc.
    input_data = {"Crop": stage, "Soil_Type": soil_type, "Crop_Stage": crop_stage}
    for col in input_data:
        if input_data[col] not in le.classes_:
            print(f"❌ Invalid value for {col}: {input_data[col]}")
            print(f"✅ Allowed values: {list(le.classes_)}")
            return None
    encoded = [le.transform([input_data[col]])[0] for col in ["Crop", "Soil_Type", "Crop_Stage"]]
    prediction = model.predict([encoded])[0]
    print(f"💧 Water Required: {prediction:.2f} mm")
    return prediction

if __name__ == "__main__":
    predict_water("Sowing", "Clay", "Seedling")

