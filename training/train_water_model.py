import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ------------------- Base Directories --------------------
BASE_DIR = os.path.dirname(__file__)  # Current project folder
DATA_PATH = os.path.join(BASE_DIR, "dt", "water_requirement.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ Water dataset loaded successfully!")

# ------------------- Encode Labels --------------------
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
stage_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
df["Crop Stage"] = stage_encoder.fit_transform(df["Crop Stage"])

# ------------------- Features & Target --------------------
X = df[["Crop", "Soil Type", "Crop Stage"]]
y = df["Water Requirement (in mm)"]

# ------------------- Train Model --------------------
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X, y)
print("✅ Water model trained successfully!")

# ------------------- Save Model & Encoders --------------------
joblib.dump(model, os.path.join(MODEL_DIR, "water_model.pkl"))
joblib.dump(crop_encoder, os.path.join(MODEL_DIR, "water_crop_encoder.pkl"))
joblib.dump(soil_encoder, os.path.join(MODEL_DIR, "water_soil_encoder.pkl"))
joblib.dump(stage_encoder, os.path.join(MODEL_DIR, "water_stage_encoder.pkl"))

print("✅ Water model and encoders saved successfully!")
