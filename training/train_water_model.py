# train_water_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
<<<<<<< HEAD
import pickle
=======
import joblib
>>>>>>> origin/main
import os

# ------------------- Base Directories --------------------
BASE_DIR = os.path.dirname(__file__)  # Current project folder
DATA_PATH = os.path.join(BASE_DIR, "dt", "water_requirement.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
<<<<<<< HEAD

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ Water dataset loaded!")

=======

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ Water dataset loaded successfully!")

>>>>>>> origin/main
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
<<<<<<< HEAD
model = RandomForestRegressor(random_state=42)
=======
model = RandomForestRegressor(random_state=42, n_estimators=100)
>>>>>>> origin/main
model.fit(X, y)
print("✅ Water model trained successfully!")

# ------------------- Save Model & Encoders --------------------
<<<<<<< HEAD
with open(os.path.join(MODEL_DIR, "water_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "water_crop_encoder.pkl"), "wb") as f:
    pickle.dump(crop_encoder, f)

with open(os.path.join(MODEL_DIR, "water_soil_encoder.pkl"), "wb") as f:
    pickle.dump(soil_encoder, f)

with open(os.path.join(MODEL_DIR, "stage_encoder.pkl"), "wb") as f:
    pickle.dump(stage_encoder, f)
=======
joblib.dump(model, os.path.join(MODEL_DIR, "water_model.pkl"))
joblib.dump(crop_encoder, os.path.join(MODEL_DIR, "water_crop_encoder.pkl"))
joblib.dump(soil_encoder, os.path.join(MODEL_DIR, "water_soil_encoder.pkl"))
joblib.dump(stage_encoder, os.path.join(MODEL_DIR, "water_stage_encoder.pkl"))
>>>>>>> origin/main

print("✅ Water model and encoders saved successfully!")
