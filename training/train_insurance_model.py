import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ------------------- Base Directories --------------------
BASE_DIR = os.path.dirname(__file__)  # Current project folder
DATA_PATH = os.path.join(BASE_DIR, "dt", "insurance_prediction.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ Insurance dataset loaded successfully!")

# ------------------- Encode Labels --------------------
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil_Type"] = soil_encoder.fit_transform(df["Soil_Type"])

# ------------------- Features and Target --------------------
X = df[["Crop", "Soil_Type"]]
y = df["Risk_Level"]

# ------------------- Train Model --------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("✅ Insurance model trained successfully!")

# ------------------- Save Model & Encoders (with joblib) --------------------
joblib.dump(model, os.path.join(MODEL_DIR, "insurance_model.pkl"))
joblib.dump(crop_encoder, os.path.join(MODEL_DIR, "ins_crop_encoder.pkl"))
joblib.dump(soil_encoder, os.path.join(MODEL_DIR, "ins_soil_encoder.pkl"))

print("✅ All insurance models and encoders saved successfully!")
