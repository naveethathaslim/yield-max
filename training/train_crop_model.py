import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ------------------- Base Directories --------------------
BASE_DIR = os.path.dirname(__file__)  # Current project folder
DATA_PATH = os.path.join(BASE_DIR, "dt", "crop_recommendation.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully!")

# ------------------- Encode Labels --------------------
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

df["Soil_Type"] = soil_encoder.fit_transform(df["Soil_Type"])
df["Crop"] = crop_encoder.fit_transform(df["Crop"])

# ------------------- Features and Target --------------------
X = df[["Soil_Type", "Nitrogen", "Phosphorus", "Potassium", "Rainfall", "pH", "Temperature"]]
y = df["Crop"]

# ------------------- Train Model --------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ------------------- Save Model & Encoders --------------------
joblib.dump(model, os.path.join(MODEL_DIR, "crop_model.joblib"))
joblib.dump(soil_encoder, os.path.join(MODEL_DIR, "soil_encoder.joblib"))
joblib.dump(crop_encoder, os.path.join(MODEL_DIR, "crop_encoder.joblib"))

print("✅ Crop model and encoders saved successfully to /models folder!")
