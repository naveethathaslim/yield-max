# train_insurance_model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
<<<<<<< HEAD
import pickle
=======
import joblib
>>>>>>> origin/main
import os

# ------------------- Base Directories --------------------
BASE_DIR = os.path.dirname(__file__)  # Current project folder
DATA_PATH = os.path.join(BASE_DIR, "dt", "insurance_prediction.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)
<<<<<<< HEAD

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ CSV loaded successfully!")

=======

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ Insurance dataset loaded successfully!")

>>>>>>> origin/main
# ------------------- Encode Labels --------------------
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil_Type"] = soil_encoder.fit_transform(df["Soil_Type"])

# ------------------- Features and Target --------------------
<<<<<<< HEAD
X = df[['Crop', 'Soil_Type']]
y = df['Risk_Level']

# ------------------- Train Model --------------------
model = RandomForestClassifier(random_state=42)
=======
X = df[["Crop", "Soil_Type"]]
y = df["Risk_Level"]

# ------------------- Train Model --------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
>>>>>>> origin/main
model.fit(X, y)
print("✅ Insurance model trained successfully!")

<<<<<<< HEAD
# ------------------- Save Model & Encoders --------------------
with open(os.path.join(MODEL_DIR, "insurance_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "crop_encoder.pkl"), "wb") as f:
    pickle.dump(crop_encoder, f)

with open(os.path.join(MODEL_DIR, "soil_encoder.pkl"), "wb") as f:
    pickle.dump(soil_encoder, f)

print("✅ Model trained and saved successfully for Risk_Level prediction!")
=======
# ------------------- Save Model & Encoders (with joblib) --------------------
joblib.dump(model, os.path.join(MODEL_DIR, "insurance_model.pkl"))
joblib.dump(crop_encoder, os.path.join(MODEL_DIR, "ins_crop_encoder.pkl"))
joblib.dump(soil_encoder, os.path.join(MODEL_DIR, "ins_soil_encoder.pkl"))

print("✅ All insurance models and encoders saved successfully!")
>>>>>>> origin/main
