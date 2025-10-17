import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
<<<<<<< HEAD
import pickle
=======
import joblib
>>>>>>> origin/main
import os

# ------------------- Base Directories --------------------
BASE_DIR = os.path.dirname(__file__)  # Current project folder
DATA_PATH = os.path.join(BASE_DIR, "dt", "fertilizer_recommendation.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
<<<<<<< HEAD

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ Fertilizer dataset loaded!")

=======

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- Load Dataset --------------------
df = pd.read_csv(DATA_PATH)
print("✅ Fertilizer dataset loaded successfully!")

>>>>>>> origin/main
# ------------------- Encode Labels --------------------
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
stage_encoder = LabelEncoder()
def_encoder = LabelEncoder()
fert_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil_Type"] = soil_encoder.fit_transform(df["Soil_Type"])
df["Crop_Stage"] = stage_encoder.fit_transform(df["Crop_Stage"])
df["Nutrient_Deficiency"] = def_encoder.fit_transform(df["Nutrient_Deficiency"])
df["Recommended_Fertilizer"] = fert_encoder.fit_transform(df["Recommended_Fertilizer"])

# ------------------- Features and Target --------------------
X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# ------------------- Train Model --------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("✅ Fertilizer model trained successfully!")

<<<<<<< HEAD
# ------------------- Save Model & Encoders --------------------
with open(os.path.join(MODEL_DIR, "fertilizer_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "fert_crop_encoder.pkl"), "wb") as f:
    pickle.dump(crop_encoder, f)

with open(os.path.join(MODEL_DIR, "fert_soil_encoder.pkl"), "wb") as f:
    pickle.dump(soil_encoder, f)

with open(os.path.join(MODEL_DIR, "fert_stage_encoder.pkl"), "wb") as f:
    pickle.dump(stage_encoder, f)

with open(os.path.join(MODEL_DIR, "fert_def_encoder.pkl"), "wb") as f:
    pickle.dump(def_encoder, f)

with open(os.path.join(MODEL_DIR, "fert_label_encoder.pkl"), "wb") as f:
    pickle.dump(fert_encoder, f)
=======
# ------------------- Save Model & Encoders (with joblib) --------------------
joblib.dump(model, os.path.join(MODEL_DIR, "fertilizer_model.pkl"))
joblib.dump(crop_encoder, os.path.join(MODEL_DIR, "fert_crop_encoder.pkl"))
joblib.dump(soil_encoder, os.path.join(MODEL_DIR, "fert_soil_encoder.pkl"))
joblib.dump(stage_encoder, os.path.join(MODEL_DIR, "fert_stage_encoder.pkl"))
joblib.dump(def_encoder, os.path.join(MODEL_DIR, "fert_def_encoder.pkl"))
joblib.dump(fert_encoder, os.path.join(MODEL_DIR, "fert_label_encoder.pkl"))
>>>>>>> origin/main

print("✅ All fertilizer models and encoders saved successfully!")
