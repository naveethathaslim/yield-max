# train_fertilizer_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("C:/Users/thasl/OneDrive/Desktop/yield-max/dt/fertilizer_recommendation.csv")
print("✅ Fertilizer dataset loaded!")

# Encode categorical columns
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

# Features and target
X = df.drop("Recommended_Fertilizer", axis=1)
y = df["Recommended_Fertilizer"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("fertilizer_model.pkl", "wb"))
pickle.dump(crop_encoder, open("fert_crop_encoder.pkl", "wb"))
pickle.dump(soil_encoder, open("fert_soil_encoder.pkl", "wb"))
pickle.dump(stage_encoder, open("fert_stage_encoder.pkl", "wb"))
pickle.dump(def_encoder, open("fert_def_encoder.pkl", "wb"))
pickle.dump(fert_encoder, open("fert_label_encoder.pkl", "wb"))

print("✅ Fertilizer model and encoders saved successfully!")
