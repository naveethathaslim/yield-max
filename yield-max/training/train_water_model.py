import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("dt/water_requirement.csv")
print("✅ Water dataset loaded!")

# Encode categorical columns
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
stage_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
df["Crop Stage"] = stage_encoder.fit_transform(df["Crop Stage"])

# Features & target
X = df[["Crop", "Soil Type", "Crop Stage"]]
y = df["Water Requirement (in mm)"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model and encoders
with open("water_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("water_crop_encoder.pkl", "wb") as f:
    pickle.dump(crop_encoder, f)

with open("water_soil_encoder.pkl", "wb") as f:
    pickle.dump(soil_encoder, f)

with open("stage_encoder.pkl", "wb") as f:
    pickle.dump(stage_encoder, f)

print("✅ Water model and encoders saved!")

