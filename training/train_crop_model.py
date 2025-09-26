# train_crop_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("C:/Users/thasl/OneDrive/Desktop/yield-max/dt/crop_recommendation.csv")

# Encode
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

df["Soil_Type"] = soil_encoder.fit_transform(df["Soil_Type"])
df["Crop"] = crop_encoder.fit_transform(df["Crop"])

# Features and target
X = df[["Soil_Type", "Nitrogen", "Phosphorus", "Potassium", "Rainfall", "pH", "Temperature"]]
y = df["Crop"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model & encoders
pickle.dump(model, open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/crop_model.pkl", "wb"))
pickle.dump(soil_encoder, open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/soil_encoder.pkl", "wb"))
pickle.dump(crop_encoder, open("C:/Users/thasl/OneDrive/Desktop/yield-max/models/crop_encoder.pkl", "wb"))

print("✅ Crop model and encoders saved!")
