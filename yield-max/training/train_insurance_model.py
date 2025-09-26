import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("dt/insurance_prediction.csv")

print("✅ CSV loaded successfully!")

# Encode Crop and Soil_Type separately
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

df['Crop'] = crop_encoder.fit_transform(df['Crop'])
df['Soil_Type'] = soil_encoder.fit_transform(df['Soil_Type'])

# Features and target
X = df[['Crop', 'Soil_Type']]
y = df['Risk_Level']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and both encoders
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("crop_encoder.pkl", "wb") as f:
    pickle.dump(crop_encoder, f)

with open("soil_encoder.pkl", "wb") as f:
    pickle.dump(soil_encoder, f)

print("✅ Model trained and saved successfully for Risk_Level prediction!")

