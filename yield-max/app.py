import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import os

# ------------------- Debug: check current directory --------------------
print("Current working dir:", os.getcwd())
print("Files in folder:", os.listdir())

# ------------------- Load All Models & Encoders --------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Crop
crop_model = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/crop_model.pkl")
insurance_model = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/insurance_model.pkl")
soil_encoder = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/soil_encoder.pkl")
crop_encoder = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/crop_encoder.pkl")

# Water
water_model = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/water_model.pkl")
water_crop_encoder = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/water_crop_encoder.pkl")
water_soil_encoder = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/water_soil_encoder.pkl")
stage_encoder = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/stage_encoder.pkl")

# Fertilizer
fert_model = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fertilizer_model.pkl")
crop_enc = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_crop_encoder.pkl")
soil_enc = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_soil_encoder.pkl")
stage_enc = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_stage_encoder.pkl")
def_enc = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_def_encoder.pkl")
fert_label_enc = load_pickle("C:/Users/thasl/OneDrive/Desktop/yield-max/models/fert_label_encoder.pkl")

# ------------------- App UI Setup --------------------
st.set_page_config(page_title="Yield Max 🌾", layout="wide")

st.markdown("""
<style>
.main { background-color: #f6fff5; }
.block-container { padding: 2rem 3rem; }
h1, h2, h3 { color: #2e7d32; }
.stButton>button { background-color: #43a047; color: white; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Yield Max: Smart Farming Assistant")
st.markdown("🚀 Empowering farmers with AI for smarter **crop, fertilizer, water**, and **insurance** decisions.")
st.markdown("---")

# ------------------- Tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Crop", "💊 Fertilizer", "💧 Water", "🛡️ Insurance"])

# ------------------- Crop Tab --------------------
with tab1:
    st.subheader("📈 Crop Recommendation")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            soil = st.selectbox("🌱 Soil Type", soil_encoder.classes_, key="crop_soil")
            nitrogen = st.number_input("Nitrogen (N)", min_value=0)
            phosphorus = st.number_input("Phosphorus (P)", min_value=0)
            potassium = st.number_input("Potassium (K)", min_value=0)
        with col2:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
            ph = st.number_input("pH Value", min_value=0.0, max_value=14.0)
            temperature = st.number_input("Temperature (°C)", min_value=-10.0)

    if st.button("🔍 Predict Best Crop", key="predict_crop"):
        soil_encoded = soil_encoder.transform([soil])[0]
        input_data = pd.DataFrame([[soil_encoded, nitrogen, phosphorus, potassium, rainfall, ph, temperature]],
                                  columns=['Soil_Type', 'Nitrogen', 'Phosphorus', 'Potassium', 'Rainfall', 'pH', 'Temperature'])
        prediction = crop_model.predict(input_data)[0]
        crop_name = crop_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Recommended Crop: **{crop_name}**")

# ------------------- Fertilizer Tab --------------------
with tab2:
    st.subheader("💊 Fertilizer Recommendation")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            crop = st.selectbox("🌿 Crop", crop_enc.classes_, key="fert_crop")
            soil = st.selectbox("🌱 Soil Type", soil_enc.classes_, key="fert_soil")
        with col2:
            stage = st.selectbox("📅 Crop Stage", stage_enc.classes_, key="fert_stage")
            deficiency = st.selectbox("⚠️ Nutrient Deficiency", def_enc.classes_, key="fert_def")

    if st.button("🧪 Get Fertilizer Recommendation", key="predict_fert"):
        input_data = pd.DataFrame([[crop, soil, stage, deficiency]],
                                  columns=["Crop", "Soil_Type", "Crop_Stage", "Nutrient_Deficiency"])
        input_data["Crop"] = crop_enc.transform(input_data["Crop"])
        input_data["Soil_Type"] = soil_enc.transform(input_data["Soil_Type"])
        input_data["Crop_Stage"] = stage_enc.transform(input_data["Crop_Stage"])
        input_data["Nutrient_Deficiency"] = def_enc.transform(input_data["Nutrient_Deficiency"])

        prediction = fert_model.predict(input_data)[0]
        fert_name = fert_label_enc.inverse_transform([prediction])[0]
        st.success(f"🌱 Recommended Fertilizer: **{fert_name}**")

# ------------------- Water Tab --------------------
with tab3:
    st.subheader("💧 Water Requirement Estimator")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            crop = st.selectbox("🌾 Crop", water_crop_encoder.classes_, key="water_crop")
            soil = st.selectbox("🌱 Soil Type", water_soil_encoder.classes_, key="water_soil")
        with col2:
            stage = st.selectbox("📅 Crop Stage", stage_encoder.classes_, key="water_stage")
            area = st.number_input("📐 Field Area (acres)", min_value=0.0, key="field_area")

    if st.button("💧 Estimate Water Need", key="predict_water"):
        crop_encoded = water_crop_encoder.transform([crop])[0]
        soil_encoded = water_soil_encoder.transform([soil])[0]
        stage_encoded = stage_encoder.transform([stage])[0]
        input_df = pd.DataFrame([[crop_encoded, soil_encoded, stage_encoded]],
                                columns=["Crop", "Soil Type", "Crop Stage"])
        prediction = water_model.predict(input_df)[0]
        total_water = prediction * area
        st.success(f"💧 Estimated Water Needed: **{total_water:.2f} mm** for {area} acres")

# ------------------- Insurance Tab --------------------
with tab4:
    st.subheader("🛡️ Insurance Risk Predictor")
    with st.container():
        crop = st.selectbox("🌾 Crop", crop_encoder.classes_, key="insurance_crop")
        soil = st.selectbox("🌱 Soil Type", soil_encoder.classes_, key="insurance_soil")

    if st.button("🧠 Predict Risk Level", key="predict_insurance"):
        encoded_crop = crop_encoder.transform([crop])[0]
        encoded_soil = soil_encoder.transform([soil])[0]
        input_df = pd.DataFrame([[encoded_crop, encoded_soil]], columns=["Crop", "Soil_Type"])
        prediction = insurance_model.predict(input_df)[0]
        st.success(f"📉 Predicted Insurance Risk Level: **{prediction}**")

# ------------------- Footer --------------------
st.markdown("---")
st.markdown("<center>Made with 💖 by <b>Naveetha Thaslim</b> | Yield Max © 2025</center>", unsafe_allow_html=True)
