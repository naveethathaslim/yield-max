# app.py
import streamlit as st
import pandas as pd
import os
import joblib

<<<<<<< HEAD
# ------------------- Helper Functions --------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------- Base Directory --------------------
BASE_DIR = os.path.dirname(__file__)  # Current project folder
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Models folder

# ------------------- Load All Models & Encoders --------------------
# Crop
crop_model = load_pickle(os.path.join(MODEL_DIR, "crop_model.pkl"))
insurance_model = load_pickle(os.path.join(MODEL_DIR, "insurance_model.pkl"))
soil_encoder = load_pickle(os.path.join(MODEL_DIR, "soil_encoder.pkl"))
crop_encoder = load_pickle(os.path.join(MODEL_DIR, "crop_encoder.pkl"))

# Water
water_model = load_pickle(os.path.join(MODEL_DIR, "water_model.pkl"))
water_crop_encoder = load_pickle(os.path.join(MODEL_DIR, "water_crop_encoder.pkl"))
water_soil_encoder = load_pickle(os.path.join(MODEL_DIR, "water_soil_encoder.pkl"))
stage_encoder = load_pickle(os.path.join(MODEL_DIR, "stage_encoder.pkl"))

# Fertilizer
fert_model = load_pickle(os.path.join(MODEL_DIR, "fertilizer_model.pkl"))
crop_enc = load_pickle(os.path.join(MODEL_DIR, "fert_crop_encoder.pkl"))
soil_enc = load_pickle(os.path.join(MODEL_DIR, "fert_soil_encoder.pkl"))
stage_enc = load_pickle(os.path.join(MODEL_DIR, "fert_stage_encoder.pkl"))
def_enc = load_pickle(os.path.join(MODEL_DIR, "fert_def_encoder.pkl"))
fert_label_enc = load_pickle(os.path.join(MODEL_DIR, "fert_label_encoder.pkl"))
=======
# ------------------- Helper --------------------
def load_joblib(file_name):
    return joblib.load(os.path.join(MODEL_DIR, file_name))

# ------------------- Base Directories --------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ------------------- Load Models & Encoders --------------------
# Crop
crop_model = load_joblib("crop_model.pkl")
crop_encoder = load_joblib("crop_encoder.pkl")
soil_encoder = load_joblib("soil_encoder.pkl")

# Fertilizer
fert_model = load_joblib("fertilizer_model.pkl")
fert_crop_enc = load_joblib("fert_crop_encoder.pkl")
fert_soil_enc = load_joblib("fert_soil_encoder.pkl")
fert_stage_enc = load_joblib("fert_stage_encoder.pkl")
fert_def_enc = load_joblib("fert_def_encoder.pkl")
fert_label_enc = load_joblib("fert_label_encoder.pkl")
>>>>>>> origin/main

# Water
water_model = load_joblib("water_model.pkl")
water_crop_enc = load_joblib("water_crop_encoder.pkl")
water_soil_enc = load_joblib("water_soil_encoder.pkl")
stage_enc = load_joblib("stage_encoder.pkl")

# Insurance
insurance_model = load_joblib("insurance_model.pkl")
ins_crop_enc = load_joblib("crop_encoder.pkl")
ins_soil_enc = load_joblib("soil_encoder.pkl")

# ------------------- App UI --------------------
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
st.markdown("🚀 AI-powered Crop, Fertilizer, Water & Insurance guidance for farmers")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Crop", "💊 Fertilizer", "💧 Water", "🛡️ Insurance"])

# ------------------- Crop Tab --------------------
with tab1:
    st.subheader("📈 Crop Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        soil = st.selectbox("🌱 Soil Type", soil_encoder.classes_)
        nitrogen = st.number_input("Nitrogen (N)", min_value=0)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0)
        potassium = st.number_input("Potassium (K)", min_value=0)
    with col2:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
        ph = st.number_input("pH Value", min_value=0.0, max_value=14.0)
        temperature = st.number_input("Temperature (°C)", min_value=-10.0)

    if st.button("🔍 Predict Best Crop"):
        soil_encoded = soil_encoder.transform([soil])[0]
        input_df = pd.DataFrame([[soil_encoded, nitrogen, phosphorus, potassium, rainfall, ph, temperature]],
                                columns=['Soil_Type', 'Nitrogen', 'Phosphorus', 'Potassium', 'Rainfall', 'pH', 'Temperature'])
        pred = crop_model.predict(input_df)[0]
        crop_name = crop_encoder.inverse_transform([pred])[0]
        st.success(f"✅ Recommended Crop: **{crop_name}**")

# ------------------- Fertilizer Tab --------------------
with tab2:
    st.subheader("💊 Fertilizer Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("🌿 Crop", fert_crop_enc.classes_)
        soil = st.selectbox("🌱 Soil Type", fert_soil_enc.classes_)
    with col2:
        stage = st.selectbox("📅 Crop Stage", fert_stage_enc.classes_)
        deficiency = st.selectbox("⚠️ Nutrient Deficiency", fert_def_enc.classes_)

    if st.button("🧪 Get Fertilizer Recommendation"):
        input_df = pd.DataFrame([[crop, soil, stage, deficiency]],
                                columns=["Crop", "Soil_Type", "Crop_Stage", "Nutrient_Deficiency"])
        input_df["Crop"] = fert_crop_enc.transform(input_df["Crop"])
        input_df["Soil_Type"] = fert_soil_enc.transform(input_df["Soil_Type"])
        input_df["Crop_Stage"] = fert_stage_enc.transform(input_df["Crop_Stage"])
        input_df["Nutrient_Deficiency"] = fert_def_enc.transform(input_df["Nutrient_Deficiency"])

        pred = fert_model.predict(input_df)[0]
        fert_name = fert_label_enc.inverse_transform([pred])[0]
        st.success(f"🌱 Recommended Fertilizer: **{fert_name}**")

# ------------------- Water Tab --------------------
with tab3:
    st.subheader("💧 Water Requirement Estimator")
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("🌾 Crop", water_crop_enc.classes_)
        soil = st.selectbox("🌱 Soil Type", water_soil_enc.classes_)
    with col2:
        stage = st.selectbox("📅 Crop Stage", stage_enc.classes_)
        area = st.number_input("📐 Field Area (acres)", min_value=0.0)

    if st.button("💧 Estimate Water Need"):
        crop_enc_val = water_crop_enc.transform([crop])[0]
        soil_enc_val = water_soil_enc.transform([soil])[0]
        stage_enc_val = stage_enc.transform([stage])[0]
        input_df = pd.DataFrame([[crop_enc_val, soil_enc_val, stage_enc_val]],
                                columns=["Crop", "Soil Type", "Crop Stage"])
        prediction = water_model.predict(input_df)[0]
        total_water = prediction * area
        st.success(f"💧 Estimated Water Needed: **{total_water:.2f} mm** for {area} acres")

# ------------------- Insurance Tab --------------------
with tab4:
    st.subheader("🛡️ Insurance Risk Predictor")
    crop = st.selectbox("🌾 Crop", ins_crop_enc.classes_)
    soil = st.selectbox("🌱 Soil Type", ins_soil_enc.classes_)

    if st.button("🧠 Predict Risk Level"):
        crop_enc_val = ins_crop_enc.transform([crop])[0]
        soil_enc_val = ins_soil_enc.transform([soil])[0]
        input_df = pd.DataFrame([[crop_enc_val, soil_enc_val]], columns=["Crop", "Soil_Type"])
        pred = insurance_model.predict(input_df)[0]
        st.success(f"📉 Predicted Insurance Risk Level: **{pred}**")

# ------------------- Footer --------------------
st.markdown("---")
st.markdown("<center>Made with 💖 by <b>Naveetha Thaslim</b> | Yield Max © 2025</center>", unsafe_allow_html=True)
