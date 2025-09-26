import streamlit as st
import pickle
import pandas as pd
import os

# ------------------- Directories --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")  # All .pkl models inside this folder

def load_pickle(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------- Load Models & Encoders --------------------
# Crop
crop_model = load_pickle("crop_model.pkl")
crop_encoder = load_pickle("crop_encoder.pkl")
soil_encoder = load_pickle("soil_encoder.pkl")

# Water
water_model = load_pickle("water_model.pkl")
water_crop_encoder = load_pickle("water_crop_encoder.pkl")
water_soil_encoder = load_pickle("water_soil_encoder.pkl")
stage_encoder = load_pickle("stage_encoder.pkl")

# Fertilizer
fert_model = load_pickle("fertilizer_model.pkl")
crop_enc = load_pickle("fert_crop_encoder.pkl")
soil_enc = load_pickle("fert_soil_encoder.pkl")
stage_enc = load_pickle("fert_stage_encoder.pkl")
def_enc = load_pickle("fert_def_encoder.pkl")
fert_label_enc = load_pickle("fert_label_encoder.pkl")

# Insurance
insurance_model = load_pickle("insurance_model.pkl")

# ------------------- Streamlit UI --------------------
st.set_page_config(page_title="Yield Max 🌾", layout="wide")

st.markdown("""
<style>
.main { background-color: #f6fff5; }
.block-container { padding: 2rem 3rem; }
h1,h2,h3 { color: #2e7d32; }
.stButton>button { background-color: #43a047; color: white; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Yield Max: Smart Farming Assistant")
st.markdown("🚀 AI-powered recommendations for **crop, fertilizer, water**, and **insurance**.")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Crop", "💊 Fertilizer", "💧 Water", "🛡️ Insurance"])

# ------------------- Crop Tab --------------------
with tab1:
    st.subheader("📈 Crop Recommendation")
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

    if st.button("🔍 Predict Best Crop"):
        soil_enc_val = soil_encoder.transform([soil])[0]
        input_df = pd.DataFrame([[soil_enc_val, nitrogen, phosphorus, potassium, rainfall, ph, temperature]],
                                columns=['Soil_Type','Nitrogen','Phosphorus','Potassium','Rainfall','pH','Temperature'])
        pred = crop_model.predict(input_df)[0]
        crop_name = crop_encoder.inverse_transform([pred])[0]
        st.success(f"✅ Recommended Crop: **{crop_name}**")

# ------------------- Fertilizer Tab --------------------
with tab2:
    st.subheader("💊 Fertilizer Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("🌿 Crop", crop_enc.classes_, key="fert_crop")
        soil = st.selectbox("🌱 Soil Type", soil_enc.classes_, key="fert_soil")
    with col2:
        stage = st.selectbox("📅 Crop Stage", stage_enc.classes_, key="fert_stage")
        deficiency = st.selectbox("⚠️ Nutrient Deficiency", def_enc.classes_, key="fert_def")

    if st.button("🧪 Get Fertilizer Recommendation"):
        df = pd.DataFrame([[crop, soil, stage, deficiency]], 
                          columns=["Crop","Soil_Type","Crop_Stage","Nutrient_Deficiency"])
        df["Crop"] = crop_enc.transform(df["Crop"])
        df["Soil_Type"] = soil_enc.transform(df["Soil_Type"])
        df["Crop_Stage"] = stage_enc.transform(df["Crop_Stage"])
        df["Nutrient_Deficiency"] = def_enc.transform(df["Nutrient_Deficiency"])
        fert_pred = fert_model.predict(df)[0]
        fert_name = fert_label_enc.inverse_transform([fert_pred])[0]
        st.success(f"🌱 Recommended Fertilizer: **{fert_name}**")

# ------------------- Water Tab --------------------
with tab3:
    st.subheader("💧 Water Requirement Estimator")
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("🌾 Crop", water_crop_encoder.classes_, key="water_crop")
        soil = st.selectbox("🌱 Soil Type", water_soil_encoder.classes_, key="water_soil")
    with col2:
        stage = st.selectbox("📅 Crop Stage", stage_encoder.classes_, key="water_stage")
        area = st.number_input("📐 Field Area (acres)", min_value=0.0)

    if st.button("💧 Estimate Water Need"):
        crop_enc_val = water_crop_encoder.transform([crop])[0]
        soil_enc_val = water_soil_encoder.transform([soil])[0]
        stage_enc_val = stage_encoder.transform([stage])[0]
        df = pd.DataFrame([[crop_enc_val, soil_enc_val, stage_enc_val]], columns=["Crop","Soil Type","Crop Stage"])
        water_pred = water_model.predict(df)[0]
        total_water = water_pred * area
        st.success(f"💧 Estimated Water Needed: **{total_water:.2f} mm** for {area} acres")

# ------------------- Insurance Tab --------------------
with tab4:
    st.subheader("🛡️ Insurance Risk Predictor")
    crop = st.selectbox("🌾 Crop", crop_encoder.classes_, key="insurance_crop")
    soil = st.selectbox("🌱 Soil Type", soil_encoder.classes_, key="insurance_soil")

    if st.button("🧠 Predict Risk Level"):
        enc_crop = crop_encoder.transform([crop])[0]
        enc_soil = soil_encoder.transform([soil])[0]
        df = pd.DataFrame([[enc_crop, enc_soil]], columns=["Crop","Soil_Type"])
        risk = insurance_model.predict(df)[0]
        st.success(f"📉 Predicted Insurance Risk Level: **{risk}**")

# ------------------- Footer --------------------
st.markdown("---")
st.markdown("<center>Made with 💖 by <b>Naveetha Thaslim</b> | Yield Max © 2025</center>", unsafe_allow_html=True)
