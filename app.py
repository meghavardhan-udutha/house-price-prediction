# streamlit-apps/house-price/app.py

import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('house_model.pkl', 'rb'))

# Set dark theme and page title
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# Custom dark styling
st.markdown("""
    <style>
        .main {
            background-color: #111;
            color: #f0f0f0;
        }
        .stButton > button {
            background-color: #ff4c4c;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: #ff1e1e;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            color: #ff4c4c;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🏠 House Price Predictor</div>", unsafe_allow_html=True)

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    lot_area = st.number_input("📏 Lot Area", min_value=100)
    overall_qual = st.slider("🏅 Overall Quality", 1, 10, 5)
    overall_cond = st.slider("🛠️ Overall Condition", 1, 10, 5)
    year_built = st.number_input("🏗️ Year Built", min_value=1800, max_value=2025, value=2000)
    year_remod = st.number_input("🔧 Year Remodeled", min_value=1800, max_value=2025, value=2010)
    low_qual_sf = st.number_input("📉 Low Quality Finished SF")

with col2:
    gr_liv_area = st.number_input("🏠 Above Ground Living Area")
    bsmt_full = st.number_input("🚿 Basement Full Bathrooms", step=1)
    bsmt_half = st.number_input("🚽 Basement Half Bathrooms", step=1)
    full_bath = st.number_input("🛁 Full Bathrooms", step=1)
    half_bath = st.number_input("🚻 Half Bathrooms", step=1)
    bedrooms = st.number_input("🛏️ Bedrooms Above Ground", step=1)
    kitchens = st.number_input("🍳 Kitchens Above Ground", step=1)

# House style encoding
house_styles = ['1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl']
house_style = st.selectbox("🏡 House Style", house_styles)

# One-hot encoding for HouseStyle
style_encoding = [1 if house_style == style else 0 for style in house_styles]

# Combine all inputs into a single row
input_data = np.array([
    lot_area, overall_qual, overall_cond, year_built, year_remod,
    low_qual_sf, gr_liv_area, bsmt_full, bsmt_half, full_bath,
    half_bath, bedrooms, kitchens
] + style_encoding).reshape(1, -1)

# --- Predict ---
if st.button("💰 Predict House Price"):
    prediction = model.predict(input_data)[0]
    prediction = max(prediction, 0)
    st.success(f"🏷️ Estimated House Price: ₹ {(round(prediction))*85:,}")
