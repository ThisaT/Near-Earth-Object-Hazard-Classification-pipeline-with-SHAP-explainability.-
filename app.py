import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# ============================================
# Load saved artifacts
# ============================================
model = joblib.load('models/random_forest_final.pkl')
scaler = joblib.load('data/scaler.pkl')
feature_columns = joblib.load('data/feature_columns.pkl')

numeric_cols = ['H', 'e', 'a', 'i', 'q', 'ad', 'per', 'per_y', 'n',
                 'condition_code', 'data_arc', 'data_arc_years']

explainer = shap.TreeExplainer(model)

# Resolve the base value once, handling both old/new SHAP formats
raw_base = explainer.expected_value
if isinstance(raw_base, (list, np.ndarray)) and np.array(raw_base).shape and np.array(raw_base).shape[0] > 1:
    base_value = raw_base[1]
else:
    base_value = raw_base

# ============================================
# Page setup
# ============================================
st.set_page_config(page_title="NEO Hazard Classifier", page_icon="☄️")
st.title("☄️ Near-Earth Object Hazard Classifier")
st.write("Enter an asteroid's orbital and physical parameters to predict whether it's potentially hazardous.")

# ============================================
# Input form
# ============================================
st.header("Orbital & Physical Parameters")

col1, col2 = st.columns(2)

with col1:
    H = st.number_input("Absolute Magnitude (H)", value=20.0)
    e = st.number_input("Eccentricity (e)", value=0.3, min_value=0.0, max_value=1.0)
    a = st.number_input("Semi-major axis (a, AU)", value=1.5)
    i = st.number_input("Inclination (i, degrees)", value=10.0)
    q = st.number_input("Perihelion distance (q, AU)", value=1.0)
    ad = st.number_input("Aphelion distance (ad, AU)", value=2.0)
    per = st.number_input("Orbital period (per, days)", value=600.0)

with col2:
    per_y = st.number_input("Orbital period (per_y, years)", value=1.6)
    n = st.number_input("Mean motion (n, deg/day)", value=0.6)
    condition_code = st.number_input("Condition code (0-9)", value=0, min_value=0, max_value=9)
    data_arc = st.number_input("Data arc (days)", value=3000)
    data_arc_years = st.number_input("Data arc (years)", value=8.2)
    orbit_class = st.selectbox("Orbit Class", ["AMO", "APO", "ATE", "IEO"])

# ============================================
# Predict button
# ============================================
if st.button("Predict Hazard Status"):

    # Engineered features
    q_proximity_1au = abs(q - 1.0)
    orbit_spread = ad - q

    # One-hot class columns
    class_flags = {
        'class_AMO': 1 if orbit_class == 'AMO' else 0,
        'class_APO': 1 if orbit_class == 'APO' else 0,
        'class_ATE': 1 if orbit_class == 'ATE' else 0,
        'class_IEO': 1 if orbit_class == 'IEO' else 0,
    }

    # Assemble input row
    input_dict = {
        'H': H, 'e': e, 'a': a, 'i': i, 'q': q, 'ad': ad,
        'per': per, 'per_y': per_y, 'n': n,
        'condition_code': condition_code,
        'data_arc': data_arc, 'data_arc_years': data_arc_years,
        **class_flags,
        'q_proximity_1au': q_proximity_1au,
        'orbit_spread': orbit_spread
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]  # enforce exact training column order

    # Scale numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("Result")
    if prediction == 1:
        st.error(f"⚠️ Potentially Hazardous — {probability*100:.1f}% confidence")
    else:
        st.success(f"✅ Not Hazardous — {(1-probability)*100:.1f}% confidence")

    # ============================================
    # SHAP explanation for this single prediction
    # ============================================
    shap_values_raw = explainer.shap_values(input_df)

    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1][0]
    elif np.array(shap_values_raw).ndim == 3:
        shap_vals = shap_values_raw[0, :, 1]
    else:
        shap_vals = shap_values_raw[0]

    shap_df = pd.DataFrame({
        'Feature': feature_columns,
        'Impact': shap_vals
    }).sort_values('Impact', key=abs, ascending=False).head(8)

    st.subheader("Why this prediction?")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['crimson' if v > 0 else 'steelblue' for v in shap_df['Impact']]
    ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors)
    ax.set_xlabel("Impact on Hazard Prediction")
    ax.invert_yaxis()
    st.pyplot(fig)
    st.caption("Red bars push toward Hazardous, blue bars push toward Not Hazardous.")