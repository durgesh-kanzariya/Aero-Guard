import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import xgboost as xgb

# Load Model
try:
    model = joblib.load('aero_guard_model.pkl')
except:
    st.error("⚠️ Model file not found! Please run 'train_model.py' first.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Aero Guard AI",
    page_icon="✈️",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #444;
        text-align: center;
        margin-bottom: 30px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-header">✈️ Aero Guard: Predictive Maintenance System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">NASA Turbofan Jet Engine RUL Prediction (XGBoost)</div>', unsafe_allow_html=True)

# --- MAIN LAYOUT: CONTROLS ---
st.write("### ⚙️ Engine Parameters Control Panel")
col1, col2, col3 = st.columns(3)

# CARD 1: TEMPERATURES
with col1:
    with st.container(border=True):
        st.subheader("🔥 Temperatures")
        s_2 = st.slider('Total Temp (s_2)', 640.0, 645.0, 642.0)
        s_4 = st.slider('Burner Temp (s_4)', 1390.0, 1410.0, 1400.0)

# CARD 2: PRESSURES
with col2:
    with st.container(border=True):
        st.subheader("💨 Pressures")
        s_3 = st.slider('Total Pressure (s_3)', 1580.0, 1600.0, 1590.0)
        s_9 = st.slider('HPC Static Pressure (s_9)', 9000.0, 9200.0, 9050.0)

# CARD 3: SPEEDS & RATIOS
with col3:
    with st.container(border=True):
        st.subheader("⚙️ Speeds & Ratios")
        s_7 = st.slider('Phy Core Speed (s_7)', 550.0, 560.0, 554.0)
        s_8 = st.slider('Phy Fan Speed (s_8)', 2380.0, 2390.0, 2388.0)
        s_11 = st.slider('Fuel Flow Ratio (s_11)', 46.0, 49.0, 47.5)
        s_12 = st.slider('Bypass Ratio (s_12)', 520.0, 530.0, 521.0)

# --- FEATURE ENGINEERING HELPER ---
def prepare_input_for_model(sliders_dict):
    """
    Translates simple slider inputs into the complex features (Lag, Std, Mean)
    expected by the XGBoost model.
    """
    # 1. Base Sensors list (Must match train_model.py)
    sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 
               's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    
    # 2. Defaults for sensors we don't show in UI (to keep it clean)
    defaults = {
        's_13': 2388.0, 's_14': 8130.0, 's_15': 8.4, 
        's_17': 392, 's_20': 39.0, 's_21': 23.4
    }
    
    # Combine UI inputs with defaults
    current_values = {**defaults, **sliders_dict}
    
    data = {}
    
    # A. Generate Rolling Means (Assume steady state: Mean = Current Value)
    for s in sensors:
        data[f"{s}_mean"] = current_values[s]
        
    # B. Generate Rolling Std Dev (Assume steady state: Std = 0.0)
    # We add a tiny epsilon (0.001) to prevent division by zero errors inside the model
    for s in sensors:
        data[f"{s}_std"] = 0.001
        
    # C. Generate Lag Features (Assume steady state: Past = Current)
    # Only for the specific sensors we lagged in training
    for s in ['s_11', 's_12', 's_4', 's_7']:
        data[f"{s}_lag1"] = current_values[s]
        data[f"{s}_lag2"] = current_values[s]

    # Convert to DataFrame
    # IMPORTANT: Columns must be in correct order. XGBoost is strict.
    # We rely on the dict keys matching the creation order, but usually 
    # XGBoost handles named columns if using pandas.
    return pd.DataFrame(data, index=[0])

# Collect User Inputs
user_inputs = {
    's_2': s_2, 's_3': s_3, 's_4': s_4, 's_7': s_7, 
    's_8': s_8, 's_9': s_9, 's_11': s_11, 's_12': s_12
}

# Generate complex features
input_df = prepare_input_for_model(user_inputs)

# Make Prediction
prediction = model.predict(input_df)
rul = int(prediction[0])

st.markdown("---")

# --- VISUALIZATION ROW ---
viz_col1, viz_col2 = st.columns([1.5, 1])

with viz_col1:
    st.subheader("📊 Remaining Useful Life (RUL)")
    
    if rul > 100: color = "#2ecc71" # Green
    elif rul > 50: color = "#f39c12" # Orange
    else: color = "#e74c3c" # Red

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = rul,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 125, 'increasing': {'color': "#2ecc71"}},
        gauge = {
            'axis': {'range': [None, 200]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "rgba(231, 76, 60, 0.2)"},
                {'range': [50, 100], 'color': "rgba(243, 156, 18, 0.2)"},
                {'range': [100, 200], 'color': "rgba(46, 204, 113, 0.2)"}],
        }
    ))
    fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with viz_col2:
    st.subheader("📡 Sensor Health Radar")
    
    categories = ['Temp', 'Pressure', 'Core Speed', 'Fan Speed', 'Fuel Ratio']
    # Scale values roughly to 0-1 range for the chart
    values = [
        user_inputs['s_2']/6.5, 
        user_inputs['s_3']/16, 
        user_inputs['s_7']/5.6, 
        user_inputs['s_8']/24, 
        user_inputs['s_11']/0.5
    ]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current State',
        line_color='#0066cc'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("Aero Guard v3.0 (XGBoost Edition) | Developed by Durgesh Kanzariya")