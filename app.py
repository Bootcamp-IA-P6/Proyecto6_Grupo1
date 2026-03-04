import streamlit as st
import joblib, pandas as pd

# ── Cargar modelos ───────────────────────────────────────────
modelo_1 = joblib.load('models/Jonathan/model.pkl')
encoders_1 = joblib.load('models/Jonathan/encoders.pkl')
features_1 = joblib.load('models/Jonathan/feature_names.pkl')

# ── UI ───────────────────────────────────────────────────────
st.title("☕ Gemelo Digital — Clasificación de Calidad de Café")
st.subheader("Introduce las características del lote:")

# Inputs sensoriales (compartidos por todos los modelos)
col1, col2, col3 = st.columns(3)
with col1:
    aroma     = st.slider("Aroma",     6.0, 10.0, 7.5, 0.25)
    flavor    = st.slider("Flavor",    6.0, 10.0, 7.5, 0.25)
with col2:
    acidity   = st.slider("Acidity",   6.0, 10.0, 7.5, 0.25)
    body      = st.slider("Body",      6.0, 10.0, 7.5, 0.25)
with col3:
    balance   = st.slider("Balance",   6.0, 10.0, 7.5, 0.25)
    aftertaste = st.slider("Aftertaste", 6.0, 10.0, 7.5, 0.25)

# Selector de modelo
st.divider()
modelo_elegido = st.radio(
    "Selecciona el modelo de predicción:",
    ["Random Forest (Jonathan)", "XGBoost / LightGBM (Compañero B)"],
    horizontal=True
)

if st.button("🔍 Clasificar lote"):
    # ... lógica de inferencia según modelo elegido
    pass