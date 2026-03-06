# ═══════════════════════════════════════════════════════════════════════════════
# app/main.py — Gemelo Digital Q-Grader
# Proyecto Grupal — Clasificación de Calidad de Café
# Versión con selector de modelo
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Digital Q-Grader AI",
    page_icon  = "☕",
    layout     = "wide",
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — Configuración de modelos disponibles
#
# Aquí se define todo lo que es diferente entre modelos.
# Para añadir un modelo nuevo, solo hay que añadir una entrada a este dict.
# ══════════════════════════════════════════════════════════════════════════════
MODELOS = {

    "☕ Camila — XGBoost": {
        # Ruta a la carpeta models/ de este miembro
        "path": Path("models/Camila"),

        # Columnas categóricas que este modelo encodea con LabelEncoder
        "cols_cat": ["Country of Origin", "Color"],

        # Métricas del modelo para mostrar en la app
        "metricas": {
            "F1-Score"   : "0.955",
            "CV F1 Media": "0.945 ± 0.005",
            "ROC-AUC"    : "0.990",
            "Overfitting": "+0.032  ✅ OK",
        },

        # Features exclusivas — True = este modelo SÍ usa esta feature
        "usa_uniformity" : True,
        "usa_clean_cup"  : True,
        "usa_sweetness"  : True,
        "usa_altitud"    : False,
        "usa_processing" : False,
        "usa_variety"    : False,
    },

    "🌲 Jonathan — Random Forest": {
        "path": Path("models/Jonathan"),

        "cols_cat": ["Color", "Country of Origin", "Processing Method", "Variety"],

        "metricas": {
            "F1-Score"   : "0.953",
            "CV F1 Media": "0.932 ± 0.019",
            "ROC-AUC"    : "0.974",
            "Overfitting": "Por calcular  ✅ OK",
        },

        "usa_uniformity" : False,
        "usa_clean_cup"  : False,
        "usa_sweetness"  : False,
        "usa_altitud"    : True,
        "usa_processing" : True,
        "usa_variety"    : True,
    },

}


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Cargar el modelo seleccionado
#
# @st.cache_resource guarda el modelo en memoria.
# El parámetro 'nombre' hace que se recargue si el usuario cambia de modelo.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_modelo(nombre: str):
    """
    Carga model.pkl, feature_names.pkl y encoders.pkl del modelo seleccionado.
    Se recarga automáticamente si cambia el nombre del modelo.
    """
    path = MODELOS[nombre]["path"]

    modelo        = joblib.load(path / "model.pkl")
    feature_names = joblib.load(path / "feature_names.pkl")
    encoders      = joblib.load(path / "encoders.pkl")

    return modelo, feature_names, encoders


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — Cabecera y selector de modelo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ☕ Digital Q-Grader AI")
st.caption("LIVE CERTIFICATION MODULE")
st.divider()

# ─── Selector de modelo ───────────────────────────────────────────────────────
# El usuario elige qué modelo usar — la app se adapta automáticamente
modelo_elegido = st.radio(
    "**Modelo de predicción:**",
    options=list(MODELOS.keys()),
    horizontal=True,
)

# Cargamos los artefactos del modelo seleccionado
modelo, feature_names, encoders = cargar_modelo(modelo_elegido)

# Referencia a la configuración del modelo activo
cfg = MODELOS[modelo_elegido]

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4 — Formulario de inputs
#
# Los inputs que aparecen dependen del modelo seleccionado.
# cfg["usa_X"] controla qué se muestra en cada caso.
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Datos del lote")

col_izq, col_der = st.columns(2)

# ─── Columna izquierda: sensoriales (todos los modelos las usan) ──────────────
with col_izq:
    st.markdown("**Puntuaciones sensoriales**")
    st.caption("Escala 0 – 10 por atributo")

    aroma      = st.slider("Fragrance / Aroma", 0.0, 10.0, 7.5, 0.25)
    flavor     = st.slider("Flavor",            0.0, 10.0, 7.5, 0.25)
    aftertaste = st.slider("Aftertaste",        0.0, 10.0, 7.5, 0.25)
    acidity    = st.slider("Acidity",           0.0, 10.0, 7.5, 0.25)
    body       = st.slider("Body",              0.0, 10.0, 7.5, 0.25)
    balance    = st.slider("Balance",           0.0, 10.0, 7.5, 0.25)

    # Solo aparecen si el modelo activo las usa
    if cfg["usa_uniformity"]:
        uniformity = st.slider("Uniformity", 0.0, 10.0, 10.0, 0.25)
    if cfg["usa_clean_cup"]:
        clean_cup  = st.slider("Clean Cup",  0.0, 10.0, 10.0, 0.25)
    if cfg["usa_sweetness"]:
        sweetness  = st.slider("Sweetness",  0.0, 10.0, 10.0, 0.25)

# ─── Columna derecha: físicas y origen ───────────────────────────────────────
with col_der:
    st.markdown("**Características físicas**")

    moisture     = st.number_input("Moisture Percentage (%)", 0.0, 15.0, 11.0, 0.1)
    cat1_defects = st.number_input("Category One Defects",    0,   50,    0)
    cat2_defects = st.number_input("Category Two Defects",    0,   50,    4)
    quakers      = st.number_input("Quakers",                 0,   50,    0)

    # Solo aparece si el modelo activo la usa
    if cfg["usa_altitud"]:
        altitud = st.number_input("Altitud (metros)", 100, 3500, 1500)

    st.markdown("**Origen y proceso**")

    pais = st.selectbox(
        "Country of Origin",
        options=sorted(encoders["Country of Origin"].classes_.tolist())
    )

    color = st.selectbox(
        "Color",
        options=encoders["Color"].classes_.tolist()
    )

    if cfg["usa_processing"]:
        metodo = st.selectbox(
            "Processing Method",
            options=sorted(encoders["Processing Method"].classes_.tolist())
        )

    if cfg["usa_variety"]:
        variedad = st.selectbox(
            "Variety",
            options=sorted(encoders["Variety"].classes_.tolist())
        )

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5 — Función de predicción
#
# Construye el dict de inputs solo con las features que usa el modelo activo,
# aplica el encoding correcto y devuelve la predicción.
# ══════════════════════════════════════════════════════════════════════════════
def predecir(feature_names, encoders, cfg) -> dict:
    """
    Construye el input del modelo desde los valores del formulario,
    aplica encoding de categóricas y devuelve predicción + probabilidad.
    """

    # ─── Construir dict con todas las features posibles ───────────────────────
    todos_los_inputs = {
        "Aroma"               : aroma,
        "Flavor"              : flavor,
        "Aftertaste"          : aftertaste,
        "Acidity"             : acidity,
        "Body"                : body,
        "Balance"             : balance,
        "Moisture Percentage" : moisture,
        "Category One Defects": cat1_defects,
        "Category Two Defects": cat2_defects,
        "Quakers"             : quakers,
        "Country of Origin"   : pais,
        "Color"               : color,
    }

    # ─── Añadir features opcionales según el modelo activo ────────────────────
    if cfg["usa_uniformity"]:
        todos_los_inputs["Uniformity"]  = uniformity
    if cfg["usa_clean_cup"]:
        todos_los_inputs["Clean Cup"]   = clean_cup
    if cfg["usa_sweetness"]:
        todos_los_inputs["Sweetness"]   = sweetness
    if cfg["usa_altitud"]:
        todos_los_inputs["altitud_limpia"] = altitud
    if cfg["usa_processing"]:
        todos_los_inputs["Processing Method"] = metodo
    if cfg["usa_variety"]:
        todos_los_inputs["Variety"]     = variedad

    # ─── Encoding de categóricas ──────────────────────────────────────────────
    df_input = pd.DataFrame([todos_los_inputs])

    for col in cfg["cols_cat"]:
        le    = encoders[col]
        valor = str(df_input[col].iloc[0])
        df_input[col] = le.transform([valor])[0] if valor in le.classes_ else 0

    # ─── Orden correcto de columnas ───────────────────────────────────────────
    df_input = df_input[feature_names]

    # ─── Predicción ───────────────────────────────────────────────────────────
    prediccion   = modelo.predict(df_input)[0]
    probabilidad = modelo.predict_proba(df_input)[0][1]

    return {
        "etiqueta"     : "SPECIALTY COFFEE" if prediccion == 1 else "NO SPECIALTY",
        "es_specialty" : bool(prediccion == 1),
        "probabilidad" : round(float(probabilidad) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 6 — Radar chart sensorial
# ══════════════════════════════════════════════════════════════════════════════
def grafico_radar(valores: dict) -> plt.Figure:
    categorias   = list(valores.keys())
    puntuaciones = list(valores.values())
    puntuaciones += [puntuaciones[0]]
    n = len(categorias)

    angulos  = [i * 2 * np.pi / n for i in range(n)]
    angulos += [angulos[0]]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.plot(angulos, puntuaciones, color="#00FF99", linewidth=2)
    ax.fill(angulos, puntuaciones, color="#00FF99", alpha=0.2)
    ax.set_thetagrids(
        [a * 180 / np.pi for a in angulos[:-1]],
        categorias, color="white", fontsize=9
    )
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="gray", fontsize=7)
    ax.grid(color="gray", alpha=0.3)
    ax.spines["polar"].set_color("gray")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 7 — Botón de predicción y resultados
# ══════════════════════════════════════════════════════════════════════════════
if st.button("🔍  Clasificar lote", use_container_width=True, type="primary"):

    resultado = predecir(feature_names, encoders, cfg)

    st.divider()

    res_izq, res_mid, res_der = st.columns([2, 1.5, 1.5])

    # ── Resultado principal ───────────────────────────────────────────────────
    with res_izq:
        st.caption("FINAL CERTIFICATION")

        if resultado["es_specialty"]:
            st.success(f"## ✅ {resultado['etiqueta']}")
            st.caption("SCAA Protocol Compliant")
        else:
            st.error(f"## ❌ {resultado['etiqueta']}")
            st.caption("Does not meet Specialty threshold (82.5 pts)")

        st.metric(
            label = "Confidence",
            value = f"{resultado['probabilidad']}%",
            delta = "↑ Specialty" if resultado["es_specialty"] else "↓ No Specialty"
        )
        st.progress(resultado["probabilidad"] / 100)

    # ── Radar chart ───────────────────────────────────────────────────────────
    with res_mid:
        st.caption("COFFEE PROFILE MAP")
        fig = grafico_radar({
            "Aroma"     : aroma,
            "Flavor"    : flavor,
            "Aftertaste": aftertaste,
            "Acidity"   : acidity,
            "Body"      : body,
            "Balance"   : balance,
        })
        st.pyplot(fig, use_container_width=True)

    # ── Métricas del modelo activo ────────────────────────────────────────────
    # Estas métricas vienen del dict MODELOS — cambian automáticamente
    # cuando el usuario selecciona un modelo diferente
    with res_der:
        st.caption("AI RELIABILITY")
        for nombre_metrica, valor in cfg["metricas"].items():
            st.metric(nombre_metrica, valor)

    # ── Resumen sensorial ─────────────────────────────────────────────────────
    st.divider()
    st.caption("SENSORY ANALYSIS REPORT")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"**Fragrance / Aroma:** {aroma} / 10")
        st.markdown(f"**Flavor:** {flavor} / 10")
        st.markdown(f"**Aftertaste:** {aftertaste} / 10")
    with col_s2:
        st.markdown(f"**Acidity:** {acidity} / 10")
        st.markdown(f"**Body:** {body} / 10")
        st.markdown(f"**Balance:** {balance} / 10")