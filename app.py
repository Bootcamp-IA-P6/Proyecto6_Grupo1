# Importar librerias necesarias.
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title            = "Digital Q-Grader AI",
    page_icon             = "☕",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 0 — CSS
# Solo se incluyen las reglas que realmente tienen efecto en la app.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Variables de color ── */
:root {
    --bg:          #F8FAF9;
    --bg-card:     #FFFFFF;
    --bg-sidebar:  #F0F7F3;
    --green:       #16A34A;
    --green-light: #22C55E;
    --green-dim:   rgba(22, 163, 74, 0.10);
    --text:        #1A2E22;
    --text-dim:    #6B7F73;
    --border:      #D1E8DA;
    --radius:      16px;
    --radius-sm:   10px;
    --shadow:      0 2px 12px rgba(22,163,74,0.08);
}

/* ── Fondo y fuente global ── */
html, body, [data-testid="stAppViewContainer"], .main {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
.block-container { background: transparent !important; padding-top: 2rem !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] hr {
    border: none !important; height: 1px !important;
    background: var(--border) !important; margin: 1.2rem 0 !important;
}

/* ── Labels de inputs en sidebar ── */
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label {
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Barra del slider en verde ── */
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--green) !important;
}

/* ── Inputs y selectbox ── */
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] .stNumberInput input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px var(--green-dim) !important;
}

/* ── Radio toggle (selector de modelo) ── */
[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    background: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 3px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    flex: 1 1 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 8px !important;
    padding: 7px 4px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--text-dim) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] { display: none !important; }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: var(--green) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* ── Botón ── */
.stButton > button {
    background: var(--green) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 24px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
}
.stButton > button:hover {
    background: var(--green-light) !important;
    transform: translateY(-2px) !important;
}

/* ── Columnas como cards ── */
div[data-testid="column"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    box-shadow: var(--shadow) !important;
}

/* ── Métricas ── */
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--green) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
    box-shadow: var(--shadow) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: var(--text-dim) !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--green) !important;
    color: #FFFFFF !important;
}

/* ── Divider ── */
hr {
    border: none !important; height: 1px !important;
    background: var(--border) !important; margin: 1.8rem 0 !important;
}

/* ── Ocultar botón colapsar sidebar ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — Configuración de modelos
# Para añadir un modelo nuevo solo hay que añadir una entrada aquí.
# ══════════════════════════════════════════════════════════════════════════════
MODELOS = {

    " XGBoost": {
        "path"          : Path("models/Camila"),
        "cols_cat"      : ["Country of Origin", "Color"],
        "metricas"      : {
            "F1-Score"   : "0.955",
            "CV F1 Media": "0.945 ± 0.005",
            "ROC-AUC"    : "0.990",
            "Overfitting": "+0.032  ✅",
        },
        "usa_uniformity": True,
        "usa_clean_cup" : True,
        "usa_sweetness" : True,
        "usa_altitud"   : False,
        "usa_processing": False,
        "usa_variety"   : False,
    },

    " Random Forest": {
        "path"          : Path("models/Jonathan"),
        "cols_cat"      : ["Color", "Country of Origin", "Processing Method", "Variety"],
        "metricas"      : {
            "F1-Score"   : "0.953",
            "CV F1 Media": "0.932 ± 0.019",
            "ROC-AUC"    : "0.974",
            "Overfitting": "+0.014  ✅",
        },
        "usa_uniformity": False,
        "usa_clean_cup" : False,
        "usa_sweetness" : False,
        "usa_altitud"   : True,
        "usa_processing": True,
        "usa_variety"   : True,
    },

    " XGBoost 2": {
        "path"          : Path("models/Juanma"),
        "cols_cat"      : ["Country of Origin", "Color"],
        "metricas"      : {
            "F1-Score"   : "0.947",
            "CV F1 Media": "0.931 ± 0.021",
            "ROC-AUC"    : "0.970",
            "Overfitting": "+0.008  ✅",
        },
        "usa_uniformity": True,
        "usa_clean_cup" : True,
        "usa_sweetness" : True,
        "usa_altitud"   : False,
        "usa_processing": False,
        "usa_variety"   : False,
    },

}


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Cargar modelo
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_modelo(nombre: str):
    path          = MODELOS[nombre]["path"]
    modelo        = joblib.load(path / "model.pkl")
    feature_names = joblib.load(path / "feature_names.pkl")
    encoders      = joblib.load(path / "encoders.pkl")
    return modelo, feature_names, encoders


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — Historial
# ══════════════════════════════════════════════════════════════════════════════
if "historial" not in st.session_state:
    st.session_state.historial = []


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4 — Helpers
# Funciones pequeñas que se usan en varios sitios.
# ══════════════════════════════════════════════════════════════════════════════
def titulo_seccion(texto: str):

    st.markdown(f"""
        <div style="font-size:0.72rem;color:#6B7F73;text-transform:uppercase;
            letter-spacing:0.1em;font-weight:600;margin-bottom:11px;">
            {texto}
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 5 — SIDEBAR: logo + selector + inputs
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
        <div style="text-align:center; padding: 8px 0 20px 0;">
            <div style="font-size:4.5rem; color:black;">☕</div>
            <div style="font-family:'DM Sans',sans-serif;font-weight:700;
                font-size:1.45rem;color:#16A34A;margin-top:8px;">
                Digital Q-Grader AI
            </div>
            <div style="color:#6B7F73;font-size:0.72rem;
                letter-spacing:0.12em;text-transform:uppercase;margin-top:4px;">
                Scientific Lab Terminal
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    titulo_seccion("Modelo de predicción")

    modelo_elegido = st.radio(
        label            = "Modelo",
        options          = list(MODELOS.keys()),
        label_visibility = "collapsed",
        horizontal       = True,
    )

    # Cargamos modelo y su configuración
    modelo, feature_names, encoders = cargar_modelo(modelo_elegido)
    cfg          = MODELOS[modelo_elegido]
    nombre_corto = modelo_elegido.split(" ", 1)[1]  # quita el emoji

    st.markdown("---")
    titulo_seccion("Sensory Scoring")

    aroma      = st.slider("Fragrance / Aroma", 0.0, 10.0, 5.0, 0.25)
    flavor     = st.slider("Flavor",            0.0, 10.0, 5.0, 0.25)
    aftertaste = st.slider("Aftertaste",        0.0, 10.0, 5.0, 0.25)
    acidity    = st.slider("Acidity",           0.0, 10.0, 5.0, 0.25)
    body       = st.slider("Body",              0.0, 10.0, 5.0, 0.25)
    balance    = st.slider("Balance",           0.0, 10.0, 5.0, 0.25)

    if cfg["usa_uniformity"]:
        uniformity = st.slider("Uniformity", 0.0, 10.0, 5.0, 0.25)
    if cfg["usa_clean_cup"]:
        clean_cup  = st.slider("Clean Cup",  0.0, 10.0, 5.0, 0.25)
    if cfg["usa_sweetness"]:
        sweetness  = st.slider("Sweetness",  0.0, 10.0, 5.0, 0.25)

    st.markdown("---")
    titulo_seccion("Physical Data")

    moisture     = st.number_input("Moisture (%)",         0.0, 15.0, 11.0, 0.1)
    cat1_defects = st.number_input("Category One Defects", 0,   50,    0)
    cat2_defects = st.number_input("Category Two Defects", 0,   50,    4)
    quakers      = st.number_input("Quakers",              0,   50,    0)

    if cfg["usa_altitud"]:
        altitud = st.number_input("Altitud (metros)", 100, 3500, 1500)

    st.markdown("---")
    titulo_seccion("Origin")

    pais  = st.selectbox("Country of Origin",
                            options=sorted(encoders["Country of Origin"].classes_.tolist()))
    color = st.selectbox("Color",
                            options=encoders["Color"].classes_.tolist())

    if cfg["usa_processing"]:
        metodo = st.selectbox("Processing Method",
                                options=sorted(encoders["Processing Method"].classes_.tolist()))
    if cfg["usa_variety"]:
        variedad = st.selectbox("Variety",
                                options=sorted(encoders["Variety"].classes_.tolist()))

    st.markdown("---")
    clasificar_btn = st.button("🔍  New Assessment", width='stretch')

    st.markdown(f"""
        <div style="margin-top:12px;background:#F0F7F3;border:1px solid #D1E8DA;
            border-radius:10px;padding:10px 14px;font-size:0.8rem;
            color:#6B7F73;text-align:center;">
            Active Model<br>
            <span style="color:#16A34A;font-weight:700;font-size:0.88rem;">
                ✦ {nombre_corto}
            </span>
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 6 — Función de predicción
# ══════════════════════════════════════════════════════════════════════════════
def predecir() -> dict:
    inputs = {
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

    if cfg["usa_uniformity"]:  inputs["Uniformity"]         = uniformity
    if cfg["usa_clean_cup"]:   inputs["Clean Cup"]          = clean_cup
    if cfg["usa_sweetness"]:   inputs["Sweetness"]          = sweetness
    if cfg["usa_altitud"]:     inputs["altitud_limpia"]     = altitud
    if cfg["usa_processing"]:  inputs["Processing Method"]  = metodo
    if cfg["usa_variety"]:     inputs["Variety"]            = variedad

    df = pd.DataFrame([inputs])

    for col in cfg["cols_cat"]:
        le    = encoders[col]
        valor = str(df[col].iloc[0])
        df[col] = le.transform([valor])[0] if valor in le.classes_ else 0

    df         = df[feature_names]
    prediccion = modelo.predict(df)[0]
    prob       = modelo.predict_proba(df)[0][1]

    return {
        "etiqueta"     : "SPECIALTY COFFEE" if prediccion == 1 else "NO SPECIALTY",
        "es_specialty" : bool(prediccion == 1),
        "probabilidad" : round(float(prob) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 7 — Radar chart sensorial
# ══════════════════════════════════════════════════════════════════════════════
def grafico_radar(valores: dict) -> plt.Figure:
    cats  = list(valores.keys())
    vals  = list(valores.values()) + [list(valores.values())[0]]
    n     = len(cats)
    angs  = [i * 2 * np.pi / n for i in range(n)] + [0]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAF9")
    ax.plot(angs, vals, color="#16A34A", linewidth=2.5)
    ax.fill(angs, vals, color="#16A34A", alpha=0.12)
    ax.set_thetagrids([a * 180 / np.pi for a in angs[:-1]], cats,
                        color="#1A2E22", fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#6B7F73", fontsize=7)
    ax.grid(color="#D1E8DA", alpha=0.8)
    ax.spines["polar"].set_color("#D1E8DA")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 8 — Cabecera principal
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="padding: 8px 0 28px 0;">
    <h1 style="font-family:'DM Sans',sans-serif;font-weight:700;
        font-size:clamp(1.8rem,3vw,2.8rem);color:#1A2E22;
        letter-spacing:-0.02em;margin:12px 0 6px 0;">
        Digital Q-Grader AI <span style="color:#16A34A;"></span>
    </h1>
    <p style="color:#6B7F73;font-size:0.95rem;margin:0;">
        Configura el lote en el panel izquierdo y pulsa
        <strong style="color:#16A34A;">New Assessment</strong>.
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 9 — Pestañas
# ══════════════════════════════════════════════════════════════════════════════
tab_clasif, tab_hist = st.tabs(["🔬  Assessment", "🕓  History"])


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 1 — Assessment
# ─────────────────────────────────────────────────────────────────────────────
with tab_clasif:

    if clasificar_btn:

        resultado = predecir()
        prob      = resultado["probabilidad"]
        specialty = resultado["es_specialty"]

        # Guardar en historial (máximo 20 entradas)
        st.session_state.historial.insert(0, {
            "hora"        : datetime.now().strftime("%H:%M:%S"),
            "fecha"       : datetime.now().strftime("%d/%m/%Y"),
            "modelo"      : nombre_corto,
            "resultado"   : resultado["etiqueta"],
            "es_specialty": specialty,
            "probabilidad": prob,
            "pais"        : pais,
            "aroma"       : aroma,
            "flavor"      : flavor,
        })
        st.session_state.historial = st.session_state.historial[:20]

        # ── Colores según resultado ───────────────────────────────────────────
        color_r  = "#16A34A" if specialty else "#DC2626"
        bg_r     = "#F0F7F3" if specialty else "#FEF2F2"
        border_r = "#D1E8DA" if specialty else "#FECACA"
        icono    = "✅" if specialty else "❌"
        sub      = "SCAA Protocol Compliant" if specialty else "Does not meet threshold (82.5 pts)"

        # ── Tarjeta de resultado ──────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:{bg_r};border:1px solid {border_r};border-radius:20px;
            padding:32px 40px;text-align:center;margin-bottom:24px;">
            <div style="color:#6B7F73;font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.14em;font-weight:600;margin-bottom:10px;">
                Final Certification
            </div>
            <div style="font-family:'DM Sans',sans-serif;font-weight:700;
                font-size:clamp(1.8rem,4vw,2.8rem);color:{color_r};margin-bottom:8px;">
                {icono} {resultado['etiqueta']}
            </div>
            <div style="color:#6B7F73;font-size:0.85rem;margin-bottom:20px;">{sub}</div>
            <div style="display:inline-block;background:#FFFFFF;
                border:1px solid {border_r};border-radius:12px;padding:10px 32px;">
                <span style="font-family:'DM Mono',monospace;font-size:2.2rem;
                    font-weight:600;color:{color_r};">{prob}%</span>
                <span style="color:#6B7F73;font-size:0.72rem;display:block;
                    letter-spacing:0.1em;text-transform:uppercase;">Confidence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Métricas del modelo activo ────────────────────────────────────────
        cols_met = st.columns(len(cfg["metricas"]))
        for col, (k, v) in zip(cols_met, cfg["metricas"].items()):
            with col:
                st.metric(k, v)

        st.divider()

        # ── Radar + Sensory report ────────────────────────────────────────────
        col_radar, col_report = st.columns(2)

        with col_radar:
            titulo_seccion("Coffee Profile Map")
            fig = grafico_radar({
                "Aroma": aroma, "Flavor": flavor, "Aftertaste": aftertaste,
                "Acidity": acidity, "Body": body, "Balance": balance,
            })
            st.pyplot(fig, width='stretch')

        with col_report:
            titulo_seccion("Sensory Analysis Report")

            for nombre_s, valor_s in {
                "Fragrance / Aroma": aroma, "Flavor": flavor,
                "Aftertaste": aftertaste, "Acidity": acidity,
                "Body": body, "Balance": balance,
            }.items():
                pct    = int((valor_s / 10) * 100)
                c_bar  = "#16A34A" if valor_s >= 7.5 else ("#F59E0B" if valor_s >= 6.0 else "#DC2626")
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#1A2E22;font-size:0.85rem;">{nombre_s}</span>
                        <span style="font-family:'DM Mono',monospace;color:{c_bar};
                            font-size:0.85rem;font-weight:600;">{valor_s}</span>
                    </div>
                    <div style="background:#F0F7F3;border-radius:4px;height:6px;">
                        <div style="background:{c_bar};width:{pct}%;height:6px;border-radius:4px;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:16px;padding:14px;background:#F8FAF9;
                border:1px solid #D1E8DA;border-radius:10px;">
                <div style="color:#6B7F73;font-size:0.7rem;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:8px;font-weight:600;">Batch Info</div>
                <div style="color:#1A2E22;font-size:0.85rem;line-height:1.9;">
                    🌍 <strong>{pais}</strong><br>
                    🎨 Color: <strong>{color}</strong><br>
                    💧 Moisture: <strong>{moisture}%</strong><br>
                    🤖 Model: <strong>{nombre_corto}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Estado inicial ────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:70px 20px;background:#FFFFFF;
            border:1px dashed #D1E8DA;border-radius:20px;margin-top:8px;">
            <div style="font-size:3.5rem;margin-bottom:16px;">☕</div>
            <div style="color:#6B7F73;font-size:1rem;max-width:400px;
                margin-inline:auto;line-height:1.6;">
                Configura los parámetros del lote en el panel izquierdo<br>
                y pulsa <strong style="color:#16A34A;">New Assessment</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 2 — Historial
# ─────────────────────────────────────────────────────────────────────────────
with tab_hist:

    if not st.session_state.historial:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:#FFFFFF;
            border:1px dashed #D1E8DA;border-radius:20px;margin-top:8px;">
            <div style="font-size:3rem;margin-bottom:14px;">🕓</div>
            <div style="color:#6B7F73;font-size:0.95rem;">
                Aún no hay clasificaciones en esta sesión.<br>
                Haz tu primer assessment en la pestaña anterior.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        col_tit, col_clear = st.columns([5, 1])
        with col_tit:
            st.markdown(f"""
            <div style="color:#6B7F73;font-size:0.72rem;text-transform:uppercase;
                letter-spacing:0.1em;font-weight:600;margin-bottom:4px;">
                Assessment History</div>
            <div style="color:#1A2E22;font-size:1.1rem;font-weight:600;">
                {len(st.session_state.historial)} clasificaciones esta sesión</div>
            """, unsafe_allow_html=True)
        with col_clear:
            if st.button("🗑️ Limpiar"):
                st.session_state.historial = []
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Resumen rápido ────────────────────────────────────────────────────
        n_spec  = sum(1 for h in st.session_state.historial if h["es_specialty"])
        n_no    = len(st.session_state.historial) - n_spec
        p_media = round(np.mean([h["probabilidad"] for h in st.session_state.historial]), 1)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("✅ Specialty",       n_spec)
        with m2: st.metric("❌ No Specialty",    n_no)
        with m3: st.metric("📊 Confianza media", f"{p_media}%")

        st.divider()

        # ── Tarjetas del historial ────────────────────────────────────────────
        for h in st.session_state.historial:
            c_h = "#16A34A" if h["es_specialty"] else "#DC2626"
            bg_h     = "#F0F7F3" if h["es_specialty"] else "#FEF2F2"
            border_h = "#D1E8DA" if h["es_specialty"] else "#FECACA"

            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                background:{bg_h};border:1px solid {border_h};border-radius:12px;
                padding:14px 20px;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:16px;">
                    <div style="background:#FFFFFF;border:1px solid {border_h};color:{c_h};
                        border-radius:10px;padding:6px 16px;font-family:'DM Mono',monospace;
                        font-weight:700;font-size:1rem;text-align:center;">
                        {h['probabilidad']}%
                    </div>
                    <div>
                        <div style="color:#1A2E22;font-size:0.9rem;font-weight:600;">
                            {"✅" if h["es_specialty"] else "❌"} {h['resultado']}
                        </div>
                        <div style="color:#6B7F73;font-size:0.75rem;margin-top:3px;">
                            {h['modelo']} · {h['pais']} · Aroma {h['aroma']} · Flavor {h['flavor']}
                        </div>
                    </div>
                </div>
                <div style="text-align:right;color:#6B7F73;font-size:0.75rem;">
                    {h['hora']}<br>{h['fecha']}
                </div>
            </div>
            """, unsafe_allow_html=True)