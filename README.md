# ☕ Clasificación de Calidad de Café
### *Coffee Quality Classification Project — Specialty vs No Specialty*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12.12-3776AB?style=for-the-badge&logo=python&logoColor=white)   ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)    ![XGBoost](https://img.shields.io/badge/XGBoost-189C3E?style=for-the-badge&logo=xgboost&logoColor=white)    ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)   ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)

*Aplicación de Machine Learning para clasificar lotes de café como Specialty o No Specialty según los estándares de la Specialty Coffee Association (SCA)*

</div>

---

## 📌 Descripción del Proyecto

Este proyecto aplica técnicas de **clasificación supervisada** para determinar si un lote de café alcanza la categoría **Specialty** según los estándares internacionales de la *Specialty Coffee Association* (SCA). El objetivo es doble: identificar qué atributos sensoriales y físicos tienen mayor impacto en la calidad final del café, y ofrecer una herramienta predictiva interactiva desplegada con Streamlit.

El dataset es una **fusión de dos fuentes del Coffee Quality Institute (CQI)**:
- `arabica_data_cleaned.csv` — dataset histórico de evaluaciones Q-Grader
- `coffee_quality.csv` — dataset actualizado CQI mayo 2023

El dataset fusionado contiene **1.512 lotes de café arábica** con 38 variables que incluyen puntuaciones sensoriales (aroma, flavor, acidez, cuerpo, balance...), información de origen, método de procesado y altitud.

**Target:** `Total Cup Points ≥ 82.5` → **Specialty** | `< 82.5` → **No Specialty**

---

## 🤝 Metodología de Equipo

El equipo adoptó un enfoque **colaborativo y paralelo**, dividiendo el trabajo en tres fases bien definidas:

### 1️⃣ Análisis Exploratorio de Datos (EDA) — Individual
Cada miembro del equipo realizó su propio EDA de forma **independiente**, lo que permitió:
- Obtener **múltiples perspectivas** sobre la calidad y estructura del dataset fusionado.
- Detectar valores atípicos y distribuciones anómalas desde distintos ángulos.
- Construir hipótesis propias sobre las correlaciones entre variables sensoriales y la calidad final.

> Esta fase garantizó que ningún sesgo individual condicionara el análisis colectivo.

### 2️⃣ Entrenamiento de Modelos — Individual
Siguiendo la misma dinámica, cada integrante desarrolló y entrenó su **propio modelo de clasificación**, experimentando con:
- Distintas combinaciones de features sensoriales, físicas y categóricas.
- Diferentes algoritmos (Random Forest, XGBoost, Árbol de Decisión).
- Ajuste de hiperparámetros con `RandomizedSearchCV` y validación cruzada estratificada.

### 3️⃣ Selección de Modelos Finales — Colaborativa
Tras una **puesta en común de resultados** en el notebook `Comparativa_modelos.ipynb`, el equipo evaluó los tres modelos y seleccionó los dos enfoques más sólidos para integrar en la aplicación final: el **XGBoost de Camila** (mejor CV F1) y el **Random Forest de Jonathan** (mayor estabilidad).

---

## 🧠 Los Modelos

### 🥇 Modelo XGBoost — *Camila* (`models/Camila/model.pkl`)
> **"Máxima precisión con el menor overfitting"**

El modelo ganador de la comparativa. Entrenado con **XGBClassifier** sobre 15 features, logra el mejor equilibrio entre rendimiento en test y generalización.

| Métrica | Valor |
|---|---|
| **F1-Score Test** | **0.9552** |
| **CV F1 Media** | **0.9449 ± 0.0048** |
| **ROC-AUC** | **0.9904** |
| **Recall** | 0.9816 |
| **Overfitting (Δ F1)** | +0.0322 ✅ |
| **Nº Features** | 15 |

**Top 3 variables:** `Flavor`, `Aftertaste`, `Balance`

**¿Por qué XGBoost?** Su capacidad de capturar relaciones no lineales entre puntuaciones sensoriales, combinada con la regularización integrada, produce predicciones muy robustas ante lotes atípicos. El recall de 0.98 es especialmente valioso en este dominio: preferimos no perder ningún café Specialty.

---

### 🥈 Modelo Random Forest — *Jonathan* (`models/Jonathan/model.pkl`)
> **"Máxima estabilidad y menor varianza entre folds"**

Entrenado con **RandomForestClassifier** optimizado vía `RandomizedSearchCV` (`n_estimators=200`, `max_depth=15`, `max_features='log2'`, `class_weight='balanced'`).

| Métrica | Valor |
|---|---|
| **F1-Score Test** | **0.9477** |
| **CV F1 Media** | **0.9354 ± 0.0167** |
| **ROC-AUC** | **0.9741** |
| **Recall** | 0.9760 |
| **Overfitting (Δ F1)** | +0.0146 ✅ |
| **Nº Features** | 15 |

**Top 5 variables:** `Flavor` (0.212), `Aftertaste` (0.187), `Balance` (0.182), `Acidity` (0.136), `Body` (0.107)

**¿Por qué Random Forest?** Su menor varianza entre folds (±0.017 vs ±0.005 de XGBoost) lo convierte en la opción más segura cuando el lote de evaluación difiere mucho del de entrenamiento. El parámetro `class_weight='balanced'` garantiza un tratamiento equitativo de ambas clases.

---

### 🥉 Modelo XGBoost — *Juan* (`models/Juanma/model.pkl`)
> **"Menor overfitting absoluto del conjunto"**

| Métrica | Valor |
|---|---|
| **F1-Score Test** | 0.9390 |
| **CV F1 Media** | 0.9310 ± 0.0210 |
| **ROC-AUC** | 0.9700 |
| **Recall** | 0.9625 |
| **Overfitting (Δ F1)** | +0.0086 ✅ |
| **Nº Features** | 15 |

**Top 3 variables:** `Flavor` (0.395), `Balance` (0.259), `Aftertaste` (0.125)

---

### 📊 Comparativa de los Tres Modelos

| Ranking | Miembro | Algoritmo | F1 Test | ROC-AUC | CV F1 Media | CV F1 Std | Overfitting |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | **Camila** | XGBClassifier | **0.9552** | **0.9904** | **0.9449** | 0.0048 | +0.0322 ✅ |
| 🥈 2 | **Jonathan** | RandomForestClassifier | 0.9477 | 0.9741 | 0.9354 | 0.0167 | +0.0146 ✅ |
| 🥉 3 | **Juan** | XGBClassifier | 0.9390 | 0.9700 | 0.9310 | 0.0210 | +0.0086 ✅ |

> Los tres modelos superan el umbral de calidad F1 > 0.93 y presentan overfitting controlado. Ninguno supera Δ = 0.04.

---

## 📊 Hallazgos Principales del EDA

Los análisis exploratorios del equipo convergieron en los siguientes *insights* clave:

- **☕ Flavor** es el predictor más potente en los tres modelos, con importancias de entre 0.21 y 0.39. La puntuación de sabor es el factor que más discrimina entre cafés Specialty y No Specialty.

- **⚖️ Balance y Aftertaste** son los otros dos pilares del modelo. Los tres juntos — Flavor, Balance, Aftertaste — explican más del 50% del poder predictivo en todos los modelos.

- **🌍 Origen geográfico** influye de forma significativa: países como Ethiopia, Colombia y Guatemala presentan tasas de Specialty notablemente superiores a la media, lo que refleja terroir, variedad y tradición de procesado.

- **🏔️ Altitud** (`altitud_limpia`) aparece como variable relevante en el modelo de Jonathan (VIF = 13.5), lo que confirma la relación conocida entre altitud y complejidad aromática del café.

- **⚠️ Multicolinealidad alta** entre las puntuaciones sensoriales (Flavor, Aftertaste, Balance, Acidity, Body presentan VIF > 1000). No es crítica para Random Forest y XGBoost, pero invalida la Regresión Logística como modelo en este dataset.

- **📊 Dataset ligeramente desbalanceado:** 834 Specialty (55%) vs 678 No Specialty (45%) — manejable sin técnicas de resampling, gestionado con `class_weight='balanced'` en Random Forest.

---

## 🖥️ Interfaz de la Aplicación

La app se llama **Digital Q-Grader AI** y está construida con **Streamlit** con diseño personalizado en paleta verde café, tipografía **DM Sans / DM Mono** y CSS inyectado.

### Panel Lateral — Sidebar

El sidebar es el **centro de control** de la aplicación:

**`🔘 Selector de modelo`** — Radio button toggle que permite cambiar entre XGBoost (Camila) y Random Forest (Jonathan). La selección actualiza dinámicamente las métricas visibles en la zona de resultados.

**`📝 Formulario de evaluación del lote`** — Inputs para las características sensoriales y físicas del café:

| Input | Tipo | Rango |
|---|---|---|
| 🌍 País de origen | `selectbox` | 38 países |
| 🌸 Aroma | `slider` | 0 – 10 |
| ☕ Flavor | `slider` | 0 – 10 |
| 🍃 Aftertaste | `slider` | 0 – 10 |
| 🍋 Acidity | `slider` | 0 – 10 |
| 💪 Body | `slider` | 0 – 10 |
| ⚖️ Balance | `slider` | 0 – 10 |
| 🎨 Color | `selectbox` | Blue-Green, Green... |
| 💧 Moisture | `number_input` | 0 – 100% |
| + otras features | varios | — |

**`⚡ Botón "New Assessment"`** — Botón verde de ancho completo que lanza la clasificación y transforma la zona central con los resultados.

---

### Zona Principal — Resultado de Clasificación

Al pulsar el botón, se muestra el panel de resultados:

**`🎯 Tarjeta de certificación final`**

Muestra el veredicto **Specialty / No Specialty** con la probabilidad de confianza en formato grande (`DM Mono`), con código de colores:

| Resultado | Color | Significado |
|---|---|---|
| ✅ Specialty Coffee | Verde `#16A34A` | Supera el estándar SCA (≥ 82.5 pts) |
| ❌ No Specialty | Rojo `#DC2626` | Por debajo del umbral SCA |

**`📊 Métricas del modelo activo`** — Fila de tarjetas con F1-Score, CV F1 Media, ROC-AUC y estado de overfitting del modelo seleccionado.

**`🕸️ Coffee Profile Map`** (columna izquierda) — Gráfico radar de Matplotlib que muestra el perfil sensorial completo del lote: Aroma, Flavor, Aftertaste, Acidity, Body, Balance. Permite visualizar de un vistazo las fortalezas y debilidades del café evaluado.

**`📋 Sensory Analysis Report`** (columna derecha) — Barras de progreso para cada atributo sensorial con código de colores:

| Rango | Color |
|---|---|
| ≥ 7.5 | 🟢 Verde — excelente |
| 6.0 – 7.4 | 🟡 Ámbar — aceptable |
| < 6.0 | 🔴 Rojo — por debajo del estándar |

**`🕓 Pestaña Historial`** — Registro de todas las clasificaciones de la sesión con métricas de resumen (total Specialty, total No Specialty, confianza media) y tarjetas individuales por evaluación.

---

## 🎨 Diseño y Estilo

La aplicación usa CSS personalizado inyectado con `st.markdown(..., unsafe_allow_html=True)`:

- **Paleta:** verde café `#16A34A` como color dominante sobre fondo `#F8FAF9` (blanco roto cálido).
- **Sidebar:** fondo `#F0F7F3` con borde `#D1E8DA` — evoca el verde de las hojas de cafeto.
- **Tipografía:** `DM Sans` para texto corrido, `DM Mono` para valores numéricos y métricas.
- **Cards:** fondo blanco `#FFFFFF` con borde `#D1E8DA`, `border-radius: 16px` y sombra difusa verde.
- **Radio toggle:** selector personalizado que oculta los radio buttons nativos y simula un toggle pill.
- **Botón:** verde sólido con `box-shadow` verde y animación `translateY(-2px)` en hover.
- **Inputs:** fondo `#FFFFFF`, borde que resalta en verde al hacer focus con glow exterior.

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Herramientas |
|---|---|
| **Lenguaje** | Python 3.13 |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Manipulación de datos** | Pandas, NumPy |
| **Visualización** | Matplotlib, Seaborn |
| **Despliegue UI** | Streamlit |
| **Serialización de modelos** | Joblib (`.pkl`) |
| **Gestión de dependencias** | uv |

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/Bootcamp-IA-P6/Proyecto6_Grupo1.git
cd Proyecto6_Grupo1
```

### 2. Instalar dependencias
```bash
uv sync
```

### 3. Ejecutar la aplicación
```bash
uv run streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`. Selecciona el modelo en el sidebar, introduce las características del lote de café y pulsa **⚡ New Assessment**.

---

## 📁 Estructura del Proyecto

```
Proyecto6_Grupo1/
│
├── 📂 data/
│   ├── raw/
│   │   ├── arabica_data_cleaned.csv          # Dataset CQI histórico
│   │   └── coffee_quality.csv                # Dataset CQI mayo 2023
│   └── processed/
│       ├── Camila/
│       │   ├── coffee_cleaned.csv            # Dataset limpio — Camila
│       │   ├── coffee_combined_cleaned.csv   # Dataset combinado — Camila
│       │   └── coffee_preprocessed.csv       # Dataset ML-ready — Camila
│       ├── Jonathan/
│       │   ├── coffee_quality_fusion.csv     # Dataset fusionado — Jonathan
│       │   └── EDA_coffee_quality_fusion.csv # Dataset EDA — Jonathan
│       └── Juanma/
│           ├── coffee_quality_clean.csv      # Dataset limpio — Juanma
│           └── coffee_quality_fusion.csv     # Dataset fusionado — Juanma
│
├── 📂 models/
│   ├── Camila/
│   │   ├── model.pkl                         # XGBClassifier — 🥇 mejor modelo
│   │   ├── encoders.pkl                      # LabelEncoders
│   │   └── feature_names.pkl                 # Nombres de features
│   ├── Jonathan/
│   │   ├── model.pkl                         # RandomForestClassifier — 🥈
│   │   ├── encoders.pkl                      # LabelEncoders
│   │   ├── feature_names.pkl                 # Nombres de features
│   │   └── model_name.pkl                    # Nombre del modelo
│   └── Juanma/
│       └── model.pkl                         # XGBClassifier — 🥉
│
├── 📂 notebooks/
│   ├── Camila/
│   │   ├── 0_Fusion_datasets.ipynb           # Fusión de datasets — Camila
│   │   ├── 1_EDA.ipynb                       # EDA — Camila
│   │   ├── 02_preprocessing.ipynb            # Preprocesamiento — Camila
│   │   ├── 03_machine_learning.ipynb         # Modelado XGBoost — Camila 🥇
│   │   └── 04_Probar_Modelo.ipynb            # Validación del modelo — Camila
│   ├── Jonathan/
│   │   ├── 0_Fusion_datasets.ipynb           # Fusión de datasets — Jonathan
│   │   ├── 1_EDA.ipynb                       # EDA — Jonathan
│   │   └── 2_Modeling.ipynb                  # Modelado Random Forest — Jonathan 🥈
│   ├── Juanma/
│   │   ├── 0_Fusion_datasets.ipynb           # Fusión de datasets — Juanma
│   │   ├── 1_EDA.ipynb                       # EDA — Juanma
│   │   ├── 2_Modeling.ipynb                  # Modelado XGBoost — Juanma 🥉
│   │   └── Probar_modelo.ipynb               # Validación del modelo — Juanma
│   └── Comparativa_modelos.ipynb             # Comparativa final de los 3 modelos ⭐
│
├── app.py                                    # Aplicación Streamlit — Digital Q-Grader AI
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 👥 Equipo

Proyecto desarrollado en equipo como parte del aprendizaje de técnicas de Machine Learning aplicadas a datos reales del sector del café de especialidad.

| Desarrolladores | GitHub | LinkedIn |
|----------------|--------|----------|
| **Jonathan** | [GitHub](#) <!-- TODO: enlace GitHub Jonathan --> | [LinkedIn](#) <!-- TODO: enlace LinkedIn Jonathan --> |
| **Juan Manuel** | [GitHub](#) <!-- TODO: enlace GitHub Juanma --> | [LinkedIn](#) <!-- TODO: enlace LinkedIn Juanma --> |
| **Camila** | [GitHub](#) <!-- TODO: enlace GitHub Camila --> | [LinkedIn](#) <!-- TODO: enlace LinkedIn Camila --> |
| **JJ** | [GitHub](#) <!-- TODO: enlace GitHub JJ --> | [LinkedIn](#) <!-- TODO: enlace LinkedIn JJ --> |

**Bootcamp:** Inteligencia Artificial
**Organización:** Factoría F5
**Año:** 2026

---

<div align="center">
<sub>Hecho con ❤️ y mucho café · <i>porque todo buen modelo empieza con un buen Specialty ☕</i></sub>
</div>
