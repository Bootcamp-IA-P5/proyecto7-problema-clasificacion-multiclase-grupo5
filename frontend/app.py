import os
import json
from typing import Dict, Any, List

import requests
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# =============================
# Frontend Streamlit para Covertype
# Requisitos (seguimiento.txt):
# - Dos modelos a elegir: XGBoost y Random Forest
# - Inputs:
#   Elevation, Aspect, Slope,
#   Horizontal distance to Hydrology,
#   Vertical distance to Hydrology,
#   Horizontal distance to Roadways,
#   Hillshade 9am, Hillshade Noon, Hillshade 3pm,
#   4 tipos de wilderness areas,
#   40 tipos de soil
# - Salida: cover type (7 valores)
# =============================

# Carga de variables de entorno (permite configurar el backend sin tocar código)
load_dotenv() # carga las variables de entorno
# El backend define los paths en .env de la raíz: XGBOOST_URL y RANDOM_FOREST_URL
# BACKEND_BASE es el host:puerto (por compatibilidad también admite BACKEND_URL)
BACKEND_BASE = os.getenv("BACKEND_BASE", os.getenv("BACKEND_URL", "http://localhost:8000"))
XGBOOST_URL = os.getenv("XGBOOST_URL", "/predict/xgboost")
RANDOM_FOREST_URL = os.getenv("RANDOM_FOREST_URL", "/predict/random_forest")

MODEL_ENDPOINTS = {
    "xgboost": f"{BACKEND_BASE}{XGBOOST_URL}",
    "random_forest": f"{BACKEND_BASE}{RANDOM_FOREST_URL}",
}

# =============================
# Utilidades de conectividad (Paso 1)
# =============================

def check_health(base_url: str) -> Dict[str, Any]:
    """
    Realiza un health check simple al backend.
    - Hace GET al root '/'
    - Mide latencia aproximada
    - Devuelve dict con { ok: bool, status: int|None, latency_ms: float|None, body: any }
    """
    import time
    url = base_url.rstrip("/") + "/"
    t0 = time.perf_counter()
    try:
        resp = requests.get(url, timeout=5)
        dt = (time.perf_counter() - t0) * 1000
        return {"ok": resp.ok, "status": resp.status_code, "latency_ms": round(dt, 1), "body": safe_json(resp)}
    except Exception as e:
        return {"ok": False, "status": None, "latency_ms": None, "body": str(e)}


def check_endpoint_exists(url: str) -> Dict[str, Any]:
    """
    Verifica si un endpoint existe intentando OPTIONS (no modifica estado).
    - FastAPI suele responder 200 con métodos permitidos si la ruta existe.
    - Si no existe, normalmente 404 o error de conexión.
    """
    try:
        r = requests.options(url, timeout=5)
        return {"exists": r.status_code < 400, "status": r.status_code, "allow": r.headers.get("allow", "")}
    except Exception as e:
        return {"exists": False, "status": None, "error": str(e)}


def safe_json(resp: requests.Response):
    """Intenta parsear JSON, si falla devuelve texto plano acotado."""
    try:
        return resp.json()
    except Exception:
        txt = resp.text
        return txt if len(txt) < 500 else txt[:500] + "..."

# =============================
# Definición de helpers
# =============================

# Claves EXACTAS requeridas por el backend (no cambiar)
WILDERNESS_NAMES = [
    "Wilderness_Area_01",
    "Wilderness_Area_02",
    "Wilderness_Area_03",
    "Wilderness_Area_04",
]

SOIL_NAMES = [f"Soil_Type_{i:02d}" for i in range(1, 41)]

# Etiquetas amigables para UI (extraídas de covtype.txt)
WILDERNESS_LABELS = {
    1: "Rawah Wilderness Area",
    2: "Neota Wilderness Area",
    3: "Comanche Peak Wilderness Area",
    4: "Cache la Poudre Wilderness Area",
}

SOIL_LABELS = {
    1: "Cathedral family - Rock outcrop complex, extremely stony.",
    2: "Vanet - Ratake families complex, very stony.",
    3: "Haploborolis - Rock outcrop complex, rubbly.",
    4: "Ratake family - Rock outcrop complex, rubbly.",
    5: "Vanet family - Rock outcrop complex complex, rubbly.",
    6: "Vanet - Wetmore families - Rock outcrop complex, stony.",
    7: "Gothic family.",
    8: "Supervisor - Limber families complex.",
    9: "Troutville family, very stony.",
    10: "Bullwark - Catamount families - Rock outcrop complex, rubbly.",
    11: "Bullwark - Catamount families - Rock land complex, rubbly.",
    12: "Legault family - Rock land complex, stony.",
    13: "Catamount family - Rock land - Bullwark family complex, rubbly.",
    14: "Pachic Argiborolis - Aquolis complex.",
    15: "Unspecified in the USFS Soil and ELU Survey.",
    16: "Cryaquolis - Cryoborolis complex.",
    17: "Gateview family - Cryaquolis complex.",
    18: "Rogert family, very stony.",
    19: "Typic Cryaquolis - Borohemists complex.",
    20: "Typic Cryaquepts - Typic Cryaquolls complex.",
    21: "Typic Cryaquolls - Leighcan family, till substratum complex.",
    22: "Leighcan family, till substratum, extremely bouldery.",
    23: "Leighcan family, till substratum - Typic Cryaquolls complex.",
    24: "Leighcan family, extremely stony.",
    25: "Leighcan family, warm, extremely stony.",
    26: "Granile - Catamount families complex, very stony.",
    27: "Leighcan family, warm - Rock outcrop complex, extremely stony.",
    28: "Leighcan family - Rock outcrop complex, extremely stony.",
    29: "Como - Legault families complex, extremely stony.",
    30: "Como family - Rock land - Legault family complex, extremely stony.",
    31: "Leighcan - Catamount families complex, extremely stony.",
    32: "Catamount family - Rock outcrop - Leighcan family complex, extremely stony.",
    33: "Leighcan - Catamount families - Rock outcrop complex, extremely stony.",
    34: "Cryorthents - Rock land complex, extremely stony.",
    35: "Cryumbrepts - Rock outcrop - Cryaquepts complex.",
    36: "Bross family - Rock land - Cryumbrepts complex, extremely stony.",
    37: "Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.",
    38: "Leighcan - Moran families - Cryaquolls complex, extremely stony.",
    39: "Moran family - Cryorthents - Leighcan family complex, extremely stony.",
    40: "Moran family - Cryorthents - Rock land complex, extremely stony.",
}

# Nombres de clases (para tarjeta de resultado)
# Nota: Los índices ahora van de 0 a 6 según el nuevo formato
COVER_TYPE_NAMES = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz",
}

# Descripciones completas de cada tipo de cobertura forestal
COVER_TYPE_DESCRIPTIONS = {
    0: "Abeto Rojo / Abeto de Colorado - Especies de coníferas que crecen en altitudes elevadas",
    1: "Pino Contorto - Pino resistente que domina en áreas quemadas y de alta montaña",
    2: "Pino Ponderosa - Pino de corteza gruesa característico de zonas secas y soleadas",
    3: "Álamo del Río / Sauce - Especies caducifolias que crecen cerca de cuerpos de agua",
    4: "Álamo Temblón - Árbol caducifolio con hojas que tiemblan, común en áreas perturbadas",
    5: "Abeto Douglas - Conífera de crecimiento rápido común en el noroeste de América",
    6: "Krummholz - Vegetación enana y deformada que crece en la línea de árboles alpina",
}

# Nota importante sobre features:
# Los modelos entrenados en los notebooks suelen esperar el esquema del dataset UCI Covertype,
# que incluye también "Horizontal_Distance_To_Fire_Points". Aunque en seguimiento.txt no aparece,
# lo exponemos como ajuste avanzado para compatibilidad. Si tu backend lo imputa por defecto,
# puedes dejarlo en 0.
CONTINUOUS_FEATURES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    # Avanzado (dataset original)
    "Horizontal_Distance_To_Fire_Points",
]

# Endpoints consumidos (derivados de .env raíz):
#   POST {BACKEND_BASE}{XGBOOST_URL}
#   POST {BACKEND_BASE}{RANDOM_FOREST_URL}
# Body: features directos según backend/models/schema.py (CoverTypePayload)
# Resp.: { "cover_type": int }


def get_models_from_backend() -> List[str]:
    # El backend no publica /models; devolvemos los requeridos por seguimiento.txt
    return ["xgboost", "random_forest"]


def call_predict(model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
    # Enviar los features directamente al endpoint del modelo (CoverTypePayload)
    url = MODEL_ENDPOINTS[model_name]
    r = requests.post(url, json=features, timeout=30)
    if r.status_code == 501:
        # Random Forest aún no implementado en backend
        return {"error": "Random Forest no implementado (501)", "detail": r.text}
    r.raise_for_status()
    return r.json()


# =============================
# UI
# =============================

st.set_page_config(page_title="Covertype – XGBoost / RandomForest", page_icon="🌲", layout="wide")

st.title("Clasificación de Cover Type")
st.caption(
    "Selecciona modelo, ingresa variables y obtén la predicción."
)

# Panel lateral: modelo + config
with st.sidebar:
    st.header("Configuración")
    available_models = get_models_from_backend()
    model_choice = st.selectbox("Modelo", available_models, index=0, help="Elige entre XGBoost y Random Forest")

    st.subheader("Backend")
    st.text_input("BACKEND_BASE", value=BACKEND_BASE, disabled=True, help="Host:puerto del backend (env)")
    st.text_input("XGBOOST_URL", value=XGBOOST_URL, disabled=True, help="Path del endpoint XGBoost (env)")
    st.text_input("RANDOM_FOREST_URL", value=RANDOM_FOREST_URL, disabled=True, help="Path del endpoint RF (env)")

    # --- Paso 1: Prueba de conexión y endpoints ---
    st.markdown("---")
    st.subheader("Estado del backend")
    
    if "connectivity" not in st.session_state:
        st.session_state.connectivity = None  # para persistir resultado entre reruns

    if st.button("Probar conexión"):
        # 1) Health check a '/'
        health = check_health(BACKEND_BASE)
        # 2) Verificación de endpoints por OPTIONS
        xgb_info = check_endpoint_exists(MODEL_ENDPOINTS["xgboost"]) 
        rf_info = check_endpoint_exists(MODEL_ENDPOINTS["random_forest"]) 
        st.session_state.connectivity = {
            "health": health,
            "xgboost": xgb_info,
            "random_forest": rf_info,
        }

    conn = st.session_state.connectivity
    if conn:
        ok = conn["health"]["ok"]
        badge = "🟢 OK" if ok else "🔴 FALLA"
        st.write(f"Health: {badge} | Status: {conn['health']['status']} | Latencia: {conn['health']['latency_ms']} ms")
        st.caption(f"Respuesta: {conn['health']['body']}")

# Inputs principales
st.subheader("Variables de entrada")

col1, col2, col3 = st.columns(3)

with col1:
    Elevation = st.number_input("Elevation", min_value=0, max_value=5000, value=3000, step=1)
    Aspect = st.number_input("Aspect", min_value=0, max_value=360, value=100, step=1)
    Slope = st.number_input("Slope", min_value=0, max_value=90, value=10, step=1)

with col2:
    Horizontal_Distance_To_Hydrology = st.number_input(
        "Horizontal distance to Hydrology", min_value=0, max_value=10000, value=100, step=1
    )
    Vertical_Distance_To_Hydrology = st.number_input(
        "Vertical distance to Hydrology", min_value=-1000, max_value=1000, value=10, step=1
    )
    Horizontal_Distance_To_Roadways = st.number_input(
        "Horizontal distance to Roadways", min_value=0, max_value=100000, value=500, step=1
    )

with col3:
    Hillshade_9am = st.number_input("Hillshade 9am", min_value=0, max_value=255, value=200, step=1)
    Hillshade_Noon = st.number_input("Hillshade Noon", min_value=0, max_value=255, value=220, step=1)
    Hillshade_3pm = st.number_input("Hillshade 3pm", min_value=0, max_value=255, value=180, step=1)

# Wilderness y Soil: input como selección única y convertimos a one-hot (tal como esperan los modelos entrenados)
st.markdown("---")
st.subheader("Zonas y Suelos")

wilderness_sel = st.selectbox(
    "Wilderness Area",
    options=[1, 2, 3, 4],
    index=0,
    format_func=lambda i: f"{i:02d} – {WILDERNESS_LABELS[i]}",
    help="Se convertirá a one-hot Wilderness_Area_01..04",
)
soil_sel = st.selectbox(
    "Soil Type",
    options=list(range(1, 41)),
    index=9,  # 10 por defecto
    format_func=lambda i: f"{i:02d} – {SOIL_LABELS[i]}",
    help="Se convertirá a one-hot Soil_Type_01..40",
)

# Campo requerido por el backend (schema FEATURES)
Horizontal_Distance_To_Fire_Points = st.number_input(
    "Horizontal distance to Fire Points", min_value=0, max_value=100000, value=0, step=1,
    help="Requerido por el backend. Si no estás seguro, deja 0."
)

# Construcción del diccionario de features como espera el backend (por nombre),
# evitando dependencia del orden de columnas.
features: Dict[str, Any] = {
    "Elevation": Elevation,
    "Aspect": Aspect,
    "Slope": Slope,
    "Horizontal_Distance_To_Hydrology": Horizontal_Distance_To_Hydrology,
    "Vertical_Distance_To_Hydrology": Vertical_Distance_To_Hydrology,
    "Horizontal_Distance_To_Roadways": Horizontal_Distance_To_Roadways,
    "Hillshade_9am": Hillshade_9am,
    "Hillshade_Noon": Hillshade_Noon,
    "Hillshade_3pm": Hillshade_3pm,
    "Horizontal_Distance_To_Fire_Points": Horizontal_Distance_To_Fire_Points,
}

# One-hot wilderness
for i, name in enumerate(WILDERNESS_NAMES, start=1):
    features[name] = 1 if i == wilderness_sel else 0

# One-hot soil
for i, name in enumerate(SOIL_NAMES, start=1):
    features[name] = 1 if i == soil_sel else 0

st.markdown("---")

# Botón de predicción y resultado (con feedback básico)
left, right = st.columns([1, 2])
with left:
    st.write("\n")
    predict_btn = st.button("Predecir cover type", type="primary")
with right:
    st.caption("Se enviarán los valores con las claves EXACTAS del schema (Wilderness_Area_01..04, Soil_Type_01..40).")

if predict_btn:
    # Validaciones suaves previas
    if Slope > 60:
        st.warning("Slope > 60° es poco común en el dataset")
    if Elevation < 1000 or Elevation > 4500:
        st.warning("Elevation fuera de rango típico (1000–4500 m)")

    try:
        with st.spinner("Consultando backend..."):
            result = call_predict(model_choice, features)
        if "error" in result:
            st.warning(result.get("error"))
            st.text(result.get("detail", ""))
        else:
            st.success("✅ Predicción recibida del backend")
            
            # Obtener datos de la respuesta
            cover_type = result.get("cover_type")
            percentages = result.get("percentages", [])
            
            if cover_type is not None and percentages:
                # Mostrar el tipo de cobertura principal (primera fila)
                st.markdown("### 🌲 Tipo de cubierta forestal")
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Tarjeta con el tipo de cobertura principal usando componentes nativos de Streamlit
                    cover_name = COVER_TYPE_NAMES.get(cover_type, "Desconocido")
                    cover_description = COVER_TYPE_DESCRIPTIONS.get(cover_type, "Descripción no disponible")

                    with st.container():
                        st.markdown(f"###    {cover_name}")
                        st.caption(cover_description)

                        # Crear columnas para el código y confianza
                        code_col, conf_col = st.columns(2)

                        with code_col:
                            st.metric(
                                label="Código",
                                value=f"{cover_type}",
                                help="Código numérico del tipo de cobertura"
                            )

                        with conf_col:
                            if 0 <= cover_type < len(percentages):
                                confidence = percentages[cover_type]
                                st.metric(
                                    label="Confianza",
                                    value=f"{confidence:.1f}%",
                                    help=f"Probabilidad asignada por el modelo {model_choice}"
                                )

                        # Separador visual
                        st.divider()

                # Mostrar gráfico de barras (segunda fila, ancho completo)
                st.markdown("### 📊 Distribución de probabilidades")
                st.markdown("---")

                # Crear gráfico de barras con Matplotlib
                fig, ax = plt.subplots(figsize=(14, 6))

                # Preparar datos para el gráfico con nombres más descriptivos pero compactos
                full_labels = [COVER_TYPE_NAMES.get(i, str(i)) for i in range(len(percentages))]
                compact_labels = [name.split('/')[0] if '/' in name else name for name in full_labels]
                x = np.arange(len(full_labels))

                # Crear barras
                bars = ax.bar(x, percentages, color='#1f77b4', alpha=0.8)

                # Resaltar la barra del tipo predicho
                if 0 <= cover_type < len(bars):
                    bars[cover_type].set_color('#2ca02c')  # Verde para la predicción
                    bars[cover_type].set_alpha(1.0)  # Más opaca para resaltar

                # Añadir etiquetas y título
                ax.set_ylabel('Probabilidad (%)', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(compact_labels, rotation=30, ha='right', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3, axis='y')

                # Añadir etiquetas con los valores y nombres completos
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    # Mostrar porcentaje arriba de la barra
                    ax.text(
                        bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%',
                        ha='center', va='bottom',
                        fontsize=11,
                        fontweight='bold'
                    )

                # Ajustar diseño
                plt.tight_layout()

                # Mostrar el gráfico en Streamlit
                st.pyplot(fig)

                # Mostrar datos completos en una sección colapsable
                with st.expander("Ver datos completos de la respuesta"):
                    st.json(result)
            else:
                st.warning("El formato de la respuesta no es el esperado")
    except requests.HTTPError as e:
        st.error(f"Error HTTP del backend: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"No fue posible obtener la predicción: {e}")

# Nota de trazabilidad de endpoints (se mantiene)

# Información de los endpoints y cómo se usan (para trazabilidad)
st.markdown("""
### Endpoints del backend utilizados
- **POST XGBoost**: `${BACKEND_BASE}${XGBOOST_URL}`
- **POST Random Forest**: `${BACKEND_BASE}${RANDOM_FOREST_URL}` (puede devolver 501 si no está implementado)

Contrato de request: features directos según `backend/models/schema.py`.
Contrato de respuesta: 
```json
{
  "cover_type": 3,
  "percentages": [14, 22, 32, 3.5, 3.5, 25, 0]
}
```
Donde `cover_type` es un número de 0 a 6 y `percentages` es un array de 7 elementos con las probabilidades de cada clase.
""")
