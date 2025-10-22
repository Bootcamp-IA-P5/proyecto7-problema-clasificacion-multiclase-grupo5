import os
import json
from typing import Dict, Any, List

import requests
import streamlit as st
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

WILDERNESS_NAMES = [
    "Wilderness_Area_01",
    "Wilderness_Area_02",
    "Wilderness_Area_03",
    "Wilderness_Area_04",
]

SOIL_NAMES = [f"Soil_Type_{i:02d}" for i in range(1, 41)]

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
    "Frontend mínimo (Streamlit) siguiendo seguimiento.txt. Selecciona modelo, ingresa variables y obtén la predicción."
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

        st.write("Endpoints:")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("XGBoost:")
            st.code(MODEL_ENDPOINTS["xgboost"], language="text")
            st.write(f"Existe: {'✅' if conn['xgboost']['exists'] else '❌'} | Status: {conn['xgboost']['status']} | Allow: {conn['xgboost'].get('allow','')}")
        with col_b:
            st.write("Random Forest:")
            st.code(MODEL_ENDPOINTS["random_forest"], language="text")
            st.write(f"Existe: {'✅' if conn['random_forest']['exists'] else '❌'} | Status: {conn['random_forest']['status']} | Allow: {conn['random_forest'].get('allow','')}")

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
    "Wilderness Area (1-4)", options=[1, 2, 3, 4], index=0,
    help="Se convertirá a one-hot Wilderness_Area_01..04"
)
soil_sel = st.number_input(
    "Soil Type (1-40)", min_value=1, max_value=40, value=10, step=1,
    help="Se convertirá a one-hot Soil_Type_01..40"
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

left, right = st.columns([1, 2])

with left:
    st.write("\n")
    predict_btn = st.button("Predecir cover type", type="primary")

with right:
    st.caption("Se envían al backend con las claves EXACTAS del schema (Wilderness_Area_01..04, Soil_Type_01..40).")

# Resultado
if predict_btn:
    try:
        with st.spinner("Consultando backend..."):
            result = call_predict(model_choice, features)
        st.success("Predicción recibida del backend")
        st.subheader("Resultado")
        st.json(result)
        # Presentación simplificada
        cover = result.get("cover_type")
        proba = result.get("proba")
        if cover is not None:
            st.metric("Cover Type", cover)
        if isinstance(proba, list) and len(proba) in (7,):
            st.bar_chart({"Clase": list(range(1, 8)), "Prob": proba}, x="Clase", y="Prob")
    except requests.HTTPError as e:
        st.error(f"Error HTTP del backend: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"No fue posible obtener la predicción: {e}")

# Información de los endpoints y cómo se usan (para trazabilidad)
st.markdown("""
### Endpoints del backend utilizados
- **POST XGBoost**: `${BACKEND_BASE}${XGBOOST_URL}`
- **POST Random Forest**: `${BACKEND_BASE}${RANDOM_FOREST_URL}` (puede devolver 501 si no está implementado)

Contrato de request: features directos según `backend/models/schema.py`.
Contrato de respuesta: `{ "cover_type": int }`.
""")
