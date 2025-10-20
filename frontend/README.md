# Frontend (Streamlit) – Covertype

Cumple `seguimiento.txt` y está alineado con el backend real (`backend/main.py`, `backend/models/schema.py`).

- **Modelos:** XGBoost y Random Forest.
- **Inputs:**
  - Elevation, Aspect, Slope
  - Horizontal distance to Hydrology
  - Vertical distance to Hydrology
  - Horizontal distance to Roadways
  - Hillshade 9am, Hillshade Noon, Hillshade 3pm
  - Wilderness_Area_01..04 (one-hot)
  - Soil_Type_01..40 (one-hot)
  - Horizontal_Distance_To_Fire_Points (requerido por el schema del backend)
- **Output:** Cover Type (1..7)

## Estructura

- `frontend/app.py`: aplicación Streamlit con formulario y cliente HTTP.
- `frontend/requirements.txt`: dependencias del frontend.
- `frontend/.env.example`: ejemplo de variables para apuntar al backend.

## Variables de entorno (frontend)

El backend define los paths vía `.env` en la raíz del repo (ver `.env.example`). El frontend los lee así:

```
BACKEND_BASE=http://localhost:8000
XGBOOST_URL=/predict/xgboost
RANDOM_FOREST_URL=/predict/random_forest
```

Puedes crear `frontend/.env` copiando `frontend/.env.example` y ajustando `BACKEND_BASE` si es necesario.

## Endpoints consumidos

- `POST {BACKEND_BASE}{XGBOOST_URL}` → predicción con XGBoost.
- `POST {BACKEND_BASE}{RANDOM_FOREST_URL}` → predicción con Random Forest.

El payload se envía como un diccionario de features con los nombres EXACTOS de `backend/models/schema.py` (`CoverTypePayload`).

### Ejemplo de request

```json
{
  "Elevation": 3000,
  "Aspect": 100,
  "Slope": 10,
  "Horizontal_Distance_To_Hydrology": 100,
  "Vertical_Distance_To_Hydrology": 10,
  "Horizontal_Distance_To_Roadways": 500,
  "Hillshade_9am": 200,
  "Hillshade_Noon": 220,
  "Hillshade_3pm": 180,
  "Horizontal_Distance_To_Fire_Points": 0,
  "Wilderness_Area_01": 1, "Wilderness_Area_02": 0, "Wilderness_Area_03": 0, "Wilderness_Area_04": 0,
  "Soil_Type_01": 0, ..., "Soil_Type_40": 0
}
```

### Ejemplo de response

```json
{ "cover_type": 2 }
```

> Nota: El endpoint de Random Forest puede responder **501 Not Implemented** hasta que se complete su implementación en el backend.

## Ejecución local

1. (Recomendado) Crear y activar un entorno virtual:
   ```bash
   python3 -m venv frontend/env
   source frontend/env/bin/activate
   ```
2. Instalar dependencias:
   ```bash
   pip install -r frontend/requirements.txt
   ```
3. Configurar variables de entorno del frontend (opcional si usas las de ejemplo):
   ```bash
   cp frontend/.env.example frontend/.env
   # Edita BACKEND_BASE si tu backend no corre en http://localhost:8000
   ```
4. Levantar la app:
   ```bash
   streamlit run frontend/app.py
   ```
5. Abre el enlace que Streamlit imprime (ej. http://localhost:8501).

## Observaciones

- Los nombres de features usan ceros a la izquierda (`Wilderness_Area_01`, `Soil_Type_01`...), según `backend/models/schema.py`.
- El formulario del frontend genera one-hot correctas en esas claves.
- Si el backend cambia los paths o el esquema, actualiza las variables en `frontend/.env`.
