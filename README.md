# Proyecto 7 · Clasificación Multiclase · Grupo 5
¡Hola! Somos el Grupo 5 de F5. Nuestro equipo está compuesto por: Oscar Rodríguez, quien desempeña el rol de SCRUM Master; Maribel Gutiérrez, nuestra Product Owner; Jabeich Benavides, encargado del Backend; y Anthony Caceda, el Frontend maravilla.


## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de machine learning capaz de resolver un problema del mundo real utilizando algoritmos de clasificación multiclase. A lo largo del proyecto, se aplican técnicas de análisis y visualización de datos, preprocesamiento, construcción de modelos supervisados y evaluación de resultados.

La clasificación multiclase es un enfoque de aprendizaje supervisado en el que cada instancia se asigna a una única clase entre tres o más posibles. A diferencia de la clasificación binaria, este tipo de modelos deben distinguir entre múltiples categorías excluyentes.


## Estructura del proyecto
```
.
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schema.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── ml_models.py
│   ├── notebooks/
│   │   ├── EDA.ipynb
│   │   ├── Random_Forest_Model.ipynb
│   │   ├── forest_covertype_xgboost.ipynb
│   │   ├── Adaboost_investigacion.ipynb
│   │   └── stacking.ipynb
│   ├── test/
│   └── requirements.txt
├── frontend/
│   └── __init__.py
├── resources/
│   ├── __init__.py
│   └── models/
│       ├── random_forest_model.pkl
│       ├── xgboost_model.pkl
│       └── xgboost_scaler.pkl
│   
├── .env.example
├── .gitignore
└── README.md
```

## Estado actual
- **EDA**: `backend/notebooks/EDA.ipynb` con análisis exploratorio.
- **Modelado**: notebooks de `Random_Forest`, `forest_covertype_xgboost`, `Adaboost_investigacion` y `stacking`.
- **API FastAPI**: `backend/main.py` expone endpoints de predicción para XGBoost y Random Forest.
- **Esquemas**: `backend/models/schema.py` define `CoverTypePayload` y `CoverTypeResponse`.
- **Servicio de modelos**: `backend/services/ml_models.py` gestiona carga y predicción.
- **Artefactos de modelo** en `resources/models/`:
  - `xgboost_model.pkl` y `xgboost_scaler.pkl`.
  - `random_forest_model.pkl`.
- **Plantillas de issues** en `.github/ISSUE_TEMPLATE/`.

## Requisitos
- Python 3.x
- Dependencias del backend en `backend/requirements.txt`.

Instalación rápida (entorno virtual recomendado):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r backend/requirements.txt
```

## Uso (notebooks)
```bash
pip install jupyterlab  # si no lo tienes
jupyter lab
```
Abrir y ejecutar los notebooks en `backend/notebooks/`.

## API (backend)

- **Variables de entorno**: copiar `.env.example` a `.env` en la raíz del proyecto.
- **Ejecución**:
```bash
uvicorn backend.main:app --reload
```
- **Documentación interactiva**: abrir `http://127.0.0.1:8000/docs`.

- **Endpoints** (configurados por variables en `.env`):
  - Base: `BASE_URL=/predict`
  - XGBoost: `XGBOOST_URL=${BASE_URL}/xgboost`
  - Random Forest: `RANDOM_FOREST_URL=${BASE_URL}/random_forest`

- **Payload**: `CoverTypePayload` con 54 características numéricas según `backend/models/schema.py`.

- **Ejemplo (curl)**:
```bash
curl -X POST \
  http://127.0.0.1:8000/predict/xgboost \
  -H "Content-Type: application/json" \
  -d '{
    "Elevation": 2596,
    "Aspect": 51,
    "Slope": 3,
    "Horizontal_Distance_To_Hydrology": 258,
    "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 510,
    "Hillshade_9am": 221,
    "Hillshade_Noon": 232,
    "Hillshade_3pm": 148,
    "Horizontal_Distance_To_Fire_Points": 6279,
    "Wilderness_Area_01": 1,
    "Wilderness_Area_02": 0,
    "Wilderness_Area_03": 0,
    "Wilderness_Area_04": 0,
    "Soil_Type_01": 0,
    "Soil_Type_02": 0,
    "Soil_Type_03": 0,
    "Soil_Type_04": 0,
    "Soil_Type_05": 0,
    "Soil_Type_06": 0,
    "Soil_Type_07": 0,
    "Soil_Type_08": 0,
    "Soil_Type_09": 0,
    "Soil_Type_10": 0,
    "Soil_Type_11": 0,
    "Soil_Type_12": 0,
    "Soil_Type_13": 0,
    "Soil_Type_14": 0,
    "Soil_Type_15": 0,
    "Soil_Type_16": 0,
    "Soil_Type_17": 0,
    "Soil_Type_18": 0,
    "Soil_Type_19": 0,
    "Soil_Type_20": 0,
    "Soil_Type_21": 0,
    "Soil_Type_22": 0,
    "Soil_Type_23": 0,
    "Soil_Type_24": 0,
    "Soil_Type_25": 0,
    "Soil_Type_26": 0,
    "Soil_Type_27": 0,
    "Soil_Type_28": 0,
    "Soil_Type_29": 0,
    "Soil_Type_30": 0,
    "Soil_Type_31": 0,
    "Soil_Type_32": 0,
    "Soil_Type_33": 0,
    "Soil_Type_34": 0,
    "Soil_Type_35": 0,
    "Soil_Type_36": 0,
    "Soil_Type_37": 0,
    "Soil_Type_38": 0,
    "Soil_Type_39": 0,
    "Soil_Type_40": 0
  }'
```

## Próximos pasos sugeridos
- Exponer el mejor modelo como servicio (API en `backend/`).
- Definir flujo de entrenamiento/inferencia reproducible (scripts o pipeline).
- Integrar `frontend/` con la API para inferencia.
- Añadir validación, tests y CI.


## Convenciones
- Ramas: trabajo en feature branches y merge hacia `development` mediante PR/MR.