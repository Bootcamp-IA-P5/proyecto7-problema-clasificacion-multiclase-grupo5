# backend/services/ml_models.py

import os
import joblib
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
XGBOOST_MODEL_PATH = os.getenv("XGBOOST_MODEL")
XGBOOST_SCALER_PATH = os.getenv("XGBOOST_SCALER")
RANDOM_FOREST_PATH = os.getenv("RANDOM_FOREST_MODEL")
# --- Global Model Storage ---
# Stores the loaded models and scalers to avoid re-loading on every request
MODELS = {
    "xgboost_model": None,
    "xgboost_scaler": None,
    "random_forest": None, 
}

def load_models():
    """
    Loads the trained models and scalers into memory.
    This function should be called once when the application starts.
    """
    print("--- Loading ML Models ---")
    try:
        # Load XGBoost Model using joblib
        # joblib.load takes the file path directly
        MODELS["xgboost_model"] = joblib.load(XGBOOST_MODEL_PATH)
        print(f"✅ XGBoost Model loaded from: {XGBOOST_MODEL_PATH}")

        # Load XGBoost Scaler using joblib
        MODELS["xgboost_scaler"] = joblib.load(XGBOOST_SCALER_PATH)
        print(f"✅ XGBoost Scaler loaded from: {XGBOOST_SCALER_PATH}"),

        MODELS["random_forest"] = joblib.load(RANDOM_FOREST_PATH)
        print(f"✅ Random Forest Model loaded from: {RANDOM_FOREST_PATH}"),

    except FileNotFoundError as e:
        print(f"❌ ERROR: Model or Scaler file not found. Check your .env paths.")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")

def predict_xgboost(data: dict) -> int:
    """
    Performs inference using the loaded XGBoost model.

    Args:
        data: A dictionary containing the feature values.

    Returns:
        The predicted Cover_Type as an integer.
    """
    model = MODELS.get("xgboost_model")
    scaler = MODELS.get("xgboost_scaler")

    if model is None or scaler is None:
        raise RuntimeError("XGBoost model or scaler is not loaded.")

    # Convert input data dictionary into a NumPy array, maintaining feature order
    # The order is implicitly enforced by Pydantic's BaseModel iteration
    feature_values = list(data.values())
    features_array = np.array(feature_values).reshape(1, -1)

    # 1. Scale the features
    scaled_features = scaler.transform(features_array)

    # 2. Make the prediction
    # XGBoost models typically return a single prediction array
    prediction = model.predict(scaled_features)[0]

    # Convert to the required output format (integer)
    return int(prediction)

# --- Future function placeholder for Random Forest ---
def predict_random_forest(data: dict) -> int:
    """
    Performs inference using the loaded Random_forest model.

    Args:
        data: A dictionary containing the feature values.

    Returns:
        The predicted Cover_Type as an integer.
    """
    model = MODELS.get("random_forest")
    # Convert input data dictionary into a NumPy array, maintaining feature order
    # The order is implicitly enforced by Pydantic's BaseModel iteration
    features_values = list(data.values())
    features_array = np.array(features_values).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    return int(prediction)