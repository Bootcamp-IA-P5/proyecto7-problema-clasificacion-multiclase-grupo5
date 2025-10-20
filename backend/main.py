# backend/main.py

import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path

from backend.models.schema import CoverTypePayload, CoverTypeResponse
from backend.services import ml_models


# Use asynccontextmanager to manage application startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup, loads the ML models into memory.
    """
    ml_models.load_models()
    yield # Application serves requests
    # Run on shutdown (optional cleanup)

# Initialize the FastAPI app with the lifespan context manager
app = FastAPI(
    title="Forest Cover Type Prediction API",
    description="API for predicting forest cover type using ML models (XGBoost, Random Forest).",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.get("/", include_in_schema=False)
def read_root():
    """Basic health check endpoint."""
    return {"message": "Welcome to the ML Prediction API. Check /docs for endpoints."}

@app.post(
    os.getenv("XGBOOST_URL"),
    response_model=CoverTypeResponse,
    summary="Predict Cover Type using XGBoost"
)
def predict_cover_type_xgboost(payload: CoverTypePayload):
    """
    Predicts the forest Cover_Type (a numeric class) based on the input features
    using the trained XGBoost model.
    """
    try:
        # Convert the Pydantic model payload to a dictionary for the service
        prediction = ml_models.predict_xgboost(payload.model_dump())
        
        return CoverTypeResponse(cover_type=prediction)
    
    except RuntimeError as e:
        # Catch errors related to un-loaded models
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Catch other potential errors during prediction
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


def predict_cover_type_xgboost(payload: CoverTypePayload):
    """
    Predicts the forest Cover_Type (a numeric class) based on the input features
    using the trained XGBoost model.
    """
    try:
        # Convert the Pydantic model payload to a dictionary for the service
        prediction = ml_models.predict_xgboost(payload.model_dump())
        
        return CoverTypeResponse(cover_type=prediction)
    
    except RuntimeError as e:
        # Catch errors related to un-loaded models
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Catch other potential errors during prediction
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post(
    os.getenv("RANDOM_FOREST_URL"),
    response_model=CoverTypeResponse,
    summary="Predict Cover Type using Random Forest"
)

def predict_cover_type_random_forest(payload: CoverTypePayload):
    """
    [Placeholder] Endpoint to predict the forest Cover_Type using the
    trained Random Forest model.
    """
    # Current implementation raises NotImplementedError
    try:
        ml_models.predict_random_forest(payload.model_dump())
    except NotImplementedError:
        raise HTTPException(
            status_code=501, 
            detail="Random Forest prediction endpoint is not yet implemented."
        )
    except Exception as e:
        # Fallback error handling
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")