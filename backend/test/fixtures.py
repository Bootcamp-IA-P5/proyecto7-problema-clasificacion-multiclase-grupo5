import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.test.test_models import CoverTypePayload
import os

# The URL is needed for post calls
PREDICT_XGBOOST_URL = os.getenv("XGBOOST_URL", "/api/v1/predict/cover_type/xgboost")
PREDICT_RANDOM_FOREST_URL = os.getenv("RANDOM_FOREST_URL", "/api/v1/predict/cover_type/random_forest")

@pytest.fixture(scope="session")
def client() -> TestClient:
    """Fixture to provide a TestClient instance for the FastAPI app."""
    return TestClient(app)

@pytest.fixture(scope="session")
def predict_xgboost_url() -> str:
    """Fixture to provide the prediction endpoint URL for xgboost."""
    return PREDICT_XGBOOST_URL

@pytest.fixture(scope="session")
def predict_random_forest_url() -> str:
    """Fixture to provide the prediction endpoint URL for random_forest."""
    return PREDICT_RANDOM_FOREST_URL

@pytest.fixture(scope="session")
def valid_payload() -> dict:
    """Fixture to provide a standard valid payload dictionary."""
    # We use the Pydantic model to generate a valid dictionary representation
    return CoverTypePayload().model_dump()

# Define the absolute path to the function we need to mock
# This path must reflect where 'ml_models' is imported (in main.py)
@pytest.fixture(scope="session")
def xgboost_ml_models_path() -> str:
    """Fixture for the mock path to the prediction function."""
    return "backend.main.ml_models.predict_xgboost"

@pytest.fixture(scope="session")
def random_forest_ml_models_path() -> str:
    """Fixture for the mock path to the random forest prediction function."""
    return "backend.main.ml_models.predict_random_forest"
