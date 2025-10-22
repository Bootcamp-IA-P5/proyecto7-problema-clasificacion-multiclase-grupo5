from unittest import mock
import pytest
from backend.test.fixtures import(
    client,
    valid_payload,
    random_forest_ml_models_path,
    predict_random_forest_url
)
# Pytest will automatically discover and use fixtures from conftest.py or fixtures.py

# Note: We don't need to import TestClient, app, or the payload data directly,
# as we rely on the fixtures provided.


def test_successful_prediction(client, predict_random_forest_url, valid_payload, random_forest_ml_models_path):
    """
    Tests the 200 OK scenario where the prediction service returns a result.
    """
    with mock.patch(random_forest_ml_models_path, return_value=3) as mock_predict:
        response = client.post(predict_random_forest_url, json=valid_payload)

        # 1. Assert the HTTP status code
        assert response.status_code == 200

        # 2. Assert the response structure and content
        data = response.json()
        assert data["cover_type"] == 3
        
        # 3. Assert the mock was called correctly
        # Check that the prediction function received the model_dump() dictionary
        mock_predict.assert_called_once()
        assert mock_predict.call_args[0][0] == valid_payload


def test_prediction_service_unavailable_503(client, predict_random_forest_url, valid_payload, random_forest_ml_models_path):
    """
    Tests the 503 Service Unavailable scenario (model not loaded - RuntimeError).
    """
    error_message = "Random Forest model is not loaded."

    # Mock the prediction function to raise the specific RuntimeError
    with mock.patch(random_forest_ml_models_path, side_effect=RuntimeError(error_message)):
        response = client.post(predict_random_forest_url, json=valid_payload)
        
        # 1. Assert the HTTP status code
        assert response.status_code == 503

        # 2. Assert the error detail matches the exception
        assert response.json()["detail"] == error_message


def test_internal_server_error_500(client, predict_random_forest_url, valid_payload, random_forest_ml_models_path):
    """
    Tests the 500 Internal Server Error scenario for general exceptions.
    """
    error_message = "An unexpected error occurred."

    # Mock the prediction function to raise a generic Exception
    with mock.patch(random_forest_ml_models_path, side_effect=Exception(error_message)):
        response = client.post(predict_random_forest_url, json=valid_payload)

        # 1. Assert the HTTP status code
        assert response.status_code == 500

        # 2. Assert the error detail includes the specific exception message
        assert response.json()["detail"] == f"Prediction failed: {error_message}"


def test_pydantic_validation_error_422(client, predict_random_forest_url, valid_payload):
    """
    Tests the built-in FastAPI Pydantic validation error (422 Unprocessable Entity).
    """
    invalid_payload = valid_payload.copy()
    invalid_payload["Elevation"] = "not_a_number" 

    response = client.post(predict_random_forest_url, json=invalid_payload)

    assert response.status_code == 422
    data = response.json()
    assert data["detail"][0]["loc"] == ["body", "Elevation"]
    assert "should be a valid integer" in data["detail"][0]["msg"]
