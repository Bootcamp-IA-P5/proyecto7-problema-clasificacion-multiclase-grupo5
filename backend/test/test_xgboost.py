from unittest import mock
import pytest
from backend.test.fixtures import(
    client,
    valid_payload,
    xgboost_ml_models_path,
    predict_xgboost_url
)
# Pytest will automatically discover and use fixtures from conftest.py

# Note: We don't need to import TestClient, app, or the payload data directly,
# as we rely on the fixtures provided by conftest.py.


def test_successful_prediction(client, predict_xgboost_url, valid_payload, xgboost_ml_models_path):
    """
    Tests the 200 OK scenario where the prediction service returns a result.
    """
    with mock.patch(xgboost_ml_models_path, return_value=5) as mock_predict:
        response = client.post(predict_xgboost_url, json=valid_payload)

        # 1. Assert the HTTP status code
        assert response.status_code == 200

        # 2. Assert the response structure and content
        data = response.json()
        assert data["cover_type"] == 5
        
        # 3. Assert the mock was called correctly
        # Check that the prediction function received the model_dump() dictionary
        mock_predict.assert_called_once()
        assert mock_predict.call_args[0][0] == valid_payload


def test_prediction_service_unavailable_503(client, predict_xgboost_url, valid_payload, xgboost_ml_models_path):
    """
    Tests the 503 Service Unavailable scenario (model not loaded - RuntimeError).
    """
    error_message = "XGBoost model file not found or failed to load."

    # Mock the prediction function to raise the specific RuntimeError
    with mock.patch(xgboost_ml_models_path, side_effect=RuntimeError(error_message)):
        response = client.post(predict_xgboost_url, json=valid_payload)
        
        # 1. Assert the HTTP status code
        assert response.status_code == 503

        # 2. Assert the error detail matches the exception
        assert response.json()["detail"] == error_message


def test_internal_server_error_500(client, predict_xgboost_url, valid_payload, xgboost_ml_models_path):
    """
    Tests the 500 Internal Server Error scenario for general exceptions.
    """
    error_message = "Database connection failed during feature lookup."

    # Mock the prediction function to raise a generic Exception
    with mock.patch(xgboost_ml_models_path, side_effect=ValueError(error_message)):
        response = client.post(predict_xgboost_url, json=valid_payload)

        # 1. Assert the HTTP status code
        assert response.status_code == 500

        # 2. Assert the error detail includes the specific exception message
        # Note: We must replicate the f-string formatting exactly
        expected_detail = f"Prediction failed: {ValueError(error_message)}"
        assert response.json()["detail"] == expected_detail


def test_pydantic_validation_error_422(client, predict_xgboost_url, valid_payload):
    """
    Tests the built-in FastAPI Pydantic validation error (422 Unprocessable Entity).
    This test ensures the input validation is working.
    """
    invalid_payload = valid_payload.copy()
    # 'elevation' expects an int, pass a string to trigger 422
    invalid_payload["Elevation"] = "not_a_number" 

    response = client.post(predict_xgboost_url, json=invalid_payload)

    # 1. Assert the HTTP status code
    assert response.status_code == 422

    # 2. Check that the detail contains validation information
    data = response.json()
    assert data["detail"][0]["loc"] == ["body", "Elevation"]
    assert "should be a valid integer" in data["detail"][0]["msg"]
