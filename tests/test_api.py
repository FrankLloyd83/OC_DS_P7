from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from Tesch_Charly_1_API_082024 import app

client = TestClient(app)


@patch("Tesch_Charly_1_API_082024.download_container_to_tempdir")
@patch("Tesch_Charly_1_API_082024.mlflow.sklearn.load_model")
def test_predict(mock_load_model, mock_download_container_to_tempdir):
    mock_download_container_to_tempdir.return_value = "model_dir"
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_load_model.return_value = mock_model

    request_data = {"features": [0.0] * 376}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": [1]}


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"L'API": "est en ligne"}


def test_favicon():
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/x-icon"


def test_predict():
    request_data = {"features": [0.0] * 376}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
