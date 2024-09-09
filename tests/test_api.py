import os
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from Tesch_Charly_1_API_082024 import (
    app,
    download_artifacts_to_tempdir,
    get_container_client,
    get_blob_client,
    get_threshold,
    get_model,
)

# Crée un client de test pour l'API
client = TestClient(app)


# Mock du chemin de connexion à Azure
@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "fake_connection_string")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "fake_experiment_id")
    monkeypatch.setenv("MLFLOW_RUN_ID", "fake_run_id")


# Test du endpoint racine
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"L'API": "est en ligne"}


# Test du endpoint favicon
def test_favicon():
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.headers["content-type"] in [
        "image/x-icon",
        "image/vnd.microsoft.icon",
    ]


@patch("Tesch_Charly_1_API_082024.download_artifacts_to_tempdir")
@patch("Tesch_Charly_1_API_082024.get_model")
@patch("Tesch_Charly_1_API_082024.get_threshold")
@patch("Tesch_Charly_1_API_082024.shap.TreeExplainer")
def test_predict(
    mock_tree_explainer,
    mock_get_threshold,
    mock_get_model,
    mock_download_artifacts_to_tempdir,
):
    # Mock du modèle
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
    mock_get_model.return_value = mock_model

    # Mock du seuil
    mock_get_threshold.return_value = 0.12

    # Mock des SHAP values
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    mock_tree_explainer.return_value = mock_explainer

    # Données de test
    payload = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}

    # Appel de l'API
    response = client.post("/predict", json=payload)

    # Vérification de la réponse
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["proba"] == 0.6
    assert data["threshold"] == 0.12
    assert data["top_10_features"] == [0.5, 0.4, 0.3, 0.2, 0.1]
