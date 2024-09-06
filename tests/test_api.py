import os
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from Tesch_Charly_1_API_082024 import app, download_container_to_tempdir, get_model

# Crée un client de test pour l'API
client = TestClient(app)


# Mock du chemin de connexion à Azure
@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "fake_connection_string")


# Test du endpoint racine
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"L'API": "est en ligne"}


# Test du endpoint favicon
def test_favicon():
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["image/x-icon", "image/vnd.microsoft.icon"]


# Mock de la méthode download_container_to_tempdir pour éviter les appels réels à Azure
@patch("Tesch_Charly_1_API_082024.download_container_to_tempdir")
@patch("Tesch_Charly_1_API_082024.get_model")
def test_predict(mock_get_model, mock_download_container_to_tempdir):
    # Simuler un répertoire temporaire
    mock_download_container_to_tempdir.return_value = "/fake/tempdir"

    # Simuler un modèle XGBoost chargé
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_get_model.return_value = mock_model

    # Effectuer la requête POST avec des features fictives
    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3, 0.4, 0.5]})

    assert response.status_code == 200
    assert response.json() == {"prediction": [1]}

    # Vérifie que le modèle et le répertoire temporaire ont été utilisés
    mock_download_container_to_tempdir.assert_called_once_with("model-xgboost-default")
    mock_get_model.assert_called_once_with("/fake/tempdir")


# Mock du client Azure Blob pour tester la fonction de téléchargement
@patch("Tesch_Charly_1_API_082024.BlobServiceClient")
def test_download_container_to_tempdir(mock_blob_service_client):
    # Simuler un client de container et une liste de blobs
    mock_container_client = MagicMock()
    mock_blob = MagicMock()
    mock_blob.name = "model.pkl"
    mock_container_client.list_blobs.return_value = [mock_blob]

    # Simuler un client blob et son flux de téléchargement
    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value.readall.return_value = (
        b"fake model data"
    )
    mock_container_client.get_blob_client.return_value = mock_blob_client

    mock_blob_service_client.from_connection_string.return_value.get_container_client.return_value = (
        mock_container_client
    )

    # Appeler la fonction pour tester
    temp_dir = download_container_to_tempdir(mock_container_client)

    # Vérifie que le fichier a été téléchargé dans le répertoire temporaire
    assert os.path.exists(os.path.join(temp_dir, "model.pkl"))
    mock_container_client.get_blob_client.assert_called_once_with("model.pkl")
