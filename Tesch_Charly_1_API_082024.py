from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from typing import List
from azure.storage.blob import BlobServiceClient
import tempfile
import os
import shap

app = FastAPI()


class PredictionRequest(BaseModel):
    features: List[float]


def get_container_client(container_name: str):
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(
        container=container_name
    )
    return container_client


def get_blob_client(container_name: str, blob_name: str):
    container_client = get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client


def download_artifacts_to_tempdir() -> str:
    container_client = get_container_client("mlartifacts")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID").replace('"', "")
    run_id = os.getenv("MLFLOW_RUN_ID").replace('"', "")
    temp_dir = tempfile.mkdtemp()
    blobs_list = container_client.list_blobs(
        name_starts_with=f"{experiment_id}/{run_id}/artifacts/"
    )
    filtered_blobs = [blob for blob in blobs_list if not blob.name.endswith(".png")]
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        for blob in filtered_blobs:
            blob_client = container_client.get_blob_client(blob.name)
            blob_path = os.path.join(temp_dir, blob.name.split("/")[-1])
            os.makedirs(os.path.dirname(blob_path), exist_ok=True)
            with open(blob_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())

        print(f"Donwloaded model from container {container_client} to {temp_dir}")
        return temp_dir


def get_threshold() -> float:
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID").replace('"', "")
    run_id = os.getenv("MLFLOW_RUN_ID").replace('"', "")
    blob_name = f"{experiment_id}/{run_id}/metrics/optimum_threshold_train"
    blob_client = get_blob_client("mlruns", blob_name)
    blob_data = blob_client.download_blob().readall()
    threshold = float(blob_data.decode("utf-8").split(" ")[1])
    return threshold


def get_model(model_dir: str) -> mlflow.sklearn.Model:
    model = mlflow.sklearn.load_model(model_dir)
    return model


@app.get("/")
def read_root():
    return {"L'API": "est en ligne"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


@app.post("/predict")
async def predict(request: PredictionRequest):
    artifacts_temp_dir = download_artifacts_to_tempdir()
    model = get_model(artifacts_temp_dir)
    threshold = get_threshold()
    features = np.array(request.features).reshape(1, -1)
    probas = model.predict_proba(features)
    prediction = (probas[:, 1] > threshold).astype(int)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    top_10_features = np.argsort(np.abs(shap_values[0]))[::-1][:10]
    important_shap_values = shap_values[0][top_10_features].tolist()

    print("prediction", prediction, type(prediction))
    print("probas", probas, type(probas))
    print("threshold", threshold, type(threshold))
    print("top_10_features", important_shap_values, type(important_shap_values))

    return {
        "prediction": int(prediction[0]),
        "proba": float(probas[0, 1]),
        "threshold": threshold,
        "top_10_features": important_shap_values,
    }
