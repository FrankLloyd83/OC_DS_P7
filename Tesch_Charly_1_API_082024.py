from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from typing import List
from azure.storage.blob import BlobServiceClient
import tempfile
import os

connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "model-xgboost-default"

blob_service_client = BlobServiceClient.from_connection_string(connect_str)


def download_container_to_tempdir(container_name: str) -> str:
    container_client = blob_service_client.get_container_client(
        container=container_name
    )
    temp_dir = tempfile.mkdtemp()
    blobs_list = container_client.list_blobs()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
        for blob in blobs_list:
            blob_client = container_client.get_blob_client(blob.name)
            blob_path = os.path.join(temp_dir, blob.name)
            os.makedirs(os.path.dirname(blob_path), exist_ok=True)

            with open(blob_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())

        print(f"Donwloaded container {container_name} to {temp_dir}")
        return temp_dir


model_dir = download_container_to_tempdir(container_name)
model = mlflow.sklearn.load_model(model_dir)

app = FastAPI()


class PredictionRequest(BaseModel):
    features: List[float]


@app.get("/")
def read_root():
    return {"L'API": "est en ligne"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


@app.post("/predict")
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
