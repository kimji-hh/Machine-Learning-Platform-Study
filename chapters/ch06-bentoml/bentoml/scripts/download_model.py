import mlflow
import bentoml
import os

mlflow.set_tracking_uri("http://localhost:5000")
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
model_name = "random-forest-classifier"
model_stage = "Production"
bentoml.mlflow.import_model(
    "random-forest-classifier", model_uri=f"models:/{model_name}/{model_stage}"
)
