import mlflow
import bentoml

mlflow.set_tracking_uri("http://mlflow.dev.jago.data:5000/")
model_name = "random-forest-classifier"
model_stage = "Staging"
bentoml.mlflow.import_model(
    "random-forest-classifier", model_uri=f"models:/{model_name}/{model_stage}"
)
