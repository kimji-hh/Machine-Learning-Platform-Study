import mlflow
from mlflow import MlflowClient
mlflow_client = MlflowClient("http://mlflow.dev.jago.data:5000/")
for model in mlflow_client.search_model_versions(f"name='random-forest-classifier'"):
    if model.current_stage == "Staging":
        model_run_id = model.run_id
mlflow.artifacts.download_artifacts(
    f"runs:/{model_run_id}/column_list/column_list.pkl", dst_path="column_list"
)
