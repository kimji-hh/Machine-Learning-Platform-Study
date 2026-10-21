from kfp.components import InputPath, OutputPath
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
import pickle
import argparse


def perform_inference(
    model_name: str,
    model_type: str,
    model_stage: str,
    mlflow_host: str,
    input_data_path: InputPath(),
    data_output_path: OutputPath(),
):
    mlflow.set_tracking_uri(mlflow_host)
    mlflow_client = MlflowClient(mlflow_host)
    for model in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if model.current_stage == model_stage:
            model_run_id = model.run_id
    mlflow.artifacts.download_artifacts(
        f"runs:/{model_run_id}/column_list/column_list.pkl", dst_path="column_list"
    )
    input_data = pd.read_pickle(input_data_path)
    input_data.drop(columns=["user_id", "event_timestamp"], inplace=True)
    with open("column_list/column_list.pkl", "rb") as f:
        col_list = pickle.load(f)
    input_data = pd.get_dummies(input_data, drop_first=True, sparse=False, dtype=float)
    input_data = input_data.reindex(columns=col_list, fill_value=0)

    if model_type == "sklearn":
        model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
    elif model_type == "xgboost":
        model = mlflow.xgboost.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
    elif model_type == "tensorflow":
        model = mlflow.tensorflow.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
    else:
        raise NotImplementedError

    predicted_classes = [x[1] for x in model.predict_proba(input_data)]
    input_data["Predicted_Income_Class"] = predicted_classes
    Path(data_output_path).parent.mkdir(parents=True, exist_ok=True)
    input_data.to_pickle(data_output_path)


def main():
    parser = argparse.ArgumentParser(description="Arguments for fetching data from url")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_stage", type=str)
    parser.add_argument("--mlflow_host", type=str)
    parser.add_argument("--input_data_path", type=str)
    parser.add_argument("--data_output_path", type=str)
    args = parser.parse_args()

    perform_inference(
        model_name=args.model_name,
        model_type=args.model_type,
        model_stage=args.model_stage,
        mlflow_host=args.mlflow_host,
        input_data_path=args.input_data_path,
        data_output_path=args.data_output_path,
    )


if __name__ == "__main__":
    main()
