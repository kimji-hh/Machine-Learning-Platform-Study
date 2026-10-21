from kfp.components import OutputPath, InputPath
from minio import Minio
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from typing import NamedTuple
from collections import namedtuple
import os
import mlflow
from mlflow import MlflowClient
import json
import argparse
from pathlib import Path
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report
from evidently.ui.remote import RemoteWorkspace


def download_file_from_minio(
    minio_host: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    object_name: str,
    file_path: str,
):
    client = Minio(
        endpoint=minio_host, access_key=access_key, secret_key=secret_key, secure=False
    )
    client.fget_object(bucket_name, object_name, file_path)


def detect_drift(
    model_name: str,
    model_stage: str,
    mlflow_host: str,
    minio_host: str,
    access_key: str,
    secret_key: str,
    reference_dataset_name: str,
    feature_dataset_path: InputPath(),
    evidently_workspace_url: str,
    evidently_ui_project_id: str,
):
    mlflow.set_tracking_uri(mlflow_host)
    mlflow_client = MlflowClient(mlflow_host)
    evidently_workspace = RemoteWorkspace(evidently_workspace_url)

    for model in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if model.current_stage == model_stage:
            model_run_id = model.run_id

    run = mlflow.get_run(model_run_id)
    for dataset_input in run.inputs.dataset_inputs:
        for tag in dataset_input.tags:
            if tag.value == reference_dataset_name:
                dataset_source = json.loads(dataset_input.dataset.source)["uri"]

    bucket_name = dataset_source.split("/")[0]
    object_name = "/".join(dataset_source.split("/")[1:])
    file_path = object_name.split("/")[-1]
    download_file_from_minio(
        minio_host, access_key, secret_key, bucket_name, object_name, file_path
    )
    reference_df = pd.read_csv(file_path)
    feature_df = pd.read_pickle(feature_dataset_path)
    report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="Education"),
            ColumnSummaryMetric(column_name="Education"),
            ColumnDriftMetric(column_name="Marital-Status"),
            ColumnSummaryMetric(column_name="Marital-Status"),
            ColumnDriftMetric(column_name="Native_country"),
            ColumnSummaryMetric(column_name="Native_country"),
            ColumnDriftMetric(column_name="Occupation"),
            ColumnSummaryMetric(column_name="Occupation"),
            ColumnDriftMetric(column_name="Race"),
            ColumnSummaryMetric(column_name="Race"),
            ColumnDriftMetric(column_name="Relationship"),
            ColumnSummaryMetric(column_name="Relationship"),
            ColumnDriftMetric(column_name="Sex"),
            ColumnSummaryMetric(column_name="Sex"),
            ColumnDriftMetric(column_name="Workclass"),
            ColumnSummaryMetric(column_name="Workclass"),
        ],
    )
    report.run(reference_data=reference_df, current_data=feature_df)
    evidently_workspace.add_report(evidently_ui_project_id, report)


def main():
    parser = argparse.ArgumentParser(description="Arguments for fetching data from url")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_stage", type=str)
    parser.add_argument("--mlflow_host", type=str)
    parser.add_argument("--minio_host", type=str)
    parser.add_argument("--access_key", type=str)
    parser.add_argument("--secret_key", type=str)
    parser.add_argument("--reference_dataset_name", type=str)
    parser.add_argument("--share_of_drift_threshold", type=float)
    parser.add_argument("--feature_dataset_path", type=str)
    parser.add_argument("--minio_drift_dashboard_path", type=str)
    parser.add_argument("--evidently_workspace_url", type=str)
    parser.add_argument("--evidently_ui_project_id", type=str)
    args = parser.parse_args()

    detect_drift(
        model_name=args.model_name,
        model_stage=args.model_stage,
        mlflow_host=args.mlflow_host,
        minio_host=args.minio_host,
        access_key=args.access_key,
        secret_key=args.secret_key,
        reference_dataset_name=args.reference_dataset_name,
        share_of_drift_threshold=args.share_of_drift_threshold,
        feature_dataset_path=args.feature_dataset_path,
        minio_drift_dashboard_path=args.minio_drift_dashboard_path,
        evidently_workspace_url=args.evidently_workspace_url,
        evidently_ui_project_id=args.evidently_ui_project_id,
    )


if __name__ == "__main__":
    main()
