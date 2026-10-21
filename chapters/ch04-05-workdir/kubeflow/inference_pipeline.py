import kfp.dsl as dsl
from kfp.compiler import Compiler
import kfp.components as comp

fetch_data_op = comp.load_component_from_file(
    "components/read_data/component.yaml"
)  # A

retrieve_features_op = comp.load_component_from_file(
    "components/retrieve_features/component.yaml"
)

run_inference_op = comp.load_component_from_file(
    "components/run_inference/component.yaml"
)
data_drift_op = comp.load_component_from_file("components/data_drift/component.yaml")

write_data_op = comp.load_component_from_file("components/write_data/component.yaml")


@dsl.pipeline(
    name="income-classifier-pipeline",
    description="A Kubeflow pipeline to classify income categories",
)
def income_classifier_pipeline(
    minio_host: str = "",
    access_key: str = "",
    secret_key: str = "",
    entity_df_bucket: str = "",
    entity_df_filename: str = "",
    feature_store_bucket_name: str = "",
    feature_store_config_file_name: str = "",
    feature_list: str = "",
    model_name: str = "",
    model_type: str = "",
    model_stage: str = "",
    mlflow_host: str = "",
    output_bucket: str = "",
    output_file_name: str = "",
    reference_dataset_name: str = "",
    minio_drift_dashboard_path: str = "",
    evidently_ui_project_id: str = "",
    evidently_workspace_url: str = "",
    share_of_drift_threshold: float = 0.5,
):  # B
    fetch_data_task = fetch_data_op(
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=entity_df_bucket,
        file_name=entity_df_filename,
    )  # C

    retrieve_features_task = retrieve_features_op(
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=feature_store_bucket_name,
        file_name=feature_store_config_file_name,
        entity_df_path=fetch_data_task.output,  # D
        feature_list=feature_list,
    )

    detect_drift_task = data_drift_op(
        model_name=model_name,
        model_stage=model_stage,
        mlflow_host=mlflow_host,
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        reference_dataset_name=reference_dataset_name,
        share_of_drift_threshold=share_of_drift_threshold,
        feature_dataset_path=retrieve_features_task.output,
        minio_drift_dashboard_path=minio_drift_dashboard_path,
        evidently_ui_project_id=evidently_ui_project_id,
        evidently_workspace_url=evidently_workspace_url,
    )
    detect_drift_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    detect_drift_task.container.set_image_pull_policy("Always")

    run_inference_task = run_inference_op(
        model_name=model_name,
        model_type=model_type,
        model_stage=model_stage,
        mlflow_host=mlflow_host,
        input_data_path=retrieve_features_task.output,  # E
    ).after(detect_drift_task)
    run_inference_task.container.set_image_pull_policy("Always")

    write_data_task = write_data_op(
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=output_bucket,
        file_name=output_file_name,
        input_data_path=run_inference_task.output,
    )  # F


Compiler().compile(
    income_classifier_pipeline,
    "income_classifier_pipeline.yaml",
)  # G
