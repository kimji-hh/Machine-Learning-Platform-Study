from feast import FeatureStore
from kfp.components import OutputPath, InputPath
import pandas as pd
from pathlib import Path
from minio import Minio
from pathlib import Path
from feast import FeatureStore
from feast.repo_config import FeastConfigError
from pydantic import ValidationError
import argparse


def init_feature_store(
    minio_host: str, access_key: str, secret_key: str, bucket_name: str, file_name: str
):
    # download the content of the feature_store.yaml from the GCS bucket
    client = Minio(
        minio_host, access_key=access_key, secret_key=secret_key, secure=False
    )
    client.fget_object(bucket_name, file_name, "feature_store.yaml")
    config_path = Path("./") / "feature_store.yaml"
    try:
        store = FeatureStore(repo_path=".")
    except ValidationError as e:
        raise FeastConfigError(e, config_path)
    return store


def get_features(
    minio_host: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_name: str,
    entity_df_path: InputPath(),
    feature_list: str,
    data_output_path: OutputPath(),
):
    store = init_feature_store(
        minio_host, access_key, secret_key, bucket_name, file_name
    )
    print("feature store initialised")
    feature_list = feature_list.split(",")
    print(feature_list)
    entity_df = pd.read_pickle(entity_df_path)
    print(entity_df.head())
    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_list,
    ).to_df()
    print("retrieved historical features")
    print(feature_df.head())
    Path(data_output_path).parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_pickle(data_output_path)


def main():
    parser = argparse.ArgumentParser(description="Arguments for fetching data from url")
    parser.add_argument("--minio_host", type=str)
    parser.add_argument("--access_key", type=str)
    parser.add_argument("--secret_key", type=str)
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--entity_df_path", type=str)
    parser.add_argument("--feature_list", type=str)
    parser.add_argument("--data_output_path", type=str)
    args = parser.parse_args()

    get_features(
        minio_host=args.minio_host,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket_name=args.bucket_name,
        file_name=args.file_name,
        entity_df_path=args.entity_df_path,
        feature_list=args.feature_list,
        data_output_path=args.data_output_path,
    )


if __name__ == "__main__":
    main()
