from kfp.components import OutputPath
import argparse
from minio import Minio
import pandas as pd
from pathlib import Path


def get_data(
    minio_host: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_name: str,
    data_output_path: OutputPath(),
):
    client = Minio(
        endpoint=minio_host, access_key=access_key, secret_key=secret_key, secure=False
    )
    client.fget_object(bucket_name, file_name, file_name)
    df = pd.read_parquet(file_name)

    Path(data_output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(data_output_path)


def main():
    parser = argparse.ArgumentParser(description="Arguments for fetching data from url")
    parser.add_argument("--minio_host", type=str)
    parser.add_argument("--access_key", type=str)
    parser.add_argument("--secret_key", type=str)
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--data_output_path", type=str)
    args = parser.parse_args()

    get_data(
        minio_host=args.minio_host,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket_name=args.bucket_name,
        file_name=args.file_name,
        data_output_path=args.data_output_path,
    )


if __name__ == "__main__":
    main()
