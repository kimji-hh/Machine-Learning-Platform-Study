from kfp.components import OutputPath, InputPath
import argparse
from minio import Minio
import pandas as pd
from pathlib import Path


def write_data(
    minio_host: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_name: str,
    input_data_path: InputPath(),
):
    client = Minio(
        endpoint=minio_host, access_key=access_key, secret_key=secret_key, secure=False
    )
    input_data = pd.read_pickle(input_data_path)
    input_data.to_parquet(file_name, index=False)
    client.fput_object(bucket_name, file_name, file_name)


def main():
    parser = argparse.ArgumentParser(description="Arguments for fetching data from url")
    parser.add_argument("--minio_host", type=str)
    parser.add_argument("--access_key", type=str)
    parser.add_argument("--secret_key", type=str)
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--input_data_path", type=str)
    args = parser.parse_args()

    write_data(
        minio_host=args.minio_host,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket_name=args.bucket_name,
        file_name=args.file_name,
        input_data_path=args.input_data_path,
    )


if __name__ == "__main__":
    main()
