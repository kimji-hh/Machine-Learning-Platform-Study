from feast import FeatureStore
import pandas as pd
from pathlib import Path
from minio import Minio
from pathlib import Path
from feast import FeatureStore
from feast.repo_config import FeastConfigError
from pydantic import ValidationError


class DataStore:
    def __init__(self, minio_host, access_key, secret_key, bucket_name, file_name):
        self.minio_host = minio_host
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.file_name = file_name

    def init_feast_store(self):
        client = Minio(
            self.minio_host,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False,
        )
        client.fget_object(self.bucket_name, self.file_name, "feature_store.yaml")
        config_path = Path("./") / "feature_store.yaml"
        try:
            store = FeatureStore(repo_path=".")
        except ValidationError as e:
            raise FeastConfigError(e, config_path)
        return store
