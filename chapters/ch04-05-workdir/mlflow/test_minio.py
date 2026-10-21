from minio import Minio
import pandas as pd
client = Minio(
        endpoint="http://localhost:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False,
    )	
obj = client.get_object(
        "entity-datasets",
        "entity_df.csv",
    )

df = pd.read_csv(obj)

print(df.head())	
