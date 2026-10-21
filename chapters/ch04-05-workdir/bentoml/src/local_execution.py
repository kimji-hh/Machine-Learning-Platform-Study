import bentoml
import pandas as pd

input_df = pd.read_csv("x_train.csv")

runner = bentoml.mlflow.get("random-forest-classifier:latest").to_runner()
runner.init_local()
preditions = runner.predict.run(input_df)
print(preditions)
