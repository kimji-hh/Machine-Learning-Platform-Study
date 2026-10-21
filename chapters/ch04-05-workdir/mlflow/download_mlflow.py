import mlflow
mlflow.set_tracking_uri("http://mlflow.dev.jago.data:5000/")
mlflow.set_experiment("income-classifier")

mlflow.artifacts.download_artifacts("runs:/739e292ea3854e039602070ec58e22bb/column_list/column_list.pkl",dst_path="column_list")
