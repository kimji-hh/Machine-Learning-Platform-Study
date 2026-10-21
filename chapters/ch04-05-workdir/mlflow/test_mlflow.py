import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import mlflow
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)
from plotnine import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from mlflow.data.pandas_dataset import PandasDataset
import uuid
import time

import os

mlflow.set_tracking_uri("http://mlflow.dev.jago.data:5000/")
mlflow.set_experiment("income-classifier")

df = pd.read_csv("../data/income_data.csv", index_col=False)

print(df.isnull().sum())

# Let´s see how much data with the ' ?' sign we have - this data is also null.

print(df.apply(lambda x: (x == " ?").sum(), axis=0))

rows_with_unknowns = df.apply(lambda x: (x == " ?").sum(), axis=1) > 0
index = df[rows_with_unknowns].index.tolist()
df = df.drop(index, axis=0)

print(df.Target.value_counts())

# Analysis of categorical data:
# mlflow the plots


target = df.Target
feature_df = df.drop("Target", axis=1)


# Codifying the predictors and the target
encoder = OneHotEncoder(sparse=False, drop="if_binary")
target = encoder.fit_transform(np.array(target).reshape(-1, 1))

# Dummifying

dummyfied_df = pd.get_dummies(feature_df, drop_first=True, sparse=False, dtype=float)
X_train, X_test, y_train, y_test = train_test_split(
    dummyfied_df, target, train_size=0.80, shuffle=True
)
from minio import Minio

minioClient = Minio(
    "localhost:9000", access_key="minio", secret_key="minio123", secure=False
)
found = minioClient.bucket_exists("mlflow-datasets")
print(found)

from io import BytesIO


def save_df_to_minio(df, bucket_name, path):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    csv_buffer = BytesIO(csv_bytes)

    minioClient.put_object(
        f"{bucket_name}",
        f"{path}",
        data=csv_buffer,
        length=len(csv_bytes),
        content_type="application/csv",
    )


BUCKET_NAME = "mlflow-datasets"
with mlflow.start_run() as run:
    results = pd.DataFrame(
        index=[
            "Roc Auc Score test",
            "Accuracy score train",
            "Accuracy Score test",
            "time to fit",
        ]
    )
    tree = DecisionTreeClassifier()
    run_id = run.info.run_id
    feature_df_path = f"income-classifier-datasets/feature_df-{run_id}.csv"
    train_df = pd.concat([X_train, pd.Series(y_train.ravel())], axis=1)
    train_df_path = f"income-classifier-datasets/train-{run_id}.csv"
    save_df_to_minio(train_df, BUCKET_NAME, train_df_path)
    save_df_to_minio(feature_df, BUCKET_NAME, feature_df_path)
    test_df = pd.concat([X_test, pd.Series(y_test.ravel())], axis=1)
    test_df_path = f"income-classifier-datasets/test-{run_id}.csv"
    save_df_to_minio(test_df, BUCKET_NAME, test_df_path)
    # save dataframe to cloud provider
    training_dataset = mlflow.data.from_pandas(
        train_df, source=f"{BUCKET_NAME}/{train_df_path}"
    )
    test_dataset = mlflow.data.from_pandas(
        test_df, source=f"{BUCKET_NAME}/{test_df_path}"
    )
    feature_dataset = mlflow.data.from_pandas(
        feature_df, source=f"{BUCKET_NAME}/{feature_df_path}"
    )
    mlflow.log_input(training_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")
    mlflow.log_input(feature_dataset, context="reference_features")
    tree.fit(X_train, y_train.ravel())
    roc_auc_score_train = roc_auc_score(y_train == 1, tree.predict_proba(X_train)[:, 1])
    roc_auc_score_test = roc_auc_score(y_test == 1, tree.predict_proba(X_test)[:, 1])
    training_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)
    mlflow.log_metric("roc_auc_score_train", roc_auc_score_train)
    print(f"Roc Auc Score train: {roc_auc_score_train}  \n")
    mlflow.log_metric("roc_auc_score_test", roc_auc_score_test)
    print(f"Roc Auc Score test: {roc_auc_score_test}  \n")
    mlflow.log_metric("training_accuracy", training_accuracy)
    print(f"Accuracy train : {training_accuracy}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Accuracy test : {test_accuracy}")
    mlflow.sklearn.log_model(tree, artifact_path="income-classifier")
    mlflow.log_params(tree.get_params())

with mlflow.start_run():
    forest = RandomForestClassifier()

    start = time.time()
    run_id = run.info.run_id
    train_df = pd.concat([X_train, pd.Series(y_train.ravel())], axis=1)
    test_df = pd.concat([X_test, pd.Series(y_test.ravel())], axis=1)
    feature_df_path = f"income-classifier-datasets/feature_df-{run_id}.csv"
    train_df_path = f"income-classifier-datasets/train-{run_id}.csv"
    save_df_to_minio(train_df, BUCKET_NAME, train_df_path)
    save_df_to_minio(feature_df, BUCKET_NAME, feature_df_path)
    test_df_path = f"income-classifier-datasets/test-{run_id}.csv"
    save_df_to_minio(test_df, BUCKET_NAME, test_df_path)
    print(f"{BUCKET_NAME}/{feature_df_path}")
    feature_dataset = mlflow.data.from_pandas(
        feature_df, source=f"{BUCKET_NAME}/{feature_df_path}"
    )
    training_dataset = mlflow.data.from_pandas(
        train_df, source=f"{BUCKET_NAME}/{train_df_path}"
    )
    test_dataset = mlflow.data.from_pandas(
        test_df, source=f"{BUCKET_NAME}/{test_df_path}"
    )
    mlflow.log_input(training_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")
    mlflow.log_input(feature_dataset, context="reference")
    forest.fit(X_train, y_train.ravel())
    end = time.time()

    roc_auc_score_train = roc_auc_score(
        y_train == 1, forest.predict_proba(X_train)[:, 1]
    )
    roc_auc_score_test = roc_auc_score(y_test == 1, forest.predict_proba(X_test)[:, 1])
    training_accuracy = forest.score(X_train, y_train)
    test_accuracy = forest.score(X_test, y_test)
    col_list = df.columns.to_list()
    with open("column_list.pkl", "wb") as f:
        pickle.dump(col_list, f)
    mlflow.log_artifact("column_list.pkl", "column_list")

    mlflow.log_metric("roc_auc_score_train", roc_auc_score_train)
    print(f"Roc Auc Score train: {roc_auc_score_train}  \n")
    mlflow.log_metric("roc_auc_score_test", roc_auc_score_test)
    print(f"Roc Auc Score test: {roc_auc_score_test}  \n")
    mlflow.log_metric("training_accuracy", training_accuracy)
    print(f"Accuracy train : {training_accuracy}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Accuracy test : {test_accuracy}")
    mlflow.sklearn.log_model(forest, "income-classifier")

    mlflow.log_params(forest.get_params())
