import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import mlflow
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
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

df=pd.read_csv('../data/income_data.csv', index_col=False)

print(df.isnull().sum())

#Let´s see how much data with the ' ?' sign we have - this data is also null.

print(df.apply(lambda x: (x==' ?').sum() ,axis=0))

rows_with_unknowns=df.apply(lambda x: (x==' ?').sum(), axis=1)>0
index=df[rows_with_unknowns].index.tolist()
df=df.drop(index, axis=0)

print(df.Target.value_counts())

#Analysis of categorical data:
#mlflow the plots

with mlflow.start_run(run_name=f"eda-{uuid.uuid4()}"):
    for i in df.iloc[:,:-1].select_dtypes(include='object').columns:
        print(f'Variable {i}  \n ')
        print(df[i].value_counts())

        plot = ggplot(df)+ geom_bar(aes(x=df[i], fill=df.Target), position='fill')+ theme_bw() + labs(title=f'Variable {i} ~ Target')+ coord_flip()
        print(plot)
        if not os.path.exists("categorical_variable_plots"):
            os.makedirs("categorical_variable_plots")
        plot.save(f"categorical_variable_plots/Variable {i}")
        mlflow.log_artifacts("categorical_variable_plots")



