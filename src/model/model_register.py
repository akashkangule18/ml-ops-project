import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import dagshub
import pickle
import json


# making connection with dagshub
dagshub.init(repo_owner='akashkangule18', repo_name='ml-ops-project', mlflow=True) #  fetching model from the model evaluation




# starting experiment
mlflow.set_experiment("final model registerd-staging")

with mlflow.start_run(run_name = 'model_registering'):
    with open ("models/model.pkl",'rb') as file:
      model = pickle.load(file)


    # fetching trianig data for model signature
    X_train = pd.read_csv("./data/processed/train_tr_processed.csv")

    # fetching metrics report for log

    with open("reports/metrics.json","r") as file:
        metrics = json.load(file)

    # logging params
    mlflow.log_params(model.get_params())
    mlflow.log_param("test_size", 0.4)

    # logiing file 
    mlflow.log_artifact(__file__)

    # logging metric reports
    mlflow.log_artifact("reports/metrics.json")

    # logging paramas
    mlflow.log_metric("accuracy_score", metrics['accuracy_score'])
    mlflow.log_metric('precision_score', metrics['precision_score'])
    mlflow.log_metric('recall_score',metrics['recall_score'])

    # siganture
    pred = model.predict(X_train)
    signature = mlflow.models.infer_signature(X_train, pred)


    # model registeration and  
    mlflow.sklearn.log_model(
       sk_model = model,
       name = 'Random_forest_model',
       signature = signature,
       registered_model_name = 'Random_forest_classifier'
    )


    # staging
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    versions = client.search_model_versions(
    "name='Random_forest_classifier'")

    latest_version = max(int(v.version) for v in versions)

    client.set_registered_model_alias(
       name = "Random_forest_classifier",
       alias = 'Staging',
       version = latest_version
    ) 
 

    print('model registration and staging sucessfully')


    