import mlflow
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import logger
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json


from logger import logger



dagshub.init(repo_owner='akashkangule18', repo_name='ml-ops-project', mlflow=True)


# getting model
def load_model(model_path):
    try: 
        with open (model_path,'rb') as file:
            model = pickle.load(file)
        logger.info('model_load_sucessfully')
        return model
    except Exception as e:
        logger.error(f"chek the model path")
        raise

def load_data(trainig_data_path):
    try:

        # fetching the x_test and y_test for prediction
        X_train = pd.read_csv(trainig_data_path)
        logger.info('X_test and y_test fetched sucessfully')
        return X_train
    except Exception as e:
        logger.error(f"chek the test data path")
        raise

def load_metrics(metric_path):
    try:
        with open(metric_path,"r") as file:
            metrics = json.load(file)
            return metrics
    except Exception as e:
        logger.error(f"chek the test metric path")
        raise

mlflow.set_experiment("model_register")
def main ():
    with mlflow.start_run(run_name =" base model register") as parent_run:

        # load model
        model = load_model("./models/model.pkl")

        # getting training data for signature
        X_train = load_data("./data/processed/train_tr_processed.csv")

        # loading metrics.json file
        metrics = load_metrics("./reports/metrics.json")

        # log params
        mlflow.log_params(model.get_params())

        # log metrics

        mlflow.log_metric("accuracy_score",metrics['accuracy_score'])
        mlflow.log_metric('precision_score', metrics['precision_score'])
        mlflow.log_metric("recall_score", metrics['recall_score'])


        # log file
        mlflow.log_artifact(__file__)
        mlflow.log_artifact("./reports/metrics.json")

        # siganture
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

        # log and register model
        mlflow.sklearn.log_model(
            sk_model = model,
            name = "Random_Forest_Model",
            signature = signature,
            registered_model_name = "RandomForestClassifier"
        )


if __name__ == "__main__":
    main()
        

    
