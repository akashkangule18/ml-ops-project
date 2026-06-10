import pandas as pd
import numpy as np
import os
import seaborn as sns
import pickle
import json
from logger import logger

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

def load_data(x_test_data_path, y_test_data_path):
    try:

        # fetching the x_test and y_test for prediction
        X_test = pd.read_csv(x_test_data_path)
        test_data = pd.read_csv(y_test_data_path)
        y_test = test_data['Placed'].values
        logger.info('X_test and y_test fetched sucessfully')
        return X_test, y_test
    except Exception as e:
        logger.error(f"chek the test data path")
        raise

def pred(model,X_test):
    try:
        # model_prediction 
        y_pred = model.predict(X_test)
        return y_pred
    except Exception as e:
        logger.error(f"model prediction is not done yet {e}")
        raise

def eval(y_test,y_pred):
    try:
                
        # fetching the accuracy
        from sklearn.metrics import accuracy_score, precision_score,recall_score,classification_report,confusion_matrix

        acc = accuracy_score(y_test,y_pred)
        pcc = precision_score(y_test,y_pred)
        rcc = recall_score(y_test,y_pred)
        cmc = confusion_matrix(y_test,y_pred)
        logger.info('evalution done sucessfully')
        return acc, pcc, cmc, rcc
    except Exception as e:
        logger.error(f"unable to perform evaluation {e}")
        raise


def report(acc,pcc,rcc,cmc):
    try:
                    
            # getting json report
            metric_report = {
                'accuracy_score' :acc,
                'precision_score': pcc,
                'recall_score':rcc,
                'confusion_matrix':cmc.tolist()
            }
            logger.info('metrics report not generated sucessfully')
            return metric_report
    except Exception as e:
            logger.error(f"metrics report not generated sucessfully")


def save_json(file_path,metric_report):
    try:
        with open(file_path,'w') as file:
              json.dump(metric_report,file)
        logger.info('json file load sucessfully')
    
    except Exception as e:
         logger.error(f"unable to save the json file")
         raise
        



def main():
    model = load_model("./models/model.pkl")
    X_test, y_test = load_data("./data/processed/test_tr_processed.csv","./data/interim/test_interim_data.csv")
    y_pred = pred(model, X_test)
    acc, pcc, cmc, rcc = eval(y_test,y_pred)
    metric_report= report(acc, pcc, rcc, cmc)
    save_json('reports/metrics.json',metric_report)



if __name__ == '__main__':
     main()