import pandas as pd
import numpy as np
import os
import seaborn as sns
from logger import logger
import pickle
import yaml

def load_file(file_path):
  try:
    params = yaml.safe_load(open(file_path,'r'))['model_building']
    logger.info('params.yaml file load sucessfully')
    return params
  except Exception as e:
     logger.error(f'check the file path of params.yaml file {e}')
     raise

# fetching the training data

def load_data(X_training_data_path, y_training_data_path):
   try:
    X_train = pd.read_csv(X_training_data_path,)
      # fetching the testing 
    train_data = pd.read_csv(y_training_data_path)
    y_train = train_data['Placed'].values
    return X_train, y_train
   except Exception as e:
      logger.error(f'chek the trainig and testing data_path')
      raise



# model building
def model(params, X_train,y_train):
  try:

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators = params['n_estimators'],
                              min_samples_split = params['min_samples_split'],
                              max_depth = params['max_depth'])
    rf.fit(X_train,y_train)
    logger.info('Random forest classifier trained sucessfully')
    return rf
  except Exception as e:
    logger.error(f"check the written model function")

def save_model(model):
  try:

    with open ('models/model.pkl','wb') as file:
      pickle.dump(model,file)
    logger.info('random forest classifier dumped sucessfully')
  except Exception as e:
    logger.error('check the function, model is not saved')
    raise


  



def main():
   params = load_file('params.yaml')
   X_train, y_train = load_data("./data/processed/train_tr_processed.csv","./data/interim/train_interim_data.csv")
   rf = model(params, X_train, y_train)
   save_model(rf)


if __name__ == '__main__':
  main()
