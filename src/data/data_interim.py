import pandas as pd
import numpy as np
import seaborn as sns
import os
import yaml
from logger import logger

# fetching the train and test data from the raw folder for data interim
def load_data(train_file_path, test_file_path):
    try:
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        logger.info('the train and test data load sucessfully')
        return train_data, test_data
    except Exception as e:
        logger.error(f'please check the data path {e}')
        raise
   



# data_prperocessing
def preprocess(train_data, test_data):
    try:

        train_data = train_data[train_data['Communication_Skills'] > 2]
        test_data = test_data[test_data['Communication_Skills'] > 2]
        logger.info('data preprocessed sucessfully')
        return train_data, test_data
    except Exception as e:
        logger.error(f'data was not processed check the given  function {e}')
        raise



def save_data(data_path, train_data,test_data):
    try: 

        train_data = train_data.to_csv(os.path.join(data_path,'train_interim_data.csv'),index = False)
        test_data = test_data.to_csv(os.path.join(data_path,'test_interim_data.csv'),index = False)
        logger.info('the data save sucessfully')
    except Exception as e:
        logger.error(f'data is not saved... chek the data path {e}')
        raise


def main():
    train_data, test_data = load_data('./data/raw/train_data.csv','./data/raw/test_data.csv')
    train_data, test_data = preprocess(train_data, test_data)
    data_path = os.path.join('./data/interim')
    os.makedirs(data_path, exist_ok= True)
    save_data(data_path,train_data, test_data)


if __name__ == '__main__':
    main()

