import pandas as pd
import numpy as np
import seaborn as sns
import os
from logger import logger
import yaml


# getting test size
def yaml_file(file_path):
    try:
        test_size = yaml.safe_load(open(file_path,'r'))['data_ingestion']['test_size']
        logger.info('the test size has been return succefully')
        return test_size
    except Exception as e:
        logger.error(f" check the file path of yaml file {e}")
        raise


# getting dataset
# url = 'https://raw.githubusercontent.com/campusx-official/toy-datasets/main/student_performance.csv'
def load_data(url):
    try:
        df = pd.read_csv(url)
        logger.info('data load sucessfully')
        return df
    except Exception as e:
        logger.error(f"please chek the url {e}")
        raise

# save data
def save_data(file_path,train_data,test_data):
    try:
        train_data.to_csv(os.path.join(file_path,'train_data.csv'),index =False)
        test_data.to_csv(os.path.join(file_path,'test_data.csv'),index = False)
        logger.info('train and test data converted in csv  and stored sucessfully')
        return train_data, test_data
    except Exception as e:
        logger.error(f"check the data path again {e}")
        raise

def main():
    test_size = yaml_file('params.yaml')
    df = load_data('https://raw.githubusercontent.com/campusx-official/toy-datasets/main/student_performance.csv')
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(df, random_state = 42, test_size = test_size)
    data_path = os.path.join('data','raw')
    os.makedirs(data_path,exist_ok=True)
    save_data(data_path,train_data, test_data)

if __name__ == "__main__":
    main()
