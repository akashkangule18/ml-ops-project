import pandas as pd
import numpy as np
import os
import seaborn as sns
from logger import logger
import yaml

def load_file(file_path):
    try:

        n_components = yaml.safe_load(open(file_path,'r'))['features']['n_components']
        logger.info('yaml file load succefully')
        return n_components
    except Exception as e:
        logger.error(f"yaml file path is not correct {e}")
        raise


def load_data(training_data_path, testing_data_path):
    try:

        # fetching the interim data
        train_data = pd.read_csv(training_data_path)
        test_data = pd.read_csv(testing_data_path)
        logger.info('data load sucessfully')
        return train_data, test_data
    except Exception as e:
        logger.error(f"check  the training  and testing data path {e}")
        raise

def transform(n_components,train_data, test_data):
    try: 

        # deleting output columns before transformation
        train_data.drop(columns=['Placed'],inplace = True)
        test_data.drop(columns=['Placed'],inplace = True)
        logger.info('output column drop sucessfully')

        # appying transformation
        from sklearn.decomposition import PCA
        pca = PCA(n_components= n_components)
        train_tr = pca.fit_transform(train_data)
        test_tr = pca.transform(test_data)
        logger.info('pca applied sucessfully')

        # making dataframe
        features = pca.get_feature_names_out()
        train_tr_processed = pd.DataFrame(
            train_tr,
            columns = features
        )
        test_tr_processed = pd.DataFrame(
            test_tr,
            columns = features
        )
        logger.info('transformed data converted into dataframe sucessfully')
        return train_tr_processed, test_tr_processed

    except Exception as e:
        logger.error(f'chek the data transformedfunction {e}')
        raise

def save_data(data_path, train_tr_processed, test_tr_processed):
    try:

        # saving the transform data
        train_tr_processed.to_csv(os.path.join(data_path,'train_tr_processed.csv'),index = False)
        test_tr_processed.to_csv(os.path.join(data_path,'test_tr_processed.csv'),index = False)
        logger.info('transformed data saved sucessfully')
    except Exception as e:
        logger.error(f'transformed data is not saved sucessfully{e}')
        raise




def main():
   n_components =  load_file('params.yaml')
   train_data, test_data = load_data('./data/interim/train_interim_data.csv','./data/interim/test_interim_data.csv')
   train_tr_processed, test_tr_processed = transform(n_components,train_data, test_data)
   data_path = os.path.join('data','processed')
   os.makedirs(data_path, exist_ok= True)
   save_data(data_path,train_tr_processed, test_tr_processed)


if __name__ == '__main__':
    main()