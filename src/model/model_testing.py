import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import dagshub
import pickle
import json

import dagshub
dagshub.init(repo_owner='akashkangule18', repo_name='ml-ops-project', mlflow=True)

import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()
# load register model

model_uri = "models:/Random_forest_classifier@Staging"

model = mlflow.sklearn.load_model(model_uri)

sample_data = pd.DataFrame({
    'IQ':[118],
    'CGPA':[3.98],
    '10th_Marks':[59],
    '12th_Marks':[68],
    'Communication_Skills':[2.33]
})

pred = model.predict(sample_data)
print(pred)
