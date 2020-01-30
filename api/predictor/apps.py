from django.apps import AppConfig
from django.conf import settings
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle

class PredictorConfig(AppConfig):
    # create path to models
    path = os.path.join(settings.MODELS, 'models.p')
 
    # load models into separate variables
    # these will be accessible via this class
    with open(path, 'rb') as pickled:
       data = pickle.load(pickled)
    logistic_regression = data['logistic_regression']
    random_forest = data['random_forest']
    xgboost = data['xgboost']

    dataset = pd.read_csv(os.path.join(settings.DATASETS, 'main_df.csv'))
