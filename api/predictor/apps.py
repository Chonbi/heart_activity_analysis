from django.apps import AppConfig
from django.conf import settings
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle

# libraries required by pickle file
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

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
