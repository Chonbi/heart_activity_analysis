# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:05:34 2020

@author: HRN4
"""

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import random
import json

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, request

app = Flask(__name__)

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"

# Get data and models
with open("models.p", 'rb') as pickled:
   data = pickle.load(pickled)
m_logistic_regression = data['logistic_regression']
m_random_forest = data['random_forest']
m_xgb_classifier = data['xgboost']

main_df = pd.read_csv('main_df.csv')

print(m_logistic_regression)
print(m_random_forest)
print(m_xgb_classifier)
print(main_df)

# Prepare X and y
X = main_df.drop(['activity'], axis=1)
y = main_df['activity']

# Remove useless columns
X = X.drop(['Unnamed: 0', 'gender', 'age', 'height', 'skin', 'sport', 'weight'], axis=1)

# Scale X
X_columns = X.columns
X = pd.DataFrame(preprocessing.scale(X))
X = X.rename(columns={ i : X_columns[i] for i in range(0, len(X_columns) ) })

def getXandYrandomly():
    n = random.randint(0, len(y)-1)
    return X.iloc[n:n+1], y[n:n+1]

def test_model(model):
    content = request.get_json(silent=True, force=True)
    if(content):
        print(content) # Do your processing
        X_test = pd.DataFrame(content['X'])
        
        if('y' in content):
            y_test = pd.Series(content['y'])
        else:
            y_test = None
        
        X_test.columns = X.columns
    else:
        X_test, y_test = getXandYrandomly()
        
    preds = model.predict(X_test)

    response = {
        'Prediction': preds.tolist(),
        'X': X_test.values.tolist()
    }

    if y_test is None:
        print("Accuracy : unknown")
    else:
        accuracy = accuracy_score(y_test, preds)
        print("Accuracy : "+str(accuracy))
        response['Accuracy'] = accuracy
        response['y'] = y_test.values.tolist()
    
    return json.dumps(response, sort_keys=True)

@app.route("/")
def home():
    return "Home<br><a href='/"+LOGISTIC_REGRESSION+"'>/"+LOGISTIC_REGRESSION+"</a><br><a href='/"+RANDOM_FOREST+"'>/"+RANDOM_FOREST+"</a><br><a href='/"+XGBOOST+"'>/"+XGBOOST+"</a>"

@app.route("/"+LOGISTIC_REGRESSION, methods=['GET', 'POST'])
def logistic_regression():
    return test_model(m_logistic_regression)


@app.route("/"+RANDOM_FOREST, methods=['GET', 'POST'])
def random_forest():
    return test_model(m_random_forest)


@app.route("/"+XGBOOST, methods=['GET', 'POST'])
def xgboost():
    return test_model(m_xgb_classifier)


if __name__ == "__main__":
     app.run()
    