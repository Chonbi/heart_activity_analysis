# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:42:25 2020

@author: Hippolyte
"""
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier



""" 
Please execute "PPG_preprocessing.py" before this 
 file if "main_df.csv" does not exist
"""


# Get data
main_df = pd.read_csv("main_df.csv")

# Prepare X and y
X = main_df.drop(['activity'], axis=1)
y = main_df['activity']

# Remove useless columns
X = X.drop(['Unnamed: 0', 'gender', 'age', 'height', 'skin', 'sport', 'weight'], axis=1)

# Scale X
X_columns = X.columns
X = pd.DataFrame(preprocessing.scale(X))
X = X.rename(columns={ i : X_columns[i] for i in range(0, len(X_columns) ) })

# Data visualization
for c in X.columns:
    X_1D = X[c].values.reshape(-1,1)
    plt.title(c)
    plt.scatter(X_1D, y, color='black')
    plt.show()


sns.set(style="whitegrid")
    
for c in main_df.columns:
    sns.boxenplot(x="activity", y=c, color="b", scale="linear", data=main_df)
    plt.show()
    
sns.set(style="whitegrid")
sns.boxenplot(x="subject_ID", y="label",
              color="b",
              scale="linear", data=main_df)
    
# Correlation Matrix
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# OR

cg = sns.clustermap(corr, cmap ="YlGnBu", linewidths = 0.1)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 


# Seperate train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# Logistic Regression model on the whole dataset
regr = linear_model.LogisticRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print("logistic regression : "+str(accuracy_score(y_test, y_pred)))


# XGBoost
# Function used to find the best parameters
def customXGBClassifier(cb, lr, md, a, n, ss):
    xg_reg = XGBClassifier(objective ='reg:linear', 
                       colsample_bytree = cb, learning_rate = lr, 
                       max_depth = md, alpha = a, n_estimators = n, 
                       subsample = ss)
    xg_reg.fit(X_train,y_train)
    preds = xg_reg.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    print("cb: "+str(cb)+", lr: "+str(lr)+", md: "+str(md)+", a: "+str(a)+", n: "+str(n)+", ss: "+str(ss)+". Accuracy:"+str(accuracy))


# Best parameters
xg_reg = XGBClassifier(objective ='reg:linear', 
                       colsample_bytree = 0.4, learning_rate = 0.2, 
                       max_depth = 10, alpha = 10, n_estimators = 150,
                       subsample = 0.7)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print("xgboost classifier : "+str(accuracy))

## Cross Validation
#kfold = StratifiedKFold(n_splits=3, random_state=7)
#results = cross_val_score(xg_reg, X, y, cv=kfold)
#print("CV of xgboost: "+str(results.mean()))

# Random Forest
rfc = RandomForestClassifier(max_depth=50, random_state=7)
rfc.fit(X_train, y_train)
preds = rfc.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print("random forest : "+str(accuracy))


pickl = {
    'logistic_regression': regr,
    'random_forest': rfc,
    'xgboost': xg_reg
}
pickle.dump(pickl, open( 'models' + ".p", "wb" ))